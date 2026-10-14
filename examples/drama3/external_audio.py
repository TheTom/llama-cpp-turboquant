#!/usr/bin/env python3
"""One-stream experimental llama backbone + pinned SGLang-Omni audio runtime."""
import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from prepare_backbone import verified_source


class DenseAttention:
    def __init__(self, heads, kv_heads, head_dim):
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
        self.cache = {}
        self.positions = None

    def bind(self, positions):
        self.positions = positions.tolist()

    def __call__(self, layer, query, key, value):
        cache = self.cache.setdefault(layer, {})
        for row, position in enumerate(self.positions):
            cache[position] = (key[row].reshape(self.kv_heads, self.head_dim).clone(), value[row].reshape(self.kv_heads, self.head_dim).clone())
        result = []
        groups = self.heads // self.kv_heads
        for row, position in enumerate(self.positions):
            visible = [cache[index] for index in sorted(cache) if index <= position]
            keys = torch.stack([pair[0] for pair in visible]).repeat_interleave(groups, dim=1).float()
            values = torch.stack([pair[1] for pair in visible]).repeat_interleave(groups, dim=1).float()
            scores = torch.einsum("hd,nhd->hn", query[row].reshape(self.heads, self.head_dim).float(), keys) / math.sqrt(self.head_dim)
            result.append(torch.einsum("hn,nhd->hd", scores.softmax(-1), values).flatten().to(query.dtype))
        return torch.stack(result)


class Backbone:
    def __init__(self, binary, gguf, gpu_layers, context, config):
        self.process = subprocess.Popen([str(binary), str(gguf), str(gpu_layers), str(context)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        ready = json.loads(self.process.stdout.readline())
        if ready.get("n_embd") != config.text_config.hidden_size or ready.get("n_vocab") != config.text_config.vocab_size:
            self.close()
            raise ValueError("Backbone dimensions do not match Fish source")
        self.candidates = sorted({config.eos_token_id, *range(config.semantic_start_token_id, config.semantic_end_token_id + 1)})

    def token_embeddings(self, token_ids, device, dtype):
        self.process.stdin.write(json.dumps({"token_ids": token_ids.cpu().tolist()}) + "\n")
        self.process.stdin.flush()
        result = json.loads(self.process.stdout.readline())
        return torch.tensor(result["token_embeddings"], device=device, dtype=dtype)

    def forward(self, embeddings, device, dtype):
        request = {"embeddings": embeddings.float().cpu().tolist(), "candidate_ids": self.candidates}
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        result = json.loads(self.process.stdout.readline())
        if result.get("candidate_ids") != self.candidates:
            raise ValueError("Backbone candidate order mismatch")
        hidden = torch.tensor(result["hidden_states"], device=device, dtype=dtype)
        values = torch.tensor(result["candidate_logits"], device=device)
        # The official sampler indexes the original vocabulary IDs.
        logits = torch.full((hidden.shape[0], self.candidates[-1] + 1), -torch.inf, device=device)
        logits[:, self.candidates] = values
        return hidden, logits

    def close(self):
        if self.process.stdin:
            self.process.stdin.close()
        self.process.wait(timeout=30)


def checkpoint_tensors(source, index, predicate):
    for shard in sorted(set(index.values())):
        keys = [name for name, filename in index.items() if filename == shard and predicate(name)]
        if not keys:
            continue
        with safe_open(source / shard, framework="pt", device="cpu") as reader:
            for name in keys:
                yield name, reader.get_tensor(name)


def load_local_codec(source, device):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.dac import modded_dac
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    directory = Path(modded_dac.__file__).parents[2] / "configs"
    codec = instantiate(OmegaConf.load(directory / "modded_dac_vq.yaml"))
    state = torch.load(source / "codec.pth", map_location="cpu", mmap=True, weights_only=True)
    codec.load_state_dict(state, strict=False, assign=True)
    del state
    return codec.eval().to(device)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--verified-source-receipt", required=True, type=Path)
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--omni-source", required=True, type=Path)
    parser.add_argument("--text", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--references", type=Path, help="JSON list of transcript/vq_codes references encoded with the same codec")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--max-frames", type=int, default=1024)
    args = parser.parse_args()
    for variable in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[variable] = "1"
    sys.path.insert(0, str(args.omni_source.resolve()))
    from sglang_omni.models.fishaudio_drama3.audio_decoder import AudioDecoder
    from sglang_omni.models.fishaudio_drama3.hf_config import FishOmniConfig
    from sglang_omni.models.fishaudio_drama3.prompt import ReferencePart, build_prompt_rows
    from sglang_omni.models.fishaudio_drama3.weights import load_audio_decoder_weights
    from tokenizers import Tokenizer
    import soundfile

    source = args.source.resolve(strict=True)
    verified_source(source, args.verified_source_receipt)
    if args.out.exists():
        raise FileExistsError(args.out)
    config = FishOmniConfig(**json.loads((source / "config.json").read_text()))
    index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    for shard in set(index.values()):
        if not (source / shard).is_file():
            raise FileNotFoundError(source / shard)
    device, dtype = torch.device(args.device), torch.bfloat16
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        with torch.device(device):
            decoder = AudioDecoder(config).eval()
    finally:
        torch.set_default_dtype(old_dtype)
    load_audio_decoder_weights(decoder, checkpoint_tensors(source, index, lambda name: name.startswith(("audio_decoder.", "audio_decoder_mtp_shared.", "audio_decoder_mtp."))))
    references = []
    if args.references:
        for descriptor in json.loads(args.references.read_text()):
            references.append(ReferencePart(transcript=descriptor["text"], audio_codes=torch.tensor(descriptor["vq_codes"], dtype=torch.int64)))
    frames = build_prompt_rows(args.text, references, Tokenizer.from_file(str(source / "tokenizer.json")), num_codebooks=decoder.num_codebooks, codebook_size=decoder.codebook_size).to(device)
    if frames.shape[0] + args.max_frames > args.context:
        raise ValueError("Prompt and requested generation exceed context")
    attention = DenseAttention(config.audio_decoder_config.n_head, config.audio_decoder_config.n_local_heads, config.audio_decoder_config.head_dim)
    backbone = Backbone(args.binary, args.gguf, args.gpu_layers, args.context, config)
    try:
        all_hidden = []
        for start in range(0, frames.shape[0], 64):
            chunk = frames[start:start + 64]
            token_embeddings = backbone.token_embeddings(chunk[:, 0], device, dtype)
            hidden, logits = backbone.forward(decoder.embed_backbone(chunk, token_embeddings), device, dtype)
            all_hidden.append(hidden)
        positions = torch.arange(frames.shape[0], device=device)
        labels = torch.cat((frames[1:], frames.new_zeros((1, frames.shape[1]))))
        labels[-1, 0] = config.eos_token_id
        attention.bind(positions)
        decoder.prefill(torch.cat(all_hidden), positions, frames, labels, attention)
        position = frames.shape[0] - 1
        frame = frames[-1:]
        previous = torch.full((1, 16), -1, device=device, dtype=torch.int64)
        generated = []
        for _ in range(args.max_frames):
            pos = torch.tensor([position], device=device)
            attention.bind(pos)
            frame = decoder.decode_step(logits[-1:], hidden[-1:], pos, frame, previous, torch.tensor([1.0], device=device), torch.tensor([30], device=device), torch.tensor([0.9], device=device), attention)
            if int(frame[0, 0]) == config.eos_token_id:
                break
            generated.append(frame[0].cpu())
            previous = torch.cat((previous[:, 1:], frame[:, :1]), dim=-1)
            position += 1
            token_embeddings = backbone.token_embeddings(frame[:, 0], device, dtype)
            hidden, logits = backbone.forward(decoder.embed_backbone(frame, token_embeddings), device, dtype)
        else:
            raise RuntimeError("Frame limit reached without EOS; refusing to report complete speech")
        if not generated:
            raise RuntimeError("No audio frames generated")
        code_rows = torch.stack(generated)
        codec = load_local_codec(source, device)
        audio = codec.from_indices(code_rows[:, 1:].T.unsqueeze(0).to(device))
        soundfile.write(args.out, audio.squeeze().float().cpu().numpy(), codec.sample_rate)
        args.out.with_suffix(".frames.json").write_text(json.dumps(code_rows.tolist()))
    finally:
        backbone.close()


if __name__ == "__main__":
    main()

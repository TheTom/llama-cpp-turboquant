#!/usr/bin/env python3
"""One-stream experimental llama backbone + pinned SGLang-Omni audio runtime."""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

PROCESS_STARTED = time.monotonic()
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


def move_decode_modules(codec, device):
    codec.decoder.to(device)
    for name in ("semantic_quantizer", "quantizer", "post_module", "upsample"):
        getattr(codec.quantizer, name).to(device)
    return codec


def load_local_codec(source, device, decode_only=False):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.dac import modded_dac
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    directory = Path(modded_dac.__file__).parents[2] / "configs"
    codec = instantiate(OmegaConf.load(directory / "modded_dac_vq.yaml"))
    state = torch.load(source / "codec.pth", map_location="cpu", mmap=True, weights_only=True)
    codec.load_state_dict(state, strict=False, assign=True)
    del state
    codec.eval()
    return move_decode_modules(codec, device) if decode_only else codec.to(device)


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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gguf-sha256", help="Previously computed SHA256 of this immutable GGUF; otherwise hash it once here")
    args = parser.parse_args()
    timings = {}
    def stamp():
        if torch.cuda.is_available() and args.device.startswith("cuda"):
            torch.cuda.synchronize()
        return time.monotonic()
    def atomic_json(path, value):
        temporary = path.with_name(path.name + ".partial")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(path)
    torch.manual_seed(args.seed)
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
    model_hash = args.gguf_sha256
    if model_hash is None:
        digest = hashlib.sha256()
        with args.gguf.open("rb") as stream:
            while block := stream.read(8 * 1024 * 1024):
                digest.update(block)
        model_hash = digest.hexdigest()
    if len(model_hash) != 64 or any(c not in "0123456789abcdef" for c in model_hash):
        raise ValueError("Expected a lowercase SHA256 digest")
    load_started = stamp()
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
    timings["model_and_audio_tower_load_seconds"] = stamp() - load_started
    try:
        prefill_started = stamp()
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
        timings["prompt_prefill_seconds"] = stamp() - prefill_started
        position = frames.shape[0] - 1
        frame = frames[-1:]
        previous = torch.full((1, 16), -1, device=device, dtype=torch.int64)
        generated = []
        generation_started = stamp()
        eos = False
        for _ in range(args.max_frames):
            pos = torch.tensor([position], device=device)
            attention.bind(pos)
            frame = decoder.decode_step(logits[-1:], hidden[-1:], pos, frame, previous, torch.tensor([1.0], device=device), torch.tensor([30], device=device), torch.tensor([0.9], device=device), attention)
            if int(frame[0, 0]) == config.eos_token_id:
                eos = True
                break
            generated.append(frame[0].cpu())
            if len(generated) == 1:
                timings["first_audio_code_frame_since_process_seconds"] = stamp() - PROCESS_STARTED
                timings["first_audio_code_frame_since_generation_seconds"] = stamp() - generation_started
            previous = torch.cat((previous[:, 1:], frame[:, :1]), dim=-1)
            position += 1
            token_embeddings = backbone.token_embeddings(frame[:, 0], device, dtype)
            hidden, logits = backbone.forward(decoder.embed_backbone(frame, token_embeddings), device, dtype)
        else:
            raise RuntimeError("Frame limit reached without EOS; refusing to report complete speech")
        if not generated:
            raise RuntimeError("No audio frames generated")
        timings["generation_loop_seconds"] = stamp() - generation_started
        code_rows = torch.stack(generated)
        codec_started = stamp()
        codec = load_local_codec(source, device, decode_only=True)
        timings["codec_load_seconds"] = stamp() - codec_started
        vocode_started = stamp()
        audio = codec.from_indices(code_rows[:, 1:].T.unsqueeze(0).to(device))
        samples = audio.squeeze().float().cpu().numpy()
        timings["vocode_seconds"] = stamp() - vocode_started
        if samples.ndim != 1 or samples.size == 0 or not torch.isfinite(audio).all() or codec.sample_rate <= 0:
            raise RuntimeError("Invalid synthesized waveform")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        partial = args.out.with_name(args.out.stem + ".partial.wav")
        write_started = stamp()
        soundfile.write(partial, samples, codec.sample_rate, format="WAV")
        info = soundfile.info(partial)
        if info.frames != samples.size or info.samplerate != codec.sample_rate:
            raise RuntimeError("Written waveform failed sanity checks")
        partial.replace(args.out)
        timings["wav_write_seconds"] = stamp() - write_started
        timings["first_complete_wav_since_process_seconds"] = stamp() - PROCESS_STARTED
        duration = samples.size / codec.sample_rate
        timings["process_to_wav_seconds"] = stamp() - PROCESS_STARTED
        timings["generation_rtf"] = timings["generation_loop_seconds"] / duration
        timings["total_rtf"] = timings["process_to_wav_seconds"] / duration
        atomic_json(args.out.with_suffix(".frames.json"), code_rows.tolist())
        report = {"source_revision": source.name, "gguf_sha256": model_hash, "prompt": args.text,
                  "eos": eos, "frames": len(generated), "audio_seconds": duration, "sample_rate": codec.sample_rate,
                  "timings": timings, "timing_scope": "monotonic Python process entry through complete WAV; code-frame latency is not audible TTFA",
                  "cold_warm": "new backbone process and audio tower/codec per invocation; filesystem cache state uncontrolled",
                  "settings": {"seed": args.seed, "context": args.context, "max_frames": args.max_frames, "temperature": 1.0, "top_k": 30, "top_p": 0.9, "native_microbatch": 8},
                  "hardware": {"device": str(device), "name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU", "torch": torch.__version__},
                  "backend": {"backbone": "TheTom/llama-cpp-turboquant", "gpu_layers": args.gpu_layers, "audio": "SGLang-Omni PyTorch DenseAttention bridge", "codec": "local source DAC F32 decode-only modules"}}
        atomic_json(args.out.with_suffix(".run.json"), report)
        atomic_json(args.out.with_suffix(".complete.json"), {"wav": args.out.name, "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(), "eos": True, "audio_seconds": duration})
    finally:
        backbone.close()


if __name__ == "__main__":
    main()

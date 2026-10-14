#!/usr/bin/env python3
"""Encode one local voice clip with the same local codec used for synthesis."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from prepare_backbone import verified_source
from external_audio import load_local_codec


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--verified-source-receipt", required=True, type=Path)
    parser.add_argument("--omni-source", required=True, type=Path)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--text", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    for variable in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[variable] = "1"
    verified_source(args.source, args.verified_source_receipt)
    if args.out.exists():
        raise FileExistsError(args.out)
    sys.path.insert(0, str(args.omni_source.resolve()))
    import soundfile
    import torchaudio
    audio, sample_rate = soundfile.read(args.audio, dtype="float32", always_2d=True)
    if audio.shape[0] == 0 or audio.shape[0] > sample_rate * 60:
        raise ValueError("Reference audio must last more than zero and at most 60 seconds")
    device = torch.device(args.device)
    codec = load_local_codec(args.source, device)
    waveform = torch.from_numpy(audio.mean(axis=1)).unsqueeze(0)
    waveform = torchaudio.functional.resample(waveform, sample_rate, codec.sample_rate)
    lengths = torch.tensor([waveform.shape[-1]], device=device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        codes, code_lengths = codec.encode(waveform.to(device), lengths)
    codes = codes[0, :, :int(code_lengths[0])].cpu().tolist()
    reference = {"text": args.text, "vq_codes": codes,
                 "source_audio_sha256": hashlib.sha256(args.audio.read_bytes()).hexdigest(),
                 "codec_sha256": hashlib.sha256((args.source / "codec.pth").read_bytes()).hexdigest(),
                 "encode_dtype": "bfloat16_autocast" if device.type == "cuda" else "float32"}
    args.out.write_text(json.dumps([reference], ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

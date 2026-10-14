#!/usr/bin/env python3
"""Stage a local Fish backbone for conversion without copying weights."""
import argparse
import json
import re
from pathlib import Path


def verified_source(source: Path, receipt: Path, expected_revision: str | None = None):
    source = source.resolve(strict=True)
    verification = json.loads(receipt.read_text())
    if Path(verification["source"]).resolve() != source:
        raise ValueError("Verified source receipt does not name this checkpoint")
    hashes = verification.get("sha256")
    revision = verification.get("revision", "")
    if not isinstance(hashes, dict) or verification.get("file_count") != len(hashes) or not re.fullmatch("[0-9a-f]{40}", revision):
        raise ValueError("Invalid verified source receipt")
    if expected_revision is not None and revision != expected_revision:
        raise ValueError("Candidate and verified source revisions differ")
    total = 0
    for name, digest in hashes.items():
        if Path(name).name != name or not re.fullmatch("[0-9a-f]{64}", digest):
            raise ValueError("Invalid verified source filename or digest")
        total += (source / name).stat().st_size
    if total != verification.get("total_bytes"):
        raise ValueError("Verified source file sizes changed")
    return verification


def stage(source: Path, destination: Path, receipt: Path):
    verified_source(source, receipt)
    source = source.resolve(strict=True)
    config = json.loads((source / "config.json").read_text())
    if config.get("architectures") != ["FishOmniModel"] or config.get("model_type") != "fish_omni":
        raise ValueError("Expected a FishOmniModel checkpoint")
    quant = config.get("quantization_config", {})
    if quant.get("quant_method") != "bsgemm-moe" or quant.get("weight_block_size") != [128, 128]:
        raise ValueError("Only the published block-scaled FP8 Drama-3 source is supported")
    index = json.loads((source / "model.safetensors.index.json").read_text())
    files = set(index["weight_map"].values())
    for name in files:
        if Path(name).name != name or not (source / name).is_file():
            raise ValueError(f"Missing source shard or invalid filename: {name}")
    required = ("tokenizer.json", "tokenizer_config.json")
    for name in required:
        if not (source / name).is_file():
            raise ValueError(f"Missing source tokenizer: {name}")
    destination.mkdir(parents=True, exist_ok=False)
    config["architectures"] = ["FishOmniBackboneForCausalLM"]
    config["quantization_config"] = {**quant, "quant_method": "fp8"}
    config["drama3_export_scope"] = "backbone_only_external_audio_runtime_required"
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for name in [*sorted(files), "model.safetensors.index.json", *required, "chat_template.jinja", "LICENSE"]:
        if (source / name).is_file():
            (destination / name).symlink_to(source / name)
    print(f"Staged backbone metadata and source links at {destination}; no audio runtime is included")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--verified-source-receipt", required=True, type=Path)
    args = parser.parse_args()
    stage(args.source, args.destination, args.verified_source_receipt)

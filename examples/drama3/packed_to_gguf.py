#!/usr/bin/env python3
"""Import C0 packed expert bytes directly; never requantize them."""
import argparse
import hashlib
import json
import os
import re
import sys
from contextlib import ExitStack
from pathlib import Path

if __name__ == "__main__":
    for variable in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[variable] = "1"

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
from conversion.base import gguf
from conversion.fish_omni import FishOmniBackboneModel
from prepare_backbone import stage, verified_source

PACKED = {"tq4_1s": gguf.GGMLQuantizationType.TQ4_1S, "q4_k": gguf.GGMLQuantizationType.Q4_K}
EXPERT = re.compile(r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verified-source-receipt", required=True, type=Path)
    parser.add_argument("--staged-metadata", required=True, type=Path)
    parser.add_argument("--temp-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(args.temp_dir.resolve())
    import tempfile
    tempfile.tempdir = str(args.temp_dir.resolve())
    stage(args.source, args.staged_metadata, args.verified_source_receipt)
    manifest = json.loads((args.candidate / "packed-manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("candidate") != "C0" or manifest.get("model_type") != "fish_omni":
        raise ValueError("Only the version-1 C0 Fish packed manifest is supported")
    verified_source(args.source, args.verified_source_receipt, manifest["source_revision"])
    entries = manifest["entries"]
    if not isinstance(entries, list):
        raise ValueError("Expected packed manifest entries array")
    entry_map = {entry["name"]: entry for entry in entries}
    if len(entry_map) != len(entries):
        raise ValueError("Duplicate packed entry names")
    source_index = json.loads((args.source / "model.safetensors.index.json").read_text())["weight_map"]
    removed_scales = {entry["source_scale_tensor"] for entry in entries if entry["format"] in PACKED}
    if set(entry_map) != set(source_index) - removed_scales:
        raise ValueError("Packed candidate and source tensor sets differ")
    for shard in manifest["shards"]:
        filename = shard["file"]
        if Path(filename).name != filename:
            raise ValueError("Invalid candidate shard filename")
        path = args.candidate / filename
        if path.stat().st_size != shard["bytes"]:
            raise ValueError("Candidate shard size differs from its receipt")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(8 * 1024 * 1024):
                digest.update(block)
        if digest.hexdigest() != shard["sha256"]:
            raise ValueError("Candidate shard hash differs from its receipt")
    model = FishOmniBackboneModel(args.staged_metadata, gguf.LlamaFileType.MOSTLY_BF16, args.output, use_temp_file=True, model_name="Drama-3-backbone-C0")
    writer = model.gguf_writer
    groups = {}
    for entry in entries:
        if entry["format"] in PACKED:
            match = EXPERT.fullmatch(entry["name"])
            if not match or not entry.get("source_scale_consumed"):
                raise ValueError("Packed import supports only scaled routed expert weights")
            layer, expert, projection = int(match[1]), int(match[2]), match[3]
            expected = "q4_k" if projection == "down_proj" else "tq4_1s"
            if not 2 <= layer < model.block_count - 2 or entry["format"] != expected:
                raise ValueError("Packed entry does not match the protected C0 policy")
            groups.setdefault((layer, projection), {})[expert] = entry
    if set(groups) != {(layer, projection) for layer in range(2, model.block_count - 2) for projection in ("gate_proj", "up_proj", "down_proj")}:
        raise ValueError("Missing packed interior expert group")
    receipts = []
    pool = ExitStack()
    readers = {shard: pool.enter_context(safe_open(args.candidate / shard, framework="pt", device="cpu")) for shard in {entry["shard"] for entry in entries}}

    def tensor(entry):
        value = readers[entry["shard"]].get_tensor(entry["name"])
        if list(value.shape) != entry["storage_shape"]:
            raise ValueError(f"Storage shape mismatch: {entry['name']}")
        return value

    for (layer, projection), experts in sorted(groups.items()):
        if set(experts) != set(range(model.hparams["num_experts"])):
            raise ValueError("Incomplete routed expert group")
        values = [tensor(experts[index]) for index in range(len(experts))]
        if any(value.dtype != torch.uint8 for value in values):
            raise ValueError("Packed expert must have U8 storage")
        encoded = torch.stack(values).numpy()
        suffix = {"gate_proj": "ffn_gate_exps", "up_proj": "ffn_up_exps", "down_proj": "ffn_down_exps"}[projection]
        name = f"blk.{layer}.{suffix}.weight"
        fmt = PACKED[experts[0]["format"]]
        expected_shape = [len(experts), *experts[0]["original_shape"]]
        if list(gguf.quant_shape_from_byte_shape(encoded.shape, fmt)) != expected_shape:
            raise ValueError("Packed logical shape mismatch")
        writer.add_tensor(name, encoded, raw_dtype=fmt)
        receipts.append({"name": name, "shape": expected_shape, "type": fmt.name, "packed_sha256": hashlib.sha256(encoded).hexdigest()})
        del encoded, values
    consumed = {entry["source_scale_tensor"] for entry in entries if entry["format"] == "fp8_e4m3"}
    for entry in entries:
        original_name = entry["name"]
        if entry["format"] in PACKED or original_name in consumed:
            continue
        routed = FishOmniBackboneModel.filter_tensors((original_name, lambda: None))
        if routed is None:
            continue
        name = routed[0]
        value = tensor(entry)
        if entry["format"] == "fp8_e4m3":
            scale = tensor(entry_map[entry["source_scale_tensor"]]).float()
            if value.dtype != torch.float8_e4m3fn or not torch.isfinite(scale).all() or not (scale > 0).all():
                raise ValueError("Invalid source FP8 weight/scale pair")
            rows, cols = value.shape
            if list(scale.shape) != [(rows + 127) // 128, (cols + 127) // 128]:
                raise ValueError("Source FP8 block geometry mismatch")
            value = value.float() * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)[:rows, :cols]
        elif entry["format"] not in ("bf16", "f32"):
            raise ValueError(f"Unsupported protected format: {entry['format']}")
        value = value.float()
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite source tensor: {original_name}")
        match = re.search(r"\.layers\.(\d+)\.", name)
        block = int(match[1]) if match else None
        for mapped, transformed in model.modify_tensors(value, name, block):
            data = transformed.numpy()
            keep_float = entry["format"] == "f32" or data.ndim == 1 or mapped.endswith("_norm.weight") or mapped.endswith((".ffn_gate_inp.weight", ".ffn_gate_inp_shexp.weight", ".ssm_conv1d.weight"))
            output_type = gguf.GGMLQuantizationType.F32 if keep_float else gguf.GGMLQuantizationType.BF16
            writer.add_tensor(mapped, gguf.quantize(data, output_type), raw_dtype=output_type)
    if model._experts is not None and any(model._experts):
        raise ValueError("Incomplete protected expert group")
    pool.close()
    model.prepare_metadata(vocab_only=False)
    writer.write_header_to_file(args.output)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    reader = gguf.GGUFReader(args.output)
    actual = {item.name: item for item in reader.tensors}
    for receipt in receipts:
        item = actual[receipt["name"]]
        if item.tensor_type.name != receipt["type"] or hashlib.sha256(item.data).hexdigest() != receipt["packed_sha256"]:
            raise ValueError(f"Packed round-trip bytes differ: {receipt['name']}")
    args.output.with_suffix(".import.json").write_text(json.dumps({"scope": "backbone_only", "packed_byte_roundtrip": receipts, "bytes": args.output.stat().st_size, "protected_expert_runtime_format": "BF16 expanded from original FP8 blocks; no Q8 requantization", "speech_quality": "pending"}, indent=2) + "\n")


if __name__ == "__main__":
    main()

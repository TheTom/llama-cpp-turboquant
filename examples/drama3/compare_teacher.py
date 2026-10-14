#!/usr/bin/env python3
"""Replay captured official mixed embeddings; report error without declaring parity."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from external_audio import Backbone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--gpu-layers", type=int, default=0)
    args = parser.parse_args()
    teacher = torch.load(args.teacher, map_location="cpu", weights_only=True)
    embeddings = teacher["input_embeddings"].float()
    positions = teacher["positions"].flatten()
    if not torch.equal(positions, torch.arange(embeddings.shape[0])):
        raise ValueError("Only a single contiguous prompt starting at zero is supported")
    raw = json.loads((args.source / "config.json").read_text())
    config = SimpleNamespace(**raw)
    config.text_config = SimpleNamespace(**raw["text_config"])
    backbone = Backbone(args.binary, args.gguf, args.gpu_layers, max(256, embeddings.shape[0] + 16), config)
    hidden = []
    try:
        for start in range(0, len(embeddings), 64):
            result, logits = backbone.forward(embeddings[start:start + 64], torch.device("cpu"), torch.float32)
            hidden.append(result)
    finally:
        backbone.close()
    hidden = torch.cat(hidden)
    expected = teacher["hidden_states"].float()
    reference = teacher["last_logits"][-1, backbone.candidates].float()
    actual = logits[-1, backbone.candidates]
    if hidden.shape != expected.shape or not torch.isfinite(hidden).all() or not torch.isfinite(actual).all():
        raise ValueError("Invalid replay shape or nonfinite native output")
    def errors(actual, reference):
        return {"relative_l2": float(torch.linalg.vector_norm(actual - reference) / torch.linalg.vector_norm(reference)),
                "rmse": float((actual - reference).square().mean().sqrt()),
                "max_absolute": float((actual - reference).abs().max()),
                "cosine": float(torch.nn.functional.cosine_similarity(actual.flatten(), reference.flatten(), dim=0))}
    top = min(10, len(reference))
    report = {"scope": "Candidate backend and precision difference against captured official source forward; not isolated quantization error",
              "reference_commit": teacher["reference_commit"], "rows": len(hidden), "hidden": errors(hidden, expected),
              "semantic_logits": errors(actual, reference),
              "semantic_top1_same": int(actual.argmax()) == int(reference.argmax()),
              "semantic_top10_overlap": len(set(actual.topk(top).indices.tolist()) & set(reference.topk(top).indices.tolist())),
              "teacher": str(args.teacher), "gguf": str(args.gguf), "gpu_layers": args.gpu_layers}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

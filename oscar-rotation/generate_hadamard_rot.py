#!/usr/bin/env python3
"""Generate data-free Hadamard rotation matrices for any model.

Output matches the format expected by export_rot_kv_gguf.py:
  {
    "format_version": 1,
    "objective": "hadamard",
    "source_grouping": "layer",
    "layers": {
        <layer_id>: {
            "layer_id": int,
            "rotation": Tensor[head_dim, head_dim] (orthogonal),
            "eigenvalues": Tensor[head_dim] (all ones for Hadamard),
        }
    }
  }
"""

import argparse
import os
import torch


def hadamard_matrix(n: int) -> torch.Tensor:
    """Generate a normalized Hadamard matrix of order n (power of 2)."""
    h = torch.tensor([[1]], dtype=torch.float32)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1),
                       torch.cat([h, -h], dim=1)], dim=0)
    # Normalize so H^T H = I
    return h / torch.sqrt(torch.tensor(n, dtype=torch.float32))


def main():
    ap = argparse.ArgumentParser(description="Generate Hadamard rotation .pt files")
    ap.add_argument("--head-dim", type=int, required=True,
                    help="Head dimension (must be power of 2)")
    ap.add_argument("--num-layers", type=int, required=True,
                    help="Number of transformer layers")
    ap.add_argument("--output-dir", default=".",
                    help="Output directory for .pt files")
    ap.add_argument("--prefix", default="",
                    help="Optional prefix for filenames (e.g. qwen3-30b-a3b-)")
    args = ap.parse_args()

    hd = args.head_dim
    # Verify power of 2
    assert hd & (hd - 1) == 0, f"head_dim={hd} must be power of 2"

    os.makedirs(args.output_dir, exist_ok=True)

    # Generate K rotation (same Hadamard for all layers)
    k_rot = hadamard_matrix(hd)
    v_rot = hadamard_matrix(hd)

    # Build checkpoint dicts
    k_checkpoint = {
        "format_version": 1,
        "objective": "hadamard",
        "source_grouping": "layer",
        "layers": {},
    }
    v_checkpoint = {
        "format_version": 1,
        "objective": "hadamard",
        "source_grouping": "layer",
        "layers": {},
    }
    eigenvalues = torch.ones(hd, dtype=torch.float32)

    for layer_id in range(args.num_layers):
        k_checkpoint["layers"][layer_id] = {
            "layer_id": layer_id,
            "rotation": k_rot.clone(),
            "eigenvalues": eigenvalues.clone(),
        }
        v_checkpoint["layers"][layer_id] = {
            "layer_id": layer_id,
            "rotation": v_rot.clone(),
            "eigenvalues": eigenvalues.clone(),
        }

    prefix = args.prefix
    k_path = os.path.join(args.output_dir, f"{prefix}k_rotation_hadamard.pt")
    v_path = os.path.join(args.output_dir, f"{prefix}v_rotation_hadamard.pt")

    torch.save(k_checkpoint, k_path)
    torch.save(v_checkpoint, v_path)

    print(f"K rotation: {k_path} ({os.path.getsize(k_path)/1e6:.1f} MB)")
    print(f"V rotation: {v_path} ({os.path.getsize(v_path)/1e6:.1f} MB)")
    print(f"  {args.num_layers} layers, {hd}x{hd} Hadamard matrices")


if __name__ == "__main__":
    main()

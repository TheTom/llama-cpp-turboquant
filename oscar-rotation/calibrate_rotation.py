#!/usr/bin/env python3
"""OSCAR calibrated rotation pipeline (G1 + G3).

Reads covariance binaries dumped by llama-oscar-calib, eigendecomposes them
to produce per-layer rotation matrices R_K and R_V, then composes them with
Hadamard (H) and bit-reversal permutation (P_br) to produce the final
R·H·P_br rotation matrices for K and V.

Usage:
  # Step 1: Run the C++ calibration tool
  ./llama-oscar-calib -m model.gguf -f calibration.txt -o covariances/

  # Step 2: Run this script to compute rotations
  python3 calibrate_rotation.py \\
      --cov-dir covariances/ \\
      --head-dim 128 \\
      --num-layers 28 \\
      --output-dir rotations/ \\
      --composition r_h_pbr

  # Step 3: Bake into GGUF
  python3 export_rot_kv_gguf.py \\
      --base model.gguf \\
      --rot-dir rotations/ \\
      --out model-rot.gguf

The output .pt files are compatible with export_rot_kv_gguf.py and
generate_and_bake_rot.py.
"""

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Hadamard matrix (power of 2 only)
# ---------------------------------------------------------------------------
def hadamard_matrix(n: int) -> torch.Tensor:
    """Normalized Hadamard matrix H such that H^T H = I."""
    assert n & (n - 1) == 0, f"head_dim={n} must be a power of 2"
    h = torch.tensor([[1.0]], dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / math.sqrt(n)).float()


# ---------------------------------------------------------------------------
# Bit-reversal permutation matrix
# ---------------------------------------------------------------------------
def bit_reversal_perm(n: int) -> torch.Tensor:
    """Permutation matrix P_br where row i has a 1 at column bit_reverse(i).

    For n=128 (7 bits): P_br[i, bit_reverse_7(i)] = 1.
    """
    assert n & (n - 1) == 0, f"head_dim={n} must be a power of 2"
    bits = int(math.log2(n))
    perm = torch.zeros(n, n, dtype=torch.float32)
    for i in range(n):
        rev = 0
        x = i
        for _ in range(bits):
            rev = (rev << 1) | (x & 1)
            x >>= 1
        perm[i, rev] = 1.0
    return perm


# ---------------------------------------------------------------------------
# Load covariance binaries from llama-oscar-calib output
# ---------------------------------------------------------------------------
def load_covariances(cov_dir: str, num_layers: int, head_dim: int) -> tuple:
    """Load per-layer Q and V covariance matrices.

    Returns (q_covs, v_covs) each of shape [num_layers, head_dim, head_dim].
    """
    q_covs = []
    v_covs = []
    for layer in range(num_layers):
        q_path = os.path.join(cov_dir, f"layer_{layer:02d}_qcov.bin")
        v_path = os.path.join(cov_dir, f"layer_{layer:02d}_vcov.bin")

        if not os.path.exists(q_path):
            raise FileNotFoundError(f"Missing Q covariance: {q_path}")
        if not os.path.exists(v_path):
            raise FileNotFoundError(f"Missing V covariance: {v_path}")

        q_cov = np.fromfile(q_path, dtype=np.float32).reshape(head_dim, head_dim)
        v_cov = np.fromfile(v_path, dtype=np.float32).reshape(head_dim, head_dim)

        # Symmetrize (should already be symmetric, but guard against FP drift)
        q_cov = (q_cov + q_cov.T) / 2.0
        v_cov = (v_cov + v_cov.T) / 2.0

        q_covs.append(torch.from_numpy(q_cov))
        v_covs.append(torch.from_numpy(v_cov))

    return torch.stack(q_covs), torch.stack(v_covs)


# ---------------------------------------------------------------------------
# Eigendecomposition to get rotation matrices
# ---------------------------------------------------------------------------
def compute_rotation(cov: torch.Tensor) -> torch.Tensor:
    """Compute rotation R = U from eigendecomposition of covariance.

    cov is [d, d] symmetric PSD. Returns R = U (eigenvectors) as [d, d].
    R is orthogonal: R^T R = I.
    """
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    # eigh returns eigenvalues in ascending order. Sort by descending variance
    # so the first eigenvector captures the most variance.
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvectors


# ---------------------------------------------------------------------------
# Composition: R · H · P_br (G3)
# ---------------------------------------------------------------------------
def compose_rotation(R: torch.Tensor, H: torch.Tensor, P_br: torch.Tensor,
                     composition: str) -> torch.Tensor:
    """Compose rotation matrix with Hadamard and bit-reversal.

    Args:
        R: Rotation matrix from eigendecomposition [d, d].
        H: Hadamard matrix [d, d].
        P_br: Bit-reversal permutation matrix [d, d].
        composition: One of 'r', 'r_h', 'r_pbr', 'r_h_pbr', 'h_pbr', 'h', 'pbr'.

    Returns:
        Composed rotation matrix [d, d].
    """
    if composition == "r":
        return R
    elif composition == "r_h":
        return R @ H
    elif composition == "r_pbr":
        return R @ P_br
    elif composition == "r_h_pbr":
        return R @ H @ P_br
    elif composition == "h_pbr":
        return H @ P_br
    elif composition == "h":
        return H
    elif composition == "pbr":
        return P_br
    else:
        raise ValueError(f"Unknown composition: {composition}")


# ---------------------------------------------------------------------------
# Save rotation checkpoint (compatible with export_rot_kv_gguf.py format)
# ---------------------------------------------------------------------------
def save_rotation(rotations: torch.Tensor, eigenvalues_list: list,
                  output_path: str, objective: str):
    """Save rotations in the .pt format expected by export_rot_kv_gguf.py.

    Args:
        rotations: [num_layers, d, d] rotation matrices.
        eigenvalues_list: list of [d] eigenvalue tensors per layer.
        output_path: Path to save the .pt file.
        objective: Description string (e.g. 'qqt_sst_r_h_pbr').
    """
    checkpoint = {
        "format_version": 1,
        "objective": objective,
        "source_grouping": "layer",
        "layers": {},
    }
    for il in range(rotations.shape[0]):
        checkpoint["layers"][il] = {
            "layer_id": il,
            "rotation": rotations[il],
            "eigenvalues": eigenvalues_list[il],
        }
    torch.save(checkpoint, output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Compute calibrated OSCAR rotations from covariance matrices")
    ap.add_argument("--cov-dir", required=True,
                    help="Directory with layer_NN_qcov.bin / layer_NN_vcov.bin from llama-oscar-calib")
    ap.add_argument("--head-dim", type=int, required=True,
                    help="Head dimension (must be power of 2)")
    ap.add_argument("--num-layers", type=int, required=True,
                    help="Number of transformer layers")
    ap.add_argument("--output-dir", default="rotations",
                    help="Output directory for .pt files")
    ap.add_argument("--composition", default="r_h_pbr",
                    choices=["r", "r_h", "r_pbr", "r_h_pbr", "h_pbr", "h", "pbr"],
                    help="Composition of rotation with H and P_br (default: r_h_pbr)")
    ap.add_argument("--prefix", default="",
                    help="Optional prefix for output filenames")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    d = args.head_dim

    print(f"Loading covariances from {args.cov_dir}...")
    q_covs, v_covs = load_covariances(args.cov_dir, args.num_layers, d)
    print(f"  Loaded {args.num_layers} layers, head_dim={d}")

    # Build H and P_br matrices.
    H = hadamard_matrix(d)
    P_br = bit_reversal_perm(d)
    print(f"  Hadamard {d}x{d} orthogonality error: {(H @ H.T - torch.eye(d)).abs().max():.2e}")
    print(f"  P_br is permutation: {torch.allclose(P_br @ P_br.T, torch.eye(d))}")

    # Eigendecompose per layer.
    print(f"\nComputing rotations (composition={args.composition})...")
    k_rotations = []
    v_rotations = []
    k_eigenvalues = []
    v_eigenvalues = []

    for layer in range(args.num_layers):
        # K rotation: R_K = eigendecompose(Q^T Q)
        R_K = compute_rotation(q_covs[layer])
        k_eig = torch.linalg.eigvalsh(q_covs[layer])
        k_eig_sorted = k_eig.flip(0)  # descending

        # V rotation: R_V = eigendecompose(V^T V)
        R_V = compute_rotation(v_covs[layer])
        v_eig = torch.linalg.eigvalsh(v_covs[layer])
        v_eig_sorted = v_eig.flip(0)

        # Compose with H and P_br.
        final_K = compose_rotation(R_K, H, P_br, args.composition)
        final_V = compose_rotation(R_V, H, P_br, args.composition)

        # Orthogonality check.
        k_err = (final_K @ final_K.T - torch.eye(d)).abs().max().item()
        v_err = (final_V @ final_V.T - torch.eye(d)).abs().max().item()
        if layer == 0:
            print(f"  Layer {layer}: K orth error={k_err:.2e}, V orth error={v_err:.2e}")

        k_rotations.append(final_K)
        v_rotations.append(final_V)
        k_eigenvalues.append(k_eig_sorted)
        v_eigenvalues.append(v_eig_sorted)

    k_rotations = torch.stack(k_rotations)
    v_rotations = torch.stack(v_rotations)

    # Save in the format expected by export_rot_kv_gguf.py.
    prefix = args.prefix
    comp = args.composition

    k_path = os.path.join(args.output_dir, f"{prefix}k_rotation_qqt_{comp}.pt")
    v_path = os.path.join(args.output_dir, f"{prefix}v_rotation_sst_{comp}.pt")

    save_rotation(k_rotations, k_eigenvalues, k_path, f"qqt_{comp}")
    save_rotation(v_rotations, v_eigenvalues, v_path, f"sst_{comp}")

    # Also save as the canonical names export_rot_kv_gguf.py expects.
    k_canonical = os.path.join(args.output_dir, "k_rotation_qqt_r_h_pbr.pt")
    v_canonical = os.path.join(args.output_dir, "v_rotation_sst_r_h_pbr.pt")
    if k_path != k_canonical:
        torch.save(torch.load(k_path, weights_only=False), k_canonical)
        torch.save(torch.load(v_path, weights_only=False), v_canonical)

    print(f"\nSaved K rotation: {k_path} ({os.path.getsize(k_path)/1e6:.1f} MB)")
    print(f"Saved V rotation: {v_path} ({os.path.getsize(v_path)/1e6:.1f} MB)")
    print(f"  {args.num_layers} layers, {d}x{d} matrices, composition={comp}")
    print(f"\nNext step: python3 export_rot_kv_gguf.py --base model.gguf --rot-dir {args.output_dir} --out model-rot.gguf")


if __name__ == "__main__":
    main()

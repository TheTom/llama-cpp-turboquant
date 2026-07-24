#!/usr/bin/env python3
"""OSCAR calibrated rotation pipeline (G1 + G3).

Reads covariance binaries dumped by llama-oscar-calib, eigendecomposes them
to produce per-layer rotation matrices R_K and R_V, then composes them with
Hadamard (H) and bit-reversal permutation (P_br) to produce the final
R·H·P_br rotation matrices for K and V.

The output .pt files are compatible with export_rot_kv_gguf.py and
generate_and_bake_rot.py.

==========================================================================
END-TO-END USAGE (3 methods)
==========================================================================

Method A: One-command pipeline (recommended)
--------------------------------------------
  python3 oscar-rotation/generate_and_bake_rot.py \
      --base model.gguf --out model-rot.gguf

  Uses data-free Hadamard rotation. No calibration needed.
  Works for any model with power-of-2 head_dim (64, 128, 256, 512).

Method B: One-command calibrated pipeline
------------------------------------------
  python3 oscar-rotation/generate_and_bake_rot.py \
      --base model.gguf --out model-rot.gguf \
      --method calibrated --dump-path calibration.txt

  Requires llama-oscar-calib to be built first:
    cmake -B build && cmake --build build --target llama-oscar-calib

  --dump-path can be a text file (used as calibration prompt) or a
  directory of QKV dumps from the legacy OSCAR paper pipeline.

Method C: Step-by-step (for advanced use / debugging)
------------------------------------------------------
  # Step 1: Build the calibration tool
  cmake -B build && cmake --build build --target llama-oscar-calib

  # Step 2: Dump Q/V covariances from model activations
  ./build/bin/llama-oscar-calib -m model.gguf -f calibration.txt -o covariances/

  # Step 3: Eigendecompose and compose R·H·P_br
  python3 oscar-rotation/calibrate_rotation.py \
      --cov-dir covariances/ \
      --head-dim 128 --num-layers 28 \
      --output-dir rotations/ --composition r_h_pbr

  # Step 4a: Bake rotations into GGUF (adds attn_k_rot + attn_v_rot tensors)
  python3 oscar-rotation/export_rot_kv_gguf.py \
      --base model.gguf --rot-dir rotations/ --out model-rot.gguf

  # Step 4b: OR bake with V rotation absorbed into W_o (G5, zero runtime cost)
  python3 oscar-rotation/export_rot_kv_gguf.py \
      --base model.gguf --rot-dir rotations/ --out model-rot.gguf --absorb-v

==========================================================================
RUNNING THE MODEL WITH ROTATIONS
==========================================================================

  # With rotation tensors (attn_k_rot + attn_v_rot baked in):
  llama-cli -m model-rot.gguf --flash-attn on \
      --cache-type-k oscar2 --cache-type-v oscar2 \
      -p "prompt" -n 100

  # With absorbed V rotation (--absorb-v was used, no attn_v_rot):
  llama-cli -m model-rot.gguf --flash-attn on \
      --cache-type-k oscar2 --cache-type-v oscar2 \
      -p "prompt" -n 100

==========================================================================
COMPOSITION MODES (--composition)
==========================================================================

  r_h_pbr  R · H · P_br  (default, best quality per OSCAR paper)
  r_h      R · H          (rotation + Hadamard, no bit-reversal)
  r_pbr    R · P_br       (rotation + bit-reversal, no Hadamard)
  r        R only          (pure spectral rotation)
  h_pbr    H · P_br       (data-free Hadamard + bit-reversal)
  h        H only          (data-free Hadamard)
  pbr      P_br only       (bit-reversal permutation only)
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
# G7: Uresidual mode - refine rotation via quantization error alignment
# ---------------------------------------------------------------------------
def generate_synthetic_activations(cov: torch.Tensor, n_samples: int = 4096) -> torch.Tensor:
    """Generate synthetic activations matching a covariance matrix.

    Uses Cholesky decomposition: X = Z @ L^T where L = cholesky(cov)
    and Z ~ N(0, I). Returns [n_samples, d].
    """
    d = cov.shape[0]
    # Regularize to ensure positive-definite for Cholesky.
    cov_reg = cov + 1e-6 * torch.eye(d, dtype=cov.dtype)
    L = torch.linalg.cholesky(cov_reg)
    Z = torch.randn(n_samples, d, dtype=cov.dtype)
    return Z @ L.T


def simulate_int2_quantize(x: torch.Tensor) -> torch.Tensor:
    """Simulate oscar2 INT2 quantization and dequantization.

    Per-block (row-wise) quantization with 4 centroids:
        scale = (max - min) / 3
        codes = clamp(round((x - min) / scale), 0, 3)
        x_dequant = codes * scale + min

    Args:
        x: [n_samples, d] activations.

    Returns:
        x_dequant: [n_samples, d] dequantized activations.
    """
    # Per-row min/max (matches oscar2 block quantization).
    row_min = x.min(dim=1, keepdim=True).values
    row_max = x.max(dim=1, keepdim=True).values
    scale = (row_max - row_min) / 3.0
    # Avoid division by zero for constant rows.
    scale = scale.clamp(min=1e-10)
    codes = torch.clamp(torch.round((x - row_min) / scale), 0, 3)
    return codes * scale + row_min


def compute_uresidual_rotation(
    cov: torch.Tensor,
    R_ref: torch.Tensor,
    n_samples: int = 4096,
    device: str = "cpu",
) -> torch.Tensor:
    """Compute one uresidual refinement step.

    1. Generate synthetic activations from cov.
    2. Apply R_ref: x_rot = x @ R_ref^T
    3. Simulate INT2 quantize/dequantize.
    4. Compute error covariance C_E = E^T E / N.
    5. Eigendecompose C_E (error directions, descending).
    6. Project cov into rotated space and eigendecompose (target, ascending).
    7. R_resid = U_target @ U_error^T  (map largest errors to least important dims).
    8. Return R_resid @ R_ref.

    Args:
        cov: [d, d] original covariance (Q^T Q or V^T V).
        R_ref: [d, d] current reference rotation (orthogonal).
        n_samples: number of synthetic samples.

    Returns:
        R_new: [d, d] refined rotation.
    """
    d = cov.shape[0]

    # Step 1: Generate synthetic activations.
    x = generate_synthetic_activations(cov, n_samples).to(device)

    # Step 2: Apply reference rotation (x_rot = x @ R_ref^T, since R is column-orthogonal).
    x_rot = x @ R_ref.T

    # Step 3: INT2 quantize/dequantize.
    x_dequant = simulate_int2_quantize(x_rot)

    # Step 4: Error and error covariance.
    error = x_rot - x_dequant  # [n_samples, d]
    C_E = (error.T @ error) / n_samples  # [d, d]

    # Step 5: Error directions (largest error variance first).
    eigvals_err, U_err = torch.linalg.eigh(C_E)
    idx_err = torch.argsort(eigvals_err, descending=True)
    U_err = U_err[:, idx_err]

    # Step 6: Project original cov into rotated space, eigendecompose.
    cov_rot = R_ref @ cov @ R_ref.T
    eigvals_target, U_target = torch.linalg.eigh(cov_rot)
    # Ascending: least important (smallest variance) first.
    idx_target = torch.argsort(eigvals_target, ascending=True)
    U_target = U_target[:, idx_target]

    # Step 7: R_resid maps largest error dims -> least important target dims.
    R_resid = U_target @ U_err.T
    # Ensure orthogonality (should already be, but guard against FP drift).
    U_check, S_check, Vh_check = torch.linalg.svd(R_resid)
    R_resid = U_check @ Vh_check

    # Step 8: Compose: new rotation = R_resid @ R_ref.
    return R_resid @ R_ref


def run_uresidual_refinement(
    cov: torch.Tensor,
    R_ref: torch.Tensor,
    n_iters: int,
    n_samples: int = 4096,
) -> torch.Tensor:
    """Run multiple uresidual refinement iterations.

    Each iteration recomputes synthetic activations under the current rotation
    and refines it by aligning quantization error with low-importance dims.
    """
    R = R_ref.clone()
    for i in range(n_iters):
        R_new = compute_uresidual_rotation(cov, R, n_samples)
        orth_err = (R_new @ R_new.T - torch.eye(cov.shape[0])).abs().max().item()
        if i == 0 or (i + 1) == n_iters:
            print(f"    uresidual iter {i+1}/{n_iters}: orth err={orth_err:.2e}")
        R = R_new
    return R


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
    ap.add_argument("--uresidual-iters", type=int, default=0,
                    help="Number of uresidual refinement iterations (0=disabled, 1-2 recommended)")
    ap.add_argument("--uresidual-samples", type=int, default=4096,
                    help="Number of synthetic samples for uresidual quantization simulation")
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

    do_uresidual = args.uresidual_iters > 0
    if do_uresidual:
        print(f"\nG7 uresidual mode: {args.uresidual_iters} iteration(s), "
              f"{args.uresidual_samples} synthetic samples")

    for layer in range(args.num_layers):
        # K rotation: R_K = eigendecompose(Q^T Q)
        R_K = compute_rotation(q_covs[layer])
        k_eig = torch.linalg.eigvalsh(q_covs[layer])
        k_eig_sorted = k_eig.flip(0)  # descending

        # V rotation: R_V = eigendecompose(V^T V)
        R_V = compute_rotation(v_covs[layer])
        v_eig = torch.linalg.eigvalsh(v_covs[layer])
        v_eig_sorted = v_eig.flip(0)

        # G7: Uresidual refinement (before composition with H/P_br).
        # Aligns quantization error directions with low-importance Q/V dims.
        if do_uresidual:
            R_K = run_uresidual_refinement(
                q_covs[layer], R_K, args.uresidual_iters, args.uresidual_samples)
            R_V = run_uresidual_refinement(
                v_covs[layer], R_V, args.uresidual_iters, args.uresidual_samples)

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

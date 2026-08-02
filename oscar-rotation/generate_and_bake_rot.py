#!/usr/bin/env python3
"""Generate and bake rotation matrices into any GGUF model.

This is the main entry point for the OSCAR rotation pipeline.
It handles everything: model config reading, rotation generation (or
calibration), and GGUF baking in a single command.

========================================================================
QUICK START
========================================================================

  # A) Data-free Hadamard (no calibration, works out of the box)
  python3 oscar-rotation/generate_and_bake_rot.py \
      --base model.gguf --out model-rot.gguf

  # B) Calibrated rotation (better quality, needs llama-oscar-calib built)
  python3 oscar-rotation/generate_and_bake_rot.py \
      --base model.gguf --out model-rot.gguf \
      --method calibrated --dump-path calibration.txt

  # C) Use existing rotation files (skip generation)
  python3 oscar-rotation/generate_and_bake_rot.py \
      --base model.gguf --out model-rot.gguf \
      --rot-dir rotations/

========================================================================
THEN RUN THE MODEL
========================================================================

  llama-cli -m model-rot.gguf --flash-attn on \
      --cache-type-k oscar2 --cache-type-v oscar2 \
      -p "your prompt" -n 100

========================================================================
FLAGS
========================================================================

  --base         Base GGUF model (required)
  --out          Output rotated GGUF (required)
  --method       'hadamard' (default) or 'calibrated'
  --dump-path    For calibrated: text file as calibration prompt, or
                 directory of QKV dumps from legacy OSCAR pipeline
  --rot-dir      Pre-existing rotation .pt directory (skips generation)

========================================================================
WHAT IT DOES INTERNALLY
========================================================================

  hadamard mode:
    1. Reads model config (arch, layers, head dims) from GGUF metadata
    2. Generates per-layer Hadamard rotation matrices (.pt files)
    3. Bakes them into the GGUF as attn_k_rot / attn_v_rot tensors

  calibrated mode:
    1. Reads model config from GGUF metadata
    2. Runs llama-oscar-calib to dump Q/V covariances from model
    3. Runs calibrate_rotation.py to eigendecompose and compose R.H.P_br
    4. Bakes rotations into the GGUF

========================================================================
REQUIREMENTS
========================================================================

  pip install torch numpy
  # Also requires the repo's gguf-py (imported relative to this file).
  # For calibrated mode: build llama-oscar-calib first:
  #   cmake -B build && cmake --build build --target llama-oscar-calib
"""

import argparse, os, sys, torch, math
from pathlib import Path

# Legacy: add the paper repo to path for compute_kv_rotation imports (fallback only)
PAPER_ROT = Path("/mnt/storage/Projects/oscar-paper/rotation")
if PAPER_ROT.exists():
    sys.path.insert(0, str(PAPER_ROT))
# The primary calibrated pipeline uses our self-contained
# calibrate_rotation.py + llama-oscar-calib (no external repo needed).

# Add our gguf-py for GGUF reading
TQ_GGUF = Path(__file__).parent.parent / "gguf-py"
sys.path.insert(0, str(TQ_GGUF))


def read_model_config(path: str) -> dict:
    """Read GGUF metadata to extract architecture parameters.

    Returns per-layer head dimensions (Gemma-4 SWA can have variable head dim).
    """
    import gguf
    import numpy as np
    r = gguf.GGUFReader(path)
    arch = r.get_field("general.architecture").parts[-1].tobytes().decode()
    n_layers = int(r.get_field(f"{arch}.block_count").parts[-1].item())
    n_head = int(r.get_field(f"{arch}.attention.head_count").parts[-1].item())
    n_head_kv = int(r.get_field(f"{arch}.attention.head_count_kv").parts[-1].item())

    # Derive per-layer head dimensions from attn_q weight shapes.
    # Gemma-4 SWA can have variable head_dim per layer; this is authoritative.
    q_shapes = {}
    for t in r.tensors:
        if ".attn_q.weight" in t.name or ".q_proj.weight" in t.name:
            parts = t.name.split(".")
            # blk.N.attn_q.weight -> N
            try:
                idx = parts.index("blk")
                layer = int(parts[idx + 1])
                q_shapes[layer] = t.shape[-1]
            except (ValueError, IndexError):
                pass

    if not q_shapes:
        raise ValueError("Could not find any attn_q/q_proj tensors")

    per_layer_hd = {}
    for layer, qdim in q_shapes.items():
        per_layer_hd[layer] = qdim // n_head

    return {
        "arch": arch,
        "n_layers": n_layers,
        "n_head": n_head,
        "n_head_kv": n_head_kv,
        "per_layer_head_dim": per_layer_hd,
    }


def generate_hadamard(cfg: dict, output_dir: str):
    """Generate Hadamard rotation .pt files compatible with export_rot_kv_gguf.py."""
    nl = cfg["n_layers"]
    per_layer_hd = cfg["per_layer_head_dim"]

    # Unique head dims across all layers
    unique_hd = sorted(set(per_layer_hd.values()))
    for hd in unique_hd:
        assert hd & (hd - 1) == 0, f"head_dim={hd} must be power of 2 for Hadamard"

    # Cache: generate each unique-sized Hadamard once
    had_cache = {}
    for hd in unique_hd:
        h = torch.tensor([[1.0]], dtype=torch.float64)
        while h.shape[0] < hd:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        h = h / math.sqrt(hd)
        h = h.float()
        err = (h @ h.T - torch.eye(hd)).abs().max().item()
        print(f"  Hadamard {hd}x{hd} orthogonality error: {err:.2e}")
        had_cache[hd] = h

    eigvals_cache = {hd: torch.ones(hd, dtype=torch.float32) for hd in unique_hd}

    for target in ("k", "v"):
        result = {
            "format_version": 1,
            "objective": f"hadamard_{target}",
            "source_grouping": "layer",
            "layers": {},
        }
        for layer_id in range(nl):
            hd = per_layer_hd.get(layer_id, unique_hd[0])
            result["layers"][layer_id] = {
                "layer_id": layer_id,
                "rotation": had_cache[hd].clone(),
                "eigenvalues": eigvals_cache[hd].clone(),
            }
        # Save with both our canonical name AND the name export_rot_kv expects
        path = Path(output_dir) / f"{target}_rotation_hadamard.pt"
        torch.save(result, str(path))
        # Also save as the filename export_rot_kv_gguf.py expects
        expected = Path(output_dir) / (f"k_rotation_qqt_r_h_pbr.pt" if target == "k"
                                       else f"v_rotation_sst_r_h_pbr.pt")
        torch.save(result, str(expected))
        print(f"  {path.name} -> {expected.name}")


def main():
    ap = argparse.ArgumentParser(description="Generate and bake rotation matrices into any GGUF")
    ap.add_argument("--base", required=True, help="Base GGUF model path")
    ap.add_argument("--out", required=True, help="Output rot-kv GGUF path")
    ap.add_argument("--method", default="hadamard", choices=["hadamard", "calibrated"],
                    help="'hadamard' (default, data-free) or 'calibrated' (needs --dump-path)")
    ap.add_argument("--dump-path", default=None,
                    help="QKV dump directory (required for --method calibrated)")
    ap.add_argument("--rot-dir", default=None,
                    help="Directory with existing .pt rotation files (skips generation)")
    args = ap.parse_args()

    base_path = Path(args.base)
    out_path = Path(args.out)
    rot_dir = Path(args.rot_dir) if args.rot_dir else Path(out_path.parent / f"{out_path.stem}_rot")

    # Create temp rotation dir
    os.makedirs(rot_dir, exist_ok=True)

    if args.method == "hadamard":
        print(f"Reading model: {base_path.name}")
        cfg = read_model_config(str(base_path))
        per_layer_hd = cfg["per_layer_head_dim"]
        # Summarize per-layer head dim distribution
        hd_counts = {}
        for hd in per_layer_hd.values():
            hd_counts[hd] = hd_counts.get(hd, 0) + 1
        hd_summary = ", ".join(f"{n} layers: {hd}x{hd}" for hd, n in sorted(hd_counts.items()))
        print(f"  Architecture: {cfg['arch']}, {cfg['n_layers']} layers, {cfg['n_head']} heads")
        print(f"  Per-layer head dim: {hd_summary}")
        print(f"  Generating Hadamard rotation...")
        generate_hadamard(cfg, str(rot_dir))
    elif args.method == "calibrated":
        # Self-contained calibrated rotation pipeline (G1+G3).
        # Step 1: llama-oscar-calib dumps Q/V covariances from model activations.
        # Step 2: calibrate_rotation.py eigendecomposes and composes R·H·P_br.
        cfg = read_model_config(str(base_path))
        per_layer_hd = cfg["per_layer_head_dim"]
        first_hd = list(per_layer_hd.values())[0]
        unique_hd = sorted(set(per_layer_hd.values()))
        if len(unique_hd) > 1:
            print(f"WARNING: mixed head dims {unique_hd}; calibration uses first={first_hd}")

        cov_dir = str(rot_dir / "covariances")
        os.makedirs(cov_dir, exist_ok=True)

        # Step 1: Run llama-oscar-calib to dump covariances.
        import shutil
        calib_bin = shutil.which("llama-oscar-calib")
        if not calib_bin:
            # Try build/bin relative to repo root.
            repo_root = Path(__file__).parent.parent
            candidate = repo_root / "build" / "bin" / "llama-oscar-calib"
            if candidate.exists():
                calib_bin = str(candidate)

        if calib_bin:
            import subprocess
            calib_text = args.dump_path if args.dump_path else ""
            calib_cmd = [
                calib_bin,
                "-m", str(base_path),
                "-o", cov_dir,
            ]
            if calib_text:
                # --dump-path doubles as calibration text file in the new pipeline.
                if os.path.isfile(calib_text):
                    calib_cmd += ["-f", calib_text]
                else:
                    calib_cmd += ["-p", calib_text]
            else:
                # Use a default calibration prompt.
                calib_cmd += ["-p",
                    "The quick brown fox jumps over the lazy dog. "
                    "Machine learning is transforming the world. "
                    "Quantization reduces model size while preserving quality."]
            print(f"Step 1: Running calibration dump: {' '.join(calib_cmd[:3])}...")
            subprocess.check_call(calib_cmd)
        else:
            print("WARNING: llama-oscar-calib not found. Falling back to external OSCAR repo.")
            # Legacy path: try the external paper repo.
            PAPER_ROT = Path("/mnt/storage/Projects/oscar-paper/rotation")
            if not PAPER_ROT.exists() or not args.dump_path:
                print("ERROR: llama-oscar-calib not built and no external OSCAR repo.")
                print("Build with: cmake -B build -DGGML_CUDA=ON && cmake --build build --target llama-oscar-calib")
                sys.exit(1)
            import subprocess
            cmd = [
                sys.executable, str(PAPER_ROT / "compute_kv_rotation.py"),
                "--method", "qqt_sst",
                "--dump-path", args.dump_path,
                "--head-dim", str(first_hd),
                "--composition", "r_h_pbr",
                "--output-dir", str(rot_dir),
            ]
            print(f"Running: {' '.join(cmd)}")
            subprocess.check_call(cmd)

        # Step 2: Run calibrate_rotation.py to eigendecompose and compose R·H·P_br.
        calib_py = Path(__file__).parent / "calibrate_rotation.py"
        if not calib_py.exists():
            print(f"ERROR: {calib_py} not found")
            sys.exit(1)
        import subprocess
        rot_cmd = [
            sys.executable, str(calib_py),
            "--cov-dir", cov_dir,
            "--head-dim", str(first_hd),
            "--num-layers", str(cfg["n_layers"]),
            "--output-dir", str(rot_dir),
            "--composition", "r_h_pbr",
        ]
        print(f"Step 2: Computing rotations: {' '.join(rot_cmd[:4])}...")
        subprocess.check_call(rot_cmd)

    # Bake into GGUF
    print(f"\nBaking rotation into GGUF...")
    export_script = Path(__file__).parent / "export_rot_kv_gguf.py"
    if not export_script.exists():
        print(f"ERROR: {export_script} not found")
        sys.exit(1)

    import subprocess
    cmd = [
        sys.executable, str(export_script),
        "--base", str(base_path),
        "--rot-dir", str(rot_dir),
        "--out", str(out_path),
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

    print(f"\nDone! Rotated model: {out_path}")
    if args.method == "hadamard":
        print("Note: Hadamard rotation is data-free and improves INT2 quantization,")
        print("  but for best quality use --method calibrated with QKV dumps.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate and bake rotation matrices into any GGUF model.

Two modes:
  hadamard (default)  Data-free Hadamard rotation, no calibration needed.
                      Works for any model with power-of-2 head_dim.

  calibrated          Requires QKV dumps from running the model on calibration
                      data (e.g. GPQA). Run the proper dump pipeline first,
                      then pass --dump-path.

Usage:
  # Hadamard (no calibration)
  python3 generate_and_bake_rot.py \
      --base model.gguf --out model-rot.gguf

  # Calibrated (needs QKV dumps from save_qkv pipeline)
  python3 generate_and_bake_rot.py \
      --base model.gguf --out model-rot.gguf \
      --method calibrated --dump-path /path/to/qkv_dumps
"""

import argparse, os, sys, torch, math
from pathlib import Path

# Add the paper repo to path for compute_kv_rotation imports
PAPER_ROT = Path("/mnt/storage/Projects/oscar-paper/rotation")
if PAPER_ROT.exists():
    sys.path.insert(0, str(PAPER_ROT))

# Add our gguf-py for GGUF reading
TQ_GGUF = Path(__file__).parent.parent / "gguf-py"
sys.path.insert(0, str(TQ_GGUF))


def read_model_config(path: str) -> dict:
    """Read GGUF metadata to extract architecture parameters."""
    import gguf
    r = gguf.GGUFReader(path)
    arch = r.get_field("general.architecture").parts[-1].tobytes().decode()
    n_layers = int(r.get_field(f"{arch}.block_count").parts[-1])
    n_head = int(r.get_field(f"{arch}.attention.head_count").parts[-1])

    # Find head_dim from a Q projection weight
    head_dim = None
    for t in r.tensors:
        if any(k in t.name for k in ["q_proj.weight", "attn_q.weight"]):
            head_dim = t.shape[1] // n_head
            break
    if head_dim is None:
        raise ValueError("Could not determine head_dim (no q_proj/attn_q tensor found)")

    return {"arch": arch, "n_layers": n_layers, "n_head": n_head, "head_dim": int(head_dim)}


def generate_hadamard(cfg: dict, output_dir: str):
    """Generate Hadamard rotation .pt files compatible with export_rot_kv_gguf.py."""
    hd = cfg["head_dim"]
    nl = cfg["n_layers"]
    assert hd & (hd - 1) == 0, f"head_dim={hd} must be power of 2 for Hadamard"

    # Build normalized Hadamard matrix
    h = torch.tensor([[1.0]], dtype=torch.float64)
    while h.shape[0] < hd:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    h = h / math.sqrt(hd)
    h = h.float()
    err = (h @ h.T - torch.eye(hd)).abs().max().item()
    print(f"Hadamard orthogonality error: {err:.2e}")

    eigvals = torch.ones(hd, dtype=torch.float32)

    for target in ("k", "v"):
        result = {
            "format_version": 1,
            "objective": f"hadamard_{target}",
            "source_grouping": "layer",
            "layers": {},
        }
        for layer_id in range(nl):
            result["layers"][layer_id] = {
                "layer_id": layer_id,
                "rotation": h.clone(),
                "eigenvalues": eigvals.clone(),
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
        print(f"  Architecture: {cfg['arch']}")
        print(f"  Layers: {cfg['n_layers']}, Head dim: {cfg['head_dim']}")
        print(f"  Generating Hadamard rotation...")
        generate_hadamard(cfg, str(rot_dir))

    elif args.method == "calibrated":
        if not args.dump_path:
            print("ERROR: --dump-path is required for --method calibrated")
            sys.exit(1)
        # Use the paper's compute_kv_rotation.py
        try:
            from compute_kv_rotation import write_hadamard_rotation, main as compute_main
            cfg = read_model_config(str(base_path))
            # Run the paper's compute script via its main
            import subprocess
            cmd = [
                sys.executable, str(PAPER_ROT / "compute_kv_rotation.py"),
                "--method", "qqt_sst",
                "--dump-path", args.dump_path,
                "--head-dim", str(cfg["head_dim"]),
                "--composition", "r_h_pbr",
                "--output-dir", str(rot_dir),
            ]
            print(f"Running: {' '.join(cmd)}")
            subprocess.check_call(cmd)
        except ImportError:
            print("ERROR: compute_kv_rotation.py not found. Clone FutureMLS-Lab/OSCAR first.")
            sys.exit(1)

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

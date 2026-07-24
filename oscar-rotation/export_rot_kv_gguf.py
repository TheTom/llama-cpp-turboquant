#!/usr/bin/env python3
"""Bake the OSCAR calibrated K/V rotations into a base GGUF.

Produces a *-rot-kv.gguf that contains per-layer rotation tensors which
the llama.cpp graph applies at runtime. The base model's weights are copied
through unchanged (no re-quantization).

==========================================================================
USAGE
==========================================================================

  # Standard: add attn_k_rot + attn_v_rot tensors to GGUF
  python3 export_rot_kv_gguf.py \
      --base model.gguf \
      --rot-dir rotations/ \
      --out model-rot.gguf

  # G5 absorb: bake R_V into W_o, drop attn_v_rot (zero V rotation cost)
  python3 export_rot_kv_gguf.py \
      --base model.gguf \
      --rot-dir rotations/ \
      --out model-rot.gguf \
      --absorb-v

  # Use pre-existing Hadamard rotations from the qwen3-4b data dir
  python3 export_rot_kv_gguf.py \
      --base qwen3-4b-q4km.gguf \
      --rot-dir oscar-rotation/qwen3-4b-thinking-2507 \
      --out qwen3-4b-q4km-rot-kv.gguf

==========================================================================
FLAGS
==========================================================================

  --base       Base GGUF model (required)
  --rot-dir    Directory with k_rotation_qqt_r_h_pbr.pt and
               v_rotation_sst_r_h_pbr.pt (default: qwen3-4b-thinking-2507/)
  --out        Output GGUF path (required)
  --absorb-v   (G5) Absorb V rotation into W_o instead of adding
               attn_v_rot tensors. Eliminates per-token V rotation cost.
               W_o_new = W_o @ R_V per head.

==========================================================================
REQUIREMENTS
==========================================================================

  pip install torch numpy
  # Also requires the repo's gguf-py (imported relative to this file).

==========================================================================
OUTPUT
==========================================================================

  Without --absorb-v:
    blk.{i}.attn_k_rot.weight  (F32, [d, d])  - applied to K at runtime
    blk.{i}.attn_v_rot.weight  (F32, [d, d])  - applied to V at runtime

  With --absorb-v:
    blk.{i}.attn_k_rot.weight  (F32, [d, d])  - applied to K at runtime
    blk.{i}.attn_output.weight (modified)      - R_V absorbed, no runtime V rot

  Run with: llama-cli -m model-rot.gguf --flash-attn on \
              --cache-type-k oscar2 --cache-type-v oscar2 ...
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gguf-py"))
import gguf, torch, numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType, GGMLQuantizationType


def load_rot(path):
    rk = torch.load(path, map_location="cpu", weights_only=False)
    # store M^T so ggml_mul_mat(rot, K) == K @ M
    return {il: np.ascontiguousarray(rk["layers"][il]["rotation"].float().numpy().T.astype(np.float32))
            for il in range(len(rk["layers"]))}


def load_rot_raw(path):
    """Load rotation matrices without transpose (for absorption into W_o)."""
    rk = torch.load(path, map_location="cpu", weights_only=False)
    return {il: rk["layers"][il]["rotation"].float()
            for il in range(len(rk["layers"]))}


def absorb_v_rotation(v_rot_raw: dict, reader, nlayers: int, n_head: int, n_head_kv: int) -> dict:
    """Absorb R_V into W_o: W_o' = W_o @ R_V per KV head.

    This eliminates the per-token V rotation at runtime by baking it into
    the output projection weight. Returns a dict of modified W_o tensors.

    Args:
        v_rot_raw: {layer_idx: R_V tensor [d, d]} (NOT transposed)
        reader: GGUFReader for the base model
        nlayers: number of layers
        n_head: number of query heads
        n_head_kv: number of KV heads

    Returns:
        dict of {tensor_name: modified_numpy_array} for W_o tensors that were absorbed.
    """
    gqa_ratio = n_head // n_head_kv
    absorbed = {}

    for t in reader.tensors:
        # Match attn_output weight: blk.{i}.attn_output.weight
        if '.attn_output.weight' not in t.name and '.o_proj.weight' not in t.name:
            continue

        # Parse layer index.
        parts = t.name.split('.')
        try:
            idx = parts.index('blk')
            layer = int(parts[idx + 1])
        except (ValueError, IndexError):
            continue

        if layer >= nlayers or layer not in v_rot_raw:
            continue

        # W_o shape: [n_embd, n_head * d] (row-major in GGUF).
        # For each KV head h, the columns [h*gqa_ratio*d : (h+1)*gqa_ratio*d]
        # need W_o[:, slice] = W_o[:, slice] @ R_V_h.
        R_V = v_rot_raw[layer].numpy().astype(np.float32)  # [d, d]
        d = R_V.shape[0]

        # Load W_o as numpy array.
        wo = t.data.astype(np.float32)
        if wo.ndim != 2:
            print(f"  skip {t.name}: not 2D (shape={wo.shape})")
            continue

        n_embd = wo.shape[0]
        expected_cols = n_head * d
        if wo.shape[1] != expected_cols:
            print(f"  skip {t.name}: shape {wo.shape} != expected ({n_embd}, {expected_cols})")
            continue

        # Absorb: for each KV head, apply R_V to the gqa_ratio query head slices.
        for kvh in range(n_head_kv):
            for g in range(gqa_ratio):
                h = kvh * gqa_ratio + g
                col_start = h * d
                col_end = col_start + d
                # W_o[:, h*d:(h+1)*d] = W_o[:, h*d:(h+1)*d] @ R_V
                wo[:, col_start:col_end] = wo[:, col_start:col_end] @ R_V

        absorbed[t.name] = wo
        print(f"  absorbed R_V into {t.name} (layer {layer})")

    return absorbed


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base GGUF (e.g. Qwen3-4B-Thinking-2507 Q4_K_M)")
    ap.add_argument("--rot-dir", default=os.path.join(here, "qwen3-4b-thinking-2507"),
                    help="dir with k_rotation_qqt_r_h_pbr.pt and v_rotation_sst_r_h_pbr.pt")
    ap.add_argument("--out", required=True, help="output *-rot-kv.gguf")
    ap.add_argument("--absorb-v", action="store_true",
                    help="""(G5) Absorb V rotation into W_o instead of adding attn_v_rot tensors. Eliminates per-token V rotation cost at runtime.""")
    args = ap.parse_args()

    reader = GGUFReader(args.base)
    arch = reader.get_field("general.architecture").contents()
    writer = GGUFWriter(args.out, arch)

    SKIP = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"}
    for key, field in reader.fields.items():
        if key in SKIP:
            continue
        vtype = field.types[0]
        sub_type = field.types[-1] if vtype == GGUFValueType.ARRAY else None
        writer.add_key_value(key, field.contents(), vtype, sub_type=sub_type)

    k_rot = load_rot(os.path.join(args.rot_dir, "k_rotation_qqt_r_h_pbr.pt"))
    v_rot = load_rot(os.path.join(args.rot_dir, "v_rotation_sst_r_h_pbr.pt"))
    nlayers = len(k_rot)

    # G5: Absorb V rotation into W_o if requested.
    absorbed_w_o = {}
    if args.absorb_v:
        print("G5: Absorbing V rotation into W_o...")
        v_rot_raw = load_rot_raw(os.path.join(args.rot_dir, "v_rotation_sst_r_h_pbr.pt"))
        # Read n_head / n_head_kv from GGUF metadata.
        n_head = int(reader.get_field(f"{arch}.attention.head_count").parts[-1].item())
        n_head_kv = int(reader.get_field(f"{arch}.attention.head_count_kv").parts[-1].item())
        absorbed_w_o = absorb_v_rotation(v_rot_raw, reader, nlayers, n_head, n_head_kv)

    # Determine which tensors to copy (skip W_o if absorbed).
    for t in reader.tensors:
        if t.name in absorbed_w_o:
            # Add tensor info with modified data size.
            new_data = absorbed_w_o[t.name]
            writer.add_tensor_info(t.name, new_data.shape, new_data.dtype,
                                   new_data.nbytes, t.tensor_type)
        else:
            writer.add_tensor_info(t.name, t.data.shape, t.data.dtype,
                                   t.data.nbytes, t.tensor_type)

    # Add K rotation tensors (always).
    for il in range(nlayers):
        writer.add_tensor_info(f"blk.{il}.attn_k_rot.weight", k_rot[il].shape, k_rot[il].dtype,
                               k_rot[il].nbytes, GGMLQuantizationType.F32)

    # Add V rotation tensors only if NOT absorbed.
    if not args.absorb_v:
        for il in range(nlayers):
            writer.add_tensor_info(f"blk.{il}.attn_v_rot.weight", v_rot[il].shape, v_rot[il].dtype,
                                   v_rot[il].nbytes, GGMLQuantizationType.F32)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    for t in reader.tensors:
        if t.name in absorbed_w_o:
            writer.write_tensor_data(absorbed_w_o[t.name], tensor_endianess=reader.endianess)
        else:
            writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)

    for il in range(nlayers):
        writer.write_tensor_data(k_rot[il], tensor_endianess=reader.endianess)
    if not args.absorb_v:
        for il in range(nlayers):
            writer.write_tensor_data(v_rot[il], tensor_endianess=reader.endianess)

    writer.close()
    n_rot = nlayers * (1 if args.absorb_v else 2)
    print(f"wrote {args.out}: {len(reader.tensors)} base tensors + {n_rot} rotation tensors")
    if args.absorb_v:
        print(f"  G5: V rotation absorbed into {len(absorbed_w_o)} W_o tensors (no attn_v_rot at runtime)")


if __name__ == "__main__":
    main()

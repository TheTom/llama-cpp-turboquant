#!/usr/bin/env python3
"""Bake the OSCAR calibrated K/V rotations into a base GGUF.

Produces a *-rot-kv.gguf that contains the per-layer rotation tensors
(blk.{i}.attn_k_rot.weight / attn_v_rot.weight) which qwen3.cpp applies in-graph.
The base model's weights are copied through unchanged (no re-quantization).

Usage:
    python3 export_rot_kv_gguf.py \
        --base   /path/to/qwen3-4b-thinking-q4km.gguf \
        --rot-dir oscar-rotation/qwen3-4b-thinking-2507 \
        --out    /path/to/qwen3-4b-thinking-q4km-rot-kv.gguf

Requires: torch, numpy, and the repo's gguf-py (imported relative to this file).
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gguf-py"))
import gguf, torch, numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType, GGMLQuantizationType


def load_rot(path):
    rk = torch.load(path, map_location="cpu")
    # store M^T so ggml_mul_mat(rot, K) == K @ M
    return {il: np.ascontiguousarray(rk["layers"][il]["rotation"].float().numpy().T.astype(np.float32))
            for il in range(len(rk["layers"]))}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base GGUF (e.g. Qwen3-4B-Thinking-2507 Q4_K_M)")
    ap.add_argument("--rot-dir", default=os.path.join(here, "qwen3-4b-thinking-2507"),
                    help="dir with k_rotation_qqt_r_h_pbr.pt and v_rotation_sst_r_h_pbr.pt")
    ap.add_argument("--out", required=True, help="output *-rot-kv.gguf")
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

    for t in reader.tensors:
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    for il in range(nlayers):
        writer.add_tensor_info(f"blk.{il}.attn_k_rot.weight", k_rot[il].shape, k_rot[il].dtype,
                               k_rot[il].nbytes, GGMLQuantizationType.F32)
        writer.add_tensor_info(f"blk.{il}.attn_v_rot.weight", v_rot[il].shape, v_rot[il].dtype,
                               v_rot[il].nbytes, GGMLQuantizationType.F32)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in reader.tensors:
        writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
    for il in range(nlayers):
        writer.write_tensor_data(k_rot[il], tensor_endianess=reader.endianess)
        writer.write_tensor_data(v_rot[il], tensor_endianess=reader.endianess)
    writer.close()
    print(f"wrote {args.out}: {len(reader.tensors)} base + {nlayers} K + {nlayers} V rotation tensors")


if __name__ == "__main__":
    main()

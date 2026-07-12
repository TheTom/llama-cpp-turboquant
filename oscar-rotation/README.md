# OSCAR rotation matrices + GGUF baking

The rotated GGUF (`*-rot-kv.gguf`, ~2.3 GB) is too large for GitHub, so this dir ships the
**4.6 MB calibrated rotation matrices** and a script to **regenerate the rotated GGUF** from a
base model. The base weights are copied through unchanged — only the per-layer
`attn_k_rot` / `attn_v_rot` tensors are added.

## Files
- `qwen3-4b-thinking-2507/k_rotation_qqt_r_h_pbr.pt`, `v_rotation_sst_r_h_pbr.pt`
  — per-layer (36 × 128×128) calibrated `R·H·P` rotations (`R_k` from query covariance,
  `R_v` from value covariance), computed on GPQA calibration data.
- `export_rot_kv_gguf.py` — bakes them into a base GGUF.

## Regenerate the rotated GGUF

```bash
# 1. get the base model (standard Qwen3-4B-Thinking-2507, Q4_K_M GGUF), e.g. from HF.
# 2. bake in the rotations:
pip install torch numpy            # gguf-py is already in this repo
python3 oscar-rotation/export_rot_kv_gguf.py \
    --base  qwen3-4b-thinking-q4km.gguf \
    --out   qwen3-4b-thinking-q4km-rot-kv.gguf
```

Then run per the top-level [README](../README.md) (env vars + `--cache-type-k/v q2_0`).

> For a one-step download of the full 2.3 GB rotated GGUF, host it on HuggingFace Hub instead
> (GitHub can't store files this large).

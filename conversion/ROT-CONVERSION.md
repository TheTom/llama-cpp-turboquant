# ROT Model Conversion Guide

Convert any GGUF model into a ROT (rotated KV cache) model for use with
OSCAR2, Q2_0, or any INT2 KV cache quantization.

## What ROT does

Per-layer orthogonal rotation matrices are baked into the GGUF as
`blk.{i}.attn_k_rot.weight` and `blk.{i}.attn_v_rot.weight` tensors.
These rotate K and Q (and optionally V) before quantization, making the
data more quantization-friendly by reducing outlier dimensions. Dot
products are preserved since the rotation is orthogonal.

Without rotation, INT2 types (oscar2, q2_0) produce garbled output on
many models. With rotation, they match the unquantized baseline.

## Quick start

```bash
# Data-free Hadamard rotation (works for any model, power-of-2 head_dim)
python3 oscar-rotation/generate_and_bake_rot.py \
    --base /path/to/model.gguf \
    --out /path/to/model-rot-kv.gguf \
    --method hadamard
```

Then use the rotated model with any INT2 cache type:

```bash
./build/bin/llama-cli \
    -m /path/to/model-rot-kv.gguf \
    --cache-type-k oscar2 --cache-type-v oscar2 \
    -fa on -ngl 99 ...
```

## Requirements

- Python 3.10+
- `torch` (any recent version)
- `numpy`
- `gguf-py` (included in this repo at `gguf-py/`)

## Scripts

All scripts live in `oscar-rotation/`:

| Script | Purpose |
|--------|---------|
| `generate_and_bake_rot.py` | **Recommended** — creates and bakes rotations in one step |
| `calibrate_rotation.py` | Compute calibrated rotations from covariance dumps (G1/G3/G7) |
| `generate_hadamard_rot.py` | Standalone Hadamard matrix generator (for inspection/advanced use) |
| `export_rot_kv_gguf.py` | Bake pre-computed `.pt` rotation files into a GGUF (G5 absorb-v) |
| `tools/oscar-calib/` | C++ tool to dump Q/V covariances from model activations |

## Usage

### Hadamard rotation (data-free, recommended)

Works for any model where the attention head dimension is a power of 2
(64, 128, 256, 512). No calibration data needed.

```bash
python3 oscar-rotation/generate_and_bake_rot.py \
    --base /path/to/model.gguf \
    --out /path/to/model-rot-kv.gguf \
    --method hadamard
```

The script reads the model's architecture and head dimension from the
GGUF metadata, generates normalized Hadamard matrices of the correct
size, and bakes them in.

For models with per-layer variable head dimensions (e.g. Gemma-4 which
has SWA layers at 256 and full-attention layers at 512), the script
auto-detects each layer's head dimension from the weight tensor shapes
and generates correctly-sized rotations per layer.

### Calibrated rotation (self-contained, highest quality)

Produces higher-quality rotations than Hadamard by computing spectral
covariance rotations from actual model activations. The full pipeline
is self-contained (no external OSCAR paper repo needed).

**One-command calibrated pipeline:**

```bash
# Build the calibration tool first
cmake -B build -DGGML_CUDA=ON && cmake --build build --target llama-oscar-calib

# Run calibration + bake in one step
python3 oscar-rotation/generate_and_bake_rot.py \
    --base /path/to/model.gguf \
    --out /path/to/model-rot-kv.gguf \
    --method calibrated \
    --dump-path /path/to/calibration.txt
```

`--dump-path` can be a text file (used as calibration prompt) or a
directory of QKV dumps from a previous run.

**Step-by-step (for advanced use / debugging):**

```bash
# Step 1: Dump Q/V covariances from model activations
./build/bin/llama-oscar-calib -m model.gguf -f calibration.txt -o covariances/

# Step 2: Eigendecompose and compose R·H·P_br
python3 oscar-rotation/calibrate_rotation.py \
    --cov-dir covariances/ --head-dim 128 --num-layers 28 \
    --output-dir rotations/ --composition r_h_pbr

# Step 3: Bake into GGUF (with V rotation absorbed into W_o for zero runtime cost)
python3 oscar-rotation/export_rot_kv_gguf.py \
    --base model.gguf --rot-dir rotations/ --out model-rot.gguf --absorb-v
```

### Uresidual refinement (G7 — iterative quality improvement)

The OSCAR paper's `uresidual` mode refines the base rotation by aligning
quantization error directions with low-importance Q/V covariance dims.
This is an iterative process that converges in 1-2 iterations.

```bash
# Calibrate with 1 uresidual iteration
python3 oscar-rotation/calibrate_rotation.py \
    --cov-dir covariances/ --head-dim 128 --num-layers 28 \
    --output-dir rotations/ --composition r_h_pbr \
    --uresidual-iters 1
```

How it works:
1. Generate synthetic activations from empirical covariance (Cholesky)
2. Apply current reference rotation
3. Simulate INT2 quantize/dequantize (per-row, 4 centroids matching oscar2)
4. Eigendecompose error covariance (largest error variance first)
5. Project target covariance into rotated space (ascending importance)
6. Map largest error dims to least important target dims
7. Update rotation and repeat

`--uresidual-samples N` controls the synthetic sample count (default: 4096).

### Absorbing V rotation into W_o (G5)

The V rotation can be baked into the output projection weight at
quantization time, eliminating runtime overhead entirely:

```bash
python3 oscar-rotation/export_rot_kv_gguf.py \
    --base model.gguf --rot-dir rotations/ --out model-rot.gguf \
    --absorb-v
```

When `--absorb-v` is used, `attn_v_rot` tensors are omitted from the
output GGUF. The rotation is "free" — no runtime cost.

### Using existing `.pt` rotation files

If you already have rotation `.pt` files (e.g. from a previous
calibration run), bake them in directly:

```bash
python3 oscar-rotation/export_rot_kv_gguf.py \
    --base /path/to/model.gguf \
    --rot-dir /path/to/rotation_pt_files \
    --out /path/to/model-rot-kv.gguf
```

The `.pt` files should be named `k_rotation_qqt_r_h_pbr.pt` and
`v_rotation_sst_r_h_pbr.pt` per the paper convention.

### Composition modes

The `--composition` flag controls which transforms are applied:

| Mode | Formula | Notes |
|------|---------|-------|
| `r_h_pbr` | R · H · P_br | Default, best quality per OSCAR paper |
| `r_h` | R · H | Rotation + Hadamard, no bit-reversal |
| `r_pbr` | R · P_br | Rotation + bit-reversal, no Hadamard |
| `r` | R only | Pure spectral rotation |
| `h_pbr` | H · P_br | Data-free Hadamard + bit-reversal |
| `h` | H only | Data-free Hadamard |
| `pbr` | P_br only | Bit-reversal permutation only |

## Running the rotated model

```bash
# With oscar2 (INT2 asymmetric, 7.1x compression)
./build/bin/llama-cli \
    -m /path/to/model-rot-kv.gguf \
    --cache-type-k oscar2 --cache-type-v oscar2 \
    -fa on -ngl 99 -c 4096 \
    -p "What is the capital of France?" -n 50

# With q2_0 (INT2 Lloyd-Max, 6.4x compression)
./build/bin/llama-cli \
    -m /path/to/model-rot-kv.gguf \
    --cache-type-k q2_0 --cache-type-v q2_0 \
    -fa on -ngl 99 -c 4096 \
    -p "What is the capital of France?" -n 50
```

### Sliding window for long context

For long-context inference, cap the attention window to keep generation
speed constant:

```bash
GGML_KV_WINDOW=8192 ./build/bin/llama-cli \
    -m /path/to/model-rot-kv.gguf \
    --cache-type-k oscar2 --cache-type-v oscar2 \
    -fa on -ngl 99 -c 1048576 \
    -p "What is the capital of France?" -n 50
```

This attends to only the last 8K tokens regardless of total cache size.
All MMA-path cache types (f16, q4_0/1, q5_0/1, q8_0, turbo2/3/4, oscar2)
respect this setting. Set to 0 (default) for full attention.

## Env vars for advanced tuning

| Env var | Purpose |
|---------|---------|
| `GGML_KV_WINDOW=N` | Sliding window: only attend to last N tokens (0 = off) |
| `LLAMA_KV_V_ROT=1` | Enable V rotation on Gemma-4 (default: off) |
| `LLAMA_KV_NO_HADAMARD=1` | Skip in-quant Hadamard (for calibrated rotation models) |

## How it works

The rotation matrices are stored as F32 tensors (one per layer, per
target) that are applied during graph construction:

- **K rotation** (`attn_k_rot`): Applied to both Q and K before the
  attention dot product. Since the same orthogonal rotation is applied
  to both, dot products are preserved: `(M@q)^T @ (M@k) = q^T @ k`.
- **V rotation** (`attn_v_rot`): Applied to V before quantization and
  undone on the attention output before the output projection. Gated
  by `LLAMA_KV_V_ROT=1` on Gemma-4; always enabled on Qwen3/Qwen3MoE.

Supported model architectures: `gemma4`, `qwen3`, `qwen35` (Qwen3.5),
`qwen3moe`.

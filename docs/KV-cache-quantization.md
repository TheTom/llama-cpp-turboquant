# KV Cache Quantization with TurboQuant

TurboQuant adds five runtime-only KV cache quantization types that compress
the K/V cache far beyond the standard `q8_0` while keeping decode quality via a
Walsh-Hadamard rotation (WHT) that Gaussianizes the cache vectors before
quantization:

| Type               | Enum                       | Size            | Compression vs f16 |
|--------------------|----------------------------|-----------------|--------------------|
| `turbo2`           | `GGML_TYPE_TURBO2_0` (43)  | 2.125 bits/value | 7.5x              |
| `turbo3`           | `GGML_TYPE_TURBO3_0` (44)  | 3.125 bits/value | 5.1x              |
| `turbo4`           | `GGML_TYPE_TURBO4_0` (47)  | 4.125 bits/value | 3.9x              |
| `turbo5`           | `GGML_TYPE_TURBO5_0` (51)  | 5.125 bits/value | 3.1x              |
| `turbo6`           | `GGML_TYPE_TURBO6_0` (52)  | 6.125 bits/value | 2.6x              |

These are KV-cache-only types: they are never stored in model files. The
corresponding model-weight quantization types are `TQ3_1S` (45) and `TQ4_1S`
(46) - 3/4-bit WHT-rotated Lloyd-Max quantization, block size 32, exposed in
`llama-quantize` as `TQ3_1S` / `TQ4_1S`.

## Usage

```bash
llama-cli -m model.gguf -c 8192 -ngl 99 \
    --cache-type-k q8_0 --cache-type-v turbo3
```

The flag parser accepts any combination of `f32`, `f16`, `bf16`, `q8_0`,
`q4_0`, `q4_1`, `iq4_nl`, `q5_0`, `q5_1` and `turbo2` to `turbo6` for K and V.
Which combinations have backend kernels is narrower and backend-dependent;
`turbo5` and `turbo6` support a smaller set of pairs, listed below. Mixing
quantized V with unquantized K is the common configuration.

On Metal, flash attention accepts exactly three mixed pairs: turbo K + turbo V,
`q8_0` K + turbo V, and turbo K + `q8_0` V, where turbo is `turbo2/3/4`. Any
other mixed pair aborts in `ggml_metal_op_flash_attn_ext`.

Turbo KV types require flash attention. If a turbo cache type is requested
with flash attention disabled, it is enabled automatically (a warning is
printed) before the quantized-V check runs, so a turbo V cache never reaches
that error. A quantized V cache of another type with flash attention explicitly
disabled is an error, matching upstream behavior.

The same flags work in `llama-server`, `llama-bench`, and `llama-perplexity`.

## turbo5 and turbo6

`turbo5` and `turbo6` sit between `q8_0` and `turbo4`. They use the same WHT
rotation as the other turbo types plus a Lloyd-Max codebook for the rotated
N(0, 1/128) distribution and a per-block corrected L2 norm:

| Type     | Block             | Layout                                                        | Centroids |
|----------|-------------------|---------------------------------------------------------------|-----------|
| `turbo5` | 128 values / 82 B | `f16` norm + 64 B of 4-bit magnitudes + 16 B of sign bits      | 32 (antisymmetric) |
| `turbo6` | 128 values / 98 B | `f16` norm + 64 B of 4-bit low bits + 32 B of 2-bit high bits | 64        |

The `turbo5` codebook is antisymmetric, so it stores a magnitude index and a
sign bit per value instead of a 5-bit code, which keeps the decode to a single
byte-permute lookup in the CUDA kernels.

Like `turbo2`/`turbo3`/`turbo4`, a symmetric `turbo5` or `turbo6` cache has its
K rewritten to `q8_0` by the auto-asymmetric rule on models with GQA ratio 6 or
more. On Qwen2.5-0.5B (GQA 7:1) `turbo6` K gives about 22 PPL and `turbo5` K
about 32 against 8.0 for `f16`, while `turbo6` V is within noise of `f16`.

Kernels exist on the CPU and CUDA backends (HIP builds the same CUDA kernels).
Metal, Vulkan and SYCL have no `SET_ROWS` kernel for these types, so on those
backends context creation fails with an error unless the KV cache is kept in
host memory (`-nkvo`). `GET_ROWS` is not implemented for any turbo type. The
blocks hold 128 values; head dims that are not a multiple of 128 are zero-padded
up to one, as for the other turbo types.

CUDA flash attention (head dims 128 and 256 after padding) covers these K/V pairs
in a default build:

- `turbo5` or `turbo6` K with V of the same type, `turbo3`, `turbo4`, `q8_0` or `f16`
- `turbo6` K with `turbo5` V
- `q8_0` K with `turbo5` or `turbo6` V (the auto-asymmetric result)

`-DGGML_CUDA_FA_ALL_QUANTS=ON` adds the remaining pairs with `f16`, `q8_0`,
`turbo2`, `turbo3` and `turbo4`. The fused GQA-packed MMA kernel, which is the fast
path for long contexts, additionally covers `turbo5`/`turbo5`, `turbo6`/`turbo6`,
`turbo5`/`turbo3`, `turbo6`/`turbo3` (head dims 128 and 256) and `turbo5`/`turbo4`,
`turbo6`/`turbo4`, `turbo6`/`turbo5` (head dim 256).

## Model-specific quality

Models with attention sinks can be unusually sensitive to K-cache quantization. GPT-OSS is a known case: even `q8_0` K changes the output distribution substantially, and lower-bit K types degrade it further despite normal codec and kernel accuracy. Use `f16` K for GPT-OSS and other sink-heavy models. Validate a quantized V cache separately against an `f16` K/V baseline before deploying it.

Short output samples are not a sufficient quality check for this class of model because the text can remain fluent while token probabilities move significantly. Use `llama-perplexity --kl-divergence` or an equivalent logit comparison when selecting cache types.

## Rotation

K and V vectors are rotated by a fixed 128x128 orthonormal Walsh-Hadamard
matrix before quantization and inverse-rotated after dequantization. Head
dimensions that are not multiples of 128 are zero-padded to the next multiple
of 128 for turbo types. MLA models have no separate V cache (V is a view of
K), so V rotation and padding are skipped for them, and the K and V cache types
must be identical.

## Environment knobs

This table is the source of truth for these variables. If it and the code ever
disagree, the code wins and both need fixing in the same PR.

| Variable                        | Default | Effect                                                          |
|---------------------------------|---------|-----------------------------------------------------------------|
| `TURBO_LAYER_ADAPTIVE`          | auto    | Layer-adaptive KV precision. Unset: Boundary V (mode 7) auto-enables when V is `turbo2` and the model has 8+ layers; `0` opts out |
| `TURBO_AUTO_ASYMMETRIC`         | `1`     | Auto-select asymmetric K/V types for large-GQA models (`0` disables) |
| `TURBO_SPARSE_V`                | `1`     | Metal: sparse-V dequant skip for `turbo3` V (`0` disables)     |
| `GGML_TQ_NATIVE`                | unset   | CUDA: `1` keeps `TQ4_1S` weights native and uses the fused TQ kernels instead of converting them to `q8_0` at load; decode about 30% faster and weight VRAM about 1.7x smaller, prefill about 2x slower |
| `GGML_CUDA_FUSE_CHAIN`          | unset   | CUDA: `0` disables elementwise chain fusion, so ADD/MUL/DIV/SCALE/CLAMP/SQR and the SILU/SIGMOID/SOFTPLUS/GELU/RELU/NEG unaries run as separate kernels |
| `GGML_CUDA_Q8CACHE`             | unset   | CUDA: `0` disables the per-graph shared `q8_1` activation cache in `mmvq`, so a repeated activation is quantized again per use |
| `LLAMA_ATTN_ROT_K_OVERRIDE` / `LLAMA_ATTN_ROT_V_OVERRIDE` | off | Enable upstream #21038 attention rotation for K / V; needs a quantized type and a head dim that is a multiple of 64 |
| `LLAMA_ATTN_ROT_K_NROT`         | `64`    | Rotation tile size for the optional upstream K path (`0` = upstream's largest power-of-two divisor of the head dim) |
| `LLAMA_ATTN_ROT_DISABLE`        | `0`     | Hard lock-out: force rotation off on both sides (`1` disables) |

`TURBO_LAYER_ADAPTIVE` is not simply `0`. With the variable unset, Boundary V
(mode 7) turns on automatically when the V cache type is `turbo2` and the model
has 8 or more layers: the first two and last two V layers become `q8_0` and
every other V layer becomes `turbo2`. Setting the variable explicitly selects
the mode; modes 5, 6 and 7 assign the V type per layer regardless of `-ctv`,
so mode 7 on a `-ctv turbo3` or `-ctv turbo4` run downgrades the non-edge V
layers to `turbo2`. The full mode legend is in `src/llama-kv-cache.cpp`
(`kv_adaptive_mode`).

Upstream attention rotation is off by default: TurboQuant manages rotation
itself (the WHT applied at cache write is equivalent and interacts with the
cache types). `LLAMA_ATTN_ROT_*` only affects the optional upstream rotation
path for models that benefit from it.

## Model-weight quantization (TQ3_1S / TQ4_1S)

```bash
llama-quantize model-f16.gguf model-tq4.gguf TQ4_1S
```

`TQ3_1S` and `TQ4_1S` are first-class weight types with CPU, CUDA/HIP (warp
cooperative mmvq), Metal, and Vulkan kernels. ggml-sycl has no TQ kernels, and
its generic dequantize path leaves TQ weights in the WHT-rotated domain, so TQ
models are unsupported on SYCL. For TQ MoE weights, the small-batch
`MUL_MAT_ID` path routes experts on device and stays CUDA-graph capturable;
only the large-batch fallback dequantizes to f16 and synchronizes the stream.

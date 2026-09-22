# KV Cache Quantization with TurboQuant

TurboQuant adds five runtime-only KV cache quantization types that compress
the K/V cache far beyond the standard `q8_0` while keeping decode quality via a
Walsh-Hadamard rotation (WHT) that Gaussianizes the cache vectors before
quantization:

| Type               | Enum                       | Size            | Compression vs f16 |
|--------------------|----------------------------|-----------------|--------------------|
| `turbo2`           | `GGML_TYPE_TURBO2_0` (43)  | 2 bits/value    | 6.4x               |
| `turbo3`           | `GGML_TYPE_TURBO3_0` (44)  | 3.25 bits/value | 4.9x               |
| `turbo4`           | `GGML_TYPE_TURBO4_0` (47)  | 4.25 bits/value | 3.8x               |
| `tq5_0`            | `GGML_TYPE_TQ5_0` (52)     | 5.125 bits/value| 3.1x               |
| `tq6_0`            | `GGML_TYPE_TQ6_0` (51)     | 6.125 bits/value| 2.6x               |

These are KV-cache-only types: they are never stored in model files. The
corresponding model-weight quantization types are `TQ3_1S` (45) and `TQ4_1S`
(46) - 3/4-bit WHT-rotated Lloyd-Max quantization, block size 32, exposed in
`llama-quantize` as `TQ3_1S` / `TQ4_1S`.

## Usage

```bash
llama-cli -m model.gguf -c 8192 -ngl 99 \
    --cache-type-k q8_0 --cache-type-v turbo3
```

Any combination of `f16`, `q8_0`, `turbo2`, `turbo3`, `turbo4`, `tq5_0`,
`tq6_0` for K and V is supported; mixing quantized V with unquantized K is the
common configuration.

Turbo KV types require flash attention. If a turbo cache type is requested
with flash attention disabled, it is enabled automatically (a warning is
printed). A quantized V cache with flash attention explicitly disabled is an
error, matching upstream behavior for all quantized V types.

The same flags work in `llama-server`, `llama-bench`, and `llama-perplexity`.

## tq6_0 and tq5_0 (high-precision K)

`tq6_0` and `tq5_0` sit between `q8_0` and `turbo4` and are meant for the K
side, which is more sensitive than V because the GQA broadcast amplifies K
error across every query head that shares a KV head. Both use the same WHT
rotation as the other turbo types plus a Lloyd-Max codebook for the rotated
N(0, 1/128) distribution and a per-block corrected L2 norm:

| Type    | Block          | Layout                                                        | Centroids |
|---------|----------------|---------------------------------------------------------------|-----------|
| `tq6_0` | 128 values / 98 B | `f16` norm + 64 B of 4-bit low bits + 32 B of 2-bit high bits | 64        |
| `tq5_0` | 128 values / 82 B | `f16` norm + 64 B of 4-bit magnitudes + 16 B of sign bits      | 32 (antisymmetric) |

The codebooks are antisymmetric, so `tq5_0` stores a magnitude index and a
sign bit per value instead of a 5-bit code, which keeps the decode to a single
byte-permute lookup in the CUDA kernels.

Unlike `turbo2`/`turbo3`/`turbo4`, a `tq6_0` or `tq5_0` K cache is not rewritten
to `q8_0` by the auto-asymmetric rule: at 5-6 bits the K error is small enough
that a high GQA ratio does not need the upgrade.

Both types are CUDA-only for inference. There is a portable reference codec on
the CPU backend (used by `test-quantize-fns` and for `GET_ROWS`/`CPY`), but no
CPU, Metal, Vulkan or SYCL flash attention kernels, so they require a CUDA
build. Their blocks hold 128 values, so only head dims 128 and 256 are
supported; other head dims fall back to no flash-attention kernel.

Flash attention covers every K/V pair through the vec kernel. The fused
GQA-packed MMA kernel, which is the fast path for long contexts, additionally
covers `tq6_0`/`tq6_0`, `tq5_0`/`tq5_0`, `tq6_0`/`turbo3`, `tq5_0`/`turbo3`
(head dims 128 and 256) and `tq6_0`/`turbo4`, `tq5_0`/`turbo4`, `tq6_0`/`tq5_0`
(head dim 256).

## Model-specific quality

Models with attention sinks can be unusually sensitive to K-cache quantization. GPT-OSS is a known case: even `q8_0` K changes the output distribution substantially, and lower-bit K types degrade it further despite normal codec and kernel accuracy. Use `f16` K for GPT-OSS and other sink-heavy models. Validate a quantized V cache separately against an `f16` K/V baseline before deploying it.

Short output samples are not a sufficient quality check for this class of model because the text can remain fluent while token probabilities move significantly. Use `llama-perplexity --kl-divergence` or an equivalent logit comparison when selecting cache types.

## Rotation

K and V vectors are rotated by a fixed 128x128 orthonormal Walsh-Hadamard
matrix before quantization and inverse-rotated after dequantization. Head
dimensions that are not multiples of 128 are zero-padded to the next multiple
of 128 for turbo types. MLA models have no separate V cache (V is a view of
K), so V rotation and padding are skipped for them.

## Environment knobs

| Variable                        | Default | Effect                                                          |
|---------------------------------|---------|-----------------------------------------------------------------|
| `TURBO_LAYER_ADAPTIVE`          | `0`     | Layer-adaptive KV precision; `7` = Boundary V (first/last layers in `q8_0`, middle in turbo) |
| `TURBO_AUTO_ASYMMETRIC`         | `1`     | Auto-select asymmetric K/V types for large-GQA models (`0` disables) |
| `TURBO_SPARSE_V`                | `1`     | Sparse-V dequant skip in flash attention (`0` disables)        |
| `LLAMA_ATTN_ROT_K_OVERRIDE`     | off     | Enable upstream #21038 attention rotation for K                |
| `LLAMA_ATTN_ROT_V_OVERRIDE`     | off     | Enable upstream #21038 attention rotation for V                |
| `LLAMA_ATTN_ROT_DISABLE`        | `0`     | Hard lock-out: force rotation off on both sides (`1` disables) |

Upstream attention rotation is off by default: TurboQuant manages rotation
itself (the WHT applied at cache write is equivalent and interacts with the
cache types). `LLAMA_ATTN_ROT_*` only affects the optional upstream rotation
path for models that benefit from it.

## Model-weight quantization (TQ3_1S / TQ4_1S)

```bash
llama-quantize model-f16.gguf model-tq4.gguf TQ4_1S
```

`TQ3_1S` and `TQ4_1S` are first-class weight types with CUDA/HIP (warp
cooperative mmvq), Metal, and Vulkan kernels. MoE models disable CUDA graphs
for TQ `MUL_MAT_ID` automatically.

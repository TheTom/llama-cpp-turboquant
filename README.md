# llama.cpp (TurboQuant fork)

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++ with aggressive KV cache and weight quantization</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (upstream). This tree adds the **TurboQuant** feature set on top of an upstream base: upstream changes are cherry-picked on an ongoing basis where they add benefit, rather than being merged wholesale.

[manifesto](https://github.com/ggml-org/llama.cpp/discussions/205) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## About this fork

**TurboQuant** compresses the attention KV cache far beyond the standard `q8_0`, and adds two WHT-rotated model-weight types. The method is data-oblivious online vector quantization ([TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874)) applied with a fixed 128x128 Walsh-Hadamard rotation (`GGML_OP_TURBO_WHT`): cache vectors are rotated before quantization (which Gaussianizes their distribution) and inverse-rotated after dequantization. Head dims that are not multiples of 128 are zero-padded.

Five fork-only GGML types (registered in `ggml/include/ggml.h`):

| Type     | Enum                       | Purpose                     | Size                   |
|----------|----------------------------|-----------------------------|------------------------|
| `turbo2` | `GGML_TYPE_TURBO2_0` (43)  | KV cache only               | 2 bits/value           |
| `turbo3` | `GGML_TYPE_TURBO3_0` (44)  | KV cache only               | 3.25 bits/value        |
| `turbo4` | `GGML_TYPE_TURBO4_0` (47)  | KV cache only               | 4.25 bits/value        |
| `TQ3_1S` | `GGML_TYPE_TQ3_1S` (45)    | model weights, Lloyd-Max    | 3 bits, block size 32  |
| `TQ4_1S` | `GGML_TYPE_TQ4_1S` (46)    | model weights, Lloyd-Max    | 4 bits, block size 32  |

Turbo KV cache types are **runtime-only**: they are never stored in model files. `TQ3_1S` / `TQ4_1S` are first-class weight types usable in any GGUF via `llama-quantize`.

## Quick start

```sh
# Build from source (required for TurboQuant: upstream pre-built binaries do NOT include it)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)

# Or build and run with Docker (docs/docker.md)
# Or download a binary built from THIS repository.
# Upstream releases (github.com/ggml-org/llama.cpp/releases) reject turbo types.
```

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server with a compressed KV cache
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF -c 32768 -ngl 99 -ctk q8_0 -ctv turbo3
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve`" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
</table>

## TurboQuant KV cache

```bash
# CLI
llama-cli -m model.gguf -c 8192 -ngl 99 --cache-type-k q8_0 --cache-type-v turbo3

# OpenAI-compatible server (same flags)
llama-server -m model.gguf -c 32768 -ngl 99 --cache-type-k q8_0 --cache-type-v turbo3
```

| Type      | Bits/value | KV cache size vs f16 K/V |
|-----------|------------|--------------------------|
| `turbo2`  | 2.0        | up to 6.4x smaller       |
| `turbo3`  | 3.25       | up to 4.9x smaller       |
| `turbo4`  | 4.25       | up to 3.8x smaller       |

Any combination of `f16`, `q8_0`, `q4_0`, `q4_1`, `q5_0`, `q5_1`, `iq4_nl` and `turbo2/3/4` for K and V is accepted by `llama-cli`, `llama-server`, `llama-bench` (`-ctk` / `-ctv`) and `llama-perplexity`. K `q8_0` + V `turbo3` is the common starting point: the KV cache drops to about 35% of f16 K/V size.

Notes:

- Turbo types **require flash attention**. If FA is not enabled it is turned on automatically (a warning is printed). A quantized V cache with FA explicitly disabled is an error.
- **MLA models (DeepSeek)**: there is no separate V cache (V is a view of K), so V rotation and padding are skipped, and the K and V cache types must be identical.
- **Attention-sink models (GPT-OSS and similar)**: even `q8_0` K changes output distributions, and lower-bit K degrades them further. Use `f16` K for these models. Validate a quantized V cache against an `f16` baseline with `llama-perplexity --kl-divergence`: short samples can stay fluent while token probabilities move.

Choosing a type:

- `turbo4`: quality-first (highest fidelity, 3.8x smaller than f16).
- `turbo3`: capacity sweet spot (4.9x).
- `turbo2`: maximum context in minimum VRAM (6.4x), most quality loss.

Full usage, model-specific quality notes and all environment knobs: [docs/KV-cache-quantization.md](docs/KV-cache-quantization.md) is the authoritative reference; this README is a summary.

## TQ weight types (TQ3_1S / TQ4_1S)

```bash
# Quantize a model to WHT-rotated Lloyd-Max weight types
llama-quantize model-f16.gguf model-tq4.gguf TQ4_1S
llama-quantize model-f16.gguf model-tq3.gguf TQ3_1S
```

`TQ3_1S` (4.00 bpw) and `TQ4_1S` (5.00 bpw) are supported on CPU, CUDA/HIP (warp-cooperative `mmvq` with fused `dp4a`), Metal, Vulkan and SYCL, and work in `llama-cli`, `llama-server`, `llama-bench` and `llama-perplexity`.

On CUDA, TQ4_1S weights are converted to `q8_0` at load time by default (best prefill speed). Set `GGML_TQ_NATIVE=1` to keep them native: decode is about 30% faster and weight VRAM drops about 1.7x, but prefill is about 2x slower. Pick it for decode-heavy serving; keep the default for prefill-heavy or mixed workloads.

## Backend support (TurboQuant features)

| Feature                     | CUDA/HIP | Metal | Vulkan | SYCL | CPU | WebGPU |
|-----------------------------|----------|-------|--------|------|-----|--------|
| `turbo2/3/4` KV cache       | yes      | yes   | yes    | yes  | yes | no     |
| Flash attention (turbo)     | yes      | yes   | yes    | yes  | yes | no     |
| `TQ3_1S` / `TQ4_1S` weights | yes      | yes   | yes    | yes  | yes | no     |
| `GGML_TQ_NATIVE`            | yes      | n/a   | n/a    | n/a  | n/a | n/a    |

## Environment knobs

| Variable                                  | Default | Effect |
|-------------------------------------------|---------|--------|
| `TURBO_LAYER_ADAPTIVE`                    | `0`     | Layer-adaptive KV precision; `7` = Boundary V (first/last layers `q8_0`, middle layers turbo) |
| `TURBO_AUTO_ASYMMETRIC`                   | `1`     | Auto-select asymmetric K/V types for large-GQA models (`0` disables) |
| `TURBO_SPARSE_V`                          | `1`     | Sparse-V dequant skip in flash attention, Metal only (`0` disables) |
| `GGML_TQ_NATIVE`                          | unset   | CUDA: `1` disables the load-time TQ4_1S to q8_0 conversion and uses fused native TQ kernels |
| `GGML_CUDA_FUSE_CHAIN`                    | unset   | CUDA: `0` disables elementwise chain fusion (SILU/GELU/ADD/MUL/SCALE/CLAMP run in one kernel) |
| `GGML_CUDA_Q8CACHE`                       | unset   | CUDA: `0` disables the per-graph shared q8_1 activation cache in `mmvq` |
| `LLAMA_ATTN_ROT_K_OVERRIDE` / `LLAMA_ATTN_ROT_V_OVERRIDE` | off | Enable optional upstream attention rotation for K / V (TurboQuant manages its own rotation) |
| `LLAMA_ATTN_ROT_K_NROT`                   | `64`    | Rotation size for the optional upstream K path |
| `LLAMA_ATTN_ROT_DISABLE`                  | `0`     | Hard lock-out: force rotation off on both sides (`1` disables) |

This table is a summary for users; [docs/KV-cache-quantization.md](docs/KV-cache-quantization.md) is the single source of truth and may list more variables or different defaults.

## Common problems

- `Unsupported cache type: turbo3`: you are running an upstream pre-built binary. Build this repository instead.
- Crash or abort on the first turbo KV write (Metal NULL-pipeline deref, Vulkan SET_ROWS abort): stale binary; update to the latest fork build.
- NaN or garbage output with `TQ4_1S` weights on CUDA: upgrade to a current build (fused TQ paths are gated on contiguous tensors), or set `GGML_TQ_NATIVE=1`.
- Output looks fluent but token probabilities changed: attention-sink model. Use `f16` K and compare with `llama-perplexity --kl-divergence` before blaming the codec.
- `not supported [backend]` lines or `0/0 tests passed` in `test-backend-ops`: the cases for your change were skipped, not passed.

## Scope and focus

This fork is not a superset of upstream. Upstream tracks a very wide range of hardware, models and features; this fork keeps a narrower, curated focus:

- **Backends**: the TurboQuant work is developed and tested for CPU, CUDA, HIP (AMD), Metal (Apple silicon), Vulkan and SYCL. Other upstream backends (CANN, MUSA, OpenCL, OpenVINO, ZenDNN, WebGPU, RPC, etc.) are inherited from the upstream base but are not a focus and receive no extra validation here.
- **Models**: new model support is brought in from upstream when it adds benefit (for example attention shapes or GQA/MLA variants that exercise the turbo paths), not across the board.
- **Features**: upstream changes are cherry-picked selectively, and fork features (turbo cache types, TQ weight types) are where the extra kernel and test investment goes.

Plain C/C++ implementation on top of the [ggml](https://github.com/ggml-org/ggml) library, no external runtime dependencies; upstream build instructions still apply, see [docs/build.md](docs/build.md).

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [KV cache quantization with TurboQuant](docs/KV-cache-quantization.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [Models](docs/models.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches and merge PRs
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information
- **Fork development rules**: test gates, coverage limits, known pitfalls and the git sync workflow live in [AGENTS.md](AGENTS.md). Read it before touching quantization or backend code.
- Issue triage: report TurboQuant-specific issues (turbo/TQ cache types, TQ weights, kernel crashes) in this repository; report general llama.cpp issues upstream.

## License

This project is licensed under [MIT](LICENSE), the same as upstream. All TurboQuant additions in this fork are under the same license.

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [stb-image](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [miniaudio.h](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain

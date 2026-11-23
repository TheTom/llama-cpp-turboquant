# llama.cpp (TurboQuant fork)

![llama.cpp](https://raw.githubusercontent.com/ggml-org/llama-brand/refs/heads/master/logo/svg/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++ with aggressive KV cache and weight quantization</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (upstream). This tree adds the **TurboQuant** feature set on top of an upstream base: upstream changes are cherry-picked on an ongoing basis where they add benefit, rather than being merged wholesale.

[manifesto](https://github.com/ggml-org/llama.cpp/discussions/205) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## About this fork

**TurboQuant** compresses the attention KV cache far beyond the standard `q8_0`, and adds WHT-rotated and ConvRot model-weight types. The method is data-oblivious online vector quantization ([TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874)) applied with a fixed 128x128 Walsh-Hadamard rotation (`GGML_OP_TURBO_WHT`): cache vectors are rotated before quantization (which Gaussianizes their distribution) and inverse-rotated after dequantization. Head dims that are not multiples of 128 are zero-padded.

Ten fork-only GGML types (registered in `ggml/include/ggml.h`):

| Type     | Enum                       | Purpose                     | Size                   | Backends            |
|----------|----------------------------|-----------------------------|------------------------|---------------------|
| `turbo2` | `GGML_TYPE_TURBO2_0` (43)  | KV cache only               | 2.125 bits/value       | CPU, CUDA/HIP, Metal, Vulkan, SYCL |
| `turbo3` | `GGML_TYPE_TURBO3_0` (44)  | KV cache only               | 3.125 bits/value       | CPU, CUDA/HIP, Metal, Vulkan, SYCL |
| `turbo4` | `GGML_TYPE_TURBO4_0` (47)  | KV cache only               | 4.125 bits/value       | CPU, CUDA/HIP, Metal, Vulkan, SYCL |
| `turbo5` | `GGML_TYPE_TURBO5_0` (51)  | KV cache only               | 5.125 bits/value       | CPU, CUDA/HIP       |
| `turbo6` | `GGML_TYPE_TURBO6_0` (52)  | KV cache only               | 6.125 bits/value       | CPU, CUDA/HIP       |
| `TQ3_1S` | `GGML_TYPE_TQ3_1S` (45)    | model weights, WHT-rotated Lloyd-Max | 4.00 bits, block 32 | CPU, CUDA/HIP, Metal, Vulkan |
| `TQ4_1S` | `GGML_TYPE_TQ4_1S` (46)    | model weights, WHT-rotated Lloyd-Max | 5.00 bits, block 32 | CPU, CUDA/HIP, Metal, Vulkan |
| `Q8_CR`  | `GGML_TYPE_Q8_CR` (48)     | model weights, ConvRot      | 8.5 bits, group 256    | CPU, CUDA/HIP, Metal |
| `Q5_CR`  | `GGML_TYPE_Q5_CR` (49)     | model weights, ConvRot      | 5.5 bits, group 256    | CPU, CUDA/HIP, Metal |
| `Q6_CR`  | `GGML_TYPE_Q6_CR` (50)     | model weights, ConvRot      | 6.5625 bits, group 256 | CPU, CUDA/HIP, Metal |

The turbo KV cache blocks hold 128 values each: 34, 50, 66, 82 and 98 bytes for
`turbo2` to `turbo6`, which is where the bits/value figures above come from
(the `static_assert`s in `ggml/src/ggml-common.h` are the authority).
`Q8_CR` / `Q5_CR` / `Q6_CR` are
`Q8_0` / `Q5_0` / `Q6_K` blocks holding a group of 256 weights rotated with a
normalized Kronecker-Hadamard matrix; `mul_mat` rotates the other operand the
same way before the dot products. Vulkan and SYCL reject them.

Turbo KV cache types are **runtime-only**: they are never stored in model files. `TQ3_1S` / `TQ4_1S` and the `*_CR` types are first-class weight types usable in any GGUF via `llama-quantize`.
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v1)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[v1.0.0](https://huggingface.co/blog/ggml-org/llama-cpp-v1) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/pulls?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Asanmai%20OR%20author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Athelittlefireman%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3AmasterFoad%20OR%20author%3Aallozaur%20OR%20author%3AAndreas-Krebbel%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Achaxu01%20OR%20author%3Alhez%20OR%20author%3AAlekseiNikiforovIBM%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming%20OR%20author%3Aynankani)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Description

The main goal of `llama.cpp` is to enable LLM (and MLLM) inference with minimal setup and state-of-the-art performance on a wide range of hardware - locally and in the cloud.

The `llama.cpp` project is built on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Quick start

```sh
# Build from source (required for TurboQuant: upstream pre-built binaries do NOT include it)
# This is a CPU-only build; see docs/build.md for the CUDA, HIP, Metal, Vulkan and SYCL flags
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)

# Or build and run with Docker (docs/docker.md)
# Or download a binary built from THIS repository.
# Upstream releases (github.com/ggml-org/llama.cpp/releases) reject turbo types.
# The ghcr.io/ggml-org/llama.cpp images in docs/docker.md are upstream builds and do too.
```
```bash
# curl
curl -LsSf https://llama.app/install.sh | sh

# powershell
irm https://llama.app/install.ps1 | iex
```

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

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
            <img width="1392" height="958" alt="Llama Chat, the built-in chat app, against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Llama Chat, the built-in chat app, against <b>llama serve</b></i>
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
| `turbo2`  | 2.125      | up to 7.5x smaller       |
| `turbo3`  | 3.125      | up to 5.1x smaller       |
| `turbo4`  | 4.125      | up to 3.9x smaller       |
| `turbo5`  | 5.125      | up to 3.1x smaller       |
| `turbo6`  | 6.125      | up to 2.6x smaller       |

Any combination of `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0`, `q5_1` and `turbo2/3/4/5/6` for K and V is accepted by the flag parser of `llama-cli`, `llama-server`, `llama-bench` (`-ctk` / `-ctv`) and `llama-perplexity`. Which combinations actually have backend kernels is narrower and backend-dependent; [docs/KV-cache-quantization.md](docs/KV-cache-quantization.md) has the per-backend picture. K `q8_0` + V `turbo3` is the common starting point: the KV cache drops to about 36% of f16 K/V size.

Notes:

- Turbo types **require flash attention**. If FA is not enabled it is turned on automatically (a warning is printed). A quantized V cache with FA explicitly disabled is an error, but a turbo V cache never reaches that check because it turns FA on first.
- **Metal**: flash attention accepts exactly three mixed pairs: turbo K + turbo V, `q8_0` K + turbo V, and turbo K + `q8_0` V (`turbo` is `turbo2/3/4` here; `turbo5/6` have no Metal kernels). Any other mixed pair, an `f16` K cache with a turbo V cache included, trips the assert in `ggml_metal_op_flash_attn_ext` and aborts. This is the backend to keep in mind for the attention-sink note below.
- **MLA models (DeepSeek)**: there is no separate V cache (V is a view of K), so V rotation and padding are skipped, and the K and V cache types must be identical.
- **Attention-sink models (GPT-OSS and similar)**: even `q8_0` K changes output distributions, and lower-bit K degrades them further. Use `f16` K for these models. Validate a quantized V cache against an `f16` baseline with `llama-perplexity --kl-divergence`: short samples can stay fluent while token probabilities move.

Choosing a type:

- `turbo6`: highest-fidelity turbo option (2.6x smaller than f16), CUDA/HIP and CPU only.
- `turbo5`: between `turbo4` and `turbo6` (3.1x), CUDA/HIP and CPU only.
- `turbo4`: quality-first for the widely supported set (3.9x smaller than f16).
- `turbo3`: capacity sweet spot (5.1x).
- `turbo2`: maximum context in minimum VRAM (7.5x), most quality loss.

`turbo2`/`turbo3`/`turbo4` are the portable choices; `turbo5`/`turbo6` currently run on CUDA/HIP and CPU only.

Full usage, per-backend support and model-specific quality notes: [docs/KV-cache-quantization.md](docs/KV-cache-quantization.md). That doc is the source of truth for the environment knobs summarized below.

## TQ weight types (TQ3_1S / TQ4_1S)

```bash
# Quantize a model to WHT-rotated Lloyd-Max weight types
llama-quantize model-f16.gguf model-tq4.gguf TQ4_1S
llama-quantize model-f16.gguf model-tq3.gguf TQ3_1S
```

`TQ3_1S` (4.00 bpw) and `TQ4_1S` (5.00 bpw) are supported on CPU, CUDA/HIP (warp-cooperative `mmvq` with fused `dp4a`), Metal and Vulkan, and work in `llama-cli`, `llama-server`, `llama-bench` and `llama-perplexity`. ggml-sycl has no TQ kernels, and its generic dequantize path leaves the weights in the WHT-rotated domain, so TQ models are not supported on SYCL.

On CUDA, `TQ4_1S` weights are converted to `q8_0` at load time by default (best prefill speed). Set `GGML_TQ_NATIVE=1` to keep them native: decode is about 30% faster and weight VRAM drops about 1.7x, but prefill is about 2x slower. Pick it for decode-heavy serving; keep the default for prefill-heavy or mixed workloads.

## Backend support (TurboQuant features)

| Feature                     | CUDA/HIP | Metal | Vulkan | SYCL | CPU | WebGPU |
|-----------------------------|----------|-------|--------|------|-----|--------|
| `turbo2/3/4` KV cache       | yes      | yes   | yes    | yes  | yes | no     |
| `turbo5/6` KV cache         | yes      | no    | no     | no   | yes | no     |
| Flash attention (turbo2/3/4) | yes     | yes   | yes    | yes  | yes | no     |
| Flash attention (turbo5/6)  | yes      | no    | no     | no   | yes | no     |
| `TQ3_1S` / `TQ4_1S` weights | yes      | yes   | yes    | no   | yes | no     |
| `Q8_CR` / `Q5_CR` / `Q6_CR` weights | yes | yes | no    | no   | yes | no     |
| `GGML_TQ_NATIVE`            | yes      | n/a   | n/a    | n/a  | n/a | n/a    |

## Environment knobs

| Variable                                  | Default | Effect |
|-------------------------------------------|---------|--------|
| `TURBO_LAYER_ADAPTIVE`                    | auto    | Layer-adaptive KV precision; mode 7 (Boundary V) auto-enables for `turbo2` V on models with 8+ layers |
| `TURBO_AUTO_ASYMMETRIC`                   | `1`     | Auto-select asymmetric K/V types for large-GQA models (`0` disables) |
| `TURBO_SPARSE_V`                          | `1`     | Sparse-V dequant skip in flash attention, Metal and `turbo3` V only (`0` disables) |
| `GGML_TQ_NATIVE`                          | unset   | CUDA: `1` keeps `TQ4_1S` weights native instead of converting them to `q8_0` at load |
| `GGML_CUDA_FUSE_CHAIN`                    | unset   | CUDA: `0` disables elementwise chain fusion |
| `GGML_CUDA_Q8CACHE`                       | unset   | CUDA: `0` disables the per-graph shared q8_1 activation cache in `mmvq` |
| `LLAMA_ATTN_ROT_K_OVERRIDE` / `LLAMA_ATTN_ROT_V_OVERRIDE` | off | Enable optional upstream attention rotation for K / V (TurboQuant manages its own rotation) |
| `LLAMA_ATTN_ROT_K_NROT`                   | `64`    | Rotation size for the optional upstream K path (`0` = upstream default) |
| `LLAMA_ATTN_ROT_DISABLE`                  | `0`     | Hard lock-out: force rotation off on both sides (`1` disables) |

This is a summary for users. [docs/KV-cache-quantization.md](docs/KV-cache-quantization.md) is the source of truth for these variables; if the doc and the code ever disagree, the code wins and both need fixing in the same PR.

## Common problems

- `Unsupported cache type: turbo3`: you are running an upstream pre-built binary. Build this repository instead. The `ghcr.io/ggml-org/llama.cpp` images in [docs/docker.md](docs/docker.md) count as upstream binaries.
- Crash or abort on the first turbo KV write (Metal NULL-pipeline deref, Vulkan SET_ROWS abort): stale binary; update to the latest fork build.
- Abort in the Metal flash attention kernel with a turbo cache: the K/V pair is not one of turbo+turbo, `q8_0`+turbo or turbo+`q8_0` (turbo = `turbo2/3/4`). Use one of those, or an `f16` V cache with `f16` K.
- NaN or garbage output with `TQ4_1S` weights on CUDA: the fused TQ kernels, which only `GGML_TQ_NATIVE=1` enters, had NaN bugs. Build a current fork revision and leave `GGML_TQ_NATIVE` unset; the default load-time `q8_0` conversion does not use them.
- Output looks fluent but token probabilities changed: attention-sink model. Use `f16` K (not available with a turbo V cache on Metal) and compare with `llama-perplexity --kl-divergence` before blaming the codec.
- `not supported [backend]` lines or `0/0 tests passed` in `test-backend-ops`: the cases for your change were skipped, not passed.

## Scope and focus

This fork is not a superset of upstream. Upstream tracks a very wide range of hardware, models and features; this fork keeps a narrower, curated focus:

- **Backends**: the TurboQuant work is developed and tested for CPU, CUDA, HIP (AMD), Metal (Apple silicon), Vulkan and SYCL. Other upstream backends (CANN, MUSA, OpenCL, OpenVINO, ZenDNN, WebGPU, RPC, etc.) are inherited from the upstream base but are not a focus and receive no extra validation here.
- **Models**: new model support is brought in from upstream when it adds benefit (for example attention shapes or GQA/MLA variants that exercise the turbo paths), not across the board.
- **Features**: upstream changes are usually cherry-picked where they add benefit, and fork features (turbo cache types, TQ and ConvRot weight types) are where the extra kernel and test investment goes. There has been one wholesale upstream sync, `a34005435` (upstream `15586e2d7`, Aug 2026, PR #287); treat further bulk syncs as a decision to make deliberately, not a default.

Plain C/C++ implementation on top of the [ggml](https://github.com/ggml-org/ggml) library, no external runtime dependencies; upstream build instructions still apply, see [docs/build.md](docs/build.md).

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [Text-to-speech (TTS) models](docs/multimodal/tts.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md) (upstream images, no TurboQuant)
- [KV cache quantization with TurboQuant](docs/KV-cache-quantization.md)
- [SYCL backend](docs/backend/SYCL.md) (`turbo2/3/4` KV cache only, no TQ weights)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [Models](docs/models.md)
- [Release process](docs/release.md)

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
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain

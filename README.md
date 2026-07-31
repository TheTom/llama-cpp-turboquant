# llama.cpp + TurboQuant+

> Production-grade KV-cache and weight quantization for llama.cpp, with cross-backend kernel support for Apple Silicon, NVIDIA CUDA, AMD ROCm, and Vulkan.

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Status: WIP](https://img.shields.io/badge/status-work--in--progress-yellow.svg)](https://github.com/TheTom/llama-cpp-turboquant)
[![Codec papers](https://img.shields.io/badge/codec-turboquant__plus-orange.svg)](https://github.com/TheTom/turboquant_plus)

A fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) integrating the **TurboQuant+** codec stack — Walsh-Hadamard rotated polar quantization, attention-gated sparse dequantization, and layer-aware V compression policies. The codec design, calibration, and validation papers live at [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus); this repository is the llama.cpp runtime integration.

### Lineage — why the `+`

TurboQuant+ is inspired by Google's original **TurboQuant** paper (ICLR 2026), which introduced Walsh-Hadamard-rotated polar codebook quantization for KV cache and demonstrated 4.6× compression at ~1% PPL loss. This project extends that foundation substantially — adding the asymmetric K/V policy (V is free, K is everything), layer-aware Boundary V protection, attention-gated sparse V dequantization, the `TQ3_1S` / `TQ4_1S` weight quantization formats, the `turbo2` / `turbo4` tier variants, the cross-backend kernel coverage (CUDA `dp4a`, HIP/ROCm RDNA/CDNA, Vulkan coopmat, Metal TurboFlash + V2.1 fused kernels), and a body of model-family-specific quality and operational fixes. The trailing `+` denotes that ongoing extension work; the original TurboQuant codec remains the foundation.

This fork is additive: every existing llama.cpp quantization, model, and backend continues to work unchanged. New types are opt-in via the standard `--cache-type-k` / `--cache-type-v` and `llama-quantize` interfaces.

## Production deployments

This fork's TurboQuant integration is used in:

- [**LocalAI**](https://localai.io) — drop-in OpenAI-compatible local inference server
- [**Chronara**](https://chronara.io) — quantum-safe fintech infrastructure with AI-driven networks
- [**AtomicChat**](https://atomic.chat/) — on-device chat application
- and other downstream projects

## Status

| | |
|---|---|
| Default branch | `feature/turboquant-kv-cache` |
| Commits ahead of upstream | ~300 |
| Upstream tracking | continuous sync from `ggml-org/llama.cpp` master |
| Upstream PR status | not yet upstreamed; running as a long-lived feature branch |

---

## What this fork adds

### Quantization types

| Type | Domain | Approx. bits | Notes | Paper |
|---|---|---|---|---|
| `TQ3_1S` | weights | ~3.5 | smaller VRAM than `q8_0` | [weight-compression-tq4](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/weight-compression-tq4.md) |
| `TQ4_1S` | weights | ~4.5 | V2.1 fused Metal kernels; CUDA `dp4a` 3.5× faster (240 t/s vs 68 baseline) | [weight-compression-tq4](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/weight-compression-tq4.md) |
| `turbo2` | KV cache | ~2.0 | aggressive; pair with Boundary V | [block-size-experiment](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/block-size-experiment.md) |
| `turbo3` | KV cache | ~3.5 | ~4.6× compression at <1.5% PPL loss | [attn-rotation-and-ppl-artifact](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/attn-rotation-and-ppl-artifact.md) |
| `turbo4` | KV cache | ~4.5 | rehabilitated to beat `q4_0` on fidelity | [turbo4-resurrection](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/turbo4-resurrection.md) |

All turbo formats use Walsh-Hadamard rotation followed by polar codebook quantization on 128-element blocks. Why this works where MSE-driven codecs fail: [why-mse-fails-for-kv-quantization](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/why-mse-fails-for-kv-quantization.md).

### Compression policies

- **Auto-asymmetric K/V compression** — recognizes that V tolerates aggressive compression while K does not; default policy picks complementary codecs rather than symmetric. [asymmetric-kv-compression](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/asymmetric-kv-compression.md)
- **Boundary V (experimental, layer-aware)** — auto-enabled for `turbo2-V`. Protects layers where aggressive V quantization degrades quality, leaves the rest at full aggression. [layer-aware-v-compression](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/layer-aware-v-compression.md), [moe-v-compression-frontier](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/moe-v-compression-frontier.md)
- **Sparse V dequantization** — skip V dequantization for positions whose softmax attention weight falls below threshold. Enabled across all Metal targets. [sparse-v-dequant](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/sparse-v-dequant.md)

### Backend coverage

| Backend | Quant kernels | Flash Attention | Notes |
|---|---|---|---|
| **Metal** (Apple Silicon) | TQ V2.1 fused, TurboFlash | Yes — sparse V across the family; `dk=512` FA kernels for Gemma 4 | TurboFlash off by default on Apple10 (corruption regression under investigation). [m5-max-stress-test](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/m5-max-stress-test.md) |
| **CUDA** (NVIDIA) | `dp4a` for `TQ4_1S`, warp-cooperative dequant (16× less compute per block), multi-token / multi-GPU | Yes — turbo VEC FA (+9% decode); mixed `f16/bf16 + q8_0` without `GGML_CUDA_FA_ALL_QUANTS` | Load-time `TQ4_1S → q8_0` conversion path |
| **HIP / ROCm** (AMD) | Portable `ggml_cuda_dp4a`; scalar half path for `TQ4_1S` on AMD | Yes — VEC FA forced for quantized KV; pool bypass for FA f16 temp buffers | RDNA3 (gfx1100), RDNA4, CDNA3 (MI300X / gfx942), CDNA4 (MI355X / gfx950). [cross-engine-mi300x](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/cross-engine-mi300x.md) |
| **Vulkan** | `TQ4_1S` weights, `SET_ROWS` for `turbo2`/`turbo4` | coopmat flash attention with `turbo3` KV | Compute-shader path; nix-buildable |
| **SYCL** (Intel) | `SET_ROWS` for `turbo2`/`turbo3`/`turbo4`, WHT rotation (group 64+128) | VEC FA for all turbo K/V combos (D=64-512) | Intel Arc (A380, B70), oneAPI 2025.2+; cross-turbo K/V combos supported |

### Model-family support

- **Gemma 4** — `dk=512` Metal FA kernels, MoE token routing, op-concurrency handling
- **Large MoE** — kernel instantiations for up to 256-expert routing
- **Hybrid architectures (GDN, Mamba)** — speculative decoding cherry-picked from upstream feature branches
- All existing llama.cpp model families remain fully supported

### Operational fixes carried by this fork

- CPU `vec_dot` heap-allocation fix for turbo / TQ types at `n > 4096`
- Apple Silicon unified-memory explosion fix
- RPC `GGML_OP_COUNT` assertion fix
- Cross-vendor `-Werror` build fixes
- Defensive `xxd.cmake` handling for missing input files

---

## Quick start

### Prebuilt binaries

The latest TurboQuant+ prebuilds are published on the [TurboQuant+ tqp-v0.3.0 release](https://github.com/TheTom/llama-cpp-turboquant/releases/tag/tqp-v0.3.0). This release adds DFlash speculative decoding, server slot save/restore across restarts, the Vulkan turbo4 layout resync, the sm_60/P100 FAST_FP16 carve-out, and 16 upstream P0/P1 fixes on top of the PR #197 turbo KV work.

| Platform | Download | Notes |
|---|---|---|
| Linux x64 | [turboquant-plus-tqp-v0.3.0-linux-x64-cpu.tar.gz](https://github.com/TheTom/llama-cpp-turboquant/releases/download/tqp-v0.3.0/turboquant-plus-tqp-v0.3.0-linux-x64-cpu.tar.gz) | CPU build with portable x64 CPU variants |
| Linux x64 Vulkan | [turboquant-plus-tqp-v0.3.0-linux-x64-vulkan.tar.gz](https://github.com/TheTom/llama-cpp-turboquant/releases/download/tqp-v0.3.0/turboquant-plus-tqp-v0.3.0-linux-x64-vulkan.tar.gz) | Vulkan build with CPU fallback variants |
| macOS Apple Silicon | [turboquant-plus-tqp-v0.3.0-macos-arm64-metal.tar.gz](https://github.com/TheTom/llama-cpp-turboquant/releases/download/tqp-v0.3.0/turboquant-plus-tqp-v0.3.0-macos-arm64-metal.tar.gz) | Metal build for arm64 Macs |
| Windows x64 NVIDIA | [turboquant-plus-tqp-v0.3.0-windows-x64-cuda12.4.zip](https://github.com/TheTom/llama-cpp-turboquant/releases/download/tqp-v0.3.0/turboquant-plus-tqp-v0.3.0-windows-x64-cuda12.4.zip) | CUDA 12.4 build with CUDA runtime DLLs bundled |

For ROCm/HIP or custom CUDA architectures, build from source with the flags below.

### Build from source

Standard llama.cpp build flags. TurboQuant types become available automatically once the matching backend is compiled in.

```bash
# Apple Silicon (Metal)
cmake -B build -DGGML_METAL=ON && cmake --build build -j

# NVIDIA CUDA
cmake -B build -DGGML_CUDA=ON && cmake --build build -j

# AMD HIP / ROCm (multi-arch fat binary)
cmake -B build -DGGML_HIP=ON -DCMAKE_HIP_ARCHITECTURES="gfx1100;gfx942;gfx950" && cmake --build build -j

# Vulkan
cmake -B build -DGGML_VULKAN=ON && cmake --build build -j

# Intel SYCL (oneAPI)
source /opt/intel/oneapi/setvars.sh
cmake -B build -DGGML_SYCL=ON -DGGML_SYCL_F16=ON \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx && cmake --build build -j
```

## Usage

### KV-cache quantization (runtime)

KV-cache types are selected per-side via the standard `--cache-type-k` / `--cache-type-v` flags.

> **Start light, then compress.** Some model families — small models, certain MoE configurations, quant-sensitive instruction-tuned variants — are more delicate than others. Pick a light asymmetric configuration first, verify output quality (eyeball + PPL on a hold-out set) on your specific model, then ratchet up V aggression if you have memory headroom to gain. **Do not start at maximum compression** and work backwards.

The core finding from the asymmetric-kv-compression paper — [**Asymmetric K/V Cache Compression: Why V is Free and K is Everything**](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/asymmetric-kv-compression.md) — drives all the configs below: **V tolerates aggressive compression, K does not**. Always keep K at higher precision than V; never start symmetric. That paper documents the specific failure modes you'll hit if you ignore this and compress K aggressively (PPL blow-up on certain model families, attention-rotation interaction with low-bit K, etc.) — read it before considering step 6.

Higher turbo number = more bits per element = less aggressive compression. The V-side compression ladder is `turbo4` (lightest) → `turbo3` → `turbo2` (heaviest). On the K side, prefer `f16` or `q8_0`; never lead with a turbo K.

Recommendations, ordered from most conservative to most aggressive:

| Step | `--cache-type-k` | `--cache-type-v` | When | Notes |
|---|---|---|---|---|
| **1. Safest start** | `f16` | `turbo4` | First contact with any new model | K untouched, V at the lightest turbo tier. If output isn't faithful at this step, the model is unusually quant-sensitive — stop and investigate before escalating. |
| **2. Conservative** | `q8_0` | `turbo4` | Verified safe at step 1, want a memory win without much risk | Light on both sides. Typically near-indistinguishable from `f16`/`f16` outputs. |
| **3. Recommended default** | `q8_0` | `turbo3` | Most dense models, most production workloads | The "asymmetric turbo" sweet spot from the [asymmetric-kv-compression](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/asymmetric-kv-compression.md) paper. Near-lossless K, ~4.6× compressed V. Total KV ~3-4× smaller than `f16`/`f16`. |
| **4. Aggressive V** | `q8_0` | `turbo2` | Memory-bound long context, after validating quality at step 3 | Boundary V auto-engages and protects sensitive layers. Expect <2% PPL loss on dense models outside the protected layers. |
| **5. MoE-aware aggressive** | `q8_0` | `turbo2` | Large MoE models (DeepSeek, Qwen3.6, Mixtral-style) | Same flags; Boundary V's per-expert-boundary protection is what makes this work on MoE. See [moe-v-compression-frontier](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/moe-v-compression-frontier.md). |
| **6. Discouraged: symmetric K compression** | any `turbo*` | any `turbo*` | Only with model-specific quality validation in hand | Compressing K is where models break. The asymmetric paper documents the failure modes. Not a starting point. |

Example invocations:

```bash
# Step 1 — safest start (first contact with a new model)
llama-cli -m model.gguf --cache-type-k f16 --cache-type-v turbo4 -p "..."

# Step 3 — recommended default (asymmetric turbo)
llama-cli -m model.gguf --cache-type-k q8_0 --cache-type-v turbo3 -p "..."

# Step 4 — aggressive V at long context
llama-cli -m model.gguf --cache-type-k q8_0 --cache-type-v turbo2 -c 131072 -p "..."
```

If output quality drops between steps, walk back to the previous step. The compression frontier is per-model — there is no global "best" setting.

### Weight quantization (offline)

Weight quantization is selected at conversion time via `llama-quantize`:

```bash
# TQ4_1S — recommended for most CUDA / HIP deployments (dp4a 3.5× faster than baseline)
llama-quantize model.f16.gguf model.tq4_1s.gguf TQ4_1S

# TQ3_1S — smaller, accept ~1-2 PPL bump
llama-quantize model.f16.gguf model.tq3_1s.gguf TQ3_1S
```

### Server slot save/restore across restarts

`llama-server` can persist a slot's KV state to disk and reload it later, including across a full process restart:

```bash
# start the server with a save path and context checkpoints enabled
llama-server -m model.gguf --slot-save-path /path/to/slots --ctx-checkpoints 8

# save slot 0 to disk
curl -X POST 'http://localhost:8080/slots/0?action=save' \
  -H 'Content-Type: application/json' -d '{"filename": "session1.bin"}'

# ... restart the server ...

# restore slot 0 from disk in the new process
curl -X POST 'http://localhost:8080/slots/0?action=restore' \
  -H 'Content-Type: application/json' -d '{"filename": "session1.bin"}'
```

Saving also writes a `session1.bin.ckpt` sidecar next to the state file, carrying the slot's in-memory context checkpoints (upstream's `llama_state_seq_save_file` only serializes tokens + KV cells, not checkpoints — they otherwise exist only in process memory and are lost across a restart). Restoring loads the sidecar back in, so a request that rolls back mid-prompt after a restore (e.g. a BPE re-tokenization at the tail of a long conversation) can resume from a checkpoint instead of forcing a full re-prefill.

**This only matters for SWA or hybrid/recurrent-memory models** (Gemma-style sliding-window attention, Mamba/hybrid architectures). Plain dense-attention models never evict early KV positions, so the server can always truncate the cache to an earlier point directly — the checkpoint/rollback path is unreachable for them regardless of this feature, with or without a restart in between.

If a state file has no sidecar (saved by an older build, or the file was moved on its own), restore falls back to synthesizing a single checkpoint from the tail of the restored state. That covers an exact-append continuation but not a deeper rollback, which degrades to a full re-prefill — the same behavior as before this feature existed, not a regression.

### Automatic behavior

The following activate based on the selected types — no flags required:

- **Auto-asymmetric K/V** — when both sides are turbo / TQ types, the policy picks complementary configurations rather than symmetric.
- **Boundary V (layer-aware)** — auto-enables for any `turbo2-V` selection.
- **Sparse V dequantization** — on Metal targets, sparse V activates for all turbo V types.
- **Flash Attention** — auto-enabled for turbo KV with the relevant backend kernel.

See the linked papers above for parameter selection guidance on a per-model basis.

## Citation

If this fork or any of its quantization types is used in your work, please cite the corresponding paper from the [TurboQuant+ paper corpus](https://github.com/TheTom/turboquant_plus/tree/main/docs/papers).

## License

MIT, same as upstream llama.cpp.

---

## Recent API changes

- [Changelog for `libllama` API](https://github.com/ggml-org/llama.cpp/issues/9289)
- [Changelog for `llama-server` REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

## Hot topics

- **Hugging Face cache migration: models downloaded with `-hf` are now stored in the standard Hugging Face cache directory, enabling sharing with other HF tools.**
- **[guide : using the new WebUI of llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/16938)**
- [guide : running gpt-oss with llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/15396)
- [[FEEDBACK] Better packaging for llama.cpp to support downstream consumers 🤗](https://github.com/ggml-org/llama.cpp/discussions/15313)
- Support for the `gpt-oss` model with native MXFP4 format has been added | [PR](https://github.com/ggml-org/llama.cpp/pull/15091) | [Collaboration with NVIDIA](https://blogs.nvidia.com/blog/rtx-ai-garage-openai-oss) | [Comment](https://github.com/ggml-org/llama.cpp/discussions/15095)
- Multimodal support arrived in `llama-server`: [#12898](https://github.com/ggml-org/llama.cpp/pull/12898) | [documentation](./docs/multimodal.md)
- VS Code extension for FIM completions: https://github.com/ggml-org/llama.vscode
- Vim/Neovim plugin for FIM completions: https://github.com/ggml-org/llama.vim
- Hugging Face Inference Endpoints now support GGUF out of the box! https://github.com/ggml-org/llama.cpp/discussions/9669
- Hugging Face GGUF editor: [discussion](https://github.com/ggml-org/llama.cpp/discussions/9268) | [tool](https://huggingface.co/spaces/CISCai/gguf-editor)
- WebGPU support is now available in the browser, see a blog/demo introducing it [here](https://reeselevine.github.io/llamas-on-the-web/).

----

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon [In Progress]](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [stb-image](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [miniaudio.h](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain

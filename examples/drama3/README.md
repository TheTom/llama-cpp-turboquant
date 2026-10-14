# Drama-3 experimental speech bridge

This lane runs the Fish Drama-3 backbone in this fork and the unmodified audio tower/codec in SGLang-Omni. It is a one-stream experimental bridge, not native FishOmni TTS support. Keep the original checkpoint's license with any derived model distribution.

Use a verified local `fishaudio/drama-3` checkpoint. Download once on the quantization machine and copy locally as needed. Every command below uses local files; set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` in the runtime environment.


## Status and measured limits

The CUDA bridge built and ran on a DGX Spark (NVIDIA GB10, sm_121a, Linux aarch64). C0 produced 11 EOS-complete diagnostic WAVs out of 12 prompts with seed 42. The tongue-twister reached the 1024-frame limit without EOS; it is a failure, not a complete sample. A warm excited-speech example produced 8.82 seconds of audio in 8.98 seconds of generation and 10.83 seconds of request wall time. These are diagnostic measurements, with uncontrolled filesystem cache, and exclude the first model load.

Teacher replay has substantial unresolved drift: final hidden relative L2 0.61054 and semantic-logit relative L2 0.16747; semantic top-1 agreed, top-10 overlap was 4/10. This compares different backends and precision together. It does not isolate quantization error. The original-precision native control and listening/ASR/speaker evaluation are still pending. Positive listening feedback is not an objective quality pass.

The RTX 5090 with 32 GiB is a target, not a verified platform. On Spark, llama reported a 26605.74 MiB CUDA model buffer and a 985.50 MiB CPU-mapped embedding buffer; the audio tower, codec, context caches, compute buffers and allocator overhead are additional. Spark unified memory does not prove a 5090 fit.

## Run the compact release

Linux, Python 3.12, a C++ compiler, CUDA 13 toolkit/NVCC, Git, libsndfile and ffmpeg are required. For Ubuntu 24.04, install the host packages with `sudo apt-get install python3.12-venv build-essential git libsndfile1 ffmpeg`. Install CUDA 13 using [NVIDIA's Linux installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) for your platform before the native build. Keep the downloaded model's LICENSE and NOTICE.

```sh
git clone --branch codex/drama3-backbone-adapter https://github.com/TheTom/llama-cpp-turboquant.git
cd llama-cpp-turboquant
git clone https://github.com/sgl-project/sglang-omni.git ../sglang-omni
git -C ../sglang-omni checkout 2cfb3b67becb437b70751defe37e63d3af4eedb0
bash examples/drama3/setup_runtime.sh "$PWD/../drama3-venv" "$PWD/../sglang-omni"
../drama3-venv/bin/python -m pip install cmake==4.1.0
../drama3-venv/bin/cmake -S . -B build-drama3 -DGGML_CUDA=ON -DLLAMA_BUILD_EXAMPLES=ON -DCMAKE_CUDA_ARCHITECTURES=121a
../drama3-venv/bin/cmake --build build-drama3 --target llama-drama3-backbone -j2
../drama3-venv/bin/hf download thetom-ai/Drama-3-ConfigI-GGUF --local-dir models/Drama-3-ConfigI
GGML_TQ_NATIVE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPATH="$PWD/../sglang-omni" ../drama3-venv/bin/python examples/drama3/external_audio.py \
  --source models/Drama-3-ConfigI/runtime \
  --runtime-manifest models/Drama-3-ConfigI/runtime/runtime-manifest.json \
  --gguf models/Drama-3-ConfigI/Drama-3-ConfigI-C0.gguf \
  --binary build-drama3/bin/llama-drama3-backbone \
  --omni-source "$PWD/../sglang-omni" --seed 42 \
  --text 'Hello, how are you?' --out hello.wav
```

`121a` is the tested Spark build target. The [NVIDIA compute capability table](https://developer.nvidia.com/cuda/gpus) lists GB10 as 12.1 and RTX 5090 as 12.0. An RTX 5090 uses `120`; the x86 install, build, total memory and speech route remain unverified. The release must be accessible to your Hugging Face account before `hf download` succeeds. Download the release once, then copy it between machines. The runtime performs no model downloads.

The compact runtime contains only the 192 original audio tensors, codec, tokenizer and configuration, about 4.04 GiB. The backbone is the separate 28,942,612,608-byte GGUF. `--runtime-manifest` verifies every asset and the GGUF SHA256, plus the complete audio tensor names, shapes and dtypes. It does not impersonate the original checkpoint receipt. The full original checkpoint path with `--verified-source-receipt` remains supported below.

`setup_runtime.sh` pins the known working CUDA 13 Torch 2.13.0 / TorchAudio 2.11.0 pair and imported package census. It installs SGLang packages without their unrelated full serving extras. A clean environment does not need the private experiment's filtered base-package view. The private vLLM image tag used during bringup is not a public dependency or a pull command. See `check_runtime.py` for the CPU import check; importing successfully does not prove GPU or speech execution.

## Backbone conversion

```sh
python examples/drama3/prepare_backbone.py /models/drama-3 /models/drama-3-backbone --verified-source-receipt /models/download.complete.json
python convert_hf_to_gguf.py /models/drama-3-backbone --outtype bf16 --outfile /models/drama-3-backbone-bf16.gguf
```

The staging command creates metadata and symlinks, without copying weights. Put the staging directory on a filesystem that supports symlinks; a CIFS NAS may reject them. Local staging links can point at NAS weights, while output and scratch files stay on the NAS. Mount the source at the same absolute path named in its verification receipt. The full `FishOmniModel` architecture stays unsupported by the ordinary converter. The staged architecture is explicitly `FishOmniBackboneForCausalLM`: it selects the Qwen3.5 MoE tensor/graph layout with noninterleaved partial RoPE and no MTP. Audio tensors stay in the original source for the external runtime.

The published expert FP8 tensors are expanded using their 128x128 F32 block multipliers. BF16 expansion needs more memory than source FP8 and must count toward the complete runtime budget. Do not silently use `--fp8-as-q8` for protected boundary layers and then describe them as unchanged.

The policy is adapted from Tom Turney’s TurboQuant Plus paper, [Tensor-Role-Aware Weight Compression for llama.cpp: TQ4_1S and the Config I Policy](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/weight-compression-tq4.md). Drama-3 C0 uses its tensor-role and boundary-protection approach while retaining attention and boundary layers at source precision; protected block-FP8 weights expand to BF16 for the GGUF runtime.

The moderate Config I policy protects the first and last two layers, quantizes interior expert gate/up to TQ4_1S and expert down to Q4_K, and leaves hybrid linear attention untouched. The first candidate leaves full attention unchanged too. A later candidate applies TQ4_1S uniformly to all interior full-attention Q/K/V/O matrices. Packed candidate import is a separate gate: ordinary HF conversion does not understand custom packed U8 safetensors and must not reinterpret them as float weights.

`packed_to_gguf.py` imports the version-1 C0 producer manifest. It verifies candidate shard hashes, stacks experts in numeric order and writes packed TQ4_1S/Q4_K bytes without requantization. It verifies the GGUF payload hashes after writing. Protected FP8 weights expand to BF16 with their original block multipliers; this runtime expansion is separate from packed candidate size. Use a scratch directory with sufficient space on the NAS because the streaming writer temporarily spools a second copy of the payload.

```sh
python examples/drama3/packed_to_gguf.py /models/drama-3 /models/C0 /models/drama-3-backbone-C0.gguf \
  --verified-source-receipt /models/download.complete.json \
  --staged-metadata /local/C0-backbone-metadata --temp-dir /models/gguf-scratch
```

## Runtime bridge

Build with `LLAMA_BUILD_EXAMPLES=ON`, then use `build/bin/llama-drama3-backbone`. Its JSON-lines protocol accepts batches of up to 64 frame embedding rows and returns final backbone norm outputs plus logits for requested token IDs. These hidden states have already passed the backbone's final RMS norm; do not normalize them again before the tower. Embedding pooling is disabled.

The first bridge uses microbatches of eight rows to keep the native small-batch TQ expert path during prefill. Larger microbatches can enter the BF16 dequantization path and need additional scratch memory. This choice is for memory qualification; benchmark its latency separately.

Use the isolated pinned environment from the compact-release setup above. Do not install the full Omni `fishaudio` extra over it. Source codec must be the local `codec.pth`. On sm_120/sm_121, the reference's Hopper-only fused tower and Triton sampler stay disabled; the module/PyTorch paths are the oracle.

```sh
GGML_TQ_NATIVE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python examples/drama3/external_audio.py \
  --source /models/drama-3 --gguf /models/drama-3-backbone.gguf \
  --verified-source-receipt /models/download.complete.json \
  --binary build/bin/llama-drama3-backbone \
  --omni-source /src/sglang-omni --text 'Hello, how are you?' --out hello.wav
```

The reference bridge keeps tower KV across the entire sequence for four shared and nine residual layers. `max_seq_len=11` is not a sliding context limit. It uses the official prompt and sampling code, including a 16-frame repeat-aware semantic sampling history. Optional `--references` takes a JSON list with `text` and pre-encoded `vq_codes`, generated using the same local codec. It refuses to report complete speech when generation reaches the frame limit without EOS.

`encode_reference.py` creates that JSON from a local clip (up to 60 seconds) and its transcript. It records the codec/audio hashes and encoding dtype. Reuse the exact same reference JSON for baseline and candidate comparisons. Its default is CPU; CUDA reference encoding is opt-in.

The bridge reads selected embedding rows from the actual GGUF model through the llama backend and its format decoder. It keeps no duplicate source embedding table. An embedding-quantized candidate therefore uses its own quantized rows instead of accidentally hiding their error with the source BF16 table. Rows enter the external audio embedding mix in BF16, matching the reference tower's activation dtype.

Synthesis moves only the codec decoder, codebooks, post-module and upsampler to the device. The encoder, pre-module and downsampler stay on CPU because `from_indices` does not call them. On the pinned source this reduces codec device weight storage from about 1.74 GiB to 0.70 GiB without changing weight precision. Runtime scratch and caches still need measurement.

Each invocation starts a fresh backbone process and loads the audio tower and codec. `--seed` defaults to 42. A successful invocation writes the WAV through a temporary file, then `.frames.json`, `.run.json` and finally `.complete.json`. Only consume samples with that last receipt. The run report includes model hash, prompt, settings, hardware and monotonic load, prefill, code-frame, generation, codec, vocode and write timings, plus generation and total RTF. First code-frame latency is not audible TTFA: this bridge vocodes after generation finishes. The preserved `prompt_prefill_seconds` aggregate now has a `prefill_breakdown` for native embedding lookup, audio embedding mix, backbone forward including IPC, and audio decoder prefill. Native decode wall time includes host setup and GPU completion, enforced by `llama_synchronize`; it is not GPU kernel-only time. Prompt row count is recorded. These timers diagnose cold first-use costs; varied-prompt warm numbers do not establish the cause. Filesystem cache state is uncontrolled. An outer batch runner should measure subprocess wall time separately. `--gguf-sha256` can reuse a digest previously computed for the same immutable GGUF.

`compare_teacher.py` replays mixed input embeddings captured from the official reference and reports final hidden-state and semantic-logit errors and top-token agreement. Its report describes the combined backend and precision difference; it does not declare equivalence or impose an arbitrary passing threshold. Review real measurements before a listening batch. Source capture instrumentation belongs to the isolated reference lane and must preserve its forward arithmetic.

## Gates before sharing results

```sh
PYTHONPATH=gguf-py:. python examples/drama3/test_conversion.py
PYTHONPATH=gguf-py python examples/drama3/test_native_bridge.py build/bin/llama-drama3-backbone
```

CPU tests cover explicit architecture selection, namespace routing, block scales, zero-centered norms, existing Qwen MRoPE metadata, causal tower cache and row replacement. A CPU build covers the native bridge API. These do not establish speech correctness.

Source/candidate integrity, byte-identical packed GGUF import, Spark CUDA execution and EOS speech have actual evidence. Remaining gates are diagnosis of teacher-forced backbone hidden/logit drift, controlled ASR/speaker/listening quality, long-form reference continuity and total peak device memory on the target 5090. Count both runtime processes, audio/codec weights, KV, graphs and scratch; a small GGUF file alone does not demonstrate a 32 GiB fit.

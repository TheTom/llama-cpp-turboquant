# Drama-3 experimental speech bridge

This lane runs the Fish Drama-3 backbone in this fork and the unmodified audio tower/codec in SGLang-Omni. It is a one-stream experimental bridge, not native FishOmni TTS support. Keep the original checkpoint's license with any derived model distribution.

Use a verified local `fishaudio/drama-3` checkpoint. Download once on the quantization machine and copy locally as needed. Every command below uses local files; set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` in the runtime environment.

## Backbone conversion

```sh
python examples/drama3/prepare_backbone.py /models/drama-3 /models/drama-3-backbone --verified-source-receipt /models/download.complete.json
python convert_hf_to_gguf.py /models/drama-3-backbone --outtype bf16 --outfile /models/drama-3-backbone-bf16.gguf
```

The staging command creates metadata and symlinks, without copying weights. Put the staging directory on a filesystem that supports symlinks; a CIFS NAS may reject them. Local staging links can point at NAS weights, while output and scratch files stay on the NAS. Mount the source at the same absolute path named in its verification receipt. The full `FishOmniModel` architecture stays unsupported by the ordinary converter. The staged architecture is explicitly `FishOmniBackboneForCausalLM`: it selects the Qwen3.5 MoE tensor/graph layout with noninterleaved partial RoPE and no MTP. Audio tensors stay in the original source for the external runtime.

The published expert FP8 tensors are expanded using their 128x128 F32 block multipliers. BF16 expansion needs more memory than source FP8 and must count toward the complete runtime budget. Do not silently use `--fp8-as-q8` for protected boundary layers and then describe them as unchanged.

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

Prepare an isolated SGLang-Omni environment at commit `2cfb3b67becb437b70751defe37e63d3af4eedb0`, including its `fishaudio` extra. Source codec must be the local `codec.pth`. On sm_120/sm_121, the reference's Hopper-only fused tower and Triton sampler stay disabled; the module/PyTorch paths are the oracle.

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

Each invocation starts a fresh backbone process and loads the audio tower and codec. `--seed` defaults to 42. A successful invocation writes the WAV through a temporary file, then `.frames.json`, `.run.json` and finally `.complete.json`. Only consume samples with that last receipt. The run report includes model hash, prompt, settings, hardware and monotonic load, prefill, code-frame, generation, codec, vocode and write timings, plus generation and total RTF. First code-frame latency is not audible TTFA: this bridge vocodes after generation finishes. Filesystem cache state is uncontrolled. An outer batch runner should measure subprocess wall time separately. `--gguf-sha256` can reuse a digest previously computed for the same immutable GGUF.

`compare_teacher.py` replays mixed input embeddings captured from the official reference and reports final hidden-state and semantic-logit errors and top-token agreement. Its report describes quantization error; it does not declare equivalence or impose an arbitrary passing threshold. Review real measurements before a listening batch. Source capture instrumentation belongs to the isolated reference lane and must preserve its forward arithmetic.

## Gates before sharing results

```sh
PYTHONPATH=gguf-py:. python examples/drama3/test_conversion.py
PYTHONPATH=gguf-py python examples/drama3/test_native_bridge.py build/bin/llama-drama3-backbone
```

CPU tests cover explicit architecture selection, namespace routing, block scales, zero-centered norms, existing Qwen MRoPE metadata, causal tower cache and row replacement. A CPU build covers the native bridge API. These do not establish speech correctness.

Remaining gates are source/candidate integrity, byte-identical packed GGUF import, teacher-forced backbone hidden/logit parity against the official reference, actual speech and EOS, ASR/speaker/listening quality, long-form reference continuity, CUDA execution and total peak device memory on the target 5090. Count both runtime processes, audio/codec weights, KV, graphs and scratch; a small GGUF file alone does not demonstrate a 32 GiB fit.

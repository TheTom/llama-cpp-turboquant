# Block KV cache streaming

**Experimental.** Bounds KV cache VRAM at long context by keeping only a
resident subset of pages on the GPU, in one shared CUDA arena, and streaming
the rest to/from host RAM on demand.

## Enabling it

```bash
llama-server -m model.gguf --kv-stream-arena-mib 8192 ...
```

`--kv-stream-arena-mib N` (alias `--kv-stream-stage-mib`) sets the total
size, in MiB, of one physical CUDA allocation shared between resident/ring KV
pages and the active phase's compute workspace. `0` (the default) disables
streaming entirely and falls back to ordinary KV cache allocation.

When a nonzero arena is set, `llama.cpp`'s normal `-fit`/auto-memory-fit pass
is skipped (`common_params_should_fit_device_memory()`) - you are responsible
for the arena being small enough to fit alongside the model weights and
compute buffers, and large enough to hold your working set. See
[Sizing the arena](#sizing-the-arena) below.

## Requirements

Streaming only activates when **all** of the following hold; otherwise it is
silently disabled (arena size ignored, ordinary allocation used) or, for a
non-zero arena that can't be honored, context creation fails with a specific
error naming which requirement was not met:

- Flash Attention enabled (`-fa on`)
- GPU KV offload enabled (default; not `--no-kv-offload`)
- Exactly one sequence (`--parallel 1` / `-np 1`) - see
  [Why single-sequence only](#why-single-sequence-only)
- The target context, not an MTP/draft context (speculative decoding's draft
  context never receives the arena - it always uses ordinary allocation, see
  `common/speculative.cpp`)
- A standard, uniform-per-layer-geometry KV cache - see
  [Supported architectures](#supported-architectures)
- Not combined with `--swa-full` - see
  [Supported architectures](#supported-architectures)

## Supported architectures

| Shape | Support |
|---|---|
| Plain dense/MoE models (one `llama_kv_cache`) | Yes |
| `llama_memory_hybrid` (recurrent + attention layers) | Yes, on the attention sub-cache |
| iSWA (dual full + sliding-window cache - Gemma-family, and this fork's own `laguna` arch) | Yes, on the full-attention (`kv_base`) sub-cache. The sliding-window sub-cache stays always-resident (it's small by construction) and does not stream, **unless** `--swa-full` is also set, in which case both sub-caches would need to stream at once - currently rejected outright (see below), not silently broken |
| MLA (DeepSeek-V2/V3-style) | Not excluded by architecture, but never empirically validated against a real model - treat as unverified, not safe |
| DSA (GLM-DSA / DeepSeek-V3.2), DSV4 (DeepSeek-V4), MSA (MiniMax-M3) | Excluded. DSV4 was attempted and found to have a real correctness bug (streaming vs non-streaming KL-divergence diverges significantly) whose root cause is still unidentified; DSA and MSA were never attempted given that finding |

`--swa-full` makes the sliding-window sub-cache full-context-length too, so
it would need its own concurrent streaming runtime sharing the same arena -
the arena only supports one lease today. This combination is explicitly
rejected at startup with a clear error rather than allowed to fail deep
inside cache construction.

## Why single-sequence only

This is not a simple validation gate that could be relaxed by testing more -
the resident-page/eviction design itself assumes one sequence:

- Page residency is a fixed window by absolute buffer offset ("the first N
  pages are resident, the rest stream"), not "each sequence's own hot range
  stays resident." With multiple sequences sharing one buffer, only whichever
  sequence happens to sit at the lowest offsets would ever be resident.
- The low-latency decode fast path is gated on the whole ubatch containing
  exactly one query token. A batch of N concurrently-decoding sequences
  (`n_tokens == N`) fails that check and would silently fall back to the
  slower prefill-style path every step.
- The adaptive resident:ring repartitioning feedback loop tracks one global
  deadline-miss counter, not per-sequence state.

Supporting real multi-sequence continuous batching would need a sequence
dimension added to the resident-cache layout and decode-path classification,
not a flag flip. Not attempted in this branch.

## Sizing the arena

Two categories of failure exist near the VRAM boundary, and only one of them
is safe:

- **Arena itself too large to allocate at context creation** - a clean,
  caught failure. The server logs an error and exits; nothing is corrupted.
- **Arena large enough to construct, but too tight for a transient kernel
  allocation during actual inference** (`ggml_cuda_pool_vmm::alloc()`
  failing mid-attention-kernel) - a **hard process abort** (`GGML_ASSERT` /
  `SIGABRT`), not recoverable in-process. This can take down an
  already-running, already-serving server on its first request that crosses
  the margin, not just fail to start.

Use `benchmarks/benchmark_kv_stream.py` (see `benchmarks/README.md`) to find
a safe arena size empirically for your model/GPU/context combination rather
than guessing - it probes VRAM headroom and backs off automatically on
allocation failure, which a production deployment does not get for free.

## Verifying it's actually active

There is currently no INFO-level log line confirming streaming activated for
a given request (a pre-existing gap, not something this doc can fix). The
most reliable signs it's working:
- `common_init_: skipping device-memory auto-fit because a shared KV/compute
  arena is explicitly configured` at startup confirms the flag was parsed
  and is nonzero - it does not by itself confirm activation.
- Context creation failing with one of the specific errors above confirms
  streaming was attempted and rejected for a named reason.
- Absent any error, and the model architecture is on the supported list
  above, streaming is active.

## Known inefficiency: MTP verification batches use the prompt-phase layout

`tools/server/server-context.cpp` never calls `llama_set_decode_phase()`, so
phase classification always falls back to automatic batch-size detection
(`llama_kv_stream_phase_is_generation()`): exactly one token is "generation",
anything else is "prompt". A speculative-decoding (MTP) verification batch is
several tokens wide (`--spec-chain N` produces up to `N+1`), so every
verification step gets classified as "prompt" and the arena stays in its
prefill-shaped (compute-heavy, KV-light) layout for the entire session -
the "generation" (KV-heavy, compute-light) layout is only reached on a true
single-token step, which barely happens once MTP is active.

Measured on a real MTP+streaming request (Qwen3.8-27B, `-c 22000`, 8192 MiB
arena, `--spec-chain 8`, 200 generated tokens / 90 decode steps): 89 of 90
steps ran in the prompt-phase layout (1246 resident pages/layer, 10 ring
slots); the one true single-token step got the generation-phase layout
(1273 resident pages/layer, 12 ring slots). That is a real but modest gap at
this context size (+2.2% resident pages, +20% ring slots) - measure again at
larger contexts before assuming it stays this small.

Fixing this is not just wiring up `llama_set_decode_phase()` in the server:
the "generation" phase's compute-buffer reservation is sized via
`graph_reserve(n_seqs, n_seqs, n_seqs, ...)` in `sched_reserve()` - built for
exactly `n_seq_max` (1) token per step. A multi-token verification batch
correctly classified as "generation" would immediately hit the existing
guard at `llama_context.cpp` ("phase arena currently supports TG1 without
speculative batches") and fail decode. Making this work requires resizing
the generation-phase reservation to the true max speculative width, not
just relaxing that guard. Not attempted - left as a known, quantified,
low-priority inefficiency rather than risk the arena-repartitioning logic
while the feature is stable.

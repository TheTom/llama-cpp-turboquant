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

## Operational limitations

Read this before enabling the arena on a server. Three features a running
deployment normally has are unavailable while streaming, because the host
cache is authoritative and the GPU holds only a mirror of it:

| Feature | Behaviour with an arena | Why |
|---|---|---|
| **Prompt caching / `llama_state_*` save+restore** | Disabled. The server logs `prompt cache disabled: this context cannot save KV cache state` once, then **every request re-prefills its whole prompt**. | The KV cache cannot be serialized: pages live in host memory under the runtime's own layout, and `llama_kv_cache::state_write_data` refuses. |
| **Context shift** (`--context-shift`) | Disabled at startup. `get_can_shift()` reports false, so `common_init_from_params` turns it off with `KV cache shifting is not supported for this context, disabling KV cache shifting`. Generation stops at the context limit instead of sliding the window. | K-shift rewrites K in place on the GPU; `seq_add` refuses. |
| **Shared / copied sequences** (`seq_cp`) | Unavailable. | `seq_cp` aliases one slot into a second sequence, which the resident mirror's per-page bookkeeping does not model. Streaming already requires `-np 1`, so this mainly rules out shared-prompt and multi-slot features. |

The practical cost is the first row: at long context, re-prefilling every
request can easily outweigh the decode-side win, so the arena suits
single-shot or long-generation workloads far better than chat-style traffic
that would otherwise hit a warm prompt cache. Measure both before deploying.

`seq_rm` **is** supported - it only edits cell bookkeeping - so ordinary
request turnover, slot reuse and speculative-draft rollback all work
normally.

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

## Attention dispatch: direct vs F16 fallback

Streamed attention picks between two CUDA code paths per (K type, V type)
pair, independent of everything else in this document:

- **`direct_attention`** - the same native turbo2/3/4/5/6 kernel (and the
  classic F16/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 kernels) ordinary non-streamed
  Flash Attention already uses, adapted to read resident/streamed pages
  directly. No extra workspace, no precision loss beyond the KV type's own
  quantization.
- **F16 fallback** - dequantizes each streamed page to F16 into a
  conversion workspace, then runs ordinary F16 attention on it. Works for
  any KV type this cache supports streaming for at all, but costs a
  dequant pass and workspace bandwidth per page.

`direct_attention` for the streamed path requires the ggml-cuda backend to
be built with `GGML_CUDA_FA_ALL_QUANTS` (off by default - it substantially
increases build time and the CUDA library size, since it instantiates
Flash Attention across the full K/V type cross product). **Without it,
every streamed KV type pair falls back to F16**, regardless of arena size
or model. Ordinary (non-streamed) attention is unaffected either way - the
turbo-native kernel it uses is unconditionally compiled in.

**The fallback's cost is not a fixed factor - it grows with context**,
because every streamed page is dequantized again on every ubatch, so the
dequant work scales with the KV already in the cache. Prefill therefore
goes from flat to roughly O(n^1.7) as soon as the arena is on. Measured on
one RTX 5090 (32 GiB, Linux), Qwen3.8-27B-Q5_0, `-c 65536 -b 2048 -ub 512`,
same prompts and server flags in both builds, prefill t/s:

| prompt tokens | arena off | arena 1024 MiB | arena 4096 MiB |
|---:|---:|---:|---:|
| **`GGML_CUDA_FA_ALL_QUANTS=ON`** | | | |
| 3902 | 3357 | 3293 | 3279 |
| 11193 | 3336 | 3336 | 3363 |
| 17490 | 3267 | 3171 | 3202 |
| 39511 | 2938 | 2810 | 2875 |
| **default build (flag off)** | | | |
| 3902 | 3161 | 1743 | 1699 |
| 11193 | 3188 | 975 | 965 |
| 17490 | 3180 | 700 | 695 |
| 39511 | 2914 | 352 | 336 |

With the flag on, streaming costs 4-6% of prefill throughput at 40K tokens.
Without it, the same run is 8x slower than its own baseline, and the gap
keeps widening with context - which is the regime this feature exists for.
Arena size barely matters either way: 4096 MiB lands within 3% of 1024 MiB
at every length in both builds.

Every benchmark number in the PR description and `benchmarks/results/` was
measured with the flag on. A default build remains functionally correct -
output is unchanged - but treat the flag as a practical requirement rather
than an optimization. Since it is off by default, `llama_kv_cache` logs a
one-time warning at startup naming the K/V pair that fell back.

**Head dim must be 256 for either path to activate at all.** Both
`direct_attention` and F16 fallback require `Q->ne[0] == V->ne[0] == 256`
(`ggml_cuda_flash_attn_ext_streamed_supported` in `fattn.cu`) - this isn't
architecture-gated the way MLA/DSA/DSV4/MSA are, so a head-dim-128 model
(common outside this fork's own turbo/Qwen3.8 testing) still passes every
other requirement above, still gets its KV cache pinned into the streaming
buffer, and still streams pages - it just never gets a streaming-accelerated
attention kernel for it. Every `GGML_OP_FLASH_ATTN_EXT` op for that model
falls through to ordinary (non-streamed) Flash Attention instead, which
still produces correct output by reading K/V from the pinned host buffer
over PCIe, but pays that transfer on every single decode step with none of
streaming's page-residency/prefetch benefit. Correct, just slow - budget
for it rather than assume streaming accelerates every model uniformly.

## Numerical exactness

Streamed output is close to non-streamed output, not bit-identical, and by
design:

- The chunked reduction combines each streamed block's partial attention
  result via a sequential online-softmax rescale as blocks arrive, rather
  than the single combined pass ordinary (non-streamed) Flash Attention
  uses (`flash_attn_combine_results`). Floating-point addition isn't
  associative, so summing the same terms in a different order gives a
  slightly different result - mean KL-divergence around 0.000000 and
  99%+ same-top-token in every correctness check run for this PR (see the
  PR description), not exact equality.
- `ggml_cuda_kv_stream_span_tuner` (`kv-stream-span-tuner.h`) picks between
  the ordinary coalesced streamed kernel and a bounded-span pipeline by
  comparing measured wall-clock time across trial runs, once per resident/
  ring layout. Which one wins depends on transient timing noise (system
  load, thermal state, etc.) during that trial window, so the same model/
  context/hardware can select a different kernel variant - and therefore a
  slightly different numerical result within the tolerance above - from one
  run to the next.

Both are intentional performance trade-offs, not bugs, but they mean two
runs of the same request are not guaranteed to produce byte-identical
logits even with all sampling held constant.

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

Two categories of failure exist near the VRAM boundary:

- **Arena itself too large to allocate at context creation** - a clean,
  caught failure. The server logs an error and exits; nothing is corrupted.
- **Arena large enough to construct, but too tight for the streamed
  attention kernel's transient scratch workspace** (`fattn.cu`'s
  `ggml_cuda_flash_attn_ext_streamed` allocates four small buffers from the
  device's general CUDA memory pool - not from the arena itself - for its
  chunked partial-attention reduction). Construction now computes that
  workspace's worst case from the model's head count/dim and `--ubatch-size`
  and confirms that much free VRAM survives reserving the arena, refusing
  construction with a clear error if not - the same clean, caught failure as
  an oversized arena, not the hard, unrecoverable `GGML_ASSERT`/`SIGABRT`
  this used to risk mid-request. This check uses free VRAM as measured at
  that point in context construction; other allocations later in
  construction (e.g. the compute buffer) aren't yet accounted for, so it
  catches the clearly-too-tight case, not necessarily every possible one.

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

## Speculative decoding (MTP)

MTP and block KV streaming compose, and MTP keeps its full speedup on top of
streaming. The two do not share memory: `common/speculative.cpp` forces
`kv_stream_arena_mib = 0` on the draft context, so **the MTP draft KV cache
stays in ordinary VRAM** and only the target cache streams. Rejected draft
tokens are rolled back with `seq_rm` on the target, which is bookkeeping-only
and safe while streaming (see `llama_kv_cache::seq_rm`).

Measured on one RTX 5090 (32 GiB, Linux, `GGML_CUDA_FA_ALL_QUANTS=ON`):
target `Qwen3.8-27B-Q5_0`, draft `mtp-Qwen3.8-27B-Q4_0`,
`--spec-type draft-mtp`, `-c 16384 -b 2048 -ub 512 -np 1`, 96 generated
tokens at `temperature 0`, two repetitions per cell. Decode t/s:

**Short prompt (18 tokens - the whole cache is resident, nothing streams):**

| config | no arena | arena 2048 MiB |
|---|---|---|
| no MTP | 70.3 / 69.8 | 70.7 / 71.5 |
| MTP, draft depth 1 | 101.5 / 104.8 | 101.8 / 104.1 |
| MTP, draft depth 4 | 111.4 / 112.4 | 113.0 / 112.4 |

**9748-token prompt, arena 1024 MiB (39 active vs 54 resident pages/layer -
still fully resident):**

| config | no arena | arena 1024 MiB |
|---|---|---|
| no MTP | 69.0 / 69.1 | 67.7 / 68.4 |
| MTP, draft depth 4 | 127.1 / 126.4 | 129.3 / 129.4 |

**9748-token prompt, arena 512 MiB - the case where streaming actually
happens** (39 active pages/layer against 22-31 resident, `copy busy 24.2%`
on the no-MTP run):

| config | arena 512 MiB |
|---|---|
| no MTP | 66.4 |
| MTP, draft depth 4 | 85.1 |

So the arena costs nothing measurable while the working set is resident
(within run-to-run noise at every depth), and 3-4% once pages genuinely
stream. MTP is worth 1.6x at short context, 1.87x at 9748 tokens resident,
and 1.28x with the cache actively streaming - the ratio shrinks because
verification steps pay for streamed pages, not because streaming interferes
with drafting. Draft and accept counts are unchanged by the arena at a given
context (132/61 and 136/60 with and without it at arena 1024), confirming
the draft path is untouched; they do shift across arena sizes, which is the
non-bit-identical behaviour described under [Numerical
exactness](#numerical-exactness).

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

A second, sharper symptom shows up under real streaming pressure. On the
arena-512 runs above, the no-MTP request reports `samples 26, copy busy
24.2%` and settles at 31 resident pages/layer, while the MTP request reports
`samples 0, misses 0, copy busy 0.0%` and only 22 resident pages/layer: a
verification batch is never classified as generation, so no decode deadline
feedback is collected at all and the partition controller never adapts away
from its prefill-shaped split. MTP still wins 1.28x there, so this costs
throughput it could have had rather than causing a regression.

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

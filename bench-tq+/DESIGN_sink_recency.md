# Sink-token + Recency-window for TQ+ — design doc

Status: design only. Empirical data in `bench-tq+/results/` informs whether the patch is worth shipping.

## Goal

Test the hypothesis (lifted from StreamingLLM 2023, re-marketed by SHARD 2026):
keeping the first N_sink positions (typically 4) and the last N_recency
positions (typically 32-64) at fp16 — while the bulk of the KV cache stays
at the existing TQ+ turbo codec — closes most of the PPL/NIAH gap vs the
fp16/fp16 baseline.

The empirical question this addresses: *does compression error accumulate
disproportionately at the high-attention positions (sink + recent) that
TQ+'s Sparse-V mechanism cannot skip?*

## Why this isn't a one-line change

llama.cpp's KV cache architecture:

- One `ggml_tensor *k` per layer, with one ggml type (e.g. `GGML_TYPE_TURBO3_0`).
- `cpy_k(k_cur, k_idxs)` writes via `ggml_set_rows` — quantizes from `k_cur` dtype to cache type.
- `get_k()` returns a view of the typed cache tensor. Flash attention reads it directly and dequants inside the kernel.
- All four GPU backends (Metal, CUDA/dp4a, HIP/RDNA+CDNA, Vulkan coopmat) have separate kernels per type.

A per-position type override breaks the assumption that the cache tensor has uniform type. There are 4 plausible implementation paths.

## Path A — Side-car fp16 sink/recency tensor + ggml overlay op

**Idea.** Per layer, allocate two small fp16 tensors alongside the main turbo cache:
- `k_sink[il]` : `[n_embd_k_gqa, n_sink]` fp16. Static — written once during the first prefill.
- `k_window[il]`: `[n_embd_k_gqa, n_recency]` fp16. Ring buffer, rotates with current position.
- Same for V.

Memory cost: `(n_sink + n_recency) * n_embd_kv * n_layers * 2 bytes`. At Nemotron-30B-A3B with 40 layers, 8 KV heads, 128 head dim, sink=4, recency=64: ~700 KB. Trivial.

**Write path** (`cpy_k`):
- If `k_idxs[i] < n_sink`: also write `k_cur[i]` to `k_sink[il]` at column `k_idxs[i]`. The slot in the main turbo cache gets overwritten too (lossy) — that's fine, the sink tensor is the authoritative copy.
- If `k_idxs[i] >= (current_length - n_recency)`: also write to `k_window[il]` with ring-buffer rotation.
- Otherwise: only write to main turbo cache.

**Read path** (`get_k` / attention build):

This is the hard part. Two options:

**A.1 — New ggml op `ggml_kv_overlay_fp16`.**
```c
// Returns a view of k_typed with positions [0..n_sink) overlaid from
// k_sink_fp16 and positions [length-n_recency..length) overlaid from
// k_window_fp16. The op itself emits a kernel that dequants k_typed
// for the "middle" range and reads fp16 directly for the overlays.
ggml_tensor * ggml_kv_overlay_fp16(
    ggml_context * ctx,
    ggml_tensor * k_typed,
    ggml_tensor * k_sink_fp16,
    ggml_tensor * k_window_fp16,
    int n_sink,
    int n_recency_start,  // = current_length - n_recency
    int n_recency_count
);
```
Backends: each (Metal, CUDA, HIP, Vulkan) needs a new kernel that produces an fp16 K [n_embd, n_kv] tensor with the overlay applied. Then flash attention consumes the fp16 tensor.

Cost: ~one full K materialization per attention layer per token at decode time = defeats the in-kernel dequant pattern that makes turbo cheap at decode.

**A.2 — Modify flash-attn kernel to take 3 K inputs and a sink/recency bound.**
```c
ggml_flash_attn_ext_with_overlays(q, k_typed, k_sink, k_window, v_typed, v_sink, v_window,
                                  n_sink, n_recency_start, n_recency_count, mask, scale, ...)
```
Per-backend kernel work. Most invasive change but preserves in-kernel dequant.

## Path B — Use the hybrid-cache abstraction

llama.cpp already supports multi-cache configurations via `kv_cache_hybrid`. The TQ+ "Boundary V" feature already routes some layers to a different codec.

**Idea.** Treat sink + recency as a separate *stream* in the multi-stream cache. The hybrid cache would have:
- Stream 0: main turbo cache (everything except sinks + recent)
- Stream 1: fp16 sink/recency cache

Attention reads from BOTH streams via the existing multi-stream get_k machinery.

**Blockers:**
1. The stream abstraction assumes streams are *independent contexts* (e.g. multiple sessions). Using them for position-based partitioning within a single sequence requires masking logic that doesn't exist today.
2. Stream concatenation in get_k currently returns one view per stream. Flash attention would need to be invoked once per stream and the partial results combined — incorrect, attention is not commutative across the K dimension after softmax.

**Verdict:** Hybrid abstraction is the wrong primitive. Skip.

## Path C — KV cache reservation hack

**Idea.** Reserve the first `n_sink + n_recency` slots of the cache as fp16, the remainder as turbo. Use a typed-pair cache (two ggml tensors per K, two per V), and slice attention into two flash-attn calls + an online-softmax merge.

```
fa(q, k_fp16[0..n_sink], v_fp16[0..n_sink], mask_sink) → score_sink, m_sink, l_sink
fa(q, k_turbo[n_sink..n_kv], v_turbo[n_sink..n_kv], mask_main) → score_main, m_main, l_main
merge(score_sink, m_sink, l_sink, score_main, m_main, l_main) → final
```

This is the **online-softmax merge** trick from Flash-Decoding. Two passes, no kernel rewrites, just graph-builder surgery + a small merge kernel.

**Cost:**
- 2× flash-attn invocations per attention layer per decode step (each on smaller K/V slices though)
- Online softmax merge: one tiny kernel (~50 lines per backend) — reduces over `[m_sink, m_main, l_sink, l_main, score_sink, score_main]`

**Win:**
- No new ggml types
- No backend-specific quant kernel changes
- The merge kernel already exists in similar form (Flash-Decoding's reduce pass)

**Verdict:** Most realistic implementation path. The cost of 2× flash-attn calls is offset by each call being on a strictly smaller K/V slice. At decode T=1 with n_kv=4096, n_sink=4, n_recency=64: the "main" pass attends to 4028 tokens, the "sink+recency" pass attends to 68 tokens. Total work increases by 68/4096 ≈ 1.7%. Acceptable.

## Path D — Don't ship a code change, ship the prediction

Empirical data from this branch's diagnostic suite will tell us:
- whether quantization error grows with context length (recency window benefit)
- whether NIAH retrieval at certain depths is preserved or destroyed (sink/recency benefit)
- whether the existing TQ+ Sparse-V already mitigates most of the would-be improvement

If the diagnostic shows TQ+ defaults are already near-baseline at long context (the Sparse-V hypothesis), the patch may not be worth shipping at all.

## Recommendation

Implement **Path C** if and only if the diagnostic results show:
- ΔPPL grows monotonically with context length (recency would help)
- AND/OR NIAH retrieval at the 0..N_sink depth or near-recent depth degrades meaningfully vs fp16 baseline (sink would help)

The diagnostic results in `bench-tq+/results/` are the gate.

## Code-change inventory if shipping Path C

Files touched:
- `src/llama-kv-cache.h` — new `kv_cache_sink_recency` config struct; allocator changes
- `src/llama-kv-cache.cpp` — `cpy_k`/`cpy_v` per-position dispatch; allocate side-car fp16 tensors
- `src/llama-graph.cpp` — `build_attn_mha` calls two flash-attn ops + a merge op when sink+recency is active
- `ggml/include/ggml.h`, `ggml/src/ggml.c` — new `ggml_attn_merge_online` op (small, pure tensor)
- `ggml/src/ggml-metal/*.metal` — `attn_merge_online_metal` kernel (~80 lines)
- `ggml/src/ggml-cuda/*.cu` — same for CUDA (~80 lines, dp4a not needed)
- `ggml/src/ggml-hip/*` — symlink/inherit from CUDA (mostly free)
- `ggml/src/ggml-vulkan/*.comp` — Vulkan compute shader (~100 lines)
- `common/arg.cpp` — `--cache-sink-tokens N` and `--cache-recency-window N` CLI flags
- `tests/test-tq-sink-recency.cpp` — correctness test: small synthetic prompt, verify output matches f16/f16 within tolerance when N_sink covers the prompt

Estimated effort: 1-2 weeks of focused work for a complete cross-backend implementation. Sink-only (no recency) on Metal-only is ~3 days.

## Test plan (independent of implementation)

See `bench-tq+/harness/`:
- `run_ppl_sweep.py` — wikitext-2 PPL across the codec ladder
- `per_position_kld.py` — PPL as a function of context length (proxy for recency-window benefit)
- `niah.py` — needle retrieval at varying depths

All three run against the model unchanged. Comparison against a sink/recency-equipped build is what the design above would enable; for now the harness establishes the baseline.

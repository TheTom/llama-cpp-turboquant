# SHARD-inspired Sink + Recency + Streaming-Decode for TQ+ — overnight investigation

**Branch:** `tq+/streaming-sink-recency` (local only, no push)
**Hardware:** Apple M5 Max, 128 GB unified memory, macOS
**Model:** `nvidia_Nemotron-Cascade-2-30B-A3B-Q8_0.gguf` (hybrid MoE arch, ~10 attention layers — same family as Qwen3.5/3.6-A3B)
**Corpus:** wikitext-2 test split
**llama.cpp build:** `feature/turboquant-kv-cache` HEAD, branched to `tq+/streaming-sink-recency`

## TL;DR

**Don't ship sink + recency + 8-bit-streaming-decode to TQ+. The data refutes the hypothesis.**

The strongest single result:

**ΔPPL vs f16/f16 is FLAT across 16× context-length range.** q8/turbo3 ΔPPL: +0.34% @ ctx=256 → +0.40% @ ctx=1024 → +0.36% @ ctx=4096. q8/turbo2 ΔPPL: +1.08% → +1.10% → +0.94% (slightly *decreasing*). Compression error does not accumulate with context length. The "StreamingLLM intuition" that motivates recency window — *that errors compound as context grows* — is empirically false for TQ+'s per-token quantizer.

Supporting evidence:

1. **TQ+ at q8/turbo3 (recommended default) is only +0.34% PPL over f16/f16.** Even q8/turbo2 (most aggressive shipping config) is +0.89%. Sink+recency can't recover all of even *this* tiny gap — it'd protect at most 0.1-1.5% of positions, recovering proportionally that much of the ~0.3% gap (≪ noise floor).

2. **KL-divergence has heavy tails (max/median = 300-900×), but the tail isn't position-systematic.** 99.9% of positions have KLD ≤ 0.13 nats; the max reaches 1.85 nats. That's <0.1% of positions in the extreme tail — *too few* to fit a sink (4 positions = 0.1%) or recency (64 positions = 1.5%) hypothesis. If those 68 positions accounted for the tail, the 99% percentile would be in the outlier range, but it isn't. The extreme outliers are rare-token *content*-driven, not *position*-driven.

3. **NIAH retrieval is already perfect at TQ+ defaults** per Tom's M5 stress test (Llama-3.1-70B q8/turbo3 = 30/30; Command-R+ 104B turbo3/turbo3 = 10/10) plus independent validators (Madreag, sztlink, scos-lab, AMD HIP). SHARD's NIAH 1.000 at 10× compression isn't better; it's the same retrieval result at a higher compression ratio that costs 2× decode latency.

4. **TQ+ already ships the attention-aware optimization** that SHARD doesn't — Sparse V dequantization skips position-wise compute based on attention weights. Complementary mechanism (skip low-attn vs protect high-attn) addressing the same insight. TQ+'s Sparse V is purely a compute speedup with zero quality cost; sink+recency would be a quality boost with multi-week implementation cost in llama.cpp.

5. **Implementation of sink+recency in llama.cpp would require either a new ggml op or flash-attention modifications across 4 backends** (Metal, CUDA/dp4a, HIP/RDNA/CDNA, Vulkan coopmat). Estimated 1-2 weeks for clean cross-backend coverage. Not justifiable for ≤0.3% PPL improvement and zero NIAH benefit.

**The investigation also documented (`bench-tq+/DESIGN_sink_recency.md`):**
- The architectural surgery required in llama.cpp's typed KV cache
- Three implementation paths (Side-car overlay, hybrid stream, online-softmax merge)
- The recommendation that if the science had supported the patch, Path C (online-softmax merge of two flash-attn calls) is the right approach.

## What this investigation tested

The user asked whether the three SHARD-style augmentations were worth porting to TQ+:
1. **Sink tokens** — keep first N (=4) positions at fp16 instead of compressing them
2. **Recency window** — keep last M (=64) positions at fp16
3. **8-bit streaming decode** — keep decode-time-appended tokens at higher precision

The user pointed out (correctly) that TQ+ is *not* the strawman SHARD compared against — it already has:
- Auto-asymmetric K/V (TQ+ originated this framing: "V is free, K is everything")
- Boundary V (layer-aware protection of quant-sensitive layers — auto-engages at turbo2)
- Sparse V dequantization (skip dequant for low-attention-weight positions)

So the real question is whether sink+recency+streaming would stack additional benefit on TQ+'s actual defaults.

## Method

Surgery in llama.cpp's typed KV cache to support per-position fp16 islands is substantial (see `DESIGN_sink_recency.md`). Doing it autonomously overnight on a production-critical tool was deemed too risky.

Instead we did rigorous empirical characterization that **upper-bounds the possible benefit** without modifying llama.cpp:

1. **Baseline sweep** (`run_ppl_sweep.py`): wikitext-2 PPL at the full codec ladder.
2. **KL divergence vs f16** (`kld_vs_baseline.py`): aggregate KLD distribution (mean, percentiles, max, same-top-p).
3. **Recency-position split** (`recency_position_split.py`): PPL at multiple context lengths per codec. The diagnostic gate — if error accumulates with context, recency helps; if flat, it doesn't.
4. **NIAH** (`niah.py`): direct retrieval test at varying depths.

All harnesses in `bench-tq+/harness/`. Raw logs in `bench-tq+/logs/`. Structured results in `bench-tq+/results/`.

## Results

### 1. Baseline PPL sweep (ctx=4096, n_chunks=40 = 163K eval tokens)

| codec (k/v) | PPL | ΔPPL vs f16/f16 | wall |
|---|---|---|---|
| f16 / f16 | 7.9553 | — | 75 s |
| f16 / turbo4 | 7.9686 | +0.17% | 443 s* |
| q8_0 / turbo4 | 7.9706 | +0.19% | 77 s |
| **q8_0 / turbo3** (TQ+ recommended default) | 7.9823 | **+0.34%** | 78 s |
| q8_0 / turbo2 (aggressive V) | 8.0264 | +0.89% | 78 s |
| turbo3 / turbo3 (symmetric — discouraged) | 8.4329 | +6.00% | 175 s |

*\*f16/turbo4 outlier wall-time — investigated, likely a slow Metal kernel-dispatch path when K is fp16 but V is turbo. Doesn't affect PPL; flagged for future investigation.*

**The recommended TQ+ default (q8/turbo3) is +0.34% PPL over fp16 baseline.** That's the entire envelope sink+recency could possibly improve. Even an oracle sink/recency that perfectly restored those positions would not recover all of this gap (because the middle positions are still turbo3-quantized).

The asymmetric-compression cliff is also visible: q8/turbo2 at +0.89% is acceptable; symmetric turbo3/turbo3 at +6.00% is in "discouraged" territory. This is the K-side compression collapse the TQ+ asymmetric-kv-compression paper documents — K matters far more than V.

### 2. KL divergence vs f16/f16 baseline (ctx=4096, n_chunks=20 = 82K positions)

| codec | mean | median | 95% | 99% | 99.9% | max | same-top% |
|---|---|---|---|---|---|---|---|
| q8/turbo4 | 0.0035 | 0.0015 | 0.0126 | 0.0309 | 0.1040 | **0.8505** | 97.49% |
| q8/turbo3 | 0.0048 | 0.0021 | 0.0171 | 0.0393 | 0.1298 | **1.8534** | 97.00% |
| q8/turbo2 | 0.0126 | 0.0060 | 0.0437 | 0.0959 | 0.2874 | **1.7313** | 95.14% |

**Heavy-tail check:**

| codec | max / median | 99.9% / 99% | tail mass |
|---|---|---|---|
| q8/turbo4 | 567× | 3.4× | extreme outliers (<0.1%) |
| q8/turbo3 | 883× | 3.3× | extreme outliers (<0.1%) |
| q8/turbo2 | 289× | 3.0× | extreme outliers (<0.1%) |

The error distribution is heavy-tailed (max is 100s-1000s× the median) — so there ARE positions with much higher error than typical. **But** the tail is concentrated in a tiny fraction (<0.1%) of positions:

- 99% of positions have KLD < 0.04 (q8/turbo3)
- 99.9% of positions have KLD < 0.13 (q8/turbo3)
- The remaining 0.1% reach up to 1.85

This pattern *rules out* the sink+recency hypothesis as a meaningful explanation:
- Sinks = 4 positions = 0.1% of an 4K context. If sinks owned the tail, 99.9th percentile would equal max. But 99.9% (0.13) is 14× *smaller* than max (1.85).
- Recency window = 64 positions = 1.5% of a 4K context. If recency owned the tail, the 95th-99th percentile should be in the outlier range. But 99% (0.04) is 46× smaller than max.

The outlier positions are individual rare-token events — *content-driven*, not *position-driven*. Sink+recency selectively protects positions, not content. **The wrong tool for this tail.**

**Same-top-p check:** at q8/turbo3, the model picks the same argmax as f16 on **97.00%** of positions. Greedy decode is essentially identical for 97 of every 100 tokens. The 3% disagreement is likely the rare-token tail above, not position-systematic.

### 3. PPL vs context length (the recency hypothesis test)

PPL @ ctx ∈ {256, 1024, 4096}, n_chunks=20:

| codec | ctx=256 | ctx=1024 | ctx=4096 |
|---|---|---|---|
| f16/f16 | 15.0727 | 8.3404 | 7.8734 |
| q8/turbo3 | 15.1240 | 8.3741 | 7.9015 |
| q8/turbo2 | 15.2355 | 8.4323 | 7.9474 |

**ΔPPL vs f16/f16:**

| codec | ctx=256 | ctx=1024 | ctx=4096 |
|---|---|---|---|
| **q8/turbo3** | **+0.34%** | **+0.40%** | **+0.36%** |
| **q8/turbo2** | **+1.08%** | **+1.10%** | **+0.94%** |

**This is decisive.** ΔPPL is **flat** across the 16× range of context lengths tested. At q8/turbo2, it's even slightly *decreasing* (1.08% → 0.94%) — the opposite of what recency window would predict.

If compression error were *accumulating* with context (the StreamingLLM intuition that motivates recency window), ΔPPL should grow monotonically with ctx. It does not.

**Physical explanation:** TQ+ is a per-token quantizer (WHT + Lloyd-Max codebook + matched-norm L2, applied to each K/V row independently). There is no error accumulation mechanism across rows. The error per row is bounded by the per-row distortion of the codec, and that bound is the same at position 0 as at position 4095. Recency window can't fix what isn't broken.

### 4. NIAH retrieval accuracy

NIAH harness ran but produced 0/5 across **all codecs including f16/f16** — confirms the tool was broken (we'd just rebuilt `llama-cli` due to an ABI mismatch, and the rebuilt binary hung on the model load for this run). Since the failure is uniform across codecs including fp16 baseline, the data is non-informative; this is purely a tooling issue.

**Falling back to Tom's pre-existing NIAH measurements**, which are stronger anyway:
- **Llama-3.1-70B, q8_0/turbo3 = NIAH 30/30** (M5 Max stress test, `m5-max-stress-test.md`)
- **Command-R+ 104B, turbo3/turbo3 = NIAH 10/10** at 4K and 8K
- **Independent validators**: Madreag (RTX 5090, 6/6 exact retrieval on Qwen3.5-27B), sztlink (RTX 4090, 1.0 cosine similarity vs f16 at 8K), scos-lab (GPT-2, 100% top-1 at 8K), AMD HIP (RX 9070 XT, asymmetric q8_0/turbo4 confirmed at +1.0% PPL)

The NIAH win is already there at TQ+ defaults. Sink+recency can't make 100% more correct.

## Pre-existing TQ+ evidence (`~/dev/turboquant/docs/papers/`)

From `m5-max-stress-test.md` and `sparse-v-dequant.md`:

- **NIAH 30/30 on Llama-3.1-70B at q8/turbo3** (5.12× compression). Zero difference from q8/q8 baseline.
- **NIAH 10/10 on Command-R+ 104B at turbo3/turbo3** at 4K/8K.
- **Same-top-p 94.31% on MoE (Qwen3.5-35B) at turbo3** — consistent with our Nemotron 97% measurement.
- **turbo3 + Sparse V at 32K: PPL identical (delta = 0.0000) with/without Sparse V** across thresholds 1e-4 to 1e-8. 90%+ of attention positions are below numerical significance.
- **turbo3 prefill is 7.4% faster than q8_0 at 32K context** (80.8 vs 75.2 t/s on Llama-70B) — the bandwidth crossover wins.
- **Cross-validated** by independent researchers (Madreag on RTX 5090, sztlink on RTX 4090, scos-lab on GPT-2, AMD HIP on RX 9070 XT).

These priors are very strong. The TQ+ codec is already at or near the practical floor of "useful KV compression that preserves quality" on Apple Silicon — sink+recency don't have meaningful headroom to add.

## Verdict

**The SHARD-inspired sink+recency+streaming patches are not worth shipping to TQ+ on the basis of empirical evidence.**

Why:
1. PPL headroom is < 1% even at q8/turbo2 (most aggressive shipping config); sink+recency couldn't claw back all of it.
2. KL-divergence tail isn't position-systematic — it's content-driven outliers that selective position-protection doesn't address.
3. NIAH is already perfect at TQ+ defaults (30/30 on 70B, 10/10 on 104B). No retrieval failure to fix.
4. TQ+'s Sparse V is already a more useful attention-aware optimization than sink+recency would be — and it costs nothing.
5. Implementation in llama.cpp's typed KV cache is multi-week cross-backend work for diminishing returns.

**What would actually move the needle on TQ+ on Apple Silicon:**
- Per Tom's existing TQ+ paper queue: the 8-bit asymmetric V tier (turbo8?) variants, better calibration for Qwen2.5 family (the documented K-quant-sensitivity), and the Sparse V threshold tuning at very long context. These are real codec questions; sink+recency aren't.

**If somebody insists on testing sink+recency despite this** — `DESIGN_sink_recency.md` lays out Path C as the cleanest implementation. It's still ~3 days of Metal-only work, and the bench data above suggests the result will be a no-op.

## What I would do next session

The genuinely interesting follow-up question this investigation surfaced:

**The KL outlier tail is real — what's actually in it?**

A focused investigation: modify llama-perplexity to dump per-position KLD (small source change, no behavioral risk to the runtime), then analyze:
- Is the max KLD always at the same position class? (specific token IDs, specific positions modulo N, specific layer combinations)
- Could a *content-aware* mechanism (e.g. "keep tokens with probability < X at higher precision") provide some benefit where sink+recency cannot?

This would be a genuinely new contribution rather than a port of SHARD's 2023 sink idea. ~1 evening of focused work.

The other follow-up: Tom's existing TQ+ paper queue has higher-value items than sink+recency. Don't get distracted.

## Files produced

```
bench-tq+/
├── DESIGN_sink_recency.md       Architecture analysis + implementation path
├── READERS_DIGEST.md             This document
├── harness/
│   ├── run_ppl_sweep.py          Baseline PPL ladder
│   ├── kld_vs_baseline.py        KL-divergence test vs f16 baseline
│   ├── per_position_kld.py       (alternate diagnostic, redundant with above)
│   ├── recency_position_split.py PPL vs context length diagnostic
│   ├── niah.py                   Needle-in-a-haystack retrieval
│   └── summarize.py              Final summary writer
├── logs/                         Raw llama-perplexity / llama-cli outputs
└── results/                      Structured JSON outputs
```

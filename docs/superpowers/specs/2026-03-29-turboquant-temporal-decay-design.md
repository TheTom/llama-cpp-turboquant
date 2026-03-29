# TurboQuant+ Temporal Decay -- Design Spec

**Date:** 2026-03-29
**Author:** apollosenvy (Gary) + Heph (Claude Opus 4.6)
**Status:** Approved for implementation

## Problem

KV cache quantization applies uniform compression across all token positions. But attention is recency-biased -- recent tokens receive exponentially more attention weight than old ones. Compressing a token from 5 seconds ago the same as a token from 5 minutes ago wastes precision where it matters and hoards it where it doesn't.

## Solution

Two-dimensional KV cache compression:
- **Vertical axis (static):** Per-layer type selection via existing `TURBO_LAYER_ADAPTIVE` mechanism
- **Horizontal axis (dynamic):** Per-position quality degradation within the allocated format

Old tokens get progressively compressed. Recent tokens maintain full quality. Boundaries shift dynamically as context grows so tier ratios stay proportional.

## Tier System

Three tiers with dynamic boundaries proportional to current context length:

| Tier | Default % | Quality | Mechanism |
|------|-----------|---------|-----------|
| Hot  | 15% (newest) | Full quality or q8_0 promoted | Base type unchanged, or promoted to q8_0 |
| Warm | 35% | Base type at full quality | turbo3/turbo4 with QJL intact |
| Cold | 50% (oldest) | Degraded in-place | QJL signs zeroed, rnorm set to 0 |

### Boundary Computation

```
total = current_context_length
cold_end  = total * cold_pct          // e.g., 50%
warm_end  = cold_end + total * warm_pct  // e.g., 85%
hot_start = warm_end                     // e.g., 85%
```

At 8K context:  cold=0-4K, warm=4K-6.8K, hot=6.8K-8K
At 64K context: cold=0-32K, warm=32K-54.4K, hot=54.4K-64K
At 128K context: cold=0-64K, warm=64K-108.8K, hot=108.8K-128K

Boundaries are recomputed on every new token insertion (trivial: three integer multiplies).

## Re-quantization Mechanism

### Cold Tier Demotion (Warm -> Cold)

turbo3 block layout:
```
block_turbo3_0 {
    ggml_half norm;               // 2 bytes -- keep
    uint8_t   qs[QK_TURBO3/4];   // 8 bytes -- keep (3-bit centroid indices)
    uint8_t   signs[QK_TURBO3/8]; // 4 bytes -- ZERO (QJL refinement bits)
}
```

Demotion = `memset(block.signs, 0, QK_TURBO3/8)`. The dequant path naturally handles this: with signs all zero, the QJL contribution becomes `(-1) * qjl_scale` for every element (uniform bias), which effectively reduces to centroid-only reconstruction with a systematic offset. Setting `rnorm=0` in turbo4 blocks eliminates the QJL contribution entirely since `qjl_scale = sqrt(pi/2) / 128 * rnorm`.

For turbo3 (where signs encode the 3rd centroid bit, not QJL): the cold demotion instead clamps centroid indices to 4 levels by zeroing the high bit in signs. This reduces from 8-level to 4-level reconstruction.

### Hot Tier Promotion (Optional)

When `TURBO_DECAY_HOT_PROMOTE=1`:
- New tokens are quantized to q8_0 instead of the base turbo type
- When a token crosses hot->warm boundary, it's dequantized from q8_0 and re-quantized to the base turbo type
- Requires the hot zone of the cache to be allocated as q8_0

This is the expensive path. Implementation deferred to phase 2.

### Piggyback on SET_ROWS

Every token generation already calls `cpy_k()` / `cpy_v()` which dispatches SET_ROWS to write the new KV entry. During the same graph evaluation, one additional operation demotes a single block that crossed a tier boundary:

1. After writing new token at position P, compute `cold_boundary = P * cold_pct`
2. If `cold_boundary` advanced past a block boundary since last check, schedule demotion of that block
3. Demotion kernel: memset signs field to 0 for the target block. Single warp, negligible cost.

Amortized cost: one memset per token per layer. At 64 layers, that's 64 * 4-byte memsets = 256 bytes of writes per generated token. Invisible.

## Configuration

### Environment Variable

```
TURBO_DECAY=balanced          # named preset
TURBO_DECAY=15,35,50          # custom hot,warm,cold percentages (must sum to 100)
TURBO_DECAY=off               # disable (default)
```

### Presets

| Name | Hot | Warm | Cold | Use Case |
|------|-----|------|------|----------|
| conservative | 25 | 40 | 35 | Chat, moderate context |
| balanced | 15 | 35 | 50 | General purpose |
| aggressive | 5 | 25 | 70 | Maximum context, summarization |

### Hot Promotion (Phase 2)

```
TURBO_DECAY_HOT_PROMOTE=1     # hot tier uses q8_0 regardless of base type
```

Default: 0 (disabled). When enabled, the hot zone is allocated with q8_0 type and tokens are re-quantized on demotion to warm tier.

## Layer-Adaptive Integration

Temporal decay stacks on top of `TURBO_LAYER_ADAPTIVE`:

- Layers assigned q8_0 by the adaptive mechanism: **no decay** (already high quality, decay would degrade unnecessarily)
- Layers assigned turbo3/turbo4/turbo2 by the adaptive mechanism: **full decay** with the configured tier percentages
- Layers assigned turbo2 (already 2-bit): **no cold demotion** (can't degrade further within format)

This means the 2D compression map automatically does the right thing: quality-sensitive layers (first/last) keep full precision everywhere, middle layers get compressed spatially AND temporally.

## Implementation Plan

### Phase 1: Core Decay (This PR)

**llama-kv-cache.h:**
- Add `struct turbo_decay_config { int hot_pct, warm_pct, cold_pct; bool hot_promote; }`
- Add `turbo_decay_config decay_cfg` member to `llama_kv_cache`
- Add `llama_pos last_cold_boundary` tracking state per layer

**llama-kv-cache.cpp:**
- Parse `TURBO_DECAY` env var in constructor (after `TURBO_LAYER_ADAPTIVE`)
- In `cpy_k()` / `cpy_v()`: after SET_ROWS write, compute new cold boundary. If it advanced, emit demotion op for the crossed block.
- Add `decay_demote_block()` method: given layer + block index, zero the signs field via `ggml_backend_tensor_set` (CPU) or inline kernel (GPU).

**ggml-cuda/set-rows.cu:**
- Add `k_turbo_decay_demote` kernel: given a turbo3/turbo4 block pointer + block index, memset the signs field to 0. Single warp, 4-16 bytes.
- Dispatch from `cpy_k` / `cpy_v` after the main SET_ROWS kernel.

**CLI:**
- Parse `TURBO_DECAY` in common params, pass to KV cache constructor.

### Phase 2: Hot Promotion (Future)

- Split cache tensor into hot + main regions with different types
- Re-quantization kernel (q8_0 -> turbo3) on hot->warm transition
- Requires cache architecture changes (multi-tensor per layer)

### Phase 3: Attention-Guided Decay (Future)

- Track per-position attention scores during generation
- Demote positions that receive consistently low attention, regardless of age
- Promote positions that receive unexpectedly high attention back to warm tier

## Testing

1. **Correctness**: Run PPL benchmark with decay=balanced vs decay=off. PPL delta should be <1% for contexts under 8K (hot zone covers everything), increasing gradually for longer contexts.
2. **VRAM**: Measure peak VRAM at 32K, 64K, 128K context with turbo3+decay vs F16. The effective compression should exceed turbo3-alone.
3. **NIAH**: Needle-in-a-haystack at various depths. Cold-tier needles may show retrieval degradation -- quantify the depth at which decay causes failures.
4. **Speed**: Measure tg overhead from demotion kernel. Should be <0.1% of per-token time.

## Success Criteria

- turbo3 + decay=balanced fits 2x more context than turbo3-alone in 24GB VRAM
- PPL at 64K context < 10% regression vs F16
- No measurable tg speed regression
- NIAH retrieval accuracy > 95% for needles in hot+warm tiers
- NIAH retrieval accuracy documented (not necessarily high) for needles in cold tier

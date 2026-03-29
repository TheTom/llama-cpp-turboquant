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

**turbo3**: The `signs[]` field stores the high bit of the 3-bit centroid index (NOT QJL signs). The dequant path reconstructs indices as `idx = low2 | (hi1 << 2)`. Zeroing signs clamps all indices to the range [0-3] (4-level reconstruction) instead of [0-7] (8-level). This is a clean halving of reconstruction precision.

```
block_turbo3_0 {
    ggml_half norm;               // 2 bytes -- keep
    uint8_t   qs[QK_TURBO3/4];   // 8 bytes -- keep (low 2-bit centroid indices)
    uint8_t   signs[QK_TURBO3/8]; // 4 bytes -- ZERO (high 1-bit of 3-bit index)
}
```

Demotion = `memset(block.signs, 0, QK_TURBO3/8)`. Idempotent -- demoting an already-cold block is a no-op.

**turbo4 (legacy 3-bit+QJL, TURBO4_USE_4BIT=0)**: The `signs[]` field stores 1-bit QJL refinement signs. Setting `rnorm=0` eliminates the QJL contribution entirely since `qjl_scale = sqrt(pi/2) / 128 * rnorm`. The 3-bit centroid indices in `qs[]` are preserved.

**turbo4 (4-bit PolarQuant, TURBO4_USE_4BIT=1)**: No QJL component. Cold demotion is not applicable -- no degradation possible within the format. Decay skips turbo4 in 4-bit mode (analogous to turbo2 in the layer-adaptive section).

### Boundary Computation (Position Space)

Boundaries are computed in **position space**, not cell index space. The KV cache is a ring buffer where cell indices are not monotonically ordered by position.

```
min_pos = seq_pos_min(seq_id)
max_pos = seq_pos_max(seq_id)
range   = max_pos - min_pos
cold_boundary_pos = min_pos + range * cold_pct / 100
warm_boundary_pos = min_pos + range * (cold_pct + warm_pct) / 100
```

Cells with `pos < cold_boundary_pos` are cold tier candidates. Demotion scans cells by position, not by contiguous index ranges.

### Multi-Sequence Policy

When multiple sequences share KV cells, demotion uses the **maximum position** across all sequences referencing a cell. A cell is only cold if it's cold for ALL sequences. This is conservative -- shared cells are demoted last.

For simplicity, Phase 1 activates temporal decay only for single-sequence workloads (n_seq=1). Multi-sequence support deferred to Phase 2.

### Hot Tier Promotion (Optional)

When `TURBO_DECAY_HOT_PROMOTE=1`:
- New tokens are quantized to q8_0 instead of the base turbo type
- When a token crosses hot->warm boundary, it's dequantized from q8_0 and re-quantized to the base turbo type
- Requires the hot zone of the cache to be allocated as q8_0

This is the expensive path. Implementation deferred to phase 2.

### Graph-Level Decay Op

`cpy_k()` / `cpy_v()` build ggml graph nodes (they don't execute directly). The decay operation is a proper ggml op (`GGML_OP_TURBO_DECAY`) added to the graph after `ggml_set_rows`. The graph scheduler handles execution ordering.

1. In `cpy_k`/`cpy_v`, after the SET_ROWS node, compute the new cold boundary in position space.
2. If the boundary advanced since the last call, append a `ggml_turbo_decay` node targeting the cells that crossed into cold tier.
3. The decay kernel (CUDA/HIP/CPU) scans the specified position range and zeroes the signs field for each matching block.

Cost: one kernel launch per layer when boundaries advance. At steady-state generation, boundaries advance by ~1 position per token, so ~1 block per layer gets demoted. Total writes: 64 layers * 4 bytes = 256 bytes per token. Invisible.

### Edge Cases

- **Sliding window attention (ISWA)**: SWA layers already evict old tokens. Decay only applies to the base (non-SWA) cache in `llama_kv_cache_iswa`.
- **Sequence deletion (seq_rm)**: When positions are removed, `last_cold_boundary` may exceed the new max position. Reset it to 0 on seq_rm to trigger a full rescan.
- **Cache defragmentation**: Defrag moves cells without changing positions. Decay state (`last_cold_boundary`) is position-based, so defrag is transparent.
- **State serialization**: Degraded blocks are saved as-is. Decay is irreversible -- restored state has the degraded quality. This is by design.
- **Idempotency**: memset on already-zeroed signs is a no-op. Safe to re-demote.

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

- turbo3 + decay=balanced maintains <5% PPL regression at 64K context (vs turbo3-alone which degrades uniformly)
- PPL at 64K context < 10% regression vs F16
- No measurable tg speed regression
- NIAH retrieval accuracy > 95% for needles in hot+warm tiers
- NIAH retrieval accuracy documented (not necessarily high) for needles in cold tier

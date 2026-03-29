# TurboQuant+ Temporal Decay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add position-aware KV cache quality degradation that progressively compresses old tokens, enabling longer contexts in the same VRAM.

**Architecture:** Three dynamic tiers (hot/warm/cold) with proportional boundaries. Cold demotion zeros the signs field in turbo3/turbo4 blocks in-place, reducing effective precision without changing ggml types. Demotion piggybacks on the SET_ROWS dispatch path, amortizing one block per token.

**Tech Stack:** C/C++ (llama.cpp), CUDA/HIP kernels, ggml graph ops

**Spec:** `docs/superpowers/specs/2026-03-29-turboquant-temporal-decay-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/llama-kv-cache.h` | Modify | Add `turbo_decay_config` struct, decay state to `llama_kv_cache` |
| `src/llama-kv-cache.cpp` | Modify | Parse env var, compute boundaries, schedule demotions in `cpy_k`/`cpy_v` |
| `ggml/src/ggml-cuda/turbo-decay.cu` | Create | CUDA/HIP kernel for in-place signs zeroing |
| `ggml/src/ggml-cuda/turbo-decay.cuh` | Create | Header for decay kernel dispatch |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | Modify | Register TURBO_DECAY op dispatch |
| `ggml/src/ggml.c` | Modify | Add GGML_OP_TURBO_DECAY op |
| `ggml/include/ggml.h` | Modify | Declare `ggml_turbo_decay` function |
| `ggml/src/ggml-cpu/ops.cpp` | Modify | CPU fallback for decay op |
| `tests/test-turbo-decay.cpp` | Create | Unit tests for decay config, boundary math, block demotion |

---

### Task 1: Decay Config Struct and Parsing

**Files:**
- Modify: `src/llama-kv-cache.h:210-270`
- Modify: `src/llama-kv-cache.cpp:30-70` (constructor top)

- [ ] **Step 1: Add decay config struct to header**

In `src/llama-kv-cache.h`, add before the `private:` section of `llama_kv_cache` (around line 210):

```cpp
// TurboQuant+ temporal decay: position-aware KV quality degradation
struct turbo_decay_config {
    bool enabled = false;
    int hot_pct  = 15;   // newest tokens, full quality
    int warm_pct = 35;   // middle tokens, base type quality
    int cold_pct = 50;   // oldest tokens, degraded (signs zeroed)
    bool hot_promote = false;  // phase 2: hot tier uses q8_0
};
```

Add as member of `llama_kv_cache` (after `turbo_innerq_scale_inv`, around line 266):

```cpp
// TurboQuant+ temporal decay state
turbo_decay_config decay_cfg;
std::vector<llama_pos> last_cold_boundary;  // per-layer tracking
```

- [ ] **Step 2: Add static parse function in llama-kv-cache.cpp**

At the top of `llama-kv-cache.cpp` (after includes, around line 30), add:

```cpp
static turbo_decay_config parse_turbo_decay() {
    turbo_decay_config cfg;
    const char * env = getenv("TURBO_DECAY");
    if (!env || strcmp(env, "off") == 0 || strlen(env) == 0) {
        return cfg;  // disabled
    }
    cfg.enabled = true;

    // Named presets
    if (strcmp(env, "conservative") == 0) {
        cfg.hot_pct = 25; cfg.warm_pct = 40; cfg.cold_pct = 35;
    } else if (strcmp(env, "balanced") == 0) {
        cfg.hot_pct = 15; cfg.warm_pct = 35; cfg.cold_pct = 50;
    } else if (strcmp(env, "aggressive") == 0) {
        cfg.hot_pct = 5; cfg.warm_pct = 25; cfg.cold_pct = 70;
    } else {
        // Custom: "H,W,C" percentages
        int h = 0, w = 0, c = 0;
        if (sscanf(env, "%d,%d,%d", &h, &w, &c) == 3 && h + w + c == 100 &&
            h >= 0 && w >= 0 && c >= 0) {
            cfg.hot_pct = h; cfg.warm_pct = w; cfg.cold_pct = c;
        } else {
            LLAMA_LOG_WARN("TURBO_DECAY: invalid value '%s', using balanced\n", env);
            cfg.hot_pct = 15; cfg.warm_pct = 35; cfg.cold_pct = 50;
        }
    }

    // Hot promotion (phase 2)
    const char * promote = getenv("TURBO_DECAY_HOT_PROMOTE");
    cfg.hot_promote = promote && atoi(promote) > 0;

    LLAMA_LOG_INFO("turbo_decay: enabled (%d/%d/%d hot/warm/cold%s)\n",
        cfg.hot_pct, cfg.warm_pct, cfg.cold_pct,
        cfg.hot_promote ? ", hot_promote" : "");

    return cfg;
}
```

- [ ] **Step 3: Initialize decay in KV cache constructor**

In the constructor, after the `TURBO_LAYER_ADAPTIVE` block (around line 203), add:

```cpp
    // Parse temporal decay config (once)
    static const turbo_decay_config s_decay_cfg = parse_turbo_decay();
    if (il == 0) {
        decay_cfg = s_decay_cfg;
    }
```

After the layers loop (around line 260, after `layers.push_back`), initialize per-layer tracking:

```cpp
    last_cold_boundary.resize(layers.size(), 0);
```

- [ ] **Step 4: Build and verify compilation**

Run: `cmake --build build --target llama-bench -j$(nproc) 2>&1 | tail -5`
Expected: Clean build, no errors.

- [ ] **Step 5: Test env var parsing**

Run: `TURBO_DECAY=balanced build/bin/llama-bench -m ~/Models/Qwen3.5-27B-Claude-Distill/*.gguf -ctk turbo3 -ctv turbo3 -ngl 999 -p 1 -n 1 -fa 1 2>&1 | grep turbo_decay`
Expected: `turbo_decay: enabled (15/35/50 hot/warm/cold)`

- [ ] **Step 6: Commit**

```bash
git add src/llama-kv-cache.h src/llama-kv-cache.cpp
git commit -m "feat(decay): add turbo_decay_config struct and env var parsing"
```

---

### Task 2: Add ggml Decay Op

**Files:**
- Modify: `ggml/include/ggml.h`
- Modify: `ggml/src/ggml.c`

- [ ] **Step 1: Declare the op in ggml.h**

In the `enum ggml_op` (find `GGML_OP_TURBO_WHT`), add after it:

```cpp
    GGML_OP_TURBO_DECAY,
```

Add the function declaration near `ggml_turbo_wht`:

```cpp
    // TurboQuant temporal decay: zero signs field of turbo blocks at given positions
    // cold_start = first position to demote, cold_end = last position (exclusive)
    GGML_API struct ggml_tensor * ggml_turbo_decay(
            struct ggml_context * ctx,
            struct ggml_tensor  * kv_cache,     // the K or V cache tensor
            int64_t               cold_start,   // first cell to demote
            int64_t               cold_end);    // last cell to demote (exclusive)
```

- [ ] **Step 2: Implement the op factory in ggml.c**

After `ggml_turbo_wht` (around line 6235), add:

```cpp
// ggml_turbo_decay

struct ggml_tensor * ggml_turbo_decay(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int64_t               cold_start,
        int64_t               cold_end) {
    GGML_ASSERT(a->type == GGML_TYPE_TURBO3_0 || a->type == GGML_TYPE_TURBO4_0);

    struct ggml_tensor * result = ggml_view_tensor(ctx, a);

    result->op = GGML_OP_TURBO_DECAY;
    result->src[0] = a;

    // Store range in op_params
    memcpy(result->op_params + 0, &cold_start, sizeof(int64_t));
    memcpy(result->op_params + 2, &cold_end,   sizeof(int64_t));

    return result;
}
```

Also add the op name in the `GGML_OP_NAME` array and `GGML_OP_SYMBOL` array (search for `TURBO_WHT` and add `TURBO_DECAY` after it).

- [ ] **Step 3: Build and verify**

Run: `cmake --build build --target llama-bench -j$(nproc) 2>&1 | tail -5`
Expected: Clean build. The op exists but has no backend implementation yet.

- [ ] **Step 4: Commit**

```bash
git add ggml/include/ggml.h ggml/src/ggml.c
git commit -m "feat(decay): add GGML_OP_TURBO_DECAY op declaration"
```

---

### Task 3: CUDA/HIP Decay Kernel

**Files:**
- Create: `ggml/src/ggml-cuda/turbo-decay.cuh`
- Create: `ggml/src/ggml-cuda/turbo-decay.cu`
- Modify: `ggml/src/ggml-cuda/ggml-cuda.cu` (dispatch)

- [ ] **Step 1: Create the header**

Create `ggml/src/ggml-cuda/turbo-decay.cuh`:

```cpp
#include "common.cuh"

void ggml_cuda_op_turbo_decay(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
```

- [ ] **Step 2: Create the kernel**

Create `ggml/src/ggml-cuda/turbo-decay.cu`:

```cpp
// TurboQuant temporal decay: zero signs field of turbo blocks in a range
// This degrades cold-tier KV entries to lower effective precision in-place.

#include "turbo-decay.cuh"

// turbo3: signs = 4 bytes per block (QK_TURBO3=32, signs = 32/8 = 4 bytes)
// turbo4: signs = 16 bytes per block (QK_TURBO4=128, signs = 128/8 = 16 bytes)
//         also zero rnorm (2 bytes at offset 2)

template <typename block_type, int signs_bytes, bool has_rnorm>
static __global__ void k_turbo_decay(
        block_type * __restrict__ data,
        const int64_t             block_start,
        const int64_t             block_end) {

    const int64_t ib = (int64_t)blockDim.x * blockIdx.x + threadIdx.x + block_start;
    if (ib >= block_end) return;

    // Zero the signs field
    memset(data[ib].signs, 0, signs_bytes);

    // For turbo4: also zero rnorm to eliminate QJL contribution
    if constexpr (has_rnorm) {
        data[ib].rnorm = 0;
    }
}

void ggml_cuda_op_turbo_decay(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    int64_t cold_start, cold_end;
    memcpy(&cold_start, dst->op_params + 0, sizeof(int64_t));
    memcpy(&cold_end,   dst->op_params + 2, sizeof(int64_t));

    if (cold_start >= cold_end) return;

    cudaStream_t stream = ctx.stream();

    // Convert position range to block range
    // Each row of the cache is n_embd_gqa elements. Positions index rows.
    // We need to demote all blocks within the position range.
    const int64_t ne0 = src0->ne[0];  // n_embd_gqa (elements per row)

    if (src0->type == GGML_TYPE_TURBO3_0) {
        const int64_t blocks_per_row = ne0 / QK_TURBO3;
        const int64_t b_start = cold_start * blocks_per_row;
        const int64_t b_end   = cold_end * blocks_per_row;
        const int64_t n_blocks = b_end - b_start;
        const int block_size = 256;
        const int grid = (n_blocks + block_size - 1) / block_size;

        k_turbo_decay<block_turbo3_0, QK_TURBO3/8, false>
            <<<grid, block_size, 0, stream>>>(
                (block_turbo3_0 *)src0->data, b_start, b_end);
    } else if (src0->type == GGML_TYPE_TURBO4_0) {
        const int64_t blocks_per_row = ne0 / QK_TURBO4;
        const int64_t b_start = cold_start * blocks_per_row;
        const int64_t b_end   = cold_end * blocks_per_row;
        const int64_t n_blocks = b_end - b_start;
        const int block_size = 256;
        const int grid = (n_blocks + block_size - 1) / block_size;

        k_turbo_decay<block_turbo4_0, QK_TURBO4/8, true>
            <<<grid, block_size, 0, stream>>>(
                (block_turbo4_0 *)src0->data, b_start, b_end);
    }
}
```

- [ ] **Step 3: Register in ggml-cuda.cu dispatch**

In `ggml/src/ggml-cuda/ggml-cuda.cu`, find the `GGML_OP_TURBO_WHT` case and add after it:

```cpp
        case GGML_OP_TURBO_DECAY:
            ggml_cuda_op_turbo_decay(ctx, dst);
            break;
```

Also in the `supports_op` function (find the `GGML_OP_TURBO_WHT` case), add:

```cpp
        case GGML_OP_TURBO_DECAY:
            return op->src[0]->type == GGML_TYPE_TURBO3_0 || op->src[0]->type == GGML_TYPE_TURBO4_0;
```

Include the header at the top with the other turbo includes:

```cpp
#include "turbo-decay.cuh"
```

- [ ] **Step 4: Add CPU fallback in ops.cpp**

In `ggml/src/ggml-cpu/ops.cpp`, find `ggml_compute_forward_turbo_wht` and add after it:

```cpp
void ggml_compute_forward_turbo_decay(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    if (params->ith != 0) return;  // single-threaded

    const ggml_tensor * src = dst->src[0];
    int64_t cold_start, cold_end;
    memcpy(&cold_start, dst->op_params + 0, sizeof(int64_t));
    memcpy(&cold_end,   dst->op_params + 2, sizeof(int64_t));

    if (cold_start >= cold_end) return;

    const int64_t ne0 = src->ne[0];

    if (src->type == GGML_TYPE_TURBO3_0) {
        const int64_t bpr = ne0 / QK_TURBO3;
        block_turbo3_0 * data = (block_turbo3_0 *)src->data;
        for (int64_t pos = cold_start; pos < cold_end; pos++) {
            for (int64_t b = 0; b < bpr; b++) {
                memset(data[pos * bpr + b].signs, 0, QK_TURBO3 / 8);
            }
        }
    } else if (src->type == GGML_TYPE_TURBO4_0) {
        const int64_t bpr = ne0 / QK_TURBO4;
        block_turbo4_0 * data = (block_turbo4_0 *)src->data;
        for (int64_t pos = cold_start; pos < cold_end; pos++) {
            for (int64_t b = 0; b < bpr; b++) {
                memset(data[pos * bpr + b].signs, 0, QK_TURBO4 / 8);
                data[pos * bpr + b].rnorm = GGML_FP32_TO_FP16(0.0f);
            }
        }
    }
}
```

Wire it into the CPU dispatch switch (find `GGML_OP_TURBO_WHT` case):

```cpp
        case GGML_OP_TURBO_DECAY:
            {
                ggml_compute_forward_turbo_decay(params, tensor);
            } break;
```

- [ ] **Step 5: Build and verify**

Run: `cmake --build build --target llama-bench -j$(nproc) 2>&1 | tail -5`
Expected: Clean build with new kernel compiled.

- [ ] **Step 6: Commit**

```bash
git add ggml/src/ggml-cuda/turbo-decay.cu ggml/src/ggml-cuda/turbo-decay.cuh \
        ggml/src/ggml-cuda/ggml-cuda.cu ggml/src/ggml-cpu/ops.cpp
git commit -m "feat(decay): CUDA/HIP/CPU kernel for in-place signs zeroing"
```

---

### Task 4: Wire Decay into cpy_k / cpy_v

**Files:**
- Modify: `src/llama-kv-cache.cpp:1190-1300` (cpy_k and cpy_v methods)

- [ ] **Step 1: Add decay scheduling to cpy_k**

In `llama_kv_cache::cpy_k()`, after the `ggml_set_rows` call (line 1222), before the return, add:

```cpp
    // Temporal decay: demote blocks that crossed into cold tier
    if (decay_cfg.enabled && (k->type == GGML_TYPE_TURBO3_0 || k->type == GGML_TYPE_TURBO4_0)) {
        // Get current max position from cells
        const auto & cells = v_cells[0];  // stream 0
        const llama_pos max_pos = cells.seq_pos_max(0);  // sequence 0

        if (max_pos > 0) {
            const llama_pos cold_boundary = (max_pos * decay_cfg.cold_pct) / 100;
            const int32_t ikv_layer = map_layer_ids.at(il);

            if (cold_boundary > last_cold_boundary[ikv_layer] && last_cold_boundary[ikv_layer] >= 0) {
                // Demote the range [old_boundary, new_boundary) to cold tier
                result = ggml_turbo_decay(ctx, result, last_cold_boundary[ikv_layer], cold_boundary);
            }

            // Cast away const for state update (decay state is mutable cache metadata)
            const_cast<llama_kv_cache *>(this)->last_cold_boundary[ikv_layer] = cold_boundary;
        }
    }
```

- [ ] **Step 2: Add same logic to cpy_v**

Add the identical block to `llama_kv_cache::cpy_v()`, after the SET_ROWS or `ggml_set_rows` call, before the return. Use `v->type` instead of `k->type` for the type check.

- [ ] **Step 3: Build and test**

Run: `cmake --build build --target llama-bench -j$(nproc) 2>&1 | tail -5`
Expected: Clean build.

Run: `TURBO_DECAY=balanced build/bin/llama-bench -m ~/Models/Qwen3.5-27B-Claude-Distill/*.gguf -ctk turbo3 -ctv turbo3 -ngl 999 -p 128 -n 32 -fa 1 2>&1 | tail -5`
Expected: Runs without crash, shows turbo_decay log message, speed within 1% of without decay.

- [ ] **Step 4: Commit**

```bash
git add src/llama-kv-cache.cpp
git commit -m "feat(decay): wire decay op into cpy_k/cpy_v graph dispatch"
```

---

### Task 5: PPL Validation

**Files:** None (benchmark-only task)

- [ ] **Step 1: PPL without decay (baseline)**

```bash
GGUF=~/Models/Qwen3.5-27B-Claude-Distill/*.gguf
build/bin/llama-perplexity -m $GGUF -f /tmp/wikitext-2-raw.txt -ngl 999 --flash-attn on -c 2048 --chunks 8 --cache-type-k turbo3 --cache-type-v turbo3 2>&1 | grep Final
```

- [ ] **Step 2: PPL with decay=balanced**

```bash
TURBO_DECAY=balanced build/bin/llama-perplexity -m $GGUF -f /tmp/wikitext-2-raw.txt -ngl 999 --flash-attn on -c 2048 --chunks 8 --cache-type-k turbo3 --cache-type-v turbo3 2>&1 | grep Final
```

Expected: PPL within 1% of baseline (at 2K context, hot zone covers most tokens).

- [ ] **Step 3: PPL with decay=aggressive**

```bash
TURBO_DECAY=aggressive build/bin/llama-perplexity -m $GGUF -f /tmp/wikitext-2-raw.txt -ngl 999 --flash-attn on -c 2048 --chunks 8 --cache-type-k turbo3 --cache-type-v turbo3 2>&1 | grep Final
```

Expected: PPL within 5% of baseline.

- [ ] **Step 4: Speed benchmark**

```bash
TURBO_DECAY=balanced build/bin/llama-bench -m $GGUF -ctk turbo3 -ctv turbo3 -ngl 999 -p 128 -n 128 -fa 1 2>&1 | tail -5
```

Expected: tg speed within 0.5% of no-decay.

- [ ] **Step 5: Document results and commit**

```bash
git commit --allow-empty -m "test(decay): PPL validation — balanced=X.XX, aggressive=X.XX, no-decay=X.XX"
```

---

### Task 6: Long-Context Stress Test

**Files:** None (benchmark-only task)

- [ ] **Step 1: Maximum context test with decay**

Test at the longest context that fits with turbo3+decay in 24GB:

```bash
TURBO_DECAY=aggressive build/bin/llama-bench -m $GGUF -ctk turbo3 -ctv turbo3 -ngl 999 -p 16384 -n 32 -fa 1 2>&1 | tail -5
```

- [ ] **Step 2: Compare VRAM at long context**

```bash
# Without decay
build/bin/llama-bench -m $GGUF -ctk turbo3 -ctv turbo3 -ngl 999 -p 16384 -n 32 -fa 1 2>&1 | grep "MiB"

# With decay
TURBO_DECAY=aggressive build/bin/llama-bench -m $GGUF -ctk turbo3 -ctv turbo3 -ngl 999 -p 16384 -n 32 -fa 1 2>&1 | grep "MiB"
```

Note: VRAM won't differ (same tensor sizes) but the cold tier tokens will have lower effective precision, validating the mechanism works at scale.

- [ ] **Step 3: Push all commits**

```bash
git push fork rocm-turbo3-v2
```

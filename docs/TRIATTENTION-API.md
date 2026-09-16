# TriAttention API Reference

## C API (include/llama.h)

### llama_triattention_init

```c
LLAMA_API bool llama_triattention_init(
    struct llama_context * ctx,
    const char * stats_path,
    int32_t  budget,
    int32_t  divide_length,
    int32_t  offset_max,
    int32_t  mode,
    int32_t  trigger,
    int32_t  agg,
    int32_t  seed,
    bool     normalize,
    bool     protect_prefill,
    bool     disable_mlr,
    bool     disable_trig,
    bool     enable_logging);
```

Initialize TriAttention KV cache eviction on a context. Must be called after
context creation and before inference begins.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `ctx` | `llama_context *` | The context to enable TriAttention on |
| `stats_path` | `const char *` | Path to `.triattention` calibration file |
| `budget` | `int32_t` | Max KV entries after pruning (e.g., 2048) |
| `divide_length` | `int32_t` | Pruning interval in tokens (e.g., 128) |
| `offset_max` | `int32_t` | Max geometric offset (e.g., 65536) |
| `mode` | `int32_t` | 0=global, 1=per-kv-head, 2=per-layer-head |
| `trigger` | `int32_t` | 0=interval, 1=slack |
| `agg` | `int32_t` | 0=mean, 1=max aggregation |
| `seed` | `int32_t` | RNG seed for tie-breaking (0=deterministic) |
| `normalize` | `bool` | Z-score normalize before selection |
| `protect_prefill` | `bool` | Protect prompt tokens from eviction |
| `disable_mlr` | `bool` | Ablation: disable MLR weighting |
| `disable_trig` | `bool` | Ablation: norm-only scoring |
| `enable_logging` | `bool` | Log pruning events to stderr |

**Returns:** `true` if initialization succeeded, `false` on error (bad file,
model mismatch, context doesn't use KV cache).

---

## Internal C++ API (src/llama-triattention.h)

### Core lifecycle

```c
// Initialize from calibration file + config
triattention_state * triattention_init(
    const char * stats_path,
    const triattention_config * cfg,
    uint32_t kv_size,
    double   rope_theta,
    uint32_t head_dim,
    uint32_t n_kv_heads);

// Free all resources
void triattention_free(triattention_state * state);
```

### Scoring functions

```c
// Invert RoPE rotation on dequantized K vectors
// Converts from post-RoPE to pre-RoPE representation
void triattention_invert_rope(
    float * k,                  // [n_keys, head_dim] in-place
    const float * omega,        // [freq_count] RoPE frequencies
    const int32_t * positions,  // [n_keys] token positions
    uint32_t n_keys,
    uint32_t head_dim,
    uint32_t freq_count,
    uint32_t rope_style);       // 0=half, 1=interleaved

// Score keys by TriAttention formula
void triattention_score_keys(
    float       * out_scores,
    const float * pre_rope_k,
    const triattention_head_stats * stats,
    const float * omega,
    const float * freq_scale_sq,
    const float * offsets,
    const int32_t * key_positions,
    int64_t  round_start,
    uint32_t n_keys,
    uint32_t head_dim,
    uint32_t freq_count,
    uint32_t n_offsets,
    enum triattention_agg agg,
    bool disable_trig);

// Full pruning pipeline (called from KV cache)
void triattention_prune_impl(
    triattention_state * state,
    const ggml_tensor ** k_tensors,
    const uint32_t    * layer_map,
    uint32_t            n_layers,
    uint32_t            n_kv_heads,
    uint32_t            padded_head_dim,
    uint32_t            kv_size,
    uint32_t          * evicted_cells,
    uint32_t          * n_evicted);
```

### Position tracking hooks

```c
void triattention_on_token_added(triattention_state * s, uint32_t cell_idx, int32_t pos);
void triattention_on_cell_removed(triattention_state * s, uint32_t cell_idx);
void triattention_on_position_shift(triattention_state * s, int32_t shift, int32_t p0, int32_t p1);
void triattention_on_reset(triattention_state * s);
```

These must be called from the KV cache implementation whenever cells are
modified. They maintain the O(1) position lookup table used by the scoring
pipeline.

### Statistics

```c
void triattention_print_stats(const triattention_state * state);
```

Prints pruning statistics: total calls, total tokens evicted, average timing.

---

## Structs

### triattention_config

```c
struct triattention_config {
    int32_t  budget;
    int32_t  divide_length;       // pruning interval
    int32_t  offset_max;
    enum triattention_mode mode;
    enum triattention_trigger trigger;
    enum triattention_agg agg;
    int32_t  seed;
    bool     normalize;
    bool     protect_prefill;
    bool     disable_mlr;
    bool     disable_trig;
    bool     enable_logging;
};
```

### triattention_head_stats

```c
struct triattention_head_stats {
    float * q_mean_real;    // [freq_count] Re(E[q_f])
    float * q_mean_imag;    // [freq_count] Im(E[q_f])
    float * q_abs_mean;     // [freq_count] E[||q_f||]
    float * q_mean_abs;     // [freq_count] ||E[q_f]|| (precomputed at init)
    float * extra_weight;   // [freq_count] excess weight (precomputed at init)
};
```

### triattention_calibration

```c
struct triattention_calibration {
    uint32_t head_dim;
    uint32_t num_layers;
    uint32_t num_attn_heads;
    uint32_t num_kv_heads;
    uint32_t num_kv_groups;
    uint32_t freq_count;
    uint32_t n_sampled;
    double   rope_theta;
    uint32_t rope_style;
    char *   model_name;
    triattention_head_stats * head_stats;
    uint32_t * sample_layer;
    uint32_t * sample_head;
};
```

---

## CUDA GPU Scoring API (ggml/include/ggml-cuda.h)

The GPU scoring path avoids the GPU→CPU transfer of the full K tensor by
computing importance scores directly on the GPU. Only the resulting score
array (one float per position) is copied back to the CPU.

### triattention_gpu_init

```c
triattention_gpu_state * triattention_gpu_init(
    const triattention_gpu_config * config,
    const triattention_gpu_head_calib * head_calibs,
    const float * omega,
    const float * freq_scale_sq,
    const float * offsets,
    void * stream);
```

Upload calibration data and precomputed arrays to GPU memory.

### triattention_gpu_score_head

```c
void triattention_gpu_score_head(
    triattention_gpu_state * state,
    const void   * k_data_dev,
    uint64_t       n_embd_k_gqa,
    size_t         row_bytes,
    uint32_t       kv_head_idx,
    uint32_t       head_calib_idx,
    const uint32_t * cell_indices_dev,
    const int32_t  * positions_dev,
    uint32_t       n_cells,
    int64_t        round_start,
    int            agg_mode,
    float        * scores_dev,
    void * stream);
```

Launch the scoring kernel for one KV head. Each thread block processes one
cache position, with `freq_count` threads cooperating on dequantization,
inverse WHT rotation, inverse RoPE, and score computation.

### Utility functions

```c
void  triattention_gpu_scores_to_host(float * host, const float * dev, uint32_t n, void * stream);
void  triattention_gpu_upload_cells(uint32_t ** ci_dev, int32_t ** pos_dev, ...);
float * triattention_gpu_alloc_scores(uint32_t n_cells, void * stream);
void  triattention_gpu_free_dev(void * ptr);
void  triattention_gpu_free(triattention_gpu_state * state);
```

---

## Enums

### triattention_mode

| Value | Name | Description |
|-------|------|-------------|
| 0 | `TRIATTENTION_MODE_GLOBAL` | Union-based global selection |
| 1 | `TRIATTENTION_MODE_PER_KV_HEAD` | Independent per-KV-head selection |
| 2 | `TRIATTENTION_MODE_PER_LAYER_HEAD` | Independent per-layer-head selection |

### triattention_trigger

| Value | Name | Description |
|-------|------|-------------|
| 0 | `TRIATTENTION_TRIGGER_INTERVAL` | Prune every N decode tokens |
| 1 | `TRIATTENTION_TRIGGER_SLACK` | Prune at budget + window occupancy |

### triattention_agg

| Value | Name | Description |
|-------|------|-------------|
| 0 | `TRIATTENTION_AGG_MEAN` | Average score over geometric offsets |
| 1 | `TRIATTENTION_AGG_MAX` | Max score over geometric offsets |

#pragma once

// TriAttention: Trigonometric KV Cache Eviction for llama.cpp
// Based on arXiv 2604.04921 (MIT/NVIDIA/ZJU)
//
// Scores cached keys using pre-RoPE Q/K concentration statistics and
// trigonometric importance estimation, then evicts the lowest-scoring
// entries to keep the KV cache within a configurable budget.
//
// Designed to work alongside TurboQuant KV cache compression:
//   TurboQuant compresses each KV entry to 2-4 bits (reducing bits/token)
//   TriAttention evicts unimportant entries entirely (reducing token count)
//   Combined: ~40x effective KV memory reduction

#include "ggml.h"

#include <cstdint>
#include <cstdio>

// Forward declarations
class  llama_kv_cache;
struct llama_hparams;

// ============================================================================
// Binary calibration file format (.triattention)
// ============================================================================
//
// Header:
//   magic          uint32  0x54524941 ("TRIA")
//   version        uint32  1
//   head_dim       uint32  e.g. 128
//   num_layers     uint32  e.g. 36
//   num_attn_heads uint32  e.g. 36  (total attention heads, not KV heads)
//   num_kv_heads   uint32  e.g. 4   (grouped query attention KV heads)
//   rope_theta     float64 e.g. 10000.0
//   rope_style     uint32  0=half, 1=interleaved
//   n_sampled      uint32  number of (layer, head) pairs with stats
//   freq_count     uint32  head_dim / 2
//   name_len       uint32  length of model name string (including null)
//   name           char[name_len]  UTF-8 null-terminated model name
//
// Per sampled head (repeated n_sampled times):
//   layer_idx      uint32
//   head_idx       uint32  (attention head index, 0..num_attn_heads-1)
//   q_mean_real    float32[freq_count]  Re(E[q_f])
//   q_mean_imag    float32[freq_count]  Im(E[q_f])
//   q_abs_mean     float32[freq_count]  E[||q_f||]
//   r_f            float32[freq_count]  ||E[q_f]|| / E[||q_f||] (validation)

#define TRIATTENTION_MAGIC   0x54524941u  // "TRIA" in little-endian
#define TRIATTENTION_VERSION 1u

// ============================================================================
// Enums
// ============================================================================

// Pruning granularity — determines how keep-sets are computed
// Paper Section 4.2: "Importance-based Token Selection"
enum triattention_mode {
    // Union-based: all sampled heads share one global selection.
    // Each head picks top-B independently, union, then select top-B
    // from union by combined (max-over-heads) score.
    // Paper default for benchmarks.
    TRIATTENTION_MODE_GLOBAL         = 0,

    // Per-KV-head independent: each KV head selects its own top-B
    // independently. Different KV heads retain different token subsets.
    // Paper Section 4.3 / ablation: can improve quality for GQA models.
    TRIATTENTION_MODE_PER_KV_HEAD    = 1,

    // Per-layer-per-KV-head independent: each (layer, KV head) pair
    // selects independently. Most fine-grained — each layer can retain
    // different tokens per head.
    TRIATTENTION_MODE_PER_LAYER_HEAD = 2,
};

// When to trigger pruning
enum triattention_trigger {
    // R-KV style: prune every divide_length tokens when cache exceeds budget
    TRIATTENTION_TRIGGER_INTERVAL = 0,

    // Slack: let cache grow to budget + divide_length, then prune back to budget
    TRIATTENTION_TRIGGER_SLACK    = 1,
};

// Score aggregation over geometric future offsets D = {1,2,4,...,offset_max}
enum triattention_agg {
    TRIATTENTION_AGG_MEAN = 0,  // Paper default: mean over offsets
    TRIATTENTION_AGG_MAX  = 1,  // Alternative: max over offsets
};

// ============================================================================
// Data structures
// ============================================================================

// Per-(layer, head) calibration statistics
// Stores the complex mean of pre-RoPE Q vectors and the mean magnitude
// per frequency band f in [0, freq_count), where freq_count = head_dim/2
struct triattention_head_stats {
    float * q_mean_real;    // [freq_count]  Re(E[q_f])
    float * q_mean_imag;    // [freq_count]  Im(E[q_f])
    float * q_abs_mean;     // [freq_count]  E[||q_f||]

    // Precomputed at init time from the above:
    float * q_mean_abs;     // [freq_count]  ||E[q_f]|| = sqrt(re^2 + im^2)
    float * extra_weight;   // [freq_count]  E[||q_f||] - ||E[q_f]|| (norm excess, MLR-weighted)
};

// Model calibration data loaded from .triattention file
struct triattention_calibration {
    uint32_t head_dim;
    uint32_t num_layers;
    uint32_t num_attn_heads;      // total attention heads
    uint32_t num_kv_heads;        // GQA KV heads
    uint32_t num_kv_groups;       // = num_attn_heads / num_kv_heads
    double   rope_theta;
    uint32_t rope_style;          // 0 = half, 1 = interleaved
    uint32_t freq_count;          // = head_dim / 2
    uint32_t n_sampled;           // number of (layer, head) pairs

    // Per sampled head arrays — length n_sampled
    uint32_t * sampled_layer;     // [n_sampled]  layer index
    uint32_t * sampled_head;      // [n_sampled]  attention head index
    triattention_head_stats * head_stats;  // [n_sampled]

    char model_name[256];
};

// Runtime configuration — set from CLI args, immutable after init
struct triattention_config {
    uint32_t budget;              // Max KV entries to retain after pruning (default: 2048)
    uint32_t divide_length;       // Pruning interval in decode tokens (default: 128)
    uint32_t offset_max;          // Max geometric offset for scoring (default: 65536)

    enum triattention_mode    mode;       // Pruning granularity (default: GLOBAL)
    enum triattention_trigger trigger;    // Trigger strategy (default: INTERVAL)
    enum triattention_agg     agg;        // Score aggregation (default: MEAN)

    bool normalize_scores;        // Z-score normalize per head before selection (default: false)
    bool protect_prefill;         // Never evict initial prompt tokens (default: true)
    bool disable_mlr;             // Ablation: use q_abs_mean directly as extra_weight (default: false)
    bool disable_trig;            // Ablation: drop trigonometric term, norm-only scoring (default: false)
    bool enable_logging;          // Log pruning events to stderr (default: false)

    int32_t seed;                 // RNG seed for tie-breaking noise (-1 = disabled, default: 0)
};

// Runtime state — one per KV cache instance
struct triattention_state {
    triattention_calibration * cal;
    triattention_config cfg;

    // Inference tracking
    int64_t  absolute_position;   // Monotonically increasing token counter
    int64_t  prefix_length;       // Prompt length (protected if protect_prefill)
    uint32_t kv_size;             // Total KV cache capacity (from cache init)

    // Precomputed arrays (allocated once at init)
    float *   omega;              // [freq_count]  RoPE frequencies: theta^(-2f/d)
    float *   freq_scale_sq;      // [freq_count]  frequency scaling^2
    float *   offsets;            // [n_offsets]   geometric {1,2,4,...,offset_max}
    uint32_t  n_offsets;

    // Per-cell absolute position tracking
    // Critical for correct RoPE inversion after eviction:
    // after pruning, surviving cells retain their original absolute positions,
    // but may be at different cache indices than when written.
    int32_t * cell_positions;     // [kv_size]  absolute position per cell (-1 = empty)

    // Scratch buffers (allocated once, reused every prune call)
    float *    dequant_buf;       // [kv_size * head_dim]  dequantized K values
    float *    unrot_buf;         // [kv_size * head_dim]  pre-RoPE K after inversion
    float *    score_buf;         // [n_sampled * kv_size]  per-head scores
    float *    combined_buf;      // [kv_size]  final combined scores
    uint32_t * keep_indices;      // [budget]   indices to retain

    // GPU scoring state (lazily initialized on first prune)
    void *   d_gpu_state;          // triattention_gpu_state* — device calibration data
    float *  d_scores;             // device score buffer (kv_size floats, one head at a time)
    bool     use_gpu;              // true once GPU state is successfully initialized
    bool     gpu_init_tried;       // prevents re-trying init on failure

    // Monitoring statistics
    uint64_t total_prune_calls;
    uint64_t total_tokens_evicted;
    double   total_prune_time_ms;
    double   last_prune_time_ms;
};

// ============================================================================
// Public API
// ============================================================================

#ifdef __cplusplus
extern "C" {
#endif

// Initialize TriAttention state from a calibration file.
// Returns nullptr on failure (file not found, format error, model mismatch).
// The returned state must be freed with triattention_free().
//
// Parameters:
//   stats_path  — path to .triattention binary calibration file
//   cfg         — runtime configuration (copied into state)
//   kv_size     — total KV cache capacity (number of cell slots)
//   rope_theta  — model's RoPE theta for validation against calibration
//   head_dim    — model's attention head dimension for validation
//   n_kv_heads  — model's number of KV heads for validation
triattention_state * triattention_init(
    const char * stats_path,
    const triattention_config * cfg,
    uint32_t kv_size,
    double   rope_theta,
    uint32_t head_dim,
    uint32_t n_kv_heads);

// Free all memory associated with a TriAttention state.
// Safe to call with nullptr.
void triattention_free(triattention_state * state);

// ============================================================================
// Core scoring functions (CPU implementations)
// ============================================================================

// Invert RoPE rotation on post-RoPE key vectors to recover pre-RoPE keys.
// Paper Eq. 4: k_base = invert_rope(k_rotated, position, omega)
//
// For "half" style (Llama/Qwen): first d/2 = real, second d/2 = imaginary
//   out[f]           = in[f]*cos(w_f*pos) + in[f+fc]*sin(w_f*pos)
//   out[f+fc]        = in[f+fc]*cos(w_f*pos) - in[f]*sin(w_f*pos)
//
// Parameters:
//   out           — [n_keys, head_dim] pre-RoPE K output
//   post_rope_k   — [n_keys, head_dim] post-RoPE K input (dequantized)
//   positions     — [n_keys] absolute positions of each key
//   omega         — [freq_count] RoPE frequencies
//   n_keys        — number of keys to process
//   head_dim      — full head dimension (freq_count * 2)
//   freq_count    — head_dim / 2
//   rope_style    — 0=half, 1=interleaved
void triattention_invert_rope(
    float       * out,
    const float * post_rope_k,
    const int32_t * positions,
    const float * omega,
    uint32_t n_keys,
    uint32_t head_dim,
    uint32_t freq_count,
    uint32_t rope_style);

// Score cached keys for a single (layer, head) pair.
// Paper Eqs. 6-10: trigonometric scoring + MLR norm term
//
// For each key i at position p_k:
//   For each freq band f:
//     k_f = complex(pre_rope_k[i*hd+f], pre_rope_k[i*hd+f+fc])
//     amp_f = q_mean_abs[f] * |k_f|
//     phi_f = atan2(Im(E[q_f]*conj(k_f)), Re(E[q_f]*conj(k_f)))
//     extra_f = extra_weight[f] * |k_f|
//   For each offset d in offsets:
//     S_trig += sum_f amp_f * fscale_sq[f] * cos(omega[f]*(Delta+d) + phi_f)
//   S_norm = sum_f extra_f * fscale_sq[f]
//   score = aggregate(S_trig + S_norm) over offsets
//
// Parameters:
//   out_scores    — [n_keys] output importance scores
//   pre_rope_k    — [n_keys, head_dim] pre-RoPE keys (from invert_rope)
//   stats         — calibration statistics for this (layer, head)
//   omega         — [freq_count] RoPE frequencies
//   freq_scale_sq — [freq_count] frequency scaling^2
//   offsets       — [n_offsets] geometric future offsets
//   key_positions — [n_keys] absolute positions
//   round_start   — current decode position (absolute)
//   n_keys        — number of keys
//   head_dim      — full head dimension
//   freq_count    — head_dim / 2
//   n_offsets     — number of geometric offsets
//   agg           — score aggregation method (mean or max)
//   disable_trig  — if true, drop trigonometric term (norm-only ablation)
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

// ============================================================================
// Main pruning entry point
// ============================================================================

// Execute one round of TriAttention pruning on the KV cache.
// Called when trigger conditions are met (see triattention_should_prune).
//
// Algorithm overview:
//   1. Enumerate occupied cells → indices + positions
//   2. Separate protected prefix from decode tokens
//   3. For each sampled (layer, head):
//      a. Dequantize K from cache → float
//      b. Invert RoPE → pre-RoPE K
//      c. Score keys
//   4. Combine scores across heads (mode-dependent)
//   5. Select top-B tokens to keep
//   6. Evict all others via cells.rm()
//
// Returns: number of cells evicted, or 0 if no pruning needed, or -1 on error.
int32_t triattention_prune(
    triattention_state * state,
    llama_kv_cache     * kv);

// Check whether pruning should trigger based on current cache state.
// Called after each token is added to the cache.
//
// Returns true if:
//   INTERVAL mode: n_used >= budget AND (absolute_position % divide_length == 0)
//   SLACK mode:    n_used >= (budget + divide_length)
bool triattention_should_prune(
    const triattention_state * state,
    uint32_t n_used);

// ============================================================================
// Position tracking hooks
// ============================================================================
// These maintain the cell_positions array that maps cache cell indices
// to absolute token positions. Required for correct RoPE inversion
// after previous pruning rounds have compacted the cache.

// Called when a new token is written to a cache cell.
void triattention_on_token_added(
    triattention_state * state,
    uint32_t cell_idx,
    int32_t  abs_pos);

// Called when a cache cell is freed (evicted or overwritten).
void triattention_on_cell_removed(
    triattention_state * state,
    uint32_t cell_idx);

// Called when seq_add() shifts positions for a sequence.
// Must update cell_positions for all cells belonging to seq_id.
void triattention_on_position_shift(
    triattention_state * state,
    int32_t delta,
    int32_t p0,
    int32_t p1);

// Called when the cache is fully cleared/reset.
void triattention_on_reset(
    triattention_state * state);

// ============================================================================
// Monitoring
// ============================================================================

// Print lifetime statistics to the given stream.
void triattention_print_stats(
    const triattention_state * state,
    FILE * stream);

#ifdef __cplusplus
}
#endif

// ============================================================================
// C++ internal API (called from llama-kv-cache.cpp)
// ============================================================================

// Internal pruning implementation that receives K tensor pointers directly.
// Called from llama_kv_cache::triattention_try_prune() which has access to
// the cache's internal layer data.
//
// Parameters:
//   state       — TriAttention runtime state
//   k_tensors   — array of K cache tensors, indexed by internal layer id
//   n_layers    — number of layers in k_tensors array
//   layer_map   — maps internal layer index → model layer index (layers[i].il)
//   kv_size     — cache capacity
//
// Returns: number of cells evicted, or -1 on error
int32_t triattention_prune_impl(
    triattention_state * state,
    ggml_tensor * const * k_tensors,
    uint32_t              n_layers,
    const int32_t       * layer_map,
    uint32_t              kv_size);

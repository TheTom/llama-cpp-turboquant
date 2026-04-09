// TriAttention: Self-Calibrating Trigonometric KV Cache Token Eviction
//
// Inspired by "TriAttention: Efficient Long Reasoning with Trigonometric KV Compression"
// by Weian Mao, Xi Lin, Wei Huang, Yuxin Xie, Tianfu Fu, Bohan Zhuang, Song Han, Yukang Chen
// arXiv:2604.04921, 2026
//
// This is an independent implementation that self-calibrates from Q projections
// during prefill — no external calibration files or Python dependencies needed.

#include "llama-triattention.h"
#include "llama-context.h"
#include "llama-kv-cache.h"
#include "llama-kv-cells.h"
#include "llama-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <thread>

static constexpr int32_t OFFSET_MAX = 65536;

void llama_triattention::init_constants(float theta, uint32_t hd, uint32_t nh, uint32_t nkv, uint32_t nl, uint32_t nr) {
    rope_theta  = theta;
    head_dim    = hd;
    n_rot       = (nr == 0) ? hd : nr;   // default: full RoPE
    freq_count  = n_rot / 2;
    n_heads     = nh;
    n_kv_heads  = nkv;
    n_layers    = nl;

    // RoPE angular frequencies: omega[i] = 1 / (theta^(2i/n_rot))
    // Note: uses n_rot (not head_dim) as the denominator — for partial RoPE, only
    // the first n_rot dims participate in the trig scoring.
    omega.resize(freq_count);
    for (uint32_t i = 0; i < freq_count; ++i) {
        omega[i] = 1.0f / powf(theta, (float)(2 * i) / (float)n_rot);
    }

    // Geometric offsets: [1, 2, 4, 8, ..., 65536]
    offsets.clear();
    for (int32_t v = 1; v <= OFFSET_MAX; v *= 2) {
        offsets.push_back((float)v);
    }

    // Allocate Q accumulators + EMA snapshots
    const uint32_t n_total = n_layers * n_kv_heads;
    q_sum_real.assign(n_total, std::vector<float>(freq_count, 0.0f));
    q_sum_imag.assign(n_total, std::vector<float>(freq_count, 0.0f));
    q_sum_abs.assign(n_total,  std::vector<float>(freq_count, 0.0f));
    q_prev_sum_real.assign(n_total, std::vector<float>(freq_count, 0.0f));
    q_prev_sum_imag.assign(n_total, std::vector<float>(freq_count, 0.0f));
    q_prev_sum_abs.assign(n_total,  std::vector<float>(freq_count, 0.0f));
    q_samples = 0;
    q_samples_at_last_update = 0;

    LLAMA_LOG_INFO("%s: %u layers, %u heads (%u kv), head_dim=%u, n_rot=%u, freq=%u, theta=%.1f\n",
        __func__, nl, nh, nkv, hd, n_rot, freq_count, theta);
}

void llama_triattention::accumulate_q(const float * q_data, int32_t n_tokens, int32_t layer_idx, size_t stride_elems) {
    if ((uint32_t)layer_idx >= n_layers) {
        return;
    }

    // Q tensor layout (pre-RoPE): per head, a head_dim-long float vector.
    // For 2D capture (MUL_MAT output): [head_dim * n_heads, n_tokens]
    //   → head stride = head_dim, token stride = head_dim * n_heads
    // For 3D view (fused-QKV models like qwen35): [head_dim, n_heads, n_tokens]
    //   → head stride = nb[1] / sizeof(float) (may differ from head_dim if view has gaps)
    //   → token stride = nb[2] / sizeof(float)
    // stride_elems is the head stride in floats (nb[1] / sizeof(float)).
    // If 0, defaults to head_dim (contiguous).
    const uint32_t heads_per_kv = (n_kv_heads > 0) ? (n_heads / n_kv_heads) : 1;
    const uint32_t fc = freq_count;
    const uint32_t hd = head_dim;
    const uint32_t nr = n_rot;

    const size_t head_stride = (stride_elems > 0) ? stride_elems : (size_t)hd;
    const size_t token_stride = head_stride * (size_t)n_heads;

    // Only the first n_rot dims participate in RoPE. Half-layout puts:
    //   real = q[0 .. fc)        where fc = n_rot/2
    //   imag = q[fc .. n_rot)
    // For full RoPE (n_rot == head_dim), dims beyond n_rot don't exist.
    // For partial RoPE, dims [n_rot..head_dim) carry non-rotated features that
    // we ignore for trig scoring (they contribute via the MLR term in score_tokens).
    (void)nr;

    for (int32_t t = 0; t < n_tokens; ++t) {
        for (uint32_t kv_h = 0; kv_h < n_kv_heads; ++kv_h) {
            const uint32_t block_idx = layer_idx * n_kv_heads + kv_h;

            for (uint32_t f = 0; f < fc; ++f) {
                float sum_r = 0.0f, sum_i = 0.0f, sum_a = 0.0f;

                for (uint32_t g = 0; g < heads_per_kv; ++g) {
                    const uint32_t q_h = kv_h * heads_per_kv + g;
                    const float * q = q_data + (size_t)t * token_stride + (size_t)q_h * head_stride;

                    const float qr = q[f];
                    const float qi = q[fc + f];

                    sum_r += qr;
                    sum_i += qi;
                    sum_a += sqrtf(qr * qr + qi * qi + 1e-8f);
                }

                const float inv_g = 1.0f / (float)heads_per_kv;
                q_sum_real[block_idx][f] += sum_r * inv_g;
                q_sum_imag[block_idx][f] += sum_i * inv_g;
                q_sum_abs[block_idx][f]  += sum_a * inv_g;
            }
        }
    }
}

void llama_triattention::update_calibration() {
    if (q_samples <= 0) {
        return;
    }

    const uint32_t n_total = n_layers * n_kv_heads;

    if (!calibrated) {
        // First calibration: compute mean from accumulated sums
        center_real.resize(n_total);
        center_imag.resize(n_total);
        center_abs.resize(n_total);

        const float inv_n = 1.0f / (float)q_samples;
        for (uint32_t i = 0; i < n_total; ++i) {
            center_real[i].resize(freq_count);
            center_imag[i].resize(freq_count);
            center_abs[i].resize(freq_count);
            for (uint32_t f = 0; f < freq_count; ++f) {
                center_real[i][f] = q_sum_real[i][f] * inv_n;
                center_imag[i][f] = q_sum_imag[i][f] * inv_n;
                center_abs[i][f]  = q_sum_abs[i][f]  * inv_n;
            }
        }

        calibrated = true;
        LLAMA_LOG_INFO("%s: initial calibration from %d tokens (%u layers, %u kv_heads)\n",
            __func__, q_samples, n_layers, n_kv_heads);

        // Non-adaptive mode: request the cb_eval scheduler hook be removed as
        // soon as it's safe. The ggml scheduler takes a much slower per-node
        // path when any eval_callback is installed (loses graph fusion +
        // forces per-range synchronization), and this cost runs for EVERY
        // batch until we uninstall. We set a pending flag here; the actual
        // call to set_eval_callback happens in llama_context::decode() after
        // the forward pass returns — doing it from inside cb_eval itself is
        // unsafe because the scheduler is actively iterating its callback
        // field.
        static int adaptive_mode_cached = -1;
        if (adaptive_mode_cached < 0) {
            const char * env = getenv("TRIATT_ADAPTIVE");
            adaptive_mode_cached = env ? atoi(env) : 0;
        }
        if (adaptive_mode_cached == 0) {
            pending_uninstall = true;
            LLAMA_LOG_INFO("%s: scheduling cb_eval uninstall (non-adaptive mode)\n", __func__);
        }
    } else {
        // Adaptive update: EMA blend of existing centers with new batch stats
        // alpha controls how fast we adapt (0.1 = slow/stable, 0.5 = fast/reactive)
        const float alpha = ema_alpha;
        const int32_t new_samples = q_samples - q_samples_at_last_update;
        if (new_samples <= 0) {
            return;
        }

        const float inv_n = 1.0f / (float)new_samples;
        for (uint32_t i = 0; i < n_total; ++i) {
            for (uint32_t f = 0; f < freq_count; ++f) {
                // New batch mean (delta since last update)
                const float new_r = (q_sum_real[i][f] - q_prev_sum_real[i][f]) * inv_n;
                const float new_i = (q_sum_imag[i][f] - q_prev_sum_imag[i][f]) * inv_n;
                const float new_a = (q_sum_abs[i][f]  - q_prev_sum_abs[i][f])  * inv_n;

                // EMA update
                center_real[i][f] = (1.0f - alpha) * center_real[i][f] + alpha * new_r;
                center_imag[i][f] = (1.0f - alpha) * center_imag[i][f] + alpha * new_i;
                center_abs[i][f]  = (1.0f - alpha) * center_abs[i][f]  + alpha * new_a;
            }
        }

        LLAMA_LOG_INFO("%s: adaptive update from %d new tokens (total=%d, alpha=%.2f)\n",
            __func__, new_samples, q_samples, alpha);
    }

    // Snapshot current sums for next delta computation
    q_prev_sum_real = q_sum_real;
    q_prev_sum_imag = q_sum_imag;
    q_prev_sum_abs  = q_sum_abs;
    q_samples_at_last_update = q_samples;
}

// Static callback for ggml eval — captures pre-RoPE Q tensors.
//
// Two capture paths depending on model architecture:
//
//   Path A (standard transformers like qwen2, llama):
//     Name: "Qcur-N"
//     Shape: 2D [head_dim * n_heads, n_tokens]
//     Op: MUL_MAT or ADD (both pre-RoPE). We accept both.
//     Data layout: contiguous [d + h*head_dim + t*head_dim*n_heads]
//
//   Path B (fused-QKV hybrid models like qwen35, qwen35moe):
//     Name: "Qcur_normed-N" (after Q norm, before RoPE)
//     Shape: 3D [head_dim, n_heads, n_tokens]
//     Data layout: contiguous (post-norm output is dense)
//
// Both paths lead into accumulate_q, which walks tokens × heads × freq.
bool llama_triattention::eval_callback(ggml_tensor * t, bool ask, void * user_data) {
    auto * self = (llama_triattention *)user_data;

    // Fast bail: once calibrated we stop accumulating Q stats unless the
    // adaptive flag is set. This removes the cb_eval hot path from steady-state
    // prefill/decode — verified to be the dominant cost on 7B @ 32K (~14s out of
    // 16s of V3 overhead). Adaptive mode can be re-enabled via TRIATT_ADAPTIVE=1.
    static int adaptive_mode = -1;
    if (adaptive_mode < 0) {
        const char * env = getenv("TRIATT_ADAPTIVE");
        adaptive_mode = env ? atoi(env) : 0;
        if (adaptive_mode != 0) {
            LLAMA_LOG_INFO("triatt: adaptive calibration ENABLED (cb_eval runs on every forward)\n");
        }
    }
    if (self->calibrated && adaptive_mode == 0) {
        return false;
    }

    const char * name = ggml_get_name(t);
    if (!name) return false;

    // Identify tensor type by name prefix
    int layer_idx = -1;
    bool path_b = false; // Path B = 3D "Qcur_normed-N"

    if (strncmp(name, "Qcur_normed-", 12) == 0) {
        layer_idx = atoi(name + 12);
        path_b = true;
    } else if (strncmp(name, "Qcur-", 5) == 0) {
        layer_idx = atoi(name + 5);
        path_b = false;
    } else {
        return false;
    }

    const int64_t hd_i64 = (int64_t)self->head_dim;
    const int64_t nh_i64 = (int64_t)self->n_heads;

    if (path_b) {
        // 3D [head_dim, n_heads, n_tokens], pre-RoPE (post-norm)
        if (t->ne[0] != hd_i64 || t->ne[1] != nh_i64) return false;
        // Skip post-RoPE variant (shouldn't match "Qcur_normed" anyway, but belt-and-suspenders)
        if (t->op == GGML_OP_ROPE) return false;
    } else {
        // 2D [head_dim * n_heads, n_tokens], pre-RoPE (pre-reshape)
        const int64_t expected_ne0 = hd_i64 * nh_i64;
        if (t->ne[0] != expected_ne0) return false;
        if (t->op == GGML_OP_ROPE) return false;
    }

    if (ask) return true;

    const float * data = (const float *)t->data;
    if (!data) return true;

    // Extract n_tokens and per-head stride (in floats) from tensor metadata.
    int32_t n_tokens;
    size_t head_stride_elems;
    if (path_b) {
        // 3D: ne = [head_dim, n_heads, n_tokens]
        // nb[1] = stride between heads (bytes), nb[2] = stride between tokens
        n_tokens = (int32_t)t->ne[2];
        head_stride_elems = t->nb[1] / sizeof(float);
    } else {
        // 2D: ne = [head_dim * n_heads, n_tokens]
        // Heads are contiguous along dim 0: head stride = head_dim floats.
        n_tokens = (int32_t)t->ne[1];
        head_stride_elems = (size_t)self->head_dim;
    }

    self->accumulate_q(data, n_tokens, layer_idx, head_stride_elems);

    // Token counting: count once per forward pass to track total tokens processed
    // (used for warmup threshold).
    //
    // Standard transformers (qwen2, llama): attention layers start at 0, so we
    //   count on layer 0. For bias models, both MUL_MAT and ADD variants of Qcur-0
    //   fire, double-counting — but that just warms up twice as fast.
    //
    // Hybrid models (qwen35): first attention layer might be 3 (every 4th layer),
    //   not 0. Track the smallest layer_idx ever observed and count on that one.
    if (self->first_attn_layer < 0 || layer_idx < self->first_attn_layer) {
        self->first_attn_layer = layer_idx;
    }
    if (layer_idx == self->first_attn_layer) {
        self->q_samples += n_tokens;

        // If we just crossed the warmup threshold, run calibration NOW rather
        // than waiting for evict() to be called. This matters because evict()
        // only runs when the cache exceeds budget — at large contexts that
        // can be many batches after calibration is technically ready. Every
        // batch in that window runs the ggml scheduler's slow per-node
        // callback dispatch path, which is the dominant V3 overhead.
        if (!self->calibrated && self->q_samples >= self->warmup_tokens) {
            self->update_calibration();
        }
    }

    return true;
}

bool llama_triattention::should_evict(int32_t n_used) const {
    // Ensure budget > window_size to avoid infinite eviction loops
    const int32_t effective_budget = std::max(budget, window_size + 1);
    return n_used > effective_budget + divide_length;
}

void llama_triattention::score_tokens(
        const float * k_data_f32,
        int32_t       n_tokens,
        int32_t       current_pos,
        int32_t       layer_idx,
        int32_t       kv_head_idx,
        float       * scores_out) const {

    const uint32_t fc = freq_count;
    const uint32_t hd = head_dim;
    const uint32_t head_block = layer_idx * n_kv_heads + kv_head_idx;

    if (head_block >= center_real.size()) {
        return;
    }

    const float * c_r   = center_real[head_block].data();
    const float * c_i   = center_imag[head_block].data();
    const float * c_abs = center_abs[head_block].data();

    // Precompute |center_complex|
    std::vector<float> c_mag(fc);
    for (uint32_t f = 0; f < fc; ++f) {
        c_mag[f] = sqrtf(c_r[f] * c_r[f] + c_i[f] * c_i[f] + 1e-8f);
    }

    const int n_off = (int)offsets.size();

    for (int32_t t = 0; t < n_tokens; ++t) {
        const float * k = k_data_f32 + (size_t)t * hd;

        float extra_sum = 0.0f;
        std::vector<float> A_coef(fc);
        std::vector<float> B_coef(fc);

        for (uint32_t f = 0; f < fc; ++f) {
            // K_rot in "half" layout
            const float k_r = k[f];
            const float k_i = k[fc + f];

            // Complex product: center * conj(K_rot)
            const float prod_real = c_r[f] * k_r + c_i[f] * k_i;
            const float prod_imag = c_i[f] * k_r - c_r[f] * k_i;

            A_coef[f] = prod_real;
            B_coef[f] = prod_imag;

            // MLR additive term
            const float k_abs = sqrtf(k_r * k_r + k_i * k_i + 1e-8f);
            extra_sum += (c_abs[f] - c_mag[f]) * k_abs;
        }

        // Aggregate scores over offsets (mean mode)
        float mean_score = 0.0f;
        for (int o = 0; o < n_off; ++o) {
            const float t_val = (float)current_pos + offsets[o];

            float base_score = 0.0f;
            for (uint32_t f = 0; f < fc; ++f) {
                const float phase = t_val * omega[f];
                base_score += A_coef[f] * cosf(phase) - B_coef[f] * sinf(phase);
            }

            mean_score += base_score + extra_sum;
        }

        scores_out[t] += mean_score / (float)n_off;
    }
}

int32_t llama_triattention::evict(llama_kv_cache * cache, ggml_type type_k) {
    if (type_k != GGML_TYPE_F16 && type_k != GGML_TYPE_Q8_0) {
        // Unsupported K type (e.g. TQ turbo formats). We can't score on this
        // cache, BUT we still need to uninstall the cb_eval scheduler hook —
        // otherwise on stacked runs (TQ + V3) the callback leaks and the slow
        // scheduler path runs for the entire prefill. Finish calibration if
        // we've crossed the warmup threshold and request the uninstall.
        if (!calibrated && q_samples >= warmup_tokens) {
            update_calibration();
        }
        static bool warned_once = false;
        if (!warned_once) {
            LLAMA_LOG_WARN("%s: unsupported K type %d — TriAttention cannot evict on this K, "
                           "but calibration + cb_eval uninstall still happen\n",
                           __func__, type_k);
            warned_once = true;
        }
        return 0;
    }

    const auto & cells = cache->get_cells(0);
    const int32_t n_used = (int32_t)cells.get_used();

    // Warm-up phase: don't evict until we have enough Q statistics
    if (q_samples < warmup_tokens) {
        return 0; // let cache grow freely during warm-up
    }

    // Adaptive calibration: update Q stats every eviction round
    update_calibration();
    if (!calibrated) {
        return 0;
    }

    if (!should_evict(n_used)) {
        return 0;
    }

    // Detect n_kv_heads from cache if not set
    if (n_kv_heads == 0) {
        const uint64_t ne0 = cache->get_k_ne0(0);
        n_kv_heads = (uint32_t)(ne0 / head_dim);
    }

    // Collect used cells
    struct cell_info {
        uint32_t  idx;
        llama_pos pos;
        float     score;
        bool      evicted;
    };

    std::vector<cell_info> active_cells;
    active_cells.reserve(n_used);

    const uint32_t kv_size = cache->get_size();
    for (uint32_t i = 0; i < kv_size; ++i) {
        if (!cells.is_empty(i)) {
            active_cells.push_back({i, cells.pos_get(i), 0.0f, false});
        }
    }

    if ((int32_t)active_cells.size() <= budget) {
        return 0;
    }

    llama_pos max_pos = 0;
    for (const auto & c : active_cells) {
        max_pos = std::max(max_pos, c.pos);
    }
    const llama_pos window_threshold = max_pos - window_size + 1;

    // Scoring mode: TRIATT_MODE env var
    // 0 = trig scoring (default), 1 = random, 2 = recency (keep newest)
    static int scoring_mode = -1;
    if (scoring_mode < 0) {
        const char * env = getenv("TRIATT_MODE");
        scoring_mode = env ? atoi(env) : 0;
        if (scoring_mode != 0) {
            LLAMA_LOG_INFO("%s: using scoring mode %d (%s)\n", __func__, scoring_mode,
                scoring_mode == 1 ? "random" : scoring_mode == 2 ? "recency" : "unknown");
        }
    }

    if (scoring_mode == 1) {
        // Random scoring baseline
        for (auto & cell : active_cells) {
            if (cell.pos >= window_threshold) {
                cell.score = 1e10f;
            } else {
                cell.score = (float)(rand() % 10000) / 10000.0f;
            }
        }
    } else if (scoring_mode == 2) {
        // Recency baseline: score = position (higher pos = keep)
        for (auto & cell : active_cells) {
            cell.score = (cell.pos >= window_threshold) ? 1e10f : (float)cell.pos;
        }
    } else {
        // Trig scoring (default) — threaded + vectorizable hot-path version.
        //
        // Algebraic identity we already exploit (see earlier commit):
        //   score_per_cell = sum_f [A_f cos_sum_f - B_f sin_sum_f] + extra
        // where cos_sum/sin_sum are precomputed once per evict().
        //
        // Two additional optimizations in this pass:
        //   A. THREADING: the per-(layer, head) blocks are trivially parallel.
        //      Each thread gets a per-cell accumulator buffer, merges at the end.
        //   B. SPLIT ACCUMULATORS: the inner loop had two serial dependency chains
        //      (acc += ...  and  extra += ...). Splitting each into 4 parallel
        //      chains lets the compiler auto-vectorize to NEON/AVX.
        //
        // Semantics are identical to the previous pass modulo float associativity
        // (~1e-5 PPL noise from reordered summation).

        const uint32_t fc = freq_count;
        const int n_off   = (int)offsets.size();
        const float inv_n_off = 1.0f / (float)n_off;

        // Precompute cos_sum / sin_sum ONCE — constant across all cells and all blocks.
        std::vector<float> cos_sum(fc);
        std::vector<float> sin_sum(fc);
        for (uint32_t f = 0; f < fc; ++f) {
            float cs = 0.0f, ss = 0.0f;
            for (int o = 0; o < n_off; ++o) {
                const float phase = ((float)max_pos + offsets[o]) * omega[f];
                cs += cosf(phase);
                ss += sinf(phase);
            }
            cos_sum[f] = cs * inv_n_off;
            sin_sum[f] = ss * inv_n_off;
        }

        const auto * type_traits = ggml_get_type_traits(type_k);
        const auto   to_float    = type_traits->to_float;
        const size_t type_size   = ggml_type_size(type_k);
        const int64_t blk_size   = ggml_blck_size(type_k);
        const uint32_t n_layers_kv = cache->get_n_layers_kv();

        // Pre-mark protected cells once so we can skip them cheaply.
        for (auto & cell : active_cells) {
            cell.score = (cell.pos >= window_threshold) ? 1e10f : 0.0f;
        }

        // Flatten (layer, head) blocks into a single index space for round-robin
        // thread assignment.
        struct block_info {
            int32_t il;
            uint32_t h;
            const uint8_t * k_base;
            size_t cell_stride;
            size_t head_offset;
        };
        std::vector<block_info> blocks;
        blocks.reserve(n_layers_kv * n_kv_heads);

        for (uint32_t ikv = 0; ikv < n_layers_kv; ++ikv) {
            const int32_t il = cache->get_layer_il(ikv);
            if ((uint32_t)il >= n_layers) continue;

            const uint8_t * k_base = (const uint8_t *)cache->get_k_data(ikv);
            const uint64_t  ne0    = cache->get_k_ne0(ikv);
            const size_t cell_stride = (ne0 / blk_size) * type_size;

            for (uint32_t h = 0; h < n_kv_heads; ++h) {
                const size_t head_offset = ((size_t)h * head_dim / blk_size) * type_size;
                blocks.push_back({il, h, k_base, cell_stride, head_offset});
            }
        }

        // Decide thread count. Cap at 8 to avoid diminishing returns on small
        // per-thread chunks (35B-A3B has only 20 blocks; with 8 threads each gets
        // 2-3 blocks, still worth it).
        const unsigned int n_hw = std::thread::hardware_concurrency();
        const int n_threads = std::max(1, std::min<int>(n_hw > 0 ? (int)n_hw : 1, 8));

        const size_t n_cells = active_cells.size();
        std::vector<std::vector<float>> thread_scores(n_threads);
        for (int t = 0; t < n_threads; ++t) {
            thread_scores[t].assign(n_cells, 0.0f);
        }

        // Snapshot pointers the worker needs so it doesn't touch shared state.
        const float * cos_sum_ptr = cos_sum.data();
        const float * sin_sum_ptr = sin_sum.data();
        const uint32_t head_dim_local = head_dim;

        auto worker = [&](int tid) {
            std::vector<float> k_f32(head_dim_local);

            for (size_t bi = (size_t)tid; bi < blocks.size(); bi += (size_t)n_threads) {
                const auto & block = blocks[bi];
                const uint32_t head_block = (uint32_t)block.il * n_kv_heads + block.h;

                if (head_block >= center_real.size()) continue;

                const float * c_r   = center_real[head_block].data();
                const float * c_i   = center_imag[head_block].data();
                const float * c_abs = center_abs[head_block].data();

                // Lift c_mag / cb_delta out of the cell loop.
                std::vector<float> cb_delta(fc);
                for (uint32_t f = 0; f < fc; ++f) {
                    const float mag = sqrtf(c_r[f] * c_r[f] + c_i[f] * c_i[f] + 1e-8f);
                    cb_delta[f] = c_abs[f] - mag;
                }

                float * const tscores = thread_scores[tid].data();
                const float * const c_r_p = c_r;
                const float * const c_i_p = c_i;
                const float * const cb_delta_p = cb_delta.data();

                for (size_t ci = 0; ci < n_cells; ++ci) {
                    const auto & cell = active_cells[ci];
                    if (cell.pos >= window_threshold) continue;

                    const uint8_t * k_ptr = block.k_base + (size_t)cell.idx * block.cell_stride + block.head_offset;
                    to_float(k_ptr, k_f32.data(), (int64_t)head_dim_local);

                    const float * const k = k_f32.data();

                    // Split accumulators (4-way) so the compiler can vectorize.
                    float acc0=0, acc1=0, acc2=0, acc3=0;
                    float ext0=0, ext1=0, ext2=0, ext3=0;

                    uint32_t f = 0;
                    for (; f + 4 <= fc; f += 4) {
                        {
                            const float k_r = k[f+0];
                            const float k_i = k[fc + f+0];
                            const float A = c_r_p[f+0] * k_r + c_i_p[f+0] * k_i;
                            const float B = c_i_p[f+0] * k_r - c_r_p[f+0] * k_i;
                            acc0 += A * cos_sum_ptr[f+0] - B * sin_sum_ptr[f+0];
                            const float k_abs = sqrtf(k_r * k_r + k_i * k_i + 1e-8f);
                            ext0 += cb_delta_p[f+0] * k_abs;
                        }
                        {
                            const float k_r = k[f+1];
                            const float k_i = k[fc + f+1];
                            const float A = c_r_p[f+1] * k_r + c_i_p[f+1] * k_i;
                            const float B = c_i_p[f+1] * k_r - c_r_p[f+1] * k_i;
                            acc1 += A * cos_sum_ptr[f+1] - B * sin_sum_ptr[f+1];
                            const float k_abs = sqrtf(k_r * k_r + k_i * k_i + 1e-8f);
                            ext1 += cb_delta_p[f+1] * k_abs;
                        }
                        {
                            const float k_r = k[f+2];
                            const float k_i = k[fc + f+2];
                            const float A = c_r_p[f+2] * k_r + c_i_p[f+2] * k_i;
                            const float B = c_i_p[f+2] * k_r - c_r_p[f+2] * k_i;
                            acc2 += A * cos_sum_ptr[f+2] - B * sin_sum_ptr[f+2];
                            const float k_abs = sqrtf(k_r * k_r + k_i * k_i + 1e-8f);
                            ext2 += cb_delta_p[f+2] * k_abs;
                        }
                        {
                            const float k_r = k[f+3];
                            const float k_i = k[fc + f+3];
                            const float A = c_r_p[f+3] * k_r + c_i_p[f+3] * k_i;
                            const float B = c_i_p[f+3] * k_r - c_r_p[f+3] * k_i;
                            acc3 += A * cos_sum_ptr[f+3] - B * sin_sum_ptr[f+3];
                            const float k_abs = sqrtf(k_r * k_r + k_i * k_i + 1e-8f);
                            ext3 += cb_delta_p[f+3] * k_abs;
                        }
                    }

                    // Tail (fc not multiple of 4 — rare but safe)
                    float acc_tail = 0.0f, ext_tail = 0.0f;
                    for (; f < fc; ++f) {
                        const float k_r = k[f];
                        const float k_i = k[fc + f];
                        const float A = c_r_p[f] * k_r + c_i_p[f] * k_i;
                        const float B = c_i_p[f] * k_r - c_r_p[f] * k_i;
                        acc_tail += A * cos_sum_ptr[f] - B * sin_sum_ptr[f];
                        const float k_abs = sqrtf(k_r * k_r + k_i * k_i + 1e-8f);
                        ext_tail += cb_delta_p[f] * k_abs;
                    }

                    const float acc = (acc0 + acc1) + (acc2 + acc3) + acc_tail;
                    const float ext = (ext0 + ext1) + (ext2 + ext3) + ext_tail;
                    tscores[ci] += acc + ext;
                }
            }
        };

        // Dispatch
        if (n_threads == 1) {
            worker(0);
        } else {
            std::vector<std::thread> workers;
            workers.reserve(n_threads);
            for (int t = 0; t < n_threads; ++t) workers.emplace_back(worker, t);
            for (auto & w : workers) w.join();
        }

        // Merge per-thread scores into active_cells, then normalize.
        const float score_norm = 1.0f / (float)(n_layers_kv * n_kv_heads);
        for (size_t ci = 0; ci < n_cells; ++ci) {
            auto & cell = active_cells[ci];
            if (cell.pos >= window_threshold) continue;
            float s = 0.0f;
            for (int t = 0; t < n_threads; ++t) {
                s += thread_scores[t][ci];
            }
            cell.score = s * score_norm;
        }
    }

    // ----- Eviction selection -----
    //
    // The eviction policy is selected in priority order:
    //   1. CLI flag / API parameter (hybrid_mode_cli, set via --triatt-hybrid or
    //      llama_triattention_enable's hybrid_mode argument).
    //   2. TRIATT_HYBRID environment variable (fallback for scripting).
    //   3. Default: 0 (V1 paper-faithful, global sort by score).
    //
    // 0 = V1 paper-faithful: global sort, evict the highest-score cells.
    // 1 = V2 per-segment quota only (experimental).
    // 2 = V3 prefix-protect + per-segment quota (recommended for standard
    //     transformers at up to 64K context).
    static int hybrid_mode = -2;
    if (hybrid_mode == -2) {
        if (hybrid_mode_cli >= 0) {
            hybrid_mode = hybrid_mode_cli;
        } else {
            const char * env = getenv("TRIATT_HYBRID");
            hybrid_mode = env ? atoi(env) : 0;
        }
        const char * p_env = getenv("TRIATT_PREFIX");
        if (p_env) {
            prefix_protect = atoi(p_env);
        }
        if (hybrid_mode == 1) {
            LLAMA_LOG_INFO("%s: V2 hybrid policy enabled (per-segment quota, n_segments=%d)\n",
                __func__, (int)n_segments);
        } else if (hybrid_mode == 2) {
            LLAMA_LOG_INFO("%s: V3 hybrid policy enabled (prefix_protect=%d, n_segments=%d)\n",
                __func__, (int)prefix_protect, (int)n_segments);
        }
    }

    const int32_t n_to_evict = n_used - budget;
    int32_t n_evicted = 0;

    if (hybrid_mode == 0) {
        // V1: global sort, evict highest-score cells.
        // Sort ascending by score. Evict from the END (highest scores are least important).
        // Note: our trig formula produces high scores for unimportant tokens
        // (the complex product Q*conj(K) measures orthogonality, not alignment).
        std::sort(active_cells.begin(), active_cells.end(),
            [](const cell_info & a, const cell_info & b) { return a.score < b.score; });

        const int32_t n_cells = (int32_t)active_cells.size();
        for (int32_t i = n_cells - 1; i >= 0 && n_evicted < n_to_evict; --i) {
            const auto & cell = active_cells[i];
            if (cell.pos >= window_threshold) continue; // skip protected
            cache->seq_rm(0, cell.pos, cell.pos + 1);
            ++n_evicted;
        }
    } else {
        // V2 / V3: per-segment quota (with optional V3 prefix protection).
        //
        // Partition non-protected cells into n_segments buckets by position
        // (uniform splits over [prefix_lo, window_threshold)). For V3, cells
        // with pos < prefix_protect are skipped entirely (never evicted).
        //
        // Compute a per-segment target proportional to that segment's
        // eviction-eligible cell count so the *fraction* removed from each
        // segment is roughly equal.
        //
        // Bucketing uses *indices into active_cells* to avoid copying full
        // cell_info structs. The cleanup pass reuses the original active_cells
        // array (with the `evicted` flag) so we never have to re-scan the
        // underlying kv_cells — the previous implementation was O(n^2) per
        // eviction round.

        const int32_t k = std::max(1, n_segments);
        const llama_pos prefix_lo = (hybrid_mode == 2) ? (llama_pos)prefix_protect : 0;

        std::vector<std::vector<uint32_t>> buckets(k); // indices into active_cells

        const llama_pos pos_hi = std::max<llama_pos>(window_threshold, prefix_lo + 1);
        const float seg_width = (float)(pos_hi - prefix_lo) / (float)k;

        for (uint32_t ai = 0; ai < active_cells.size(); ++ai) {
            const auto & cell = active_cells[ai];
            if (cell.pos >= window_threshold || cell.pos < prefix_lo) continue;
            int seg = (int)((float)(cell.pos - prefix_lo) / seg_width);
            if (seg < 0) seg = 0;
            if (seg >= k) seg = k - 1;
            buckets[seg].push_back(ai);
        }

        int32_t total_eligible = 0;
        for (const auto & b : buckets) total_eligible += (int32_t)b.size();
        if (total_eligible <= 0) {
            LLAMA_LOG_INFO("%s: hybrid evict skipped — no eligible cells (n_used=%d, window=%d)\n",
                __func__, n_used, (int)window_threshold);
            return 0;
        }

        const float target_frac = (float)n_to_evict / (float)total_eligible;

        for (int s = 0; s < k; ++s) {
            auto & bucket = buckets[s];
            if (bucket.empty()) continue;

            // Sort bucket indices by score (ascending — highest scores last).
            std::sort(bucket.begin(), bucket.end(),
                [&active_cells](uint32_t a, uint32_t b) {
                    return active_cells[a].score < active_cells[b].score;
                });

            const int32_t bucket_target = (int32_t)((float)bucket.size() * target_frac);
            const int32_t bucket_to_evict = std::min(bucket_target, (int32_t)bucket.size());

            for (int32_t i = (int32_t)bucket.size() - 1, evicted_here = 0;
                 i >= 0 && evicted_here < bucket_to_evict && n_evicted < n_to_evict;
                 --i, ++evicted_here) {
                auto & cell = active_cells[bucket[i]];
                cache->seq_rm(0, cell.pos, cell.pos + 1);
                cell.evicted = true;
                ++n_evicted;
            }
        }

        // Cleanup pass — O(n log n) instead of the previous O(n^2).
        //
        // If rounding left us short, sort the remaining (not-yet-evicted,
        // not-prefix-protected, not-window) cells globally by score and evict
        // the highest. We reuse the original active_cells / score data directly;
        // no cache re-scan, no linear search per cell.
        if (n_evicted < n_to_evict) {
            std::vector<uint32_t> remaining;
            remaining.reserve(total_eligible);
            for (uint32_t ai = 0; ai < active_cells.size(); ++ai) {
                const auto & cell = active_cells[ai];
                if (cell.evicted) continue;
                if (cell.pos >= window_threshold) continue;
                if (cell.pos < prefix_lo) continue;
                remaining.push_back(ai);
            }
            std::sort(remaining.begin(), remaining.end(),
                [&active_cells](uint32_t a, uint32_t b) {
                    return active_cells[a].score < active_cells[b].score;
                });
            for (int32_t i = (int32_t)remaining.size() - 1;
                 i >= 0 && n_evicted < n_to_evict; --i) {
                auto & cell = active_cells[remaining[i]];
                cache->seq_rm(0, cell.pos, cell.pos + 1);
                cell.evicted = true;
                ++n_evicted;
            }
        }
    }

    const char * mode_str =
        (hybrid_mode == 0) ? "v1" :
        (hybrid_mode == 1) ? "v2" :
        (hybrid_mode == 2) ? "v3" : "unknown";
    LLAMA_LOG_INFO("%s: evicted %d/%d tokens (budget=%d, used=%d, max_pos=%d, mode=%s)\n",
        __func__, n_evicted, n_used, budget, n_used - n_evicted, (int)max_pos, mode_str);

    return n_evicted;
}

#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

// emit_ingredients_t (only meaningful when keep_rs_t): write per-token (k,v,g,beta) ingredients
// instead of a full [S_v,S_v] state snapshot per retained slot, plus one fixed-cost trailing
// final-state block -- see ggml_gated_delta_net's emit_mode==1 contract (ggml.h) and the CPU
// reference (ggml-cpu/ops.cpp). Never instantiated with keep_rs_t == false.
template <int S_v, bool KDA, bool keep_rs_t, bool emit_ingredients_t>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output layout (per-slot stride passed in as state_slot_stride) — same per-(seq,head)
    // offset scheme as before, except emit_ingredients_t slots are 4*S_v wide (k,v,g,beta)
    // instead of S_v*S_v (a full state matrix); state_out points at the base of THIS
    // (seq,head)'s slot-0 storage either way. `state` (unoffset) is kept around separately so
    // the trailing final-state block (always S_v*S_v-shaped, at a fixed offset past all K
    // ingredient slots) can be addressed too.
    const int64_t state_in_offset = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    float * state_out;
    if constexpr (emit_ingredients_t) {
        state_out = state + (sequence * H + h_idx) * (4 * S_v);
    } else {
        state_out = state + (sequence * H + h_idx) * S_v * S_v;
    }

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // emit_ingredients_t==false: most-recent-first (slot 0 = final state), matching the
            // s_copy/rs_idx row-selection convention used elsewhere. emit_ingredients_t==true:
            // chronological (slot 0 = oldest of the K retained tokens) -- ingredients are only
            // ever consumed by this op's own replay call site, which needs a straight forward
            // prefix view, not a reversed one; see the CPU reference for the full rationale.
            const int target_slot = emit_ingredients_t ? ((int) t - ((int) n_tokens - K)) : ((int) n_tokens - 1 - t);
            if (target_slot >= 0 && target_slot < K) {
                if constexpr (emit_ingredients_t) {
                    // ingredients: k (offset 0, full vector), v (offset S_v, this warp's column
                    // only), g (offset 2*S_v, raw/non-exponentiated), beta (offset 3*S_v). k/g/beta
                    // don't vary by column, so only one warp (col == 0) needs to write them --
                    // every warp doing so redundantly would cost O(S_v) writes each, i.e. O(S_v^2)
                    // total per component, which is what this design is supposed to avoid.
                    float * ingr_slot = state_out + (int64_t) target_slot * state_slot_stride;
                    if (col == 0) {
#pragma unroll
                        for (int r = 0; r < rows_per_lane; r++) {
                            const int i    = r * warp_size + lane;
                            ingr_slot[i]   = k_reg[r];
                            if constexpr (KDA) {
                                ingr_slot[2 * S_v + i] = g_t[i];
                            } else {
                                ingr_slot[2 * S_v + i] = *g_t;
                            }
                            ingr_slot[3 * S_v + i] = beta_val;
                        }
                    }
                    if (lane == 0) {
                        ingr_slot[S_v + col] = v_t[col];
                    }
                } else {
                    float * curr_state = state_out + (int64_t) target_slot * state_slot_stride;
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        const int i = r * warp_size + lane;
                        curr_state[col * S_v + i] = s_shard[r];
                    }
                }
            }

            if constexpr (emit_ingredients_t) {
                // state immediately before the K-token retained window starts, captured inline
                // (free relative to a second op call over the same prefix) when n_tokens > K.
                const int t_ckpt = (int) n_tokens - K - 1;
                if (t_ckpt >= 0 && t == t_ckpt) {
                    float * ckpt_slot = state + (int64_t) K * state_slot_stride +
                                         S_v * S_v * H * n_seqs + (sequence * H + h_idx) * S_v * S_v;
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        const int i = r * warp_size + lane;
                        ckpt_slot[col * S_v + i] = s_shard[r];
                    }
                }
            }
        }
    }

    if constexpr (emit_ingredients_t) {
        // fixed, once-per-call cost (not scaled by K): the true final state, for use as the next
        // decode's optimistic base state / this decode's own checkpoint update.
        float * final_slot = state + (int64_t) K * state_slot_stride + (sequence * H + h_idx) * S_v * S_v;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            final_slot[col * S_v + i] = s_shard[r];
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i              = r * warp_size + lane;
            state_out[col * S_v + i] = s_shard[r];
        }
    }
}

// Row-per-warp variant (port of ggml-org PR #22587 onto this fork's kernel contract).
// Each warp owns GDN_ROWS_D_V_PER_WARP rows of the transposed state (= output columns) and its
// lanes shard the QK axis in contiguous chunks, so q/k are loaded once per warp for several
// columns (vectorized) instead of once per column. Output/state semantics are identical to
// gated_delta_net_cuda above, including keep_rs_t snapshots, emit_ingredients_t ingredient slots,
// the pre-window checkpoint and the trailing final-state block.
constexpr int GDN_ROWS_D_V_PER_WARP = 4;
constexpr int GDN_ROWS_NUM_WARPS    = 4;
constexpr int GDN_ROWS_BLOCK_DV     = GDN_ROWS_NUM_WARPS * GDN_ROWS_D_V_PER_WARP;  // 16

template <int S_v, bool KDA, bool keep_rs_t, bool emit_ingredients_t>
__global__ void __launch_bounds__(ggml_cuda_get_physical_warp_size() * GDN_ROWS_NUM_WARPS, 2)
gated_delta_net_rows_cuda(const float * __restrict__ q,
                          const float * __restrict__ k,
                          const float * __restrict__ v,
                          const float * __restrict__ g,
                          const float * __restrict__ beta,
                          const float * __restrict__ curr_state,
                          float *       __restrict__ dst,
                          float *                    state,
                          int64_t       H,
                          int64_t       n_tokens,
                          int64_t       n_seqs,
                          int64_t       sq1,
                          int64_t       sq2,
                          int64_t       sq3,
                          int64_t       sv1,
                          int64_t       sv2,
                          int64_t       sv3,
                          int64_t       sb1,
                          int64_t       sb2,
                          int64_t       sb3,
                          const uint3   neqk1_magic,
                          const uint3   rq3_magic,
                          float         scale,
                          int64_t       state_slot_stride,
                          int           K) {
    constexpr int warp_size     = ggml_cuda_get_physical_warp_size();
    constexpr int active_lanes  = (S_v < warp_size) ? S_v : warp_size;
    constexpr int d_qk_per_lane = S_v / active_lanes;
    constexpr int d_v_per_warp  = GDN_ROWS_D_V_PER_WARP;
    static_assert(d_qk_per_lane >= 1, "S_v must be >= 1");
    static_assert(S_v % active_lanes == 0, "S_v must be a multiple of active_lanes");
    static_assert(S_v % GDN_ROWS_BLOCK_DV == 0, "S_v must be a multiple of GDN_ROWS_BLOCK_DV");

    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      lane     = threadIdx.x;
    const int      warp_id  = threadIdx.y;

    // this warp's state rows (= output columns) and this lane's QK chunk
    const uint32_t dv_base  = blockIdx.z * GDN_ROWS_BLOCK_DV + warp_id * d_v_per_warp;
    const uint32_t dqk_base = lane * d_qk_per_lane;
    // the warp owning column 0 writes the column-independent ingredients (k, g, beta)
    const bool     col0_warp = dv_base == 0;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float * attn_data = dst + ((int64_t) sequence * n_tokens * H + h_idx) * S_v;
    curr_state += (int64_t) sequence * H * S_v * S_v + (int64_t) h_idx * S_v * S_v;

    float * state_out;
    if constexpr (emit_ingredients_t) {
        state_out = state + ((int64_t) sequence * H + h_idx) * (4 * S_v);
    } else {
        state_out = state + ((int64_t) sequence * H + h_idx) * S_v * S_v;
    }

    auto load_qk_lane = [&] __device__(float(&reg)[d_qk_per_lane], const float * base) {
        if constexpr (S_v < warp_size) {
            reg[0] = (lane < active_lanes) ? base[lane] : 0.0f;
        } else {
            ggml_cuda_memcpy_1<d_qk_per_lane * sizeof(float)>(reg, base + dqk_base);
        }
    };
    auto store_qk_lane = [&] __device__(const float(&reg)[d_qk_per_lane], float * base) {
        if constexpr (S_v < warp_size) {
            if (lane < active_lanes) {
                base[lane] = reg[0];
            }
        } else {
            ggml_cuda_memcpy_1<d_qk_per_lane * sizeof(float)>(base + dqk_base, reg);
        }
    };
    // full-state store of this warp's rows into a [S_v, S_v] slot (row r contiguous)
    auto store_rows = [&] __device__(const float (&tile)[d_v_per_warp][d_qk_per_lane], float * slot) {
#pragma unroll
        for (int r = 0; r < d_v_per_warp; ++r) {
            store_qk_lane(tile[r], slot + (int64_t) (dv_base + r) * S_v);
        }
    };

    ggml_cuda_pdl_sync();

    // state is stored transposed: M[row][c] = S[c][row], row contiguous
    __align__(16) float s_tile[d_v_per_warp][d_qk_per_lane];
#pragma unroll
    for (int r = 0; r < d_v_per_warp; ++r) {
        load_qk_lane(s_tile[r], curr_state + (int64_t) (dv_base + r) * S_v);
    }

    __align__(16) float k_reg[d_qk_per_lane];
    load_qk_lane(k_reg, k + (int64_t) iq3 * sq3 + (int64_t) iq1 * sq1);

    for (int t = 0; t < n_tokens; ++t) {
        const float * v_t = v + (int64_t) sequence * sv3 + (int64_t) t * sv2 + (int64_t) h_idx * sv1;

        const int64_t gb_off   = (int64_t) sequence * sb3 + (int64_t) t * sb2 + (int64_t) h_idx * sb1;
        const float   beta_val = beta[gb_off];

        __align__(16) float g_reg[d_qk_per_lane];  // raw g (KDA), kept for the ingredient slot
        __align__(16) float alpha_lane[d_qk_per_lane];
        float               alpha_scalar = 0.0f;
        if constexpr (KDA) {
            load_qk_lane(g_reg, g + gb_off * S_v);
#pragma unroll
            for (int c = 0; c < d_qk_per_lane; ++c) {
                alpha_lane[c] = expf(g_reg[c]);
            }
        } else {
            alpha_scalar = expf(g[gb_off]);
        }

        // only the first d_v_per_warp lanes hold a real v[dv_base + lane]; broadcast via shuffle
        float v_local = 0.0f;
        if (lane < d_v_per_warp) {
            v_local = v_t[dv_base + lane];
        }

        // stage A: state update S = g*S + k*delta, per owned row
#pragma unroll
        for (int r = 0; r < d_v_per_warp; ++r) {
            float partial = 0.0f;
#pragma unroll
            for (int c = 0; c < d_qk_per_lane; ++c) {
                if constexpr (KDA) {
                    partial += alpha_lane[c] * s_tile[r][c] * k_reg[c];
                } else {
                    partial += s_tile[r][c] * k_reg[c];
                }
            }
            partial = warp_reduce_sum<warp_size>(partial);

            const float v_r   = __shfl_sync(0xffffffff, v_local, r, warp_size);
            const float delta = beta_val * (v_r - (KDA ? 1.0f : alpha_scalar) * partial);

#pragma unroll
            for (int c = 0; c < d_qk_per_lane; ++c) {
                if constexpr (KDA) {
                    s_tile[r][c] = alpha_lane[c] * s_tile[r][c] + delta * k_reg[c];
                } else {
                    s_tile[r][c] = alpha_scalar * s_tile[r][c] + delta * k_reg[c];
                }
            }
        }

        // retained-state outputs for token t (post-update state, k_reg still holds k_t)
        if constexpr (keep_rs_t) {
            const int target_slot = emit_ingredients_t ? ((int) t - ((int) n_tokens - K)) : ((int) n_tokens - 1 - t);
            if (target_slot >= 0 && target_slot < K) {
                if constexpr (emit_ingredients_t) {
                    float * ingr_slot = state_out + (int64_t) target_slot * state_slot_stride;
                    if (col0_warp) {
                        store_qk_lane(k_reg, ingr_slot);
                        if constexpr (KDA) {
                            store_qk_lane(g_reg, ingr_slot + 2 * S_v);
                        } else {
                            __align__(16) float g_raw[d_qk_per_lane];
                            __align__(16) float b_rep[d_qk_per_lane];
#pragma unroll
                            for (int c = 0; c < d_qk_per_lane; ++c) {
                                g_raw[c] = g[gb_off];
                                b_rep[c] = beta_val;
                            }
                            store_qk_lane(g_raw, ingr_slot + 2 * S_v);
                            store_qk_lane(b_rep, ingr_slot + 3 * S_v);
                        }
                        if constexpr (KDA) {
                            __align__(16) float b_rep[d_qk_per_lane];
#pragma unroll
                            for (int c = 0; c < d_qk_per_lane; ++c) {
                                b_rep[c] = beta_val;
                            }
                            store_qk_lane(b_rep, ingr_slot + 3 * S_v);
                        }
                    }
                    if (lane < d_v_per_warp) {
                        ingr_slot[S_v + dv_base + lane] = v_local;
                    }
                } else {
                    store_rows(s_tile, state_out + (int64_t) target_slot * state_slot_stride);
                }
            }

            if constexpr (emit_ingredients_t) {
                // state immediately before the K-token retained window starts
                const int t_ckpt = (int) n_tokens - K - 1;
                if (t_ckpt >= 0 && t == t_ckpt) {
                    float * ckpt_slot = state + (int64_t) K * state_slot_stride +
                                        (int64_t) S_v * S_v * H * n_seqs + ((int64_t) sequence * H + h_idx) * S_v * S_v;
                    store_rows(s_tile, ckpt_slot);
                }
            }
        }

        // prefetch k for the next token while issuing the q load for this token
        if (t + 1 < n_tokens) {
            load_qk_lane(k_reg, k + (int64_t) iq3 * sq3 + (int64_t) (t + 1) * sq2 + (int64_t) iq1 * sq1);
        }

        __align__(16) float q_reg[d_qk_per_lane];
        load_qk_lane(q_reg, q + (int64_t) iq3 * sq3 + (int64_t) t * sq2 + (int64_t) iq1 * sq1);

        // stage B: attention output attn[row] = (S^T q)[row]
        float attn_val = 0.0f;
#pragma unroll
        for (int r = 0; r < d_v_per_warp; ++r) {
            float partial = 0.0f;
#pragma unroll
            for (int c = 0; c < d_qk_per_lane; ++c) {
                partial += s_tile[r][c] * q_reg[c];
            }
            partial = warp_reduce_sum<warp_size>(partial);
            if (lane == r) {
                attn_val = partial;
            }
        }

        if (lane < d_v_per_warp) {
            attn_data[(int64_t) t * S_v * H + dv_base + lane] = attn_val * scale;
        }
    }

    if constexpr (emit_ingredients_t) {
        store_rows(s_tile, state + (int64_t) K * state_slot_stride + ((int64_t) sequence * H + h_idx) * S_v * S_v);
    }
    if constexpr (!keep_rs_t) {
        store_rows(s_tile, state_out);
    }
}

template <bool KDA, bool keep_rs_t, bool emit_ingredients_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    static const bool gdn_legacy = getenv("GGML_CUDA_GDN_LEGACY") != nullptr;
    if (!gdn_legacy && S_v % GDN_ROWS_BLOCK_DV == 0) {
        const dim3 rows_grid(H, n_seqs, (S_v + GDN_ROWS_BLOCK_DV - 1) / GDN_ROWS_BLOCK_DV);
        const dim3 rows_block(warp_size, GDN_ROWS_NUM_WARPS, 1);
        const ggml_cuda_kernel_launch_params rows_params = ggml_cuda_kernel_launch_params(rows_grid, rows_block, 0, stream);
#define GDN_ROWS_CASE(SV) case SV: ggml_cuda_kernel_launch(gated_delta_net_rows_cuda<SV, KDA, keep_rs_t, emit_ingredients_t>, rows_params, \
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, \
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K); return;
        switch (S_v) {
            GDN_ROWS_CASE(16)
            GDN_ROWS_CASE(32)
            GDN_ROWS_CASE(64)
            GDN_ROWS_CASE(128)
            default: break;
        }
#undef GDN_ROWS_CASE
    }

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t, emit_ingredients_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t, emit_ingredients_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t, emit_ingredients_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t, emit_ingredients_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) and emit_mode (0=full snapshots, 1=ingredients) are op params;
    // state holds s0 only [S_v, S_v, H, n_seqs].
    const int  K         = ggml_get_op_params_i32(dst, 0);
    const int  emit_mode = ggml_get_op_params_i32(dst, 1);
    const bool keep_rs   = (K > 1) || (emit_mode != 0);
    const bool emit_ingr = (emit_mode == 1);

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing.
    // emit_ingr slots are 4*S_v wide (k,v,g,beta) instead of S_v*S_v (a full state matrix);
    // the fusion path (external cache) never engages for emit_mode==1 (see
    // ggml_cuda_try_gdn_cache_fusion's explicit guard), so cache->slot_stride is always the
    // emit_mode==0 layout when cache != nullptr.
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = emit_ingr ? (4 * S_v * H * n_seqs) : (S_v * S_v * H * n_seqs);
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    if (kda) {
        if (emit_ingr) {
            launch_gated_delta_net<true, true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else if (keep_rs) {
            launch_gated_delta_net<true, true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<true, false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } else {
        if (emit_ingr) {
            launch_gated_delta_net<false, true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else if (keep_rs) {
            launch_gated_delta_net<false, true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<false, false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}

#include "common.h"
#include "dequantize.h"
#include "tq.h"
#include "turbo.h"

// ===== TurboQuant Walsh-Hadamard Transform kernel =====
// O(d log d) rotation for 128-element groups. Replaces dense 128x128 matmul.
// Each thread processes one 128-element group using half4 vectorized butterfly.
// Uses the same WHT signs already defined (turbo_wht_signs1/2, turbo_wht_signs1_h4/2_h4).

kernel void kernel_turbo_wht(
        constant ggml_metal_kargs_turbo_wht & args,
        device const float * src [[buffer(1)]],
        device float * dst [[buffer(2)]],
        uint tgpig [[threadgroup_position_in_grid]],
        uint tiitg [[thread_index_in_threadgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    // Each thread handles one 128-element group
    const int64_t group_idx = tgpig * ntg + tiitg;
    const int64_t n_groups = args.n_elements / 128;
    if (group_idx >= n_groups) return;

    const device float * in = src + group_idx * 128;
    device float * out = dst + group_idx * 128;

    // Load into half4 vectors for fast butterfly
    half4 v[32];
    const bool is_inverse = (args.direction == 1);

    // Apply first signs (s1 for fwd, s2 for inv)
    for (int i = 0; i < 32; i++) {
        float4 f = float4(in[i*4], in[i*4+1], in[i*4+2], in[i*4+3]);
        half4 s = is_inverse ? turbo_wht_signs2_h4[i] : turbo_wht_signs1_h4[i];
        v[i] = half4(f) * s;
    }

    // WHT butterfly (7 stages, vectorized half4)
    // h=1: within each half4
    for (int i = 0; i < 32; i++) {
        half4 a = v[i];
        v[i] = half4(a.x + a.y, a.x - a.y, a.z + a.w, a.z - a.w);
    }
    // h=2: within each half4
    for (int i = 0; i < 32; i++) {
        half4 a = v[i];
        v[i] = half4(a.x + a.z, a.y + a.w, a.x - a.z, a.y - a.w);
    }
    // h=4..64: between half4 vectors
    for (int h = 4; h < 128; h *= 2) {
        int vec_stride = h / 4;
        for (int i = 0; i < 32; i++) {
            int group_pos = i % (2 * vec_stride);
            if (group_pos < vec_stride) {
                int partner = i + vec_stride;
                half4 a = v[i], b = v[partner];
                v[i] = a + b;
                v[partner] = a - b;
            }
        }
    }

    // Apply second signs + normalize, write output as fp32
    const half4 inv_sqrt = half4(0.08838834764831845h);
    for (int i = 0; i < 32; i++) {
        half4 s = is_inverse ? turbo_wht_signs1_h4[i] : turbo_wht_signs2_h4[i];
        float4 f = float4(v[i] * inv_sqrt * s);
        out[i*4] = f.x;
        out[i*4+1] = f.y;
        out[i*4+2] = f.z;
        out[i*4+3] = f.w;
    }
}

// Pass 2: 1 threadgroup merges T/64 partial results (trivially fast)
//
// Benchmarked B=32, B=64, B=128 - B=64 is optimal for Apple Silicon.

// V4 architecture: 32 threads (1 SIMD group) per block, tokens processed in a loop.
// Tokens per block stays 64 for pass 2 compatibility.
#define TURBO_FLASH_BLOCK_SIZE 64
#define TURBO_FLASH_TG_SIZE 32

// Function constants for Pass 1
constant int32_t FC_turbo_flash_p1_dk [[function_constant(FC_TURBO_FLASH_P1 + 0)]];
constant int32_t FC_turbo_flash_p1_dv [[function_constant(FC_TURBO_FLASH_P1 + 1)]];
constant bool FC_turbo_flash_p1_has_mask [[function_constant(FC_TURBO_FLASH_P1 + 2)]];
constant bool FC_turbo_flash_p1_k_is_turbo3 [[function_constant(FC_TURBO_FLASH_P1 + 3)]];

// Function constants for Pass 2
constant int32_t FC_turbo_flash_p2_dv [[function_constant(FC_TURBO_FLASH_P2 + 0)]];

// ---------------------------------------------------------------------------
// Pass 1 V4: SIMD-per-token architecture (Eric Kryski pattern)
// ---------------------------------------------------------------------------
//
// Grid: (n_bh, n_blocks, 1)
// n_bh = ne01 * ne02 * ne03 (total query heads across batch)
// n_blocks = ceil(T_kv / TURBO_FLASH_BLOCK_SIZE)
//
// Threadgroup: (32, 1, 1) - exactly 1 SIMD group
//
// Each SIMD group of 32 threads processes one block of 64 KV tokens for one
// query head. Each lane handles DIMS_PER_LANE = DK/32 interleaved dimensions
// (e.g., lane 0 -> dims 0,32,64,96 for DK=128).
//
// Key differences from V3:
// - NO shared memory - Q, codebook, softmax state, V accum all in registers
// - K scoring via per-lane partial dot product + simd_sum reduction
// - Online softmax across tokens (not across threads within a token)
// - V accumulation fused with softmax in the token loop
//
// Buffer layout (unchanged from V3 for pass 2 compatibility):
// partial_out: float[n_bh * n_blocks * dv] - per-block weighted V accumulators
// partial_ms: float[n_bh * n_blocks * 2] - per-block {max_score, exp_sum}
//
template <short DK, short DV>
kernel void kernel_turbo_flash_p1(
        constant ggml_metal_kargs_turbo_flash_p1 & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device float * partial_out,
        device float * partial_ms,
        threadgroup float * shmem [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]]) {

    constexpr short DK_PER_LANE = DK / 32;
    constexpr short DV_PER_LANE = DV / 32;

    const uint lane = tiitg % 32; // SIMD lane (0-31)
    const uint bh_idx = tgpig[0];
    const uint block_id = tgpig[1];

    const int T_kv = args.ne11;
    const int n_blocks = args.n_blocks;

    // Token range for this block
    const int t_start = (int)(block_id * TURBO_FLASH_BLOCK_SIZE);
    const int t_end = min(t_start + TURBO_FLASH_BLOCK_SIZE, T_kv);

    // Decompose bh_idx back to (iq1, iq2, iq3) for pointer arithmetic
    const uint iq1 = bh_idx % args.ne01;
    const uint iq2 = (bh_idx / args.ne01) % args.ne02;
    const uint iq3 = bh_idx / (args.ne01 * args.ne02);

    // GQA: map query head to KV head
    const uint ikv2 = iq2 / (args.ne02 / args.ne_12_2);
    const uint ikv3 = iq3 / (args.ne03 / args.ne_12_3);

    // ====== Load Q into registers ======
    // Each lane loads its interleaved dims: lane, lane+32, lane+64, ...
    device const float * q_ptr = (device const float *)((device const char *)q + iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03);
    float q_vals[DK_PER_LANE];
    for (short i = 0; i < DK_PER_LANE; i++) {
        const int d = (int)lane + i * 32;
        q_vals[i] = (d < DK) ? q_ptr[d] : 0.0f;
    }

    // ====== Load turbo3 codebook into registers (8 entries) ======
    float v_cb[8];
    for (int i = 0; i < 8; i++) {
        v_cb[i] = float(turbo_centroids_3bit_h[i]);
    }

    // K codebook - same centroids, only loaded when K is turbo3
    float k_cb[8];
    if (FC_turbo_flash_p1_k_is_turbo3) {
        for (int i = 0; i < 8; i++) {
            k_cb[i] = float(turbo_centroids_3bit_h[i]);
        }
    }

    // ====== Online softmax state - all in registers ======
    float m_state = -INFINITY;
    float l_state = 0.0f;
    float o_state[DV_PER_LANE];
    for (short i = 0; i < DV_PER_LANE; i++) {
        o_state[i] = 0.0f;
    }

    // ====== K/V base pointers for this KV head ======
    device const char * k_base = (device const char *)k + ikv2*args.nb12 + ikv3*args.nb13;
    device const char * v_base = (device const char *)v + ikv2*args.nb22 + ikv3*args.nb23;

    // Mask pointer (precompute once)
    device const half * mask_ptr = nullptr;
    if (FC_turbo_flash_p1_has_mask) {
        mask_ptr = (device const half *)(mask + iq1*args.nb31 + (iq2 % args.ne32)*args.nb32 + (iq3 % args.ne33)*args.nb33);
    }

    // ====== Process all tokens in this block ======
    for (int t = t_start; t < t_end; t++) {

        // --- Check mask ---
        float mask_val = 0.0f;
        if (FC_turbo_flash_p1_has_mask) {
            mask_val = (float)mask_ptr[t];
            if (mask_val <= -MAXHALF) {
                continue; // masked out - skip entirely
            }
        }

        // --- Dequant K and compute Q*K score ---
        // Each lane computes partial dot for its interleaved dims, then simd_sum.
        float dot_partial = 0.0f;

        if (FC_turbo_flash_p1_k_is_turbo3) {
            // K is turbo3_0: same struct as V - norm, qs[], signs[]
            device const block_turbo3_0 * k_row = (device const block_turbo3_0 *)(k_base + t * args.nb11);
            const float k_norm = float(k_row[0].norm);

            for (short i = 0; i < DK_PER_LANE; i++) {
                const int d = (int)lane + i * 32;
                if (d >= DK) break;

                const int qs_byte = d / 4;
                const int qs_shift = (d % 4) * 2;
                const uint8_t q_idx = (k_row[0].qs[qs_byte] >> qs_shift) & 0x03;

                const int sign_byte = d / 8;
                const int sign_bit = d % 8;
                const uint8_t s_bit = (k_row[0].signs[sign_byte] >> sign_bit) & 1;

                const uint8_t centroid_idx = q_idx | (s_bit << 2);
                dot_partial += q_vals[i] * k_cb[centroid_idx] * k_norm;
            }
        } else {
            // K is q8_0: 32 elements per block, DK/32 blocks per row.
            device const block_q8_0 * k_row = (device const block_q8_0 *)(k_base + t * args.nb11);

            for (short i = 0; i < DK_PER_LANE; i++) {
                const int d = (int)lane + i * 32;
                if (d >= DK) break;

                // Which q8_0 block and offset within it
                const int qb = d / 32; // block index
                const int qj = d % 32; // element within block
                dot_partial += q_vals[i] * (float)k_row[qb].qs[qj] * (float)k_row[qb].d;
            }
        }
        float score = simd_sum(dot_partial) * args.scale + mask_val;

        // --- Dequant V for this token ---
        // V is turbo3: one block_turbo3_0 per 128 dims (QK_TURBO3=128).
        // Each lane unpacks its interleaved dims.
        device const block_turbo3_0 * v_row = (device const block_turbo3_0 *)(v_base + t * args.nb21);
        const float v_norm = float(v_row[0].norm);

        float v_decoded[DV_PER_LANE];
        for (short i = 0; i < DV_PER_LANE; i++) {
            const int d = (int)lane + i * 32;
            if (d >= DV) { v_decoded[i] = 0.0f; continue; }

            const int qs_byte = d / 4;
            const int qs_shift = (d % 4) * 2;
            const uint8_t q_idx = (v_row[0].qs[qs_byte] >> qs_shift) & 0x03;

            const int sign_byte = d / 8;
            const int sign_bit = d % 8;
            const uint8_t s_bit = (v_row[0].signs[sign_byte] >> sign_bit) & 1;

            const uint8_t centroid_idx = q_idx | (s_bit << 2);
            v_decoded[i] = v_cb[centroid_idx] * v_norm;
        }

        // --- Online softmax update + V accumulation ---
        float new_m = max(m_state, score);
        float exp_diff = exp(m_state - new_m);
        float exp_score = exp(score - new_m);

        for (short i = 0; i < DV_PER_LANE; i++) {
            o_state[i] = o_state[i] * exp_diff + exp_score * v_decoded[i];
        }

        l_state = l_state * exp_diff + exp_score;
        m_state = new_m;
    }

    // ====== Write partial results ======
    // Each lane writes its interleaved dims to partial_out (scatter write, no reduction)
    for (short i = 0; i < DV_PER_LANE; i++) {
        const int d = (int)lane + i * 32;
        if (d < DV) {
            partial_out[bh_idx * (uint64_t)n_blocks * DV + block_id * DV + d] = o_state[i];
        }
    }

    // Lane 0 writes block max and sum
    if (lane == 0) {
        partial_ms[bh_idx * (uint64_t)n_blocks * 2 + block_id * 2 + 0] = m_state;
        partial_ms[bh_idx * (uint64_t)n_blocks * 2 + block_id * 2 + 1] = l_state;
    }
}

// ---------------------------------------------------------------------------
// Pass 2: Merge block partial results + inverse WHT rotation
// ---------------------------------------------------------------------------
//
// Grid: (n_bh, 1, 1)
// Threadgroup: (tg_size, 1, 1) where tg_size >= DV (for WHT butterfly)
//
// Merges partial results from Pass 1 across all blocks using online softmax
// correction, then applies inverse WHT rotation (signs2 -> FWHT -> signs1).
//
template <short DV>
kernel void kernel_turbo_flash_p2(
        constant ggml_metal_kargs_turbo_flash_p2 & args,
        device const float * partial_out,
        device const float * partial_ms,
        device float * dst,
        threadgroup float * shmem [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]]) {

    const uint tid = tiitg;
    const uint bh_idx = tgpig[0];
    const int n_blocks = args.n_blocks;

    // Shared memory for merge + WHT
    threadgroup float * shared_out = shmem;

    // ====== Step 1: Find global max across all blocks ======
    // Single thread scans - n_blocks is typically small (T/64)
    threadgroup float * shared_global_max = shmem + DV;
    threadgroup float * shared_global_sum = shmem + DV + 1;

    if (tid == 0) {
        float gmax = -INFINITY;
        for (int b = 0; b < n_blocks; b++) {
            gmax = max(gmax, partial_ms[bh_idx * (uint64_t)n_blocks * 2 + b * 2 + 0]);
        }
        shared_global_max[0] = gmax;

        // Compute global exp sum with correction
        float gsum = 0.0f;
        for (int b = 0; b < n_blocks; b++) {
            float bmax = partial_ms[bh_idx * (uint64_t)n_blocks * 2 + b * 2 + 0];
            float bsum = partial_ms[bh_idx * (uint64_t)n_blocks * 2 + b * 2 + 1];
            float correction = exp(bmax - gmax);
            gsum += correction * bsum;
        }
        shared_global_sum[0] = gsum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_max = shared_global_max[0];
    float global_sum = shared_global_sum[0];
    float inv_global_sum = (global_sum > 0.0f) ? 1.0f / global_sum : 0.0f;

    // ====== Step 2: Merge partial outputs with softmax correction ======
    if ((int)tid < DV) {
        float accum = 0.0f;
        for (int b = 0; b < n_blocks; b++) {
            float bmax = partial_ms[bh_idx * (uint64_t)n_blocks * 2 + b * 2 + 0];
            float correction = exp(bmax - global_max);
            float block_val = partial_out[bh_idx * (uint64_t)n_blocks * DV + b * DV + tid];
            accum += correction * block_val;
        }
        // Normalize by global softmax sum
        shared_out[tid] = accum * inv_global_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ====== Step 3: Inverse WHT rotation (signs2 -> FWHT -> signs1) ======
    // The turbo3 V cache stores values in WHT-rotated space.
    // We need to apply the inverse rotation to get back to the original space.
    // Inverse WHT: multiply by signs2, apply FWHT butterfly, multiply by signs1, scale by 1/sqrt(dim).

    // Step 3a: Apply signs2
    if ((int)tid < DV && (int)tid < 128) {
        shared_out[tid] *= turbo_wht_signs2[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3b: FWHT butterfly - simd_shuffle_xor for first 5 stages (zero barriers)
    // then shared memory for remaining stages (stages 5-6 for dim=128)
    // Ported from Eric Kryski's V2.1 optimization.
    if ((int)tid < DV && (int)tid < 128) {
        const int dim_wht = min((int)DV, 128);
        float val = shared_out[tid];

        // Phase 1: Intra-SIMD butterfly via simd_shuffle_xor (stages 0..4)
        // Register-to-register, no shared memory or barriers needed
        const uint lane_in_simd = tid % 32;
        // log2(dim_wht) for power-of-2 dims: 128->7, 64->6, 32->5
        const int log2_dim = (dim_wht >= 128) ? 7 : (dim_wht >= 64) ? 6 : 5;
        const int simd_stages = min(5, log2_dim);
        for (int s = 0; s < simd_stages; s++) {
            const uint step = 1u << s;
            float other = simd_shuffle_xor(val, (ushort)step);
            val = (lane_in_simd & step) ? (other - val) : (other + val);
        }

        // Phase 2: Cross-SIMD butterfly via shared memory (stages 5+)
        // Only needed for dim > 32
        if (dim_wht > 32) {
            shared_out[tid] = val;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (int half_block = 32; half_block < dim_wht; half_block <<= 1) {
                const int bfly_size = half_block << 1;
                const int bfly_idx = (int)tid / bfly_size;
                const int local_idx = (int)tid % bfly_size;
                const int base_idx = bfly_idx * bfly_size;

                float a = shared_out[base_idx + (local_idx % half_block)];
                float b_val_s = shared_out[base_idx + (local_idx % half_block) + half_block];
                threadgroup_barrier(mem_flags::mem_threadgroup);

                shared_out[tid] = (local_idx < half_block) ? (a + b_val_s) : (a - b_val_s);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            val = shared_out[tid];
        }

        // Step 3c: Apply signs1 and normalize
        float inv_sqrt_dim = rsqrt((float)dim_wht);
        dst[bh_idx * DV + tid] = val * inv_sqrt_dim * turbo_wht_signs1[tid];
    }

    // For DV > 128: elements beyond the rotation group are written directly (no WHT)
    // TODO: support DV > 128 with multiple rotation groups if needed
}

// Template instantiations for TurboFlash Pass 1
// DK = head dim for K (q8_0), DV = head dim for V (turbo3)
// Common asymmetric config: DK == DV for most models
typedef decltype(kernel_turbo_flash_p1<128, 128>) turbo_flash_p1_t;

template [[host_name("kernel_turbo_flash_p1_dk64_dv64")]] kernel turbo_flash_p1_t kernel_turbo_flash_p1<64, 64>;
template [[host_name("kernel_turbo_flash_p1_dk96_dv96")]] kernel turbo_flash_p1_t kernel_turbo_flash_p1<96, 96>;
template [[host_name("kernel_turbo_flash_p1_dk128_dv128")]] kernel turbo_flash_p1_t kernel_turbo_flash_p1<128, 128>;

// Template instantiations for TurboFlash Pass 2
typedef decltype(kernel_turbo_flash_p2<128>) turbo_flash_p2_t;

template [[host_name("kernel_turbo_flash_p2_dv64")]] kernel turbo_flash_p2_t kernel_turbo_flash_p2<64>;
template [[host_name("kernel_turbo_flash_p2_dv96")]] kernel turbo_flash_p2_t kernel_turbo_flash_p2<96>;
template [[host_name("kernel_turbo_flash_p2_dv128")]] kernel turbo_flash_p2_t kernel_turbo_flash_p2<128>;


// TurboQuant set_rows kernel - block size 128 (QK_TURBO3/QK_TURBO4)
// TurboQuant SET_ROWS kernel - processes QK_TURBO3_GROUP (128) elements per iteration,
// writes QK_TURBO3_GROUP/QK_TURBO3 (4) blocks per iteration.
// The rotation operates on 128 elements, then results are split into 32-element blocks.
template<typename TI, typename block_q, int QK, void (*quantize_func)(device const float *, device block_q &)>
kernel void kernel_set_rows_turbo(
        constant ggml_metal_kargs_set_rows & args,
        device const void * src0,
        device const void * src1,
        device float * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint tiitg[[thread_index_in_threadgroup]],
        uint3 tptg [[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;
    const int32_t i12 = i03%args.ne12;
    const int32_t i11 = i02%args.ne11;
    const int32_t i01 = tgpig.x*tptg.y + tiitg/tptg.x;
    if (i01 >= args.ne01) return;

    const int32_t i10 = i01;
    const TI i1 = ((const device TI *) ((const device char *) src1 + i10*args.nb10 + i11*args.nb11 + i12*args.nb12))[0];

          device block_q * dst_row = ( device block_q *) (( device char *) dst + i1*args.nb1 + i02*args.nb2 + i03*args.nb3);
    const device float * src_row = (const device float *) ((const device char *) src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);

    // Process in groups of 4 blocks (128 elements) for rotation
    const int blocks_per_group = QK_TURBO3_GROUP / QK; // 128/32 = 4
    const int n_groups = args.nk0 / blocks_per_group;

    for (int grp = tiitg%tptg.x; grp < n_groups; grp += tptg.x) {
        const device float * grp_src = src_row + QK_TURBO3_GROUP * grp;

        // Normalize and rotate the full 128-element group
        float norm_sq = 0.0f;
        for (int j = 0; j < QK_TURBO3_GROUP; j++) norm_sq += grp_src[j] * grp_src[j];
        float grp_norm = sqrt(norm_sq);
        float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;

        float x[128];
        for (int j = 0; j < 128; j++) x[j] = grp_src[j] * inv_norm;
        turbo_rotate_forward(x, turbo_wht_signs1, turbo_wht_signs2);

        // Split into 4 blocks of 32 elements each
        // All blocks store the SAME group norm - centroids are in normalized space
        // Norm correction (ported from @spiritbuun's CUDA implementation):
        // Accumulate ||centroid_vector||^2 across all 128 elements, then store
        // grp_norm / ||centroid_vector|| instead of raw grp_norm. This makes
        // dequantized vectors have the exact original L2 norm at zero decode cost.
        float recon_norm_sq = 0.0f;

        for (int b = 0; b < blocks_per_group; b++) {
            device block_q & blk = dst_row[grp * blocks_per_group + b];
            const int off = b * QK;

            for (int j = 0; j < QK / 4; j++) blk.qs[j] = 0;
            for (int j = 0; j < QK / 8; j++) blk.signs[j] = 0;

            // Quantize rotated values to 3-bit centroids
            for (int j = 0; j < QK; j++) {
                float rv = x[off + j]; // rotated, normalized value
                uint8_t idx;
                if (rv < turbo_mid_3bit[0]) idx = 0;
                else if (rv < turbo_mid_3bit[1]) idx = 1;
                else if (rv < turbo_mid_3bit[2]) idx = 2;
                else if (rv < turbo_mid_3bit[3]) idx = 3;
                else if (rv < turbo_mid_3bit[4]) idx = 4;
                else if (rv < turbo_mid_3bit[5]) idx = 5;
                else if (rv < turbo_mid_3bit[6]) idx = 6;
                else idx = 7;

                blk.qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
                if (idx & 0x4) blk.signs[j / 8] |= (1 << (j % 8));

                // Accumulate centroid reconstruction norm for norm correction
                float c = turbo_centroids_3bit[idx];
                recon_norm_sq += c * c;
            }
        }

        // Norm correction: store corrected norm so dequant(x) has exact original L2 norm.
        // Zero decode cost - dequant already multiplies by stored norm.
        float recon_norm = sqrt(recon_norm_sq);
        float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
        for (int b = 0; b < blocks_per_group; b++) {
            dst_row[grp * blocks_per_group + b].norm = half(corrected_norm);
        }
    }
}

// TurboQuant2 SET_ROWS kernel - 2-bit PolarQuant, 4 centroids, no signs byte.
// Same 128-element group WHT rotation as turbo3, but simpler quantization.
template<typename TI>
kernel void kernel_set_rows_turbo2(
        constant ggml_metal_kargs_set_rows & args,
        device const void * src0,
        device const void * src1,
        device float * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint tiitg[[thread_index_in_threadgroup]],
        uint3 tptg [[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;
    const int32_t i12 = i03%args.ne12;
    const int32_t i11 = i02%args.ne11;
    const int32_t i01 = tgpig.x*tptg.y + tiitg/tptg.x;
    if (i01 >= args.ne01) return;

    const int32_t i10 = i01;
    const TI i1 = ((const device TI *) ((const device char *) src1 + i10*args.nb10 + i11*args.nb11 + i12*args.nb12))[0];

          device block_turbo2_0 * dst_row = ( device block_turbo2_0 *) (( device char *) dst + i1*args.nb1 + i02*args.nb2 + i03*args.nb3);
    const device float * src_row = (const device float *) ((const device char *) src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);

    // Process in groups of 4 blocks (128 elements) for rotation
    const int blocks_per_group = QK_TURBO2_GROUP / QK_TURBO2; // 128/32 = 4
    const int n_groups = args.nk0 / blocks_per_group;

    for (int grp = tiitg%tptg.x; grp < n_groups; grp += tptg.x) {
        const device float * grp_src = src_row + QK_TURBO2_GROUP * grp;

        float norm_sq = 0.0f;
        for (int j = 0; j < QK_TURBO2_GROUP; j++) norm_sq += grp_src[j] * grp_src[j];
        float grp_norm = sqrt(norm_sq);
        float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;

        float x[128];
        for (int j = 0; j < 128; j++) x[j] = grp_src[j] * inv_norm;
        turbo_rotate_forward(x, turbo_wht_signs1, turbo_wht_signs2);

        float recon_norm_sq = 0.0f;

        for (int b = 0; b < blocks_per_group; b++) {
            device block_turbo2_0 & blk = dst_row[grp * blocks_per_group + b];
            const int off = b * QK_TURBO2;

            for (int j = 0; j < QK_TURBO2 / 4; j++) blk.qs[j] = 0;

            for (int j = 0; j < QK_TURBO2; j++) {
                float rv = x[off + j];
                uint8_t idx;
                if (rv < turbo_mid_2bit[0]) idx = 0;
                else if (rv < turbo_mid_2bit[1]) idx = 1;
                else if (rv < turbo_mid_2bit[2]) idx = 2;
                else idx = 3;

                blk.qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);

                float c = turbo_centroids_2bit[idx];
                recon_norm_sq += c * c;
            }
        }

        float recon_norm = sqrt(recon_norm_sq);
        float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
        for (int b = 0; b < blocks_per_group; b++) {
            dst_row[grp * blocks_per_group + b].norm = half(corrected_norm);
        }
    }
}

// TurboQuant4 SET_ROWS kernel - processes 128 elements per block.
// Unlike turbo3 (4x32 blocks), turbo4 uses single 128-element blocks
// with packed 3-bit indices and QJL signs.
template<typename TI>
kernel void kernel_set_rows_turbo4(
        constant ggml_metal_kargs_set_rows & args,
        device const void * src0,
        device const void * src1,
        device float * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint tiitg[[thread_index_in_threadgroup]],
        uint3 tptg [[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;
    const int32_t i12 = i03%args.ne12;
    const int32_t i11 = i02%args.ne11;
    const int32_t i01 = tgpig.x*tptg.y + tiitg/tptg.x;
    if (i01 >= args.ne01) return;

    const int32_t i10 = i01;
    const TI i1 = ((const device TI *) ((const device char *) src1 + i10*args.nb10 + i11*args.nb11 + i12*args.nb12))[0];

          device block_turbo4_0 * dst_row = ( device block_turbo4_0 *) (( device char *) dst + i1*args.nb1 + i02*args.nb2 + i03*args.nb3);
    const device float * src_row = (const device float *) ((const device char *) src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);

    // Each block is one 128-element group (nk0 = ne0 / QK_TURBO4)
    const int n_blocks = args.nk0;

    for (int blk_idx = tiitg%tptg.x; blk_idx < n_blocks; blk_idx += tptg.x) {
        const device float * blk_src = src_row + QK_TURBO4 * blk_idx;
        device block_turbo4_0 & blk = dst_row[blk_idx];

        // Step 1: Compute norm + normalize
        float norm_sq = 0.0f;
        for (int j = 0; j < QK_TURBO4; j++) norm_sq += blk_src[j] * blk_src[j];
        float grp_norm = sqrt(norm_sq);
        float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;

        float x[128];
        float normalized[128];
        for (int j = 0; j < 128; j++) {
            normalized[j] = blk_src[j] * inv_norm;
            x[j] = normalized[j];
        }

        // Step 2: WHT rotate in-place
        turbo_rotate_forward(x, turbo_wht_signs1, turbo_wht_signs2);

        // Step 3: 4-bit PolarQuant - nibble packing (2 indices per byte)
        for (int j = 0; j < QK_TURBO4 / 2; j++) blk.qs[j] = 0;
        // qs[64] covers full nibble range - no signs field in 4-bit struct

        float recon_norm_sq = 0.0f;
        for (int j = 0; j < 128; j++) {
            float val = x[j];
            uint8_t idx;
            if (val < turbo_mid_4bit[ 0]) idx = 0;
            else if (val < turbo_mid_4bit[ 1]) idx = 1;
            else if (val < turbo_mid_4bit[ 2]) idx = 2;
            else if (val < turbo_mid_4bit[ 3]) idx = 3;
            else if (val < turbo_mid_4bit[ 4]) idx = 4;
            else if (val < turbo_mid_4bit[ 5]) idx = 5;
            else if (val < turbo_mid_4bit[ 6]) idx = 6;
            else if (val < turbo_mid_4bit[ 7]) idx = 7;
            else if (val < turbo_mid_4bit[ 8]) idx = 8;
            else if (val < turbo_mid_4bit[ 9]) idx = 9;
            else if (val < turbo_mid_4bit[10]) idx = 10;
            else if (val < turbo_mid_4bit[11]) idx = 11;
            else if (val < turbo_mid_4bit[12]) idx = 12;
            else if (val < turbo_mid_4bit[13]) idx = 13;
            else if (val < turbo_mid_4bit[14]) idx = 14;
            else idx = 15;

            // 4-bit nibble pack: 2 elements per byte
            blk.qs[j / 2] |= (idx & 0xF) << ((j % 2) * 4);

            float c = turbo_centroids_4bit[idx];
            recon_norm_sq += c * c;
        }

        // Norm correction
        float recon_norm = sqrt(recon_norm_sq);
        blk.norm = half((recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm);
    }
}

// TurboQuant set_rows instantiations (128-element groups: 4x32-element blocks for turbo3, dedicated kernels for turbo2/4)
typedef decltype(kernel_set_rows_turbo<int64_t, block_turbo3_0, QK_TURBO3, quantize_turbo3_0>) set_rows_turbo3_t;
template [[host_name("kernel_set_rows_f32_i64_turbo3")]] kernel set_rows_turbo3_t kernel_set_rows_turbo<int64_t, block_turbo3_0, QK_TURBO3, quantize_turbo3_0>;
template [[host_name("kernel_set_rows_f32_i32_turbo3")]] kernel set_rows_turbo3_t kernel_set_rows_turbo<int32_t, block_turbo3_0, QK_TURBO3, quantize_turbo3_0>;

typedef decltype(kernel_set_rows_turbo2<int64_t>) set_rows_turbo2_t;
template [[host_name("kernel_set_rows_f32_i64_turbo2")]] kernel set_rows_turbo2_t kernel_set_rows_turbo2<int64_t>;
template [[host_name("kernel_set_rows_f32_i32_turbo2")]] kernel set_rows_turbo2_t kernel_set_rows_turbo2<int32_t>;

typedef decltype(kernel_set_rows_turbo4<int64_t>) set_rows_turbo4_t;
template [[host_name("kernel_set_rows_f32_i64_turbo4")]] kernel set_rows_turbo4_t kernel_set_rows_turbo4<int64_t>;
template [[host_name("kernel_set_rows_f32_i32_turbo4")]] kernel set_rows_turbo4_t kernel_set_rows_turbo4<int32_t>;

// ============================================================================
// TQ weight activation pre-rotation (32-element WHT) and ConvRot
// ============================================================================

// Shared pre-rotation kernels (used by both TQ3 and TQ4 for mul_mm path)
kernel void kernel_tq3_rotate_act(
        device float * x [[buffer(0)]],
        constant int64_t & n [[buffer(1)]],
        uint tpig [[thread_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const int64_t base = ((int64_t)tpig / 32) * 32;
    if (base >= n) return;

    float val = x[base + tiisg] * tq3_signs[tiisg];
    for (ushort step = 1; step < 32; step <<= 1) {
        float other = simd_shuffle_xor(val, step);
        val = (tiisg & step) ? (other - val) : (other + val);
    }
    x[base + tiisg] = val * tq3_inv_sqrt32;
}

kernel void kernel_tq3_unrotate_act(
        device float * x [[buffer(0)]],
        constant int64_t & n [[buffer(1)]],
        uint tpig [[thread_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const int64_t base = ((int64_t)tpig / 32) * 32;
    if (base >= n) return;

    float val = x[base + tiisg];
    for (ushort step = 1; step < 32; step <<= 1) {
        float other = simd_shuffle_xor(val, step);
        val = (tiisg & step) ? (other - val) : (other + val);
    }
    x[base + tiisg] = val * tq3_inv_sqrt32 * tq3_signs[tiisg];
}

constant float convrot_h4[4][4] = {
    { 1.0f,  1.0f,  1.0f, -1.0f},
    { 1.0f,  1.0f, -1.0f,  1.0f},
    { 1.0f, -1.0f,  1.0f,  1.0f},
    {-1.0f,  1.0f,  1.0f,  1.0f},
};

kernel void kernel_convrot(
        device const float * src [[buffer(0)]],
        device       float * dst [[buffer(1)]],
        uint group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float buf[2][QK8_CR];

    for (ushort i = tid; i < QK8_CR; i += 64) {
        buf[0][i] = src[group * QK8_CR + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ushort rd = 0;
    ushort wr = 1;
    for (ushort len = 4, shift = 0; len <= QK8_CR; len *= 4, shift += 2) {
        const ushort quarter = len / 4;
        for (ushort i = tid; i < QK8_CR; i += 64) {
            const ushort base = i & ~(len - 1);
            const ushort row = (i >> shift) & 3;
            const ushort col = i & (quarter - 1);

            float sum = 0.0f;
            for (ushort j = 0; j < 4; ++j) {
                sum += convrot_h4[row][j] * buf[rd][base + j * quarter + col];
            }
            buf[wr][i] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ushort tmp = rd;
        rd = wr;
        wr = tmp;
    }

    for (ushort i = tid; i < QK8_CR; i += 64) {
        dst[group * QK8_CR + i] = buf[rd][i] * (1.0f / 16.0f);
    }
}

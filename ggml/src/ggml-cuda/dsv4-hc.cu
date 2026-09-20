#include "common.cuh"
#include "dsv4-hc.cuh"


static constexpr int DSV4_HC = 4;
static constexpr int DSV4_HC_SQ = DSV4_HC * DSV4_HC; // 16
static constexpr int ELEMS_PER_MATRIX = DSV4_HC_SQ;
static constexpr int TOKENS_PER_WARP = 32 / ELEMS_PER_MATRIX; // 2
static constexpr int WARPS_PER_BLOCK = 4;
static constexpr int BLOCK_SIZE = WARPS_PER_BLOCK * 32; // 128
static constexpr int TOKENS_PER_BLOCK = WARPS_PER_BLOCK * TOKENS_PER_WARP; // 8
static constexpr unsigned MASK_FULL = 0xFFFFFFFF;

// Warp-level 4x4 sinkhorn normalization.
// 16 threads = one 4x4 matrix.  2 matrices per warp (lanes 0-15, 16-31).
// Each thread holds and operates on exactly one matrix element.
// Row reductions use XOR mask 1,2 and column reductions use XOR mask 4,8.
static __global__ void dsv4_hc_comb_f32(
        const float * mixes,
        const float * scale,
        const float * base,
        float * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        float eps,
        int32_t n_iter) {
    constexpr int comb_offset = 2 * DSV4_HC;
    constexpr int ROW_REDUCE_MASK_1 = 1;  // adjacent pairs in row
    constexpr int ROW_REDUCE_MASK_2 = 2;  // quartets = full row
    constexpr int COL_REDUCE_MASK_1 = 4;  // pairs across rows (stride 4)
    constexpr int COL_REDUCE_MASK_2 = 8;  // all rows (stride 4, then 8)

    const int warp_id  = threadIdx.x / 32;
    const int lane     = threadIdx.x % 32;
    const int token_off = lane / ELEMS_PER_MATRIX; // 0 or 1
    const int elem_id   = lane % ELEMS_PER_MATRIX; // 0..15
    const int row = elem_id / DSV4_HC;   // 0..3 (isrc in old kernel)
    const int col = elem_id % DSV4_HC;    // 0..3 (idst in old kernel)

    ggml_cuda_pdl_lc();

    for (int64_t it = (int64_t) blockIdx.x * TOKENS_PER_BLOCK + warp_id * TOKENS_PER_WARP + token_off;
         it < n_tokens;
         it += gridDim.x * TOKENS_PER_BLOCK) {

        const float scale_comb = scale[2 * ss0];
        const int mix_idx = comb_offset + elem_id;

        // Load this element = mixes[combined_offset + elem_id] * scale + base
        float v = mixes[mix_idx * sm0 + it * sm1] * scale_comb + base[mix_idx * sb0];

        // ---- Row softmax (over columns within each row) ----
        // max over the 4 columns of this row
        float max_v = v;
        max_v = fmaxf(max_v, __shfl_xor_sync(MASK_FULL, max_v, ROW_REDUCE_MASK_1));
        max_v = fmaxf(max_v, __shfl_xor_sync(MASK_FULL, max_v, ROW_REDUCE_MASK_2));

        // exp and sum over the 4 columns
        float ex = expf(v - max_v);
        float sum_ex = ex;
        sum_ex += __shfl_xor_sync(MASK_FULL, sum_ex, ROW_REDUCE_MASK_1);
        sum_ex += __shfl_xor_sync(MASK_FULL, sum_ex, ROW_REDUCE_MASK_2);

        // normalize: exp(v - max) / sum + eps
        float comb_val = ex * __frcp_rn(sum_ex) + eps;

        // ---- Column normalization (over rows within each column) ----
        float col_sum = comb_val;
        col_sum += __shfl_xor_sync(MASK_FULL, col_sum, COL_REDUCE_MASK_1);
        col_sum += __shfl_xor_sync(MASK_FULL, col_sum, COL_REDUCE_MASK_2);
        col_sum += eps; // match CPU: sum = eps + elements
        comb_val *= __frcp_rn(col_sum);

        // ---- Sinkhorn iterations ----
        // unroll=1 prevents compiler from massively unrolling the sinkhorn loop
        #pragma unroll 1
        for (int32_t i = 1; i < n_iter; ++i) {
            // Row norm
            float rsum = comb_val;
            rsum += __shfl_xor_sync(MASK_FULL, rsum, ROW_REDUCE_MASK_1);
            rsum += __shfl_xor_sync(MASK_FULL, rsum, ROW_REDUCE_MASK_2);
            rsum += eps; // match CPU: sum = eps + elements
            comb_val *= __frcp_rn(rsum);

            // Col norm
            float csum = comb_val;
            csum += __shfl_xor_sync(MASK_FULL, csum, COL_REDUCE_MASK_1);
            csum += __shfl_xor_sync(MASK_FULL, csum, COL_REDUCE_MASK_2);
            csum += eps; // match CPU: sum = eps + elements
            comb_val *= __frcp_rn(csum);
        }

        // ---- Write ----
        dst[col * sd0 + row * sd1 + it * sd2] = comb_val;
    }
}

static __global__ void dsv4_hc_pre_f32(
        const float * x,
        const float * weights,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sx2,
        int64_t sw0,
        int64_t sw1,
        int64_t sd0,
        int64_t sd1) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0 = ir % n_embd;
    const int64_t it = ir / n_embd;

    float sum = x[i0*sx0 + it*sx2] * weights[it*sw1];
    for (int64_t ih = 1; ih < hc; ++ih) {
        const float xv = x[i0*sx0 + ih*sx1 + it*sx2];
        const float wv = weights[ih*sw0 + it*sw1];
        sum += xv * wv;
    }

    dst[i0*sd0 + it*sd1] = sum;
}

static __global__ void dsv4_hc_post_f32(
        const float * x,
        const float * residual,
        const float * post,
        const float * comb,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sr0,
        int64_t sr1,
        int64_t sr2,
        int64_t sp0,
        int64_t sp1,
        int64_t sc0,
        int64_t sc1,
        int64_t sc2,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * hc * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0   = ir % n_embd;
    const int64_t idst = (ir / n_embd) % hc;
    const int64_t it   = ir / (n_embd * hc);

    float sum = x[i0*sx0 + it*sx1] * post[idst*sp0 + it*sp1];
    for (int64_t isrc = 0; isrc < hc; ++isrc) {
        sum += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
    }

    dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
}

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc_mix_dim = (2 + DSV4_HC)*DSV4_HC;

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0] == DSV4_HC);
    GGML_ASSERT(dst->ne[1] == DSV4_HC);
    GGML_ASSERT(dst->ne[2] == mixes->ne[1]);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(base->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, base,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    const int64_t n_tokens = mixes->ne[1];
    const float eps = ggml_get_op_params_f32(dst, 0);
    const int32_t n_iter = ggml_get_op_params_i32(dst, 1);

    const int block_size = BLOCK_SIZE; // 128
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_tokens + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_comb_f32, launch_params,
            (const float *) mixes->data, (const float *) scale->data, (const float *) base->data, (float *) dst->data,
            n_tokens,
            nbm0 / sizeof(float), nbm1 / sizeof(float),
            nbs0 / sizeof(float),
            nbb0 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
            eps, n_iter);
}
void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    const int64_t n_embd   = x->ne[0];
    const int64_t hc       = x->ne[1];
    const int64_t n_tokens = x->ne[2];

    const int block_size = 256;
    const int64_t nr = n_embd * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_pre_f32, launch_params,
            (const float *) x->data, (const float *) weights->data, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float), nbx2 / sizeof(float),
            nbw0 / sizeof(float), nbw1 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float));
}

void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * residual = dst->src[1];
    const ggml_tensor * post     = dst->src[2];
    const ggml_tensor * comb     = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(comb->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,        nb);
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbp, post,     nb);
    GGML_TENSOR_LOCALS(size_t, nbc, comb,     nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t hc       = residual->ne[1];

    const int block_size = 256;
    const int64_t nr = n_embd * hc * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_post_f32, launch_params,
            (const float *) x->data, (const float *) residual->data,
            (const float *) post->data, (const float *) comb->data, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float),
            nbr0 / sizeof(float), nbr1 / sizeof(float), nbr2 / sizeof(float),
            nbp0 / sizeof(float), nbp1 / sizeof(float),
            nbc0 / sizeof(float), nbc1 / sizeof(float), nbc2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float));
}

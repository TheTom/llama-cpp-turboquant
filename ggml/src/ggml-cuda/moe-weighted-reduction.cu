#include "moe-weighted-reduction.cuh"

static __global__ void moe_weighted_reduction_f32(const float * __restrict__ experts,
                                                  const float * __restrict__ expert_scale,
                                                  const float * __restrict__ weights,
                                                  const float * __restrict__ add0,
                                                  const float * __restrict__ add1,
                                                  float * __restrict__ dst,
                                                  const int64_t n_embd,
                                                  const int     n_expert_used) {
    const int64_t token = blockIdx.x;
    const int64_t col   = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= n_embd) {
        return;
    }

    const uint64_t first_row   = (uint64_t) token * n_expert_used;
    const float    first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
    float          sum         = (experts[first_row * n_embd + col] * first_scale) * weights[first_row];

    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row   = first_row + expert;
        const float   scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        sum += (experts[row * n_embd + col] * scale) * weights[row];
    }
    // folded residual ADDs, applied in graph order so the result is bit-identical to separate adds
    if (add0 != nullptr) {
        sum += add0[token * n_embd + col];
    }
    if (add1 != nullptr) {
        sum += add1[token * n_embd + col];
    }
    dst[token * n_embd + col] = sum;
}

// One block per token that computes the whole row into shared memory before writing it, so dst may
// alias any input (used when folding residual ADDs puts the final output in memory the allocator
// recycled from the expert outputs). Same per-column arithmetic and order as the kernel above.
static __global__ void moe_weighted_reduction_f32_staged(const float * __restrict__ experts,
                                                         const float * __restrict__ expert_scale,
                                                         const float * __restrict__ weights,
                                                         const float * __restrict__ add0,
                                                         const float * __restrict__ add1,
                                                         float *                    dst,
                                                         const int64_t              n_embd,
                                                         const int                  n_expert_used) {
    extern __shared__ float row_buf[];
    const int64_t  token     = blockIdx.x;
    const uint64_t first_row = (uint64_t) token * n_expert_used;
    for (int64_t col = threadIdx.x; col < n_embd; col += blockDim.x) {
        const float first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
        float       sum         = (experts[first_row * n_embd + col] * first_scale) * weights[first_row];
        for (int expert = 1; expert < n_expert_used; ++expert) {
            const uint64_t row   = first_row + expert;
            const float    scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
            sum += (experts[row * n_embd + col] * scale) * weights[row];
        }
        if (add0 != nullptr) {
            sum += add0[token * n_embd + col];
        }
        if (add1 != nullptr) {
            sum += add1[token * n_embd + col];
        }
        row_buf[col] = sum;
    }
    __syncthreads();
    for (int64_t col = threadIdx.x; col < n_embd; col += blockDim.x) {
        dst[token * n_embd + col] = row_buf[col];
    }
}

static void launch_moe_weighted_reduction(const float * experts,
                                          const float * expert_scale,
                                          const float * weights,
                                          const float * add0,
                                          const float * add1,
                                          float *       dst,
                                          int64_t       n_embd,
                                          int64_t       n_tokens,
                                          int           n_expert_used,
                                          bool          staged,
                                          cudaStream_t  stream) {
    constexpr int threads = 256;
    if (staged) {
        GGML_ASSERT(n_tokens == 1 && n_embd <= GGML_CUDA_MOE_WR_STAGED_MAX_EMBD);
        moe_weighted_reduction_f32_staged<<<1, 1024, n_embd * sizeof(float), stream>>>(
            experts, expert_scale, weights, add0, add1, dst, n_embd, n_expert_used);
        return;
    }
    const dim3 blocks(n_tokens, (n_embd + threads - 1) / threads, 1);
    moe_weighted_reduction_f32
        <<<blocks, threads, 0, stream>>>(experts, expert_scale, weights, add0, add1, dst, n_embd, n_expert_used);
}

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst,
                                         const ggml_tensor *         add0,
                                         const ggml_tensor *         add1,
                                         bool                        staged) {
    GGML_ASSERT(experts->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(expert_scale == nullptr || expert_scale->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(experts));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(expert_scale == nullptr || ggml_is_contiguous(expert_scale));
    GGML_ASSERT(ggml_is_contiguous(dst));
    for (const ggml_tensor * a : { add0, add1 }) {
        GGML_ASSERT(a == nullptr || (a->type == GGML_TYPE_F32 && ggml_is_contiguous(a) && ggml_nelements(a) == ggml_nelements(dst)));
    }

    const int64_t n_embd        = experts->ne[0];
    const int64_t n_expert_used = experts->ne[1];
    const int64_t n_tokens      = experts->ne[2] * experts->ne[3];
    cudaStream_t  stream        = ctx.stream();

    launch_moe_weighted_reduction((const float *) experts->data,
                                  expert_scale ? (const float *) expert_scale->data : nullptr,
                                  (const float *) weights->data,
                                  add0 ? (const float *) add0->data : nullptr,
                                  add1 ? (const float *) add1->data : nullptr,
                                  (float *) dst->data, n_embd, n_tokens, (int) n_expert_used, staged, stream);
    CUDA_CHECK(cudaGetLastError());
}

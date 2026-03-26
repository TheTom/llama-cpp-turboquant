#include "turbo-wht.cuh"

// WHT sign arrays — must match CPU (ops.cpp) and Metal (ggml-metal.metal)
// Generated with seed=42 (rotation)
static __device__ const float turbo_cuda_wht_s1[128] = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1
};

static __device__ const float turbo_cuda_wht_s2[128] = {
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1
};

// Each thread processes one 128-element group
static __global__ void kernel_turbo_wht(
        const float * __restrict__ src,
        float       * __restrict__ dst,
        const int64_t n_groups,
        const int     direction) {

    const int64_t g = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_groups) return;

    const float * s_first  = (direction == 0) ? turbo_cuda_wht_s1 : turbo_cuda_wht_s2;
    const float * s_second = (direction == 0) ? turbo_cuda_wht_s2 : turbo_cuda_wht_s1;

    const float * in  = src + g * 128;
    float       * out = dst + g * 128;

    float x[128];

    // Apply first signs
    for (int i = 0; i < 128; i++) {
        x[i] = in[i] * s_first[i];
    }

    // WHT butterfly (7 stages)
    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }

    // Normalize + second signs
    const float inv_sqrt_128 = 0.08838834764831845f;
    for (int i = 0; i < 128; i++) {
        out[i] = x[i] * inv_sqrt_128 * s_second[i];
    }
}

void ggml_cuda_op_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const float * src_d = (const float *)src->data;
    float       * dst_d = (float *)dst->data;

    int direction;
    memcpy(&direction, dst->op_params, sizeof(int));

    const int64_t n_total = ggml_nelements(src);
    GGML_ASSERT(n_total % 128 == 0);
    const int64_t n_groups = n_total / 128;

    const int block_size = 256;
    const int n_blocks = (n_groups + block_size - 1) / block_size;

    kernel_turbo_wht<<<n_blocks, block_size, 0, ctx.stream()>>>(
        src_d, dst_d, n_groups, direction);
}

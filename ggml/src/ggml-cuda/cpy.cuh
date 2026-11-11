#include "common.cuh"

#define CUDA_CPY_BLOCK_SIZE 64

void ggml_cuda_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * src1);

#define GGML_CUDA_CPY_BATCH_MAX 8

// n (<= GGML_CUDA_CPY_BATCH_MAX) f32 -> f32 CPY nodes whose sources share one shape and stride set and
// whose destinations share another, run as one launch (blockIdx.y = copy).
void ggml_cuda_cpy_f32_batched(ggml_backend_cuda_context & ctx, const ggml_tensor * const * cpys, int n);

void ggml_cuda_dup(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

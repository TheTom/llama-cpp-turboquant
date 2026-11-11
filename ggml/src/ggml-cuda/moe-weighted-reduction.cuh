#include "common.cuh"

// staged (alias-safe) variant: one token, row held in shared memory
#define GGML_CUDA_MOE_WR_STAGED_MAX_EMBD 8192

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst,
                                         const ggml_tensor *         add0 = nullptr,
                                         const ggml_tensor *         add1 = nullptr,
                                         bool                        staged = false);

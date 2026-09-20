#pragma once

#include "common.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// Returns true when a CR inverse kernel was launched.
bool ggml_cuda_mul_mat_vec_cr(
        ggml_type type,
        const void * weights,
        const float * x,
        float * y,
        int64_t m,
        int64_t n,
        int64_t k,
        int cc,
        cudaStream_t stream);

#endif

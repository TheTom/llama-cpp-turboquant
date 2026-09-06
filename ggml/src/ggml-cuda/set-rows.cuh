#pragma once

#include "common.cuh"

#define CUDA_SET_ROWS_BLOCK_SIZE 256

void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_set_rows_staged(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * dst,
    int64_t first_row,
    int64_t row_count,
    void * mirror_data);

#pragma once

#include "common.cuh"

// Largest expert count the tail takes: the result matches SUM_ROWS bit for bit only while every
// slot sits in its own lane of reduce_rows_f32's first warp.
#define MOE_REDUCE_MAX_USED 32

// Fused MoE expert-reduce tail: replaces MUL(expert weights) -> PERMUTE -> CONT -> SUM_ROWS
// after a MUL_MAT_ID with a weighted sum over the expert slots. Weight-type agnostic: reads the
// f32 MUL_MAT_ID output directly, so it applies to every quantization. Accumulates into pool
// scratch and copies to the graph tensor last, which resolves every allocator-aliasing hazard
// without vetoes (the graph output may reuse memory of any tensor that dies inside the tail).
void ggml_cuda_op_moe_reduce(ggml_backend_cuda_context & ctx,
                             const ggml_tensor * mmid_out,   // [nrows, n_used, ntok] f32
                             const ggml_tensor * weights,    // [1, n_used, ntok] f32
                             ggml_tensor * out);             // [1, nrows, ntok] f32

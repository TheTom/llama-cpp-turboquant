#pragma once

#include "common.cuh"

/*
 * TurboQuant CUDA backend — 3-bit uniform KV cache compression
 * Based on Lucien2468/Ollama-TurboQuant-Integration
 *
 * turbo3: 3-bit uniform quantization, 32-element blocks, 14 bytes, 4.6x compression
 *   - round(x/d) clamped to [-4,3], packed 8 per 3 bytes
 *   - dp4a compatible for hardware-accelerated dot product
 *   - No rotation, no codebooks — simple and fast
 */

#define QR_TURBO3  2    /* 2 elements per dequantize call */
#define QI_TURBO3  3    /* 12 bytes / sizeof(int) = 3 ints per block */
#define QK_TURBO4  256  /* turbo4 block size (stub) */

/* ===== Declarations ===== */

void turbo_quant_init_cuda(cudaStream_t stream);
void ggml_cuda_op_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

template<typename dst_t>
void dequantize_row_turbo3_0_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream);

template<typename dst_t>
void dequantize_row_turbo3_0_no_wht_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream);

void turbo3_inverse_wht32_cuda(float * data, int64_t n_elements, cudaStream_t stream);

template<typename dst_t>
void dequantize_row_turbo4_0_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream);

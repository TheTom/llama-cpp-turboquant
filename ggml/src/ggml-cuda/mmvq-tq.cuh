#pragma once

#include "common.cuh"

// Fused TQ weight mul_mat: handles ne[1]=1 (decode) and ne[1]>1 (prefill/speculative)
void ggml_cuda_mul_mat_tq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// Legacy single-token alias
inline void ggml_cuda_mul_mat_vec_tq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_cuda_mul_mat_tq(ctx, src0, src1, dst);
}

// Capture-safe MoE (MUL_MAT_ID) matvec for TQ weights at small batch (decode / light
// speculative). Reads ids on-device; no host sync, so it can be recorded into a CUDA graph.
// Caller must ensure src1 is contiguous and dst->ne[2] (n_tokens) <= MMVQ_MAX_BATCH_SIZE.
void ggml_cuda_mul_mat_id_tq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);

// Large prefill: runtime TQ4_1S → q8_0 scratch + cuBLAS
void ggml_cuda_mul_mat_tq4_1s_cublas(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
// Phase 2: native MFMA-i8 MMQ prefill for TQ4_1S (gfx90a). Pre-rotates the activation (forward WHT)
// then runs the stock MMQ with a TQ4_1S int8-centroid load_tiles. Env-gated by GGML_TQ_MMQ.
void ggml_cuda_mul_mat_tq4_1s_mmq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
// Phase 2 (MoE): native MFMA-i8 MMQ prefill for TQ4_1S experts (gfx90a). Pre-rotates the activation
// once, then runs the stock MMQ_ID (ggml_cuda_mul_mat_q with ids). Env-gated by GGML_TQ_MMQ.
void ggml_cuda_mul_mat_id_tq4_1s_mmq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);

// Load-time conversion: TQ4_1S → q8_0 in VRAM (dequant + requantize)
void ggml_cuda_convert_tq4_1s_to_q8_0(const void * src_tq4, void * dst_q8, int64_t n_elements, cudaStream_t stream);

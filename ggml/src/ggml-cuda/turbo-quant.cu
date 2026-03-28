/*
 * TurboQuant CUDA kernels — 3-bit uniform quantization
 * Based on Lucien2468/Ollama-TurboQuant-Integration
 */

#include "turbo-quant.cuh"
#include "convert.cuh"

#include <cstdint>

void turbo_quant_init_cuda(cudaStream_t stream) {
    GGML_UNUSED(stream);
}

/* ===== turbo3 dequantize: 3-bit uniform, packed 8 per 3 bytes ===== */

template<typename dst_t>
static __global__ void dequantize_block_turbo3_0(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;
    const block_turbo3_0 * x = (const block_turbo3_0 *)vx;
    const int tid = threadIdx.x;  // 0..31
    if (tid >= 32) return;

    const float d = __half2float(x[i].d);
    const int group = tid / 8;
    const int elem  = tid % 8;
    const uint8_t * qs = x[i].qs + group * 3;

    int val;
    switch (elem) {
        case 0: val = (qs[0]      ) & 7; break;
        case 1: val = (qs[0] >>  3) & 7; break;
        case 2: val = ((qs[0] >> 6) & 3) | ((qs[1] & 1) << 2); break;
        case 3: val = (qs[1] >>  1) & 7; break;
        case 4: val = (qs[1] >>  4) & 7; break;
        case 5: val = ((qs[1] >> 7) & 1) | ((qs[2] & 3) << 1); break;
        case 6: val = (qs[2] >>  2) & 7; break;
        case 7: val = (qs[2] >>  5) & 7; break;
        default: val = 0; break;
    }

    yy[i * 32 + tid] = ggml_cuda_cast<dst_t>((val - 4) * d);
}

/* ===== turbo4 dequantize (stub — not yet implemented in Lucien approach) ===== */

template<typename dst_t>
static __global__ void dequantize_block_turbo4_0(
        const void * __restrict__ vx, dst_t * __restrict__ yy, const int64_t k) {
    /* Stub — turbo4 not yet ported to Lucien format */
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < k) yy[idx] = ggml_cuda_cast<dst_t>(0.0f);
}

/* ===== TURBO_WHT graph op (kept for backward compat, identity for Lucien approach) ===== */

void ggml_cuda_op_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    const int64_t ne = ggml_nelements(src0);
    cudaStream_t stream = ctx.stream();

    if (src0_d != dst_d) {
        cudaMemcpyAsync(dst_d, src0_d, ne * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }
    /* No-op for Lucien approach (no rotation). Just copy. */
}

/* ===== Row dequantize launchers ===== */

template<typename dst_t>
void dequantize_row_turbo3_0_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_TURBO3;
    dequantize_block_turbo3_0<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
void dequantize_row_turbo3_0_no_wht_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    /* Same as full dequant — no WHT in Lucien approach */
    dequantize_row_turbo3_0_cuda(vx, y, k, stream);
}

void turbo3_inverse_wht32_cuda(float * data, int64_t n_elements, cudaStream_t stream) {
    /* No-op for Lucien approach */
    GGML_UNUSED(data); GGML_UNUSED(n_elements); GGML_UNUSED(stream);
}

template<typename dst_t>
void dequantize_row_turbo4_0_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int threads = 256;
    const int blocks = (k + threads - 1) / threads;
    dequantize_block_turbo4_0<<<blocks, threads, 0, stream>>>(vx, y, k);
}

/* Explicit template instantiations */
template void dequantize_row_turbo3_0_cuda<float>(const void *, float *, const int64_t, cudaStream_t);
template void dequantize_row_turbo3_0_cuda<half>(const void *, half *, const int64_t, cudaStream_t);
template void dequantize_row_turbo3_0_cuda<nv_bfloat16>(const void *, nv_bfloat16 *, const int64_t, cudaStream_t);

template void dequantize_row_turbo3_0_no_wht_cuda<float>(const void *, float *, const int64_t, cudaStream_t);
template void dequantize_row_turbo3_0_no_wht_cuda<half>(const void *, half *, const int64_t, cudaStream_t);

template void dequantize_row_turbo4_0_cuda<float>(const void *, float *, const int64_t, cudaStream_t);
template void dequantize_row_turbo4_0_cuda<half>(const void *, half *, const int64_t, cudaStream_t);
template void dequantize_row_turbo4_0_cuda<nv_bfloat16>(const void *, nv_bfloat16 *, const int64_t, cudaStream_t);

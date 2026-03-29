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

/* ===== turbo3 dequantize: unpack 3-bit from 4 blocks + inverse WHT128 (cooperative) ===== */
/* One CUDA block per 128-element group (4 turbo3 blocks). 128 threads. */

template<typename dst_t>
static __global__ void dequantize_block_turbo3_0(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t grp = blockIdx.x;  /* group index (4 turbo3 blocks per group) */
    const block_turbo3_0 * x = (const block_turbo3_0 *)vx;
    const int tid = threadIdx.x;  /* 0..127 */
    if (tid >= 128) return;

    /* Each thread unpacks one element from one of 4 blocks */
    const int blk_idx = tid / 32;
    const int elem_in_blk = tid % 32;
    const block_turbo3_0 * blk = &x[grp*4 + blk_idx];

    const float d = __half2float(blk->d);
    const int group = elem_in_blk / 8;
    const int elem  = elem_in_blk % 8;
    const uint8_t * qs = blk->qs + group * 3;

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

    /* Dequant to rotated space using Lloyd-Max centroids */
    const float cn[8] = {
        -0.18165533f, -0.11483849f, -0.06487426f, -0.02106513f,
         0.02106513f,  0.06487426f,  0.11483849f,  0.18165533f
    };
    __shared__ float shmem[128];
    shmem[tid] = cn[val] * d;  /* d = group norm, cn[val] = Lloyd-Max centroid */
    __syncthreads();

    /* Cooperative inverse WHT128 (7 butterfly stages) */
    for (int step = 1; step < 128; step <<= 1) {
        int partner = tid ^ step;
        float a = shmem[tid];
        float b = shmem[partner];
        __syncthreads();
        if (tid < partner) {
            shmem[tid]     = a + b;
            shmem[partner] = a - b;
        }
        __syncthreads();
    }

    /* Normalize and undo sign flips */
    yy[grp * 128 + tid] = ggml_cuda_cast<dst_t>(shmem[tid] * 0.08838834764831845f * turbo3_wht128_signs[tid]);
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

/* ===== Custom set_rows for turbo3 with WHT128 ===== */
/* Each thread processes one 128-element rotation group (4 blocks). */

template<typename idx_t>
static __global__ void turbo3_set_rows_kernel(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_turbo3_0 * __restrict__ dst,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne02,
        const int64_t ne03,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t s1,
        const int64_t s2,
        const int64_t s3) {

    const int64_t ne_groups = ne00 / TURBO3_GROUP;
    const int64_t ne_total = ne_groups * ne01 * ne02 * ne03;
    const int64_t i = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= ne_total) return;

    /* Decompose flat index into (group_in_row, row, dim2, dim3) */
    const int64_t i_grp = i % ne_groups;
    const int64_t i01   = (i / ne_groups) % ne01;
    const int64_t i02   = (i / (ne_groups * ne01)) % ne02;
    const int64_t i03   = (i / (ne_groups * ne01 * ne02));

    /* Get destination row from index tensor */
    const int64_t dst_row = src1[i01*s10 + i02*s11 + i03*s12];

    /* Source: 128 floats from src0 */
    const float * src_ptr = src0 + i01*s01 + i02*s02 + i03*s03 + i_grp*TURBO3_GROUP;

    /* Destination: 4 consecutive turbo3 blocks */
    char * dst_base = (char *)dst + dst_row*s1 + i02*s2 + i03*s3;
    block_turbo3_0 * dst_ptr = (block_turbo3_0 *)dst_base + i_grp * (TURBO3_GROUP / QK_TURBO3);

    /* WHT128 rotate + quantize into 4 blocks */
    turbo3_quantize_group_128(src_ptr, dst_ptr);
}

template<typename idx_t>
void turbo3_set_rows_cuda(
        const float * src0_d, const idx_t * src1_d, block_turbo3_0 * dst_d,
        int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
        size_t nb01, size_t nb02, size_t nb03,
        size_t nb10, size_t nb11, size_t nb12,
        size_t nb1, size_t nb2, size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % TURBO3_GROUP == 0);
    const int64_t ne_groups = ne00 / TURBO3_GROUP;
    const int64_t ne_total = ne_groups * ne01 * ne02 * ne03;
    const int threads = 64;
    const int blocks = (ne_total + threads - 1) / threads;

    turbo3_set_rows_kernel<<<blocks, threads, 0, stream>>>(
        src0_d, src1_d, dst_d,
        ne00, ne01, ne02, ne03,
        (int64_t)(nb01/sizeof(float)), (int64_t)(nb02/sizeof(float)), (int64_t)(nb03/sizeof(float)),
        (int64_t)(nb10/sizeof(idx_t)), (int64_t)(nb11/sizeof(idx_t)), (int64_t)(nb12/sizeof(idx_t)),
        (int64_t)nb1, (int64_t)nb2, (int64_t)nb3);
}

template void turbo3_set_rows_cuda<int32_t>(const float*, const int32_t*, block_turbo3_0*, int64_t, int64_t, int64_t, int64_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, cudaStream_t);
template void turbo3_set_rows_cuda<int64_t>(const float*, const int64_t*, block_turbo3_0*, int64_t, int64_t, int64_t, int64_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, cudaStream_t);

/* ===== Row dequantize launchers ===== */

template<typename dst_t>
void dequantize_row_turbo3_0_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    GGML_ASSERT(k % TURBO3_GROUP == 0);
    const int n_groups = k / TURBO3_GROUP;
    dequantize_block_turbo3_0<<<n_groups, 128, 0, stream>>>(vx, y);
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

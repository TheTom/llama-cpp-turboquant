/*
 * TurboQuant CUDA kernels — dequantize with inverse WHT
 * Based on animehacker/llama-turboquant reference implementation.
 * Per-block WHT32 rotation: no graph-level ops needed.
 */

#include "turbo-quant.cuh"
#include "convert.cuh"

#include <cstdint>

void turbo_quant_init_cuda(cudaStream_t stream) {
    GGML_UNUSED(stream);
}

/* ===== turbo3 dequantize: centroid lookup + cooperative inverse WHT32 ===== */

template<typename dst_t>
static __global__ void dequantize_block_turbo3_0(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const float centroids[8] = {
        -2.1573f, -1.3336f, -0.7434f, -0.2428f,
         0.2428f,  0.7434f,  1.3336f,  2.1573f
    };
    const int8_t signs[32] = {
        +1, -1, +1, +1, -1, -1, +1, -1, +1, +1, -1, +1, -1, +1, -1, -1,
        +1, -1, -1, +1, +1, -1, +1, -1, -1, +1, +1, +1, -1, -1, +1, -1
    };

    const int64_t i = blockIdx.x;
    const block_turbo3_0 * x = (const block_turbo3_0 *)vx;
    const int tid = threadIdx.x;
    if (tid >= 32) return;

    const float d = __half2float(x[i].gamma);

    // Step 1: Each thread dequantizes its value (3-bit index from qs + qr)
    const int low2 = (x[i].qs[tid / 4] >> (2 * (tid % 4))) & 3;
    const int hi1  = (x[i].qr[tid / 8] >> (tid % 8)) & 1;
    const int idx  = low2 | (hi1 << 2);

    __shared__ float shmem[32];
    shmem[tid] = d * centroids[idx];
    __syncthreads();

    // Step 2: Cooperative inverse WHT (5 butterfly stages)
    for (int step = 1; step < 32; step <<= 1) {
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

    // Step 3: Normalize and undo sign flips
    const float inv_sqrt32 = 0.17677669529663688f;
    yy[i * QK_TURBO3 + tid] = ggml_cuda_cast<dst_t>(shmem[tid] * inv_sqrt32 * signs[tid]);
}

/* ===== turbo4 dequantize: PolarQuant (angle grid + sign), no rotation ===== */
/* 256 elements per block. 256 threads, 1 element each. Simple and fast. */

template<typename dst_t>
static __global__ void dequantize_block_turbo4_0(
        const void * __restrict__ vx, dst_t * __restrict__ yy, const int64_t k) {
    const int64_t i = blockIdx.x;  /* block index */
    const int tid = threadIdx.x;   /* 0..255, one element each */

    if (i * QK_TURBO4 + tid >= k) return;

    const block_turbo4_0 * x = (const block_turbo4_0 *) vx;
    const float d = __half2float(x[i].d);

    /* Unpack 3-bit angle: 2-bit lo from al[], 1-bit hi from ah[] */
    const uint8_t lo  = (x[i].al[tid / 4] >> ((tid % 4) * 2)) & 0x3;
    const uint8_t hi  = (x[i].ah[tid / 8] >> (tid % 8)) & 0x1;
    const uint8_t idx = lo | (hi << 2);

    /* Sign from QJL signs[] */
    const uint8_t sign = (x[i].signs[tid / 8] >> (tid % 8)) & 0x1;

    /* Dequant: d * grid[idx] * (1 - 2*sign) */
    const float grid_val = turbo4_polar_grid(idx);
    yy[i * QK_TURBO4 + tid] = ggml_cuda_cast<dst_t>(d * grid_val * (1.0f - 2.0f * sign));
}

/* ===== Optimization #2: Flat no-WHT dequant — no shared memory, no syncthreads ===== */
/* Each thread handles 4 elements independently. 256 threads per block, 1024 elements/block.
 * No cooperative computation needed since there's no WHT butterfly.
 * 32x fewer kernel launches than the per-block version. */

template<typename dst_t>
static __global__ void dequantize_flat_turbo3_0_no_wht(
        const void * __restrict__ vx, dst_t * __restrict__ yy, const int64_t n_elements) {
    /* Optimization #3: Shared memory centroid LUT — loaded once per block */
    __shared__ float sh_cn[8];
    if (threadIdx.x < 8) {
        const float cn[8] = {-2.1573f, -1.3336f, -0.7434f, -0.2428f,
                              0.2428f,  0.7434f,  1.3336f,  2.1573f};
        sh_cn[threadIdx.x] = cn[threadIdx.x];
    }
    __syncthreads();

    const int64_t elem = (int64_t)blockIdx.x * blockDim.x * 4 + threadIdx.x * 4;
    if (elem >= n_elements) return;

    const block_turbo3_0 * x = (const block_turbo3_0 *)vx;
    const int64_t ib = elem / QK_TURBO3;
    const int j_base = elem % QK_TURBO3;
    const float d = __half2float(x[ib].gamma);

    /* Batch-read one qs byte (4 elements) + half signs byte */
    const uint8_t qb = x[ib].qs[j_base / 4];
    const uint8_t sb = x[ib].qr[j_base / 8];
    const int sshift = j_base % 8;

    #pragma unroll
    for (int l = 0; l < 4 && elem + l < n_elements; l++) {
        const int low2 = (qb >> (l * 2)) & 3;
        const int hi1  = (sb >> (sshift + l)) & 1;
        yy[elem + l] = ggml_cuda_cast<dst_t>(d * sh_cn[low2 | (hi1 << 2)]);
    }
}

/* ===== Optimization #1: Fused V dequant + attention matmul ===== */
/* Computes kqv = (V_turbo3_dequanted_no_wht)^T * attn_weights without writing V_f32 to DRAM.
 * Each CUDA block computes one output element kqv[head_dim_idx, query_idx].
 * V data is read directly from turbo3 blocks, dequanted in registers. */

static __global__ void turbo3_fused_v_attn_kernel(
        const void * __restrict__ v_data,     // turbo3 V cache [n_kv, head_dim] per head
        const float * __restrict__ attn,      // attention weights [n_kv, n_q] per head
        float * __restrict__ output,          // output [head_dim, n_q] per head
        const int64_t n_kv,                   // number of KV positions
        const int64_t head_dim,               // head dimension (e.g. 128)
        const int64_t n_q,                    // number of query positions (1 during decode)
        const int64_t v_stride_row,           // byte stride between V rows (n_kv dim)
        const int64_t attn_stride_row) {      // stride between attn rows

    /* Shared memory centroid LUT */
    __shared__ float sh_cn[8];
    if (threadIdx.x < 8) {
        const float cn[8] = {-2.1573f, -1.3336f, -0.7434f, -0.2428f,
                              0.2428f,  0.7434f,  1.3336f,  2.1573f};
        sh_cn[threadIdx.x] = cn[threadIdx.x];
    }
    __syncthreads();

    /* Each block handles one (head_dim_idx, q_idx) output element.
     * Threads cooperatively iterate over n_kv positions. */
    const int64_t hd_idx = blockIdx.x;   // which head_dim element
    const int64_t q_idx  = blockIdx.y;   // which query position

    if (hd_idx >= head_dim || q_idx >= n_q) return;

    /* Which turbo3 block and position within block for this head_dim element */
    const int ib_local = hd_idx / QK_TURBO3;   // block index within one V row
    const int j_in_block = hd_idx % QK_TURBO3; // element within block
    const int qs_byte_idx = j_in_block / 4;
    const int qs_shift = (j_in_block % 4) * 2;
    const int qr_byte_idx = j_in_block / 8;
    const int qr_shift = j_in_block % 8;

    float sum = 0.0f;

    /* Iterate over KV positions, each thread handles a stride */
    for (int64_t k = threadIdx.x; k < n_kv; k += blockDim.x) {
        /* Dequant V[k, hd_idx] from turbo3 — in registers, no DRAM write */
        const char * v_row = (const char *)v_data + k * v_stride_row;
        const block_turbo3_0 * blk = (const block_turbo3_0 *)v_row + ib_local;

        const float d = __half2float(blk->gamma);
        const int low2 = (blk->qs[qs_byte_idx] >> qs_shift) & 3;
        const int hi1  = (blk->qr[qr_byte_idx] >> qr_shift) & 1;
        const float v_val = d * sh_cn[low2 | (hi1 << 2)];

        /* Multiply by attention weight */
        sum += v_val * attn[k * attn_stride_row + q_idx];
    }

    /* Warp reduction */
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    /* Block reduction via shared memory */
    __shared__ float sh_partial[32];  // one per warp
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    if (lane_id == 0) sh_partial[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane_id < (blockDim.x + 31) / 32) ? sh_partial[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane_id == 0) {
            output[hd_idx * n_q + q_idx] = sum;
        }
    }
}

/* Launch fused V*attn kernel */
void turbo3_fused_v_attn_cuda(
        const void * v_data, const float * attn, float * output,
        int64_t n_kv, int64_t head_dim, int64_t n_q, int64_t n_heads,
        int64_t v_stride_row, int64_t v_stride_head,
        int64_t attn_stride_row, int64_t attn_stride_head,
        int64_t out_stride_head,
        cudaStream_t stream) {

    const int threads = min((int)n_kv, 256);
    dim3 grid(head_dim, n_q);

    for (int64_t h = 0; h < n_heads; h++) {
        const char * v_head = (const char *)v_data + h * v_stride_head;
        const float * attn_head = attn + h * attn_stride_head;
        float * out_head = output + h * out_stride_head;

        turbo3_fused_v_attn_kernel<<<grid, threads, 0, stream>>>(
            v_head, attn_head, out_head,
            n_kv, head_dim, n_q, v_stride_row, attn_stride_row);
    }
}

/* ===== Inverse WHT32 kernel for output (post-matmul) ===== */
/* Each CUDA block processes one 32-element WHT block cooperatively. */

static __global__ void turbo3_inverse_wht32_kernel(float * __restrict__ data, const int64_t n_elements) {
    const int8_t signs[32] = {
        +1, -1, +1, +1, -1, -1, +1, -1, +1, +1, -1, +1, -1, +1, -1, -1,
        +1, -1, -1, +1, +1, -1, +1, -1, -1, +1, +1, +1, -1, -1, +1, -1
    };

    const int64_t block_idx = blockIdx.x;
    const int tid = threadIdx.x;
    if (block_idx * 32 + tid >= n_elements) return;

    __shared__ float shmem[32];
    shmem[tid] = data[block_idx * 32 + tid];
    __syncthreads();

    for (int step = 1; step < 32; step <<= 1) {
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

    const float inv_sqrt32 = 0.17677669529663688f;
    data[block_idx * 32 + tid] = shmem[tid] * inv_sqrt32 * signs[tid];
}

void turbo3_inverse_wht32_cuda(float * data, int64_t n_elements, cudaStream_t stream) {
    GGML_ASSERT(n_elements % 32 == 0);
    const int nb = n_elements / 32;
    turbo3_inverse_wht32_kernel<<<nb, 32, 0, stream>>>(data, n_elements);
}

/* ===== TURBO_WHT graph op (for post-matmul inverse WHT on output) ===== */

void ggml_cuda_op_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;

    int direction;
    memcpy(&direction, dst->op_params, sizeof(int));

    const int64_t ne = ggml_nelements(src0);
    cudaStream_t stream = ctx.stream();

    if (src0_d != dst_d) {
        cudaMemcpyAsync(dst_d, src0_d, ne * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }

    /* direction=1 → inverse WHT32 (for V output), direction=0 → forward (not used here) */
    GGML_ASSERT(ne % 32 == 0);
    turbo3_inverse_wht32_cuda(dst_d, ne, stream);
    /* Note: forward WHT would need a separate kernel. For now only inverse is used. */
}

/* ===== Row dequantize launchers ===== */

template<typename dst_t>
void dequantize_row_turbo3_0_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_TURBO3;
    dequantize_block_turbo3_0<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
void dequantize_row_turbo3_0_no_wht_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    /* Flat kernel: 256 threads, 4 elements/thread = 1024 elements/block */
    const int threads = 256;
    const int elems_per_block = threads * 4;
    const int nb = (k + elems_per_block - 1) / elems_per_block;
    dequantize_flat_turbo3_0_no_wht<<<nb, threads, 0, stream>>>(vx, y, k);
}

template<typename dst_t>
void dequantize_row_turbo4_0_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    GGML_ASSERT(k % QK_TURBO4 == 0);
    const int nb = k / QK_TURBO4;
    dequantize_block_turbo4_0<<<nb, 256, 0, stream>>>(vx, y, k);  /* 256 threads = 1 per element */
}

/* Explicit template instantiations */
template void dequantize_row_turbo3_0_cuda<float>(const void * vx, float * y, const int64_t k, cudaStream_t stream);
template void dequantize_row_turbo3_0_cuda<half>(const void * vx, half * y, const int64_t k, cudaStream_t stream);
template void dequantize_row_turbo3_0_cuda<nv_bfloat16>(const void * vx, nv_bfloat16 * y, const int64_t k, cudaStream_t stream);

template void dequantize_row_turbo3_0_no_wht_cuda<float>(const void * vx, float * y, const int64_t k, cudaStream_t stream);
template void dequantize_row_turbo3_0_no_wht_cuda<half>(const void * vx, half * y, const int64_t k, cudaStream_t stream);

template void dequantize_row_turbo4_0_cuda<float>(const void * vx, float * y, const int64_t k, cudaStream_t stream);
template void dequantize_row_turbo4_0_cuda<half>(const void * vx, half * y, const int64_t k, cudaStream_t stream);
template void dequantize_row_turbo4_0_cuda<nv_bfloat16>(const void * vx, nv_bfloat16 * y, const int64_t k, cudaStream_t stream);

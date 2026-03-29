// TurboQuant temporal decay: zero signs field of turbo blocks in a position range.
// For turbo3: signs = high bit of 3-bit centroid index (4 bytes per 32-element block).
//   Zeroing reduces 8-level to 4-level reconstruction.
// For turbo4 (legacy 3-bit+QJL): signs = QJL refinement bits (16 bytes per 128-element block).
//   Also zeros rnorm to eliminate QJL contribution.

#include "turbo-decay.cuh"

// turbo3 kernel: zero signs field to collapse 3-bit -> 2-bit reconstruction
static __global__ void k_turbo3_decay(
        block_turbo3_0 * __restrict__ data,
        const int64_t                 block_start,
        const int64_t                 block_end) {

    const int64_t ib = (int64_t)blockDim.x * blockIdx.x + threadIdx.x + block_start;
    if (ib >= block_end) return;

    memset(data[ib].signs, 0, QK_TURBO3 / 8);
}

#if !TURBO4_USE_4BIT
// turbo4 legacy kernel: zero signs + rnorm to eliminate QJL contribution
static __global__ void k_turbo4_decay(
        block_turbo4_0 * __restrict__ data,
        const int64_t                 block_start,
        const int64_t                 block_end) {

    const int64_t ib = (int64_t)blockDim.x * blockIdx.x + threadIdx.x + block_start;
    if (ib >= block_end) return;

    memset(data[ib].signs, 0, QK_TURBO4 / 8);
    data[ib].rnorm = 0;
}
#endif

void ggml_cuda_op_turbo_decay(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    int64_t cold_start, cold_end;
    memcpy(&cold_start, dst->op_params + 0, sizeof(int64_t));
    memcpy(&cold_end,   dst->op_params + 2, sizeof(int64_t));

    if (cold_start >= cold_end) return;

    cudaStream_t stream = ctx.stream();
    const int64_t ne0 = src0->ne[0];  // n_embd_gqa (elements per row)

    if (src0->type == GGML_TYPE_TURBO3_0) {
        const int64_t blocks_per_row = ne0 / QK_TURBO3;
        const int64_t b_start = cold_start * blocks_per_row;
        const int64_t b_end   = cold_end * blocks_per_row;
        const int64_t n_blocks = b_end - b_start;
        if (n_blocks <= 0) return;
        const int block_size = 256;
        const int grid = (int)((n_blocks + block_size - 1) / block_size);

        k_turbo3_decay<<<grid, block_size, 0, stream>>>(
            (block_turbo3_0 *)src0->data, b_start, b_end);
    }
#if !TURBO4_USE_4BIT
    else if (src0->type == GGML_TYPE_TURBO4_0) {
        const int64_t blocks_per_row = ne0 / QK_TURBO4;
        const int64_t b_start = cold_start * blocks_per_row;
        const int64_t b_end   = cold_end * blocks_per_row;
        const int64_t n_blocks = b_end - b_start;
        if (n_blocks <= 0) return;
        const int block_size = 256;
        const int grid = (int)((n_blocks + block_size - 1) / block_size);

        k_turbo4_decay<<<grid, block_size, 0, stream>>>(
            (block_turbo4_0 *)src0->data, b_start, b_end);
    }
#endif
}

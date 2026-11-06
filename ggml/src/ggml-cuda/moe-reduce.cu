#include "moe-reduce.cuh"

// out_scr[row, tok] = sum_s w[s, tok] * x[row, s, tok]
// One thread per (row, tok); slot loop inside. Consecutive threads read consecutive rows of the
// same slot, so every pass over x is coalesced. w is tiny (n_used floats per token) and served
// from cache after the first block touches it.
//
// Bit-identical to the unfused MUL -> CONT -> SUM_ROWS: MUL rounds each product on its own, and
// reduce_rows_f32 puts slot s in lane s (n_used <= 32 < block size, other lanes 0), so its xor
// butterfly reduces the slots as a pairwise tree over the next power of two: at each offset,
// p[j] += p[j + off]. Same products, same tree, explicit _rn ops so no FMA contraction.
template <int NP>   // n_used rounded up to a power of two
static __global__ void k_moe_wsum(
        const float * __restrict__ x,
        const float * __restrict__ w,
        float       * __restrict__ out_scr,
        const int     nrows,
        const int     n_used,
        const int     ntok,
        const int64_t sx_slot,      // f32 elems between x slots
        const int64_t sx_tok,       // f32 elems between x tokens
        const int64_t sw_slot,      // f32 elems between w slots
        const int64_t sw_tok) {     // f32 elems between w tokens
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nrows * ntok) {
        return;
    }
    const int row = i % nrows;
    const int tok = i / nrows;

    const float * xr = x + row + tok * sx_tok;
    const float * wr = w + tok * sw_tok;

    float p[NP];
#pragma unroll
    for (int s = 0; s < NP; ++s) {
        p[s] = s < n_used ? __fmul_rn(xr[s * sx_slot], wr[s * sw_slot]) : 0.0f;
    }
#pragma unroll
    for (int off = NP/2; off > 0; off >>= 1) {
#pragma unroll
        for (int j = 0; j < off; ++j) {
            p[j] = __fadd_rn(p[j], p[j + off]);
        }
    }
    out_scr[i] = p[0];
}

// Copy the packed scratch into the graph output tensor. Runs after every read of the fused
// range, so the output may freely alias any tensor that died inside it.
static __global__ void k_moe_wsum_out(
        const float * __restrict__ scr,
        float       * __restrict__ out,
        const int     nrows,
        const int     ntok,
        const int64_t so_tok) {     // f32 elems between out tokens
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nrows * ntok) {
        return;
    }
    const int row = i % nrows;
    const int tok = i / nrows;
    out[tok * so_tok + row] = scr[i];
}

void ggml_cuda_op_moe_reduce(ggml_backend_cuda_context & ctx,
                             const ggml_tensor * mmid_out,
                             const ggml_tensor * weights,
                             ggml_tensor * out) {
    GGML_ASSERT(mmid_out->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type  == GGML_TYPE_F32);
    GGML_ASSERT(out->type      == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(out));

    const int nrows  = mmid_out->ne[0];
    const int n_used = mmid_out->ne[1];
    const int ntok   = mmid_out->ne[2];

    cudaStream_t stream = ctx.stream();

    const int64_t sx_slot = mmid_out->nb[1] / sizeof(float);
    const int64_t sx_tok  = mmid_out->nb[2] / sizeof(float);
    const int64_t sw_slot = weights->nb[1]  / sizeof(float);
    const int64_t sw_tok  = weights->nb[2]  / sizeof(float);
    const int64_t so_tok  = out->nb[2]      / sizeof(float);

    const int n_out = nrows * ntok;
    ggml_cuda_pool_alloc<float> scr(ctx.pool(), n_out);

    GGML_ASSERT(n_used >= 1 && n_used <= MOE_REDUCE_MAX_USED);
    const dim3 grid((n_out + 255) / 256);
    const float * x_d = (const float *) mmid_out->data;
    const float * w_d = (const float *) weights->data;
    if (n_used <= 2) {
        k_moe_wsum<2><<<grid, 256, 0, stream>>>(x_d, w_d, scr.get(), nrows, n_used, ntok, sx_slot, sx_tok, sw_slot, sw_tok);
    } else if (n_used <= 4) {
        k_moe_wsum<4><<<grid, 256, 0, stream>>>(x_d, w_d, scr.get(), nrows, n_used, ntok, sx_slot, sx_tok, sw_slot, sw_tok);
    } else if (n_used <= 8) {
        k_moe_wsum<8><<<grid, 256, 0, stream>>>(x_d, w_d, scr.get(), nrows, n_used, ntok, sx_slot, sx_tok, sw_slot, sw_tok);
    } else if (n_used <= 16) {
        k_moe_wsum<16><<<grid, 256, 0, stream>>>(x_d, w_d, scr.get(), nrows, n_used, ntok, sx_slot, sx_tok, sw_slot, sw_tok);
    } else {
        k_moe_wsum<32><<<grid, 256, 0, stream>>>(x_d, w_d, scr.get(), nrows, n_used, ntok, sx_slot, sx_tok, sw_slot, sw_tok);
    }
    k_moe_wsum_out<<<(n_out + 255) / 256, 256, 0, stream>>>(
        scr.get(), (float *) out->data, nrows, ntok, so_tok);
    CUDA_CHECK(cudaGetLastError());
}

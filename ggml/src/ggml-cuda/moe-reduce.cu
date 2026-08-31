#include "moe-reduce.cuh"

// out_scr[row, tok] = sum_s w[s, tok] * x[row, s, tok]
// One thread per (row, tok); slot loop inside. Consecutive threads read consecutive rows of the
// same slot, so every pass over x is coalesced. w is tiny (n_used floats per token) and served
// from cache after the first block touches it.
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

    float acc = 0.0f;
    for (int s = 0; s < n_used; ++s) {
        acc += wr[s * sw_slot] * xr[s * sx_slot];
    }
    out_scr[i] = acc;
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

    k_moe_wsum<<<(n_out + 255) / 256, 256, 0, stream>>>(
        (const float *) mmid_out->data, (const float *) weights->data, scr.get(),
        nrows, n_used, ntok, sx_slot, sx_tok, sw_slot, sw_tok);
    k_moe_wsum_out<<<(n_out + 255) / 256, 256, 0, stream>>>(
        scr.get(), (float *) out->data, nrows, ntok, so_tok);
    CUDA_CHECK(cudaGetLastError());
}

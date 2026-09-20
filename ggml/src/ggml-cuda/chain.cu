#include "chain.cuh"
#include "unary.cuh"

#define TQ_CHAIN_BLOCK 256

// The op loop is uniform across threads (k is the same for every lane), so the switch costs a
// few scalar instructions per op and never diverges. Everything between ops stays in a register.
static __global__ void k_elem_chain(const float * __restrict__ src,
                                    float * __restrict__ dst,
                                    const tq_chain_desc desc,
                                    const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * TQ_CHAIN_BLOCK + threadIdx.x;
    if (i >= n) {
        return;
    }

    float v = src[i];

#pragma unroll 1
    for (int k = 0; k < desc.n_ops; ++k) {
        const float * o_ptr = desc.other[k];
        const float   o     = o_ptr ? (desc.bcast[k] ? o_ptr[0] : o_ptr[i]) : 0.0f;

        switch (desc.code[k]) {
            case TQ_CHAIN_ADD:      v = v + o; break;
            case TQ_CHAIN_MUL:      v = v * o; break;
            case TQ_CHAIN_DIV:      v = desc.chain_is_lhs[k] ? (v / o) : (o / v); break;
            case TQ_CHAIN_SCALE:    v = v * desc.p0[k] + desc.p1[k]; break;
            case TQ_CHAIN_CLAMP:    v = fminf(fmaxf(v, desc.p0[k]), desc.p1[k]); break;
            case TQ_CHAIN_SILU:     v = v / (1.0f + expf(-v)); break;
            case TQ_CHAIN_SIGMOID:  v = 1.0f / (1.0f + expf(-v)); break;
            case TQ_CHAIN_SOFTPLUS: v = (v > 20.0f) ? v : logf(1.0f + expf(v)); break;  // same form as op_softplus
            case TQ_CHAIN_GELU:     v = ggml_cuda_op_gelu_single(v); break;  // same form as op_gelu
            case TQ_CHAIN_RELU:     v = fmaxf(v, 0.0f); break;
            case TQ_CHAIN_NEG:      v = -v; break;
            case TQ_CHAIN_SQR:      v = v * v; break;
            default: break;
        }
    }

    dst[i] = v;
}

void ggml_cuda_op_elem_chain(ggml_backend_cuda_context & ctx,
                             const float * src, float * dst, int64_t n,
                             const tq_chain_desc & desc) {
    const int64_t nblocks = (n + TQ_CHAIN_BLOCK - 1) / TQ_CHAIN_BLOCK;
    k_elem_chain<<<(unsigned) nblocks, TQ_CHAIN_BLOCK, 0, ctx.stream()>>>(src, dst, desc, n);
    CUDA_CHECK(cudaGetLastError());
}

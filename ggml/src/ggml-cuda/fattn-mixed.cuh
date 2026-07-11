#pragma once
#include "common.cuh"
#include "fattn-common.cuh"

// OSCAR INT2 q2_0 fused FA kernel — ported from
// OSCAR-llamacpp commit 95cb84d1e (ggml/src/ggml-cuda/flash-attn-mixed.cu)
//
// One thread per query row. Each thread computes the full KQ dot (no cross-thread
// reduction), then accumulates a slice of the V output. Simple online softmax.
// Supports q2_0 K/V; f16 HP tier is compiled out (n_hp=0 in the simplified port).
//
// Signature adapted for turboquant's graph: Q=src[0], K=src[1], V=src[2], mask=src[3].
// Parameters are read from op_params[0..2] (scale, max_bias, logit_softcap).

static __device__ float kq_dot_q2_0_mix(const char * K_c, const float * q, int DK) {
    const block_q2_0 * K = (const block_q2_0 *) K_c;
    constexpr float c[4] = {-0.9816f, -0.4528f, 0.4528f, 0.9816f};
    float sum = 0.0f;
    for (int i = 0; i < DK; i += 32) {
        const int ib = i / 32;
        const float mean = __half2float(K[ib].m);
        const float d = __half2float(K[ib].d);
        for (int j = 0; j < 32; ++j) {
            const int  by  = j / 4;
            const int  sub = j % 4;
            const int  code = (K[ib].qs[by] >> (2 * sub)) & 0x03;
            sum += (mean + d * c[code]) * q[i + j];
        }
    }
    return sum;
}

// T = threads per block = min(256, next_power_of_2 dividing DV)
template <int T>
static __global__ void fattn_mixed_kernel(
        const char  * q,
        const char  * k, const char * v,
        const char  * mask_data, int64_t mask_nb1, int64_t mask_nb2, int64_t mask_nb3, int64_t mask_ne2, int64_t mask_ne3,
        float * __restrict__ dst,
        int64_t nbq1, int64_t nbq2, int64_t nbq3,
        int64_t nbk1, int64_t nbk2, int64_t nbk3,
        int64_t nbv1, int64_t nbv2, int64_t nbv3,
        int64_t n_kv,
        int64_t DK, int64_t DV, int64_t N, int64_t n_head, int64_t nseq,
        float scale, float max_bias, float logit_softcap,
        int rk2, int rk3, int rv2, int rv3) {

    const int64_t ir = blockIdx.x;
    if (ir >= nseq * n_head * N) return;

    const int iq3 = ir / (n_head * N);
    const int iq2 = (ir - iq3 * n_head * N) / N;
    const int iq1 = (ir - iq3 * n_head * N - iq2 * N);

    // ALiBi slope
    const uint32_t h = iq2;
    const uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));
    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
    const float slope = (max_bias > 0.0f) ? (h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2 * (h - n_head_log2) + 1)) : 1.0f;

    const float * pq = (const float *)((const char *) q + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3));

    const int lane = threadIdx.x;
    const int slice = DV / T;
    const int j0 = lane * slice;

    float v_acc[128];  // max slice (DV=512, T=4)
    for (int j = 0; j < slice; ++j) v_acc[j] = 0.0f;

    float M = -INFINITY;
    float S = 0.0f;

    const ggml_fp16_t * mp = mask_data ? (const ggml_fp16_t *)(mask_data + iq1*mask_nb1 + (iq2%mask_ne2)*mask_nb2 + (iq3%mask_ne3)*mask_nb3) : NULL;

    const int ik2 = iq2 / rk2;  const int ik3 = iq3 / rk3;
    const int iv2 = iq2 / rv2;  const int iv3 = iq3 / rv3;

    // ---- LP tier (q2_0 K/V) ----
    for (int64_t ic = 0; ic < n_kv; ++ic) {
        const float mv = mp ? slope * __half2float(mp[ic]) : 0.0f;

        const char * kd = (const char *) k + (ic * nbk1 + ik2 * nbk2 + ik3 * nbk3);
        float s = kq_dot_q2_0_mix(kd, pq, DK) * scale;
        if (logit_softcap != 0.0f) s = logit_softcap * tanhf(s);
        s += mv;

        const float Mold = M;
        float ms = 1.0f, vs = 1.0f;
        const char * vd = (const char *) v + (ic * nbv1 + iv2 * nbv2 + iv3 * nbv3);
        if (s > M) { M = s; ms = expf(Mold - M); for (int j = 0; j < slice; ++j) v_acc[j] *= ms; }
        else       { vs = expf(s - M); }
        // dequantize q2_0 V slice: value = m + d * centroid
        for (int j = 0; j < slice; ++j) {
            const int g = (j0 + j) / 32;
            const int z = (j0 + j) % 32;
            const float mean = __half2float(((const block_q2_0 *) vd)[g].m);
            const float d   = __half2float(((const block_q2_0 *) vd)[g].d);
            const int  by  = z / 4;
            const int  sub = z % 4;
            const int  code = ((((const block_q2_0 *) vd)[g].qs[by]) >> (2 * sub)) & 0x03;
            constexpr float cc[4] = {-0.9816f, -0.4528f, 0.4528f, 0.9816f};
            const float vval = mean + d * cc[code];
            v_acc[j] += vs * vval;
        }
        S = S * ms + vs;
    }

    // No HP tier in this simplified port (n_hp = 0 → skipped)

    const float S_inv = (S == 0.0f) ? 0.0f : 1.0f / S;
    float * out = (float *)((char *) dst + (iq3 * n_head * N + iq2 * N + iq1) * DV * sizeof(float));
    for (int j = 0; j < slice; ++j) out[j0 + j] = v_acc[j] * S_inv;
}

// Host-side driver for the mixed kernel.
// Adapted from ggml_cuda_flash_attn_ext_mixed in the OSCAR fork.
static void ggml_cuda_flash_attn_ext_mixed(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    GGML_TENSOR_LOCALS(int64_t, neq, Q, ne);
    GGML_TENSOR_LOCALS(size_t,  nbq, Q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, K, ne);
    GGML_TENSOR_LOCALS(size_t,  nbk, K, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, V, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, V, nb);

    const int64_t DK = nek0;
    const int64_t DV = nev0;
    const int64_t N  = neq1;
    const int64_t n_head = neq2;
    const int64_t nseq   = neq3;

    float scale = 1.0f, max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&scale,         (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (float *) dst->op_params + 2, sizeof(float));
    if (logit_softcap != 0.0f) scale /= logit_softcap;

    const int64_t n_kv = nek1;
    const int64_t n_hp = 0;  // No HP tier in this simplified port

    const int rk2 = neq2 / nek2, rk3 = neq3 / nek3;
    const int rv2 = neq2 / nev2, rv3 = neq3 / nev3;

    // Pick thread count: DV must be divisible by T
    int T = (DV >= 256) ? 256 : (DV >= 128) ? 128 : (DV >= 64) ? 64 : 32;
    while (T > 1 && DV % T != 0) T /= 2;
    GGML_ASSERT(DV % T == 0);

    const int64_t nrows = nseq * n_head * N;
    const dim3 grid(nrows);
    const dim3 block(T);

    cudaStream_t stream = ctx.stream();

    // Dispatch by thread count (T = threads per block)
    switch (T) {
        case 256:
            fattn_mixed_kernel<256><<<grid, block, 0, stream>>>(
                (const char *) Q->data,
                (const char *) K->data, (const char *) V->data,
                mask ? (const char *) mask->data : nullptr,
                mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0, mask ? mask->nb[3] : 0,
                mask ? mask->ne[2] : 0, mask ? mask->ne[3] : 0,
                (float *) dst->data,
                nbq1, nbq2, nbq3,
                nbk1, nbk2, nbk3,
                nbv1, nbv2, nbv3,
                n_kv,
                DK, DV, N, n_head, nseq,
                scale, max_bias, logit_softcap,
                rk2, rk3, rv2, rv3);
            break;
        case 128:
            fattn_mixed_kernel<128><<<grid, block, 0, stream>>>(
                (const char *) Q->data,
                (const char *) K->data, (const char *) V->data,
                mask ? (const char *) mask->data : nullptr,
                mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0, mask ? mask->nb[3] : 0,
                mask ? mask->ne[2] : 0, mask ? mask->ne[3] : 0,
                (float *) dst->data,
                nbq1, nbq2, nbq3,
                nbk1, nbk2, nbk3,
                nbv1, nbv2, nbv3,
                n_kv,
                DK, DV, N, n_head, nseq,
                scale, max_bias, logit_softcap,
                rk2, rk3, rv2, rv3);
            break;
        case 64:
            fattn_mixed_kernel<64><<<grid, block, 0, stream>>>(
                (const char *) Q->data,
                (const char *) K->data, (const char *) V->data,
                mask ? (const char *) mask->data : nullptr,
                mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0, mask ? mask->nb[3] : 0,
                mask ? mask->ne[2] : 0, mask ? mask->ne[3] : 0,
                (float *) dst->data,
                nbq1, nbq2, nbq3,
                nbk1, nbk2, nbk3,
                nbv1, nbv2, nbv3,
                n_kv,
                DK, DV, N, n_head, nseq,
                scale, max_bias, logit_softcap,
                rk2, rk3, rv2, rv3);
            break;
        case 32:
            fattn_mixed_kernel<32><<<grid, block, 0, stream>>>(
                (const char *) Q->data,
                (const char *) K->data, (const char *) V->data,
                mask ? (const char *) mask->data : nullptr,
                mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0, mask ? mask->nb[3] : 0,
                mask ? mask->ne[2] : 0, mask ? mask->ne[3] : 0,
                (float *) dst->data,
                nbq1, nbq2, nbq3,
                nbk1, nbk2, nbk3,
                nbv1, nbv2, nbv3,
                n_kv,
                DK, DV, N, n_head, nseq,
                scale, max_bias, logit_softcap,
                rk2, rk3, rv2, rv3);
            break;
        default:
            GGML_ABORT("unexpected thread count for mixed FA kernel");
    }

    GGML_ASSERT(cudaGetLastError() == cudaSuccess);
}

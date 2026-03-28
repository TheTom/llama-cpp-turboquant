#include "common.cuh"

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const float d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const float d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const float d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

    v.x *= d;
    v.y *= d;
}

/* TurboQuant 3-bit dequantize: centroid lookup + norm multiply (no rotation).
 * QR_TURBO3 = 2 (2 elements per call), iqs ranges 0..15 for 32 elements per block. */
#define QR_TURBO3 2

static __device__ __forceinline__ void dequantize_turbo3_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_turbo3_0 * x = (const block_turbo3_0 *) vx;
    const float norm = __half2float(x[ib].norm);

    const int j0 = iqs * 2;
    const int j1 = j0 + 1;

    /* 3-bit index: lower 2 bits from qs[], upper 1 bit from signs[] */
    constexpr float centroids[8] = {
        -0.190685f, -0.117832f, -0.065717f, -0.021460f,
         0.021460f,  0.065717f,  0.117832f,  0.190685f
    };

    const uint8_t low2_0 = (x[ib].qs[j0 / 4] >> ((j0 % 4) * 2)) & 0x3;
    const uint8_t hi1_0  = (x[ib].signs[j0 / 8] >> (j0 % 8)) & 0x1;
    v.x = centroids[low2_0 | (hi1_0 << 2)] * norm;

    const uint8_t low2_1 = (x[ib].qs[j1 / 4] >> ((j1 % 4) * 2)) & 0x3;
    const uint8_t hi1_1  = (x[ib].signs[j1 / 8] >> (j1 % 8)) & 0x1;
    v.y = centroids[low2_1 | (hi1_1 << 2)] * norm;
}

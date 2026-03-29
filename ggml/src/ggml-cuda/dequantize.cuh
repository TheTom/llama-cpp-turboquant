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

/* TurboQuant turbo3: Lloyd-Max centroid dequant (float2 interface, rotated space).
 * Used by generic dequantize_block_cuda. Does NOT apply inverse WHT —
 * full dequant with inverse WHT uses dequantize_row_turbo3_0_cuda. */
#define QR_TURBO3 2

static __device__ __forceinline__ void dequantize_turbo3_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_turbo3_0 * x = (const block_turbo3_0 *) vx;
    const float norm = __half2float(x[ib].d);  /* group norm stored in d */

    const float cn[8] = {
        -0.18165533f, -0.11483849f, -0.06487426f, -0.02106513f,
         0.02106513f,  0.06487426f,  0.11483849f,  0.18165533f
    };

    const int group_idx = iqs / 4;
    const int pair_idx  = iqs % 4;
    const uint8_t * qs = x[ib].qs + group_idx * 3;

    int v0, v1;
    if (pair_idx == 0) {
        v0 = (qs[0] & 7);
        v1 = ((qs[0] >> 3) & 7);
    } else if (pair_idx == 1) {
        v0 = ((qs[0] >> 6) & 3) | ((qs[1] & 1) << 2);
        v1 = ((qs[1] >> 1) & 7);
    } else if (pair_idx == 2) {
        v0 = ((qs[1] >> 4) & 7);
        v1 = ((qs[1] >> 7) & 1) | ((qs[2] & 3) << 1);
    } else {
        v0 = ((qs[2] >> 2) & 7);
        v1 = ((qs[2] >> 5) & 7);
    }

    v.x = cn[v0] * norm;
    v.y = cn[v1] * norm;
}

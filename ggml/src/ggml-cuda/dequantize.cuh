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

/* TurboQuant turbo3: 3-bit uniform quantization dequant (float2 interface).
 * Values stored as [0,7] with offset -4. Packed 8 per 3 bytes.
 * QR_TURBO3=2: produces 2 float values per call, iqs ranges 0..15. */
#define QR_TURBO3 2

static __device__ __forceinline__ void dequantize_turbo3_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_turbo3_0 * x = (const block_turbo3_0 *) vx;
    const float d = __half2float(x[ib].d);

    const int group_idx = iqs / 4;  // which group of 8 (0..3)
    const int pair_idx  = iqs % 4;  // which pair within group (0..3)
    const uint8_t * qs = x[ib].qs + group_idx * 3;

    if (pair_idx == 0) {
        v.x = ((qs[0] & 7) - 4.0f) * d;
        v.y = (((qs[0] >> 3) & 7) - 4.0f) * d;
    } else if (pair_idx == 1) {
        v.x = ((((qs[0] >> 6) & 3) | ((qs[1] & 1) << 2)) - 4.0f) * d;
        v.y = (((qs[1] >> 1) & 7) - 4.0f) * d;
    } else if (pair_idx == 2) {
        v.x = (((qs[1] >> 4) & 7) - 4.0f) * d;
        v.y = ((((qs[1] >> 7) & 1) | ((qs[2] & 3) << 1)) - 4.0f) * d;
    } else {
        v.x = (((qs[2] >> 2) & 7) - 4.0f) * d;
        v.y = (((qs[2] >> 5) & 7) - 4.0f) * d;
    }
}

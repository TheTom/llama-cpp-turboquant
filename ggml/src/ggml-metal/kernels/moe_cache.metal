#include "common.h"
#include "dequantize.h"

// UE4M3 -> fp32, raw E4M3 value (matches the CPU ggml_ue4m3_to_fp32 without
// its * 0.5 fold; that fold pairs the x2 kvalues_mxfp4 table, while the Metal
// kvalues_mxfp4_f table is already halved).
static inline float nvfp4_ue4m3_to_fp32(uint8_t x) {
    if (x == 0 || x == 0x7F) {
        return 0.0f;
    }
    const int exp = (x >> 3) & 0xF;
    const int man = x & 0x7;
    if (exp == 0) {
        return (float)man * (1.0f / 512.0f);
    }
    const uint32_t bits = ((uint32_t)(exp + 120) << 23) | ((uint32_t)man << 20);
    return as_type<float>(bits);
}

// ============================================================================
// MoE Expert Cache matvec — reads quantized weights from a pool slab
// at per-row offsets and computes dot products with q8_1 activations.
// One thread per (hit, output_row) pair.
// Template parameter: block_t = the quantized block type.
// ============================================================================

// Per-type block dot functions. MSL type-checks every branch of a runtime if,
// so dispatch happens via overload resolution (each overload only touches its
// own block type's members). One thread per (hit, output row) pair.
// Q8_0: signed 8-bit quants, one 32-wide act block
static inline void kernel_moe_cache_mv_block_dot(
    device const block_q8_0 * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    const float d_w = (float)w_block->d;
    const float d_a = (float)a_block->d;
    int sum_q = 0;
    for (int i = 0; i < QK8_0; i++) {
        sum_q += (int)w_block->qs[i] * (int)a_block->qs[i];
    }
    sum += d_w * d_a * (float)sum_q;
}

// Q4_0: 4-bit nibbles, offset by -8, one 32-wide act block.
// qs packing (CPU reference): byte j = element j (low nibble) | element
// j+16 (high nibble), so element j pairs with act element j.
static inline void kernel_moe_cache_mv_block_dot(
    device const block_q4_0 * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    const float d_w = (float)w_block->d;
    const float d_a = (float)a_block->d;
    int sum_q = 0;
    for (int i = 0; i < 16; i++) {
        const uint8_t nibbles = w_block->qs[i];
        sum_q += ((int)(nibbles & 0xF) - 8) * (int)a_block->qs[i];
        sum_q += ((int)(nibbles >> 4)  - 8) * (int)a_block->qs[i + 16];
    }
    sum += d_w * d_a * (float)sum_q;
}

// Q4_K: 8 sub-blocks of 32 elements, each with a 6-bit scale and 6-bit min
// packed across scales[12] (get_scale_min_k4). Sub-block sb pairs with act
// block sb; formula matches the CUDA reference:
//   d_all * d8[sb] * sc6 * dot1 - d_min * d8[sb] * mn6 * dot2
// where dot1 = sum(q4*qs) and dot2 = sum(qs) over the 32 elements.
// qs packing (CPU reference): sub-block pair 2k/2k+1 shares qs[32k..32k+31],
// even sub-block in the low nibbles, odd in the high nibbles.
static inline void kernel_moe_cache_mv_block_dot(
    device const block_q4_K * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    device const uint8_t * q = w_block->qs;
    device const uint8_t * sc = w_block->scales;
    const float d_all = (float)w_block->d;
    const float d_min = (float)w_block->dmin;

    for (int sb = 0; sb < 8; sb++) {
        int sc6, mn6;
        if (sb < 4) {
            sc6 = sc[sb] & 0x3F;
            mn6 = sc[sb + 4] & 0x3F;
        } else {
            sc6 = (sc[sb + 4] & 0xF) | ((sc[sb - 4] & 0xC0) >> 2);
            mn6 = (sc[sb + 4] >> 4) | ((sc[sb] & 0xC0) >> 2);
        }
        const float dl = d_all * (float)sc6;
        const float ml = d_min * (float)mn6;
        const float d_a = (float)a_block[sb].d;

        int dot1 = 0;
        int dot2 = 0;
        for (int j = 0; j < 32; j++) {
            const uint8_t nibbles = q[(sb >> 1) * 32 + j];
            const int n0 = (sb & 1) ? (nibbles >> 4) : (nibbles & 0xF);
            const int qs = (int)a_block[sb].qs[j];
            dot1 += n0 * qs;
            dot2 += qs;
        }
        sum += d_a * (dl * (float)dot1 - ml * (float)dot2);
    }
}

// Q6_K: 16 sub-blocks of 16 elements, int8 scale per sub-block, 8 act blocks
// of 32. Element e uses weight scale scales[e/16] and act block e/32, so two
// weight sub-blocks share one act block. q6 value packing (CPU reference):
//   ql byte (e%64) + 64*(e/128), high nibble iff (e/64) is odd
//   qh byte (e%32) + 32*(e/128), bits 2*((e/32) % 4)
static inline void kernel_moe_cache_mv_block_dot(
    device const block_q6_K * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    const float d_all = (float)w_block->d;

    for (int e = 0; e < QK_K; e++) {
        const int sb = e / 16;
        const int ab = e / 32;
        const float dl = d_all * (float)w_block->scales[sb];
        const float d_a = (float)a_block[ab].d;

        const uint8_t low = w_block->ql[(e % 64) + 64 * (e / 128)];
        const uint8_t high = w_block->qh[(e % 32) + 32 * (e / 128)];
        const int q4 = ((e / 64) & 1) ? (low >> 4) : (low & 0xF);
        const int q2 = (high >> (2 * ((e / 32) % 4))) & 0x3;
        const int q6 = q4 | (q2 << 4);
        const float w_val = dl * (float)(q6 - 32);

        sum += w_val * d_a * (float)a_block[ab].qs[e % 32];
    }
}

// Q5_K: 8 sub-blocks of 32. Same scale/min packing as Q4_K (get_scale_min_k4)
// plus a per-element high bit in qh. q5 = nibble | (high << 4) with no offset
// (the min term shifts it; see vec_dot_q5_K_q8_1_impl_vmmq, which also uses
// the unshifted vl|vh). The high bit of element (sb*32 + j) sits at qh[j]
// bit (2*(sb >> 1) + (sb & 1)).
static inline void kernel_moe_cache_mv_block_dot(
    device const block_q5_K * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    device const uint8_t * q = w_block->qs;
    device const uint8_t * qh = w_block->qh;
    device const uint8_t * sc = w_block->scales;
    const float d_all = (float)w_block->d;
    const float d_min = (float)w_block->dmin;

    for (int sb = 0; sb < 8; sb++) {
        int sc6, mn6;
        if (sb < 4) {
            sc6 = sc[sb] & 0x3F;
            mn6 = sc[sb + 4] & 0x3F;
        } else {
            sc6 = (sc[sb + 4] & 0xF) | ((sc[sb - 4] & 0xC0) >> 2);
            mn6 = (sc[sb + 4] >> 4) | ((sc[sb] & 0xC0) >> 2);
        }
        const float dl = d_all * (float)sc6;
        const float ml = d_min * (float)mn6;
        const float d_a = (float)a_block[sb].d;

        int dot1 = 0;
        int dot2 = 0;
        for (int j = 0; j < 32; j++) {
            const uint8_t nibbles = q[(sb >> 1) * 32 + j];
            const int q5 = (sb & 1) ? (nibbles >> 4) : (nibbles & 0xF);
            const int high = (qh[j] >> (2 * (sb >> 1) + (sb & 1))) & 1;
            const int qs = (int)a_block[sb].qs[j];
            dot1 += (q5 | (high << 4)) * qs;
            dot2 += qs;
        }
        sum += d_a * (dl * (float)dot1 - ml * (float)dot2);
    }
}

// Q1_0: 128 elements per block, one scale, 1 bit per element (+1/-1). The
// weight block spans four 32-wide act chunks, each with its own q8_1 scale.
static inline void kernel_moe_cache_mv_block_dot(
    device const block_q1_0 * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    float s = 0.0f;
    for (int c = 0; c < 4; c++) {
        const float d_a = (float)a_block[c].d;
        int sumi = 0;
        for (int j = 0; j < 32; j++) {
            const int e = c*32 + j;
            const int bit = (w_block->qs[e >> 3] >> (e & 7)) & 1;
            sumi += (bit ? 1 : -1) * (int)a_block[c].qs[j];
        }
        s += d_a * (float)sumi;
    }
    sum += (float)w_block->d * s;
}

// Q2_0: 64 elements per block, one scale, 2 bits per element mapped
// {0,1,2,3} -> {-1,0,1,2} (CPU vec_dot_q2_0 reference). Two act chunks.
static inline void kernel_moe_cache_mv_block_dot(
    device const block_q2_0 * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    float s = 0.0f;
    for (int c = 0; c < 2; c++) {
        const float d_a = (float)a_block[c].d;
        int sumi = 0;
        for (int j = 0; j < 32; j++) {
            const int e = c*32 + j;
            const int q2 = (w_block->qs[e >> 2] >> (2 * (e & 3))) & 3;
            sumi += (q2 - 1) * (int)a_block[c].qs[j];
        }
        s += d_a * (float)sumi;
    }
    sum += (float)w_block->d * s;
}

// Q4_1: nibbles 0..15, min term via the q8_1 activation sum (CPU reference:
// (d*d8)*sumi + m*s). qs packing like Q4_0.
static inline void kernel_moe_cache_mv_block_dot(
    device const block_q4_1 * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    const float d = (float)w_block->d;
    const float m = (float)w_block->m;
    const float d_a = (float)a_block->d;
    int sumi = 0;
    for (int i = 0; i < 16; i++) {
        const uint8_t nibbles = w_block->qs[i];
        sumi += (int)(nibbles & 0xF) * (int)a_block->qs[i];
        sumi += (int)(nibbles >> 4) * (int)a_block->qs[i + 16];
    }
    sum += (d * d_a) * (float)sumi + m * (float)a_block->s;
}

// Q5_0: nibble | (high bit << 4) - 16, single scale, no min term. High bit of
// element e is bit e of the 32-bit qh field (4 bytes, CPU packing).
static inline void kernel_moe_cache_mv_block_dot(
    device const block_q5_0 * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    const float d = (float)w_block->d;
    const float d_a = (float)a_block->d;
    const uint32_t qh = (uint32_t)w_block->qh[0] | ((uint32_t)w_block->qh[1] << 8) |
                        ((uint32_t)w_block->qh[2] << 16) | ((uint32_t)w_block->qh[3] << 24);
    int sumi = 0;
    for (int i = 0; i < 16; i++) {
        const int v0 = ((int)(w_block->qs[i] & 0xF) | (((int)(qh >> i) & 1) << 4)) - 16;
        const int v1 = ((int)(w_block->qs[i] >> 4) | (((int)(qh >> (i + 16)) & 1) << 4)) - 16;
        sumi += v0 * (int)a_block->qs[i] + v1 * (int)a_block->qs[i + 16];
    }
    sum += (d * d_a) * (float)sumi;
}

// Q5_1: nibble | (high bit << 4) in 0..31, min term via the act sum (CPU
// reference: (d*d8)*sumi + m*s).
static inline void kernel_moe_cache_mv_block_dot(
    device const block_q5_1 * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    const float d = (float)w_block->d;
    const float m = (float)w_block->m;
    const float d_a = (float)a_block->d;
    const uint32_t qh = (uint32_t)w_block->qh[0] | ((uint32_t)w_block->qh[1] << 8) |
                        ((uint32_t)w_block->qh[2] << 16) | ((uint32_t)w_block->qh[3] << 24);
    int sumi = 0;
    for (int i = 0; i < 16; i++) {
        const int v0 = (int)(w_block->qs[i] & 0xF) | (((int)(qh >> i) & 1) << 4);
        const int v1 = (int)(w_block->qs[i] >> 4) | (((int)(qh >> (i + 16)) & 1) << 4);
        sumi += v0 * (int)a_block->qs[i] + v1 * (int)a_block->qs[i + 16];
    }
    sum += (d * d_a) * (float)sumi + m * (float)a_block->s;
}

// Q2_K: 16 sub-blocks of 16 elements. scales[sb] packs a 4-bit scale (low) and
// a 4-bit min (high). q2 codes are raw 0..3; the min term recenters them.
// Sub-block sb pairs with act block (ab + sb/2). value packing (CPU
// vec_dot_q2_K_q8_K reference): element e uses byte qs[32*(e/128) + e%32],
// 2-bit field 2*((e/32)%4).
static inline void kernel_moe_cache_mv_block_dot(
    device const block_q2_K * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    const float d_all = (float)w_block->d;
    const float d_min = (float)w_block->dmin;
    for (int sb = 0; sb < 16; sb++) {
        const float dl = d_all * (float)(w_block->scales[sb] & 0xF);
        const float ml = d_min * (float)(w_block->scales[sb] >> 4);
        const float d_a = (float)a_block[sb / 2].d;
        int dot1 = 0;
        int dot2 = 0;
        for (int j = 0; j < 16; j++) {
            const int e = sb*16 + j;
            const int q2 = (w_block->qs[32*(e/128) + e%32] >> (2*((e/32)%4))) & 3;
            const int qs = (int)a_block[sb / 2].qs[j + 16*(sb & 1)];
            dot1 += q2 * qs;
            dot2 += qs;
        }
        sum += d_a * (dl * (float)dot1 - ml * (float)dot2);
    }
}

// Q3_K: 16 sub-blocks of 16 elements, 6-bit scale per sub-block packed across
// scales[12] (CPU vec_dot_q3_K_q8_K reference unpacking). value = (2-bit
// code) - 4 + 4*hmask bit; scale = sc6 - 32; no min term.
static inline void kernel_moe_cache_mv_block_dot(
    device const block_q3_K * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    const float d_all = (float)w_block->d;
    for (int sb = 0; sb < 16; sb++) {
        // 6-bit scales packed across scales[12] (CUDA vec_dot_q3_K_q8_1
        // unpacking): 16 low nibbles in scales[0..7], 16x2 high bits in
        // scales[8..11].
        const int sc_low  = (w_block->scales[sb % 8] >> (4 * (sb / 8))) & 0xF;
        const int sc_high = (w_block->scales[8 + sb % 4] >> (2 * (sb / 4))) & 0x3;
        const float dl = d_all * (float)((sc_low | (sc_high << 4)) - 32);
        const float d_a = (float)a_block[sb / 2].d;
        int dot1 = 0;
        for (int j = 0; j < 16; j++) {
            const int e = sb*16 + j;
            const int low2 = (w_block->qs[32*(e/128) + e%32] >> (2*((e/32)%4))) & 3;
            const int hbit = (w_block->hmask[e%32] >> (4*(e/128) + (e/32)%4)) & 1;
            dot1 += (low2 - 4 + 4*hbit) * (int)a_block[sb / 2].qs[j + 16*(sb & 1)];
        }
        sum += d_a * dl * (float)dot1;
    }
}

// IQ2_XXS: 8 act windows of 32. Each window uses 8 bytes of qs: 4 grid
// indices (aux32_g) and 4 bytes of scale+signs (aux32_s); the 4-bit scale
// sits in the top nibble, signs are 7-bit groups (ksigns_iq2xs). scale factor
// (2*scale+1)/8 (CPU reference).
static inline void kernel_moe_cache_mv_block_dot(
    device const block_iq2_xxs * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    float s = 0.0f;
    for (int ib32 = 0; ib32 < 8; ib32++) {
        device const uint16_t * q2 = w_block->qs + 4*ib32;
        const uint32_t aux32_g = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux32_s = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        const int ls = 2*((int)((aux32_s >> 28) & 0xF)) + 1;
        const float d_a = (float)a_block[ib32].d;
        int sumi = 0;
        for (int l = 0; l < 4; l++) {
            const int idx = (int)((aux32_g >> (8*l)) & 0xFF);
            const uint8_t signs = ksigns_iq2xs[(aux32_s >> (7*l)) & 127];
            constant uint8_t * grid = (constant uint8_t *)(iq2xxs_grid + idx);
            for (int i = 0; i < 8; i++) {
                const int g = (int)grid[i];
                const int sgn = (signs >> i) & 1;
                sumi += (sgn ? -g : g) * (int)a_block[ib32].qs[8*l + i];
            }
        }
        s += d_a * (float)(sumi * ls);
    }
    sum += (float)w_block->d * (1.0f/8.0f) * s;
}

// IQ2_XS: per 32-window, 4 uint16 words hold a 9-bit grid index and 7 sign
// bits each; scales[ib32] holds two 4-bit scales (per 16-element half).
// scale factor (2*scale+1)/8 (CPU reference).
static inline void kernel_moe_cache_mv_block_dot(
    device const block_iq2_xs * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    float s = 0.0f;
    for (int ib32 = 0; ib32 < 8; ib32++) {
        const int ls1 = 2*((int)(w_block->scales[ib32] & 0xF)) + 1;
        const int ls2 = 2*((int)(w_block->scales[ib32] >> 4)) + 1;
        device const uint16_t * q2 = w_block->qs + 4*ib32;
        const float d_a = (float)a_block[ib32].d;
        int sumi1 = 0;
        int sumi2 = 0;
        for (int l = 0; l < 4; l++) {
            const uint16_t q2w = q2[l];
            const int idx = (int)(q2w & 511);
            const uint8_t signs = ksigns_iq2xs[q2w >> 9];
            constant uint8_t * grid = (constant uint8_t *)(iq2xs_grid + idx);
            int sumi = 0;
            for (int i = 0; i < 8; i++) {
                const int g = (int)grid[i];
                const int sgn = (signs >> i) & 1;
                sumi += (sgn ? -g : g) * (int)a_block[ib32].qs[8*l + i];
            }
            if (l < 2) { sumi1 += sumi; } else { sumi2 += sumi; }
        }
        s += d_a * (float)(ls1*sumi1 + ls2*sumi2);
    }
    sum += (float)w_block->d * (1.0f/8.0f) * s;
}

// IQ2_S: per 32-window, 4 grid bytes + 4 sign bytes (at qs + 32), a 2-bit high
// index from qh[ib32], and two 4-bit scales. scale factor (2*scale+1)/8.
static inline void kernel_moe_cache_mv_block_dot(
    device const block_iq2_s * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    float s = 0.0f;
    for (int ib32 = 0; ib32 < 8; ib32++) {
        const int ls1 = 2*((int)(w_block->scales[ib32] & 0xF)) + 1;
        const int ls2 = 2*((int)(w_block->scales[ib32] >> 4)) + 1;
        const uint8_t qh = w_block->qh[ib32];
        const float d_a = (float)a_block[ib32].d;
        int sumi1 = 0;
        int sumi2 = 0;
        for (int l = 0; l < 4; l++) {
            const int idx = (int)w_block->qs[4*ib32 + l] | ((qh << (8 - 2*l)) & 0x300);
            const uint8_t sg = w_block->qs[32 + 4*ib32 + l];
            constant uint8_t * grid = (constant uint8_t *)(iq2s_grid + idx);
            int sumi = 0;
            for (int i = 0; i < 8; i++) {
                const int g = (int)grid[i];
                const int sgn = (sg >> i) & 1;
                sumi += (sgn ? -g : g) * (int)a_block[ib32].qs[8*l + i];
            }
            if (l < 2) { sumi1 += sumi; } else { sumi2 += sumi; }
        }
        s += d_a * (float)(ls1*sumi1 + ls2*sumi2);
    }
    sum += (float)w_block->d * (1.0f/8.0f) * s;
}

// IQ3_XXS: per 32-window, 8 grid bytes in qs[0..63] and 4 scale+sign bytes in
// qs[64..95]. Two 4-byte grid entries and 7-bit sign groups per l; scale
// factor (2*scale+1)/4 (CPU reference).
static inline void kernel_moe_cache_mv_block_dot(
    device const block_iq3_xxs * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    float s = 0.0f;
    for (int ib32 = 0; ib32 < 8; ib32++) {
        const uint32_t aux32 = (uint32_t)w_block->qs[64 + 4*ib32 + 0] |
                               ((uint32_t)w_block->qs[64 + 4*ib32 + 1] << 8) |
                               ((uint32_t)w_block->qs[64 + 4*ib32 + 2] << 16) |
                               ((uint32_t)w_block->qs[64 + 4*ib32 + 3] << 24);
        const int ls = 2*((int)((aux32 >> 28) & 0xF)) + 1;
        const float d_a = (float)a_block[ib32].d;
        int sumi = 0;
        for (int l = 0; l < 4; l++) {
            constant uint8_t * grid1 = (constant uint8_t *)(iq3xxs_grid + w_block->qs[8*ib32 + 2*l + 0]);
            constant uint8_t * grid2 = (constant uint8_t *)(iq3xxs_grid + w_block->qs[8*ib32 + 2*l + 1]);
            const uint8_t signs = ksigns_iq2xs[(aux32 >> (7*l)) & 127];
            for (int i = 0; i < 4; i++) {
                const int g1 = (int)grid1[i];
                const int g2 = (int)grid2[i];
                const int s1 = (signs >> i) & 1;
                const int s2 = (signs >> (i + 4)) & 1;
                sumi += (s1 ? -g1 : g1) * (int)a_block[ib32].qs[8*l + i];
                sumi += (s2 ? -g2 : g2) * (int)a_block[ib32].qs[8*l + i + 4];
            }
        }
        s += d_a * (float)(sumi * ls);
    }
    sum += (float)w_block->d * (1.0f/4.0f) * s;
}

// IQ3_S: per 32-window, 8 grid bytes, 4 sign bytes, qh[ib32] for the 9th
// index bit, and a 4-bit scale. scale factor 1 + 2*scale (no /8; CPU
// reference).
static inline void kernel_moe_cache_mv_block_dot(
    device const block_iq3_s * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    float s = 0.0f;
    for (int ib32 = 0; ib32 < 8; ib32++) {
        const int ls = 2*((int)(w_block->scales[ib32 >> 1] >> (4 * (ib32 & 1))) & 0xF) + 1;
        const uint8_t qh = w_block->qh[ib32];
        const float d_a = (float)a_block[ib32].d;
        int sumi = 0;
        for (int l = 0; l < 4; l++) {
            constant uint8_t * grid1 = (constant uint8_t *)(iq3s_grid + (w_block->qs[8*ib32 + 2*l + 0] | ((qh << (8 - 2*l)) & 256)));
            constant uint8_t * grid2 = (constant uint8_t *)(iq3s_grid + (w_block->qs[8*ib32 + 2*l + 1] | ((qh << (7 - 2*l)) & 256)));
            const uint8_t sg = w_block->signs[4*ib32 + l];
            for (int i = 0; i < 4; i++) {
                const int g1 = (int)grid1[i];
                const int g2 = (int)grid2[i];
                const int s1 = (sg >> i) & 1;
                const int s2 = (sg >> (i + 4)) & 1;
                sumi += (s1 ? -g1 : g1) * (int)a_block[ib32].qs[8*l + i];
                sumi += (s2 ? -g2 : g2) * (int)a_block[ib32].qs[8*l + i + 4];
            }
        }
        s += d_a * (float)(sumi * ls);
    }
    sum += (float)w_block->d * s;
}

// IQ1_S: per 32-window, 4 grid bytes in qs, qh[ib] supplies a 3-bit high index
// and the scale/delta. grid values are 4-bit per byte (iq1s_grid_gpu). The
// min term uses the q8_1 activation sum (CUDA reference).
static inline void kernel_moe_cache_mv_block_dot(
    device const block_iq1_s * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    float s = 0.0f;
    for (int ib = 0; ib < 8; ib++) {
        const uint16_t qh = w_block->qh[ib];
        const int ls = 2*((qh >> 12) & 7) + 1;
        const float delta = (qh & 0x8000) ? -1.0f - IQ1S_DELTA : -1.0f + IQ1S_DELTA;
        const float d_a = (float)a_block[ib].d;
        int sumi = 0;
        for (int l = 0; l < 4; l++) {
            const int idx = (int)w_block->qs[4*ib + l] | (((qh >> (3*l)) & 7) << 8);
            constant uint32_t * e = iq1s_grid_gpu + idx;
            const uint32_t ev = *e;
            for (int i = 0; i < 4; i++) {
                const int g0 = (int)((ev >> (8*i)) & 0xF);
                const int g1 = (int)((ev >> (8*i + 4)) & 0xF);
                sumi += g0 * (int)a_block[ib].qs[8*l + i];
                sumi += g1 * (int)a_block[ib].qs[8*l + i + 4];
            }
        }
        s += (float)ls * (d_a * (float)sumi + (float)a_block[ib].s * delta);
    }
    sum += (float)w_block->d * s;
}

// IQ1_M: no block d; the scale is reconstructed from the four uint16 scales
// words. Per 32-window: 4 grid bytes, 2 qh bytes (deltas), two 6-bit scales.
// delta terms use the raw q8 sum (CUDA reference), act scale from ds.x.
static inline void kernel_moe_cache_mv_block_dot(
    device const block_iq1_m * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    device const uint16_t * sc = (device const uint16_t *)w_block->scales;
    float s = 0.0f;
    for (int ib = 0; ib < 8; ib++) {
        const uint8_t qh0 = w_block->qh[2*ib];
        const uint8_t qh1 = w_block->qh[2*ib + 1];
        const float delta0 = (qh0 & 0x08) ? -1.0f - IQ1M_DELTA : -1.0f + IQ1M_DELTA;
        const float delta1 = (qh0 & 0x80) ? -1.0f - IQ1M_DELTA : -1.0f + IQ1M_DELTA;
        const float delta2 = (qh1 & 0x08) ? -1.0f - IQ1M_DELTA : -1.0f + IQ1M_DELTA;
        const float delta3 = (qh1 & 0x80) ? -1.0f - IQ1M_DELTA : -1.0f + IQ1M_DELTA;
        const uint16_t sc16 = sc[ib >> 1];
        const int ls1 = 2*((sc16 >> (6*(ib & 1) + 0)) & 7) + 1;
        const int ls2 = 2*((sc16 >> (6*(ib & 1) + 3)) & 7) + 1;
        const float d_a = (float)a_block[ib].d;
        int sumi0 = 0;
        int sumi1 = 0;
        float sumf0 = 0.0f;
        float sumf1 = 0.0f;
        for (int l = 0; l < 4; l++) {
            const uint8_t qhl = (l < 2) ? qh0 : qh1;
            const int idx = (int)w_block->qs[4*ib + l] | ((qhl << (8 - 4*(l % 2))) & 0x700);
            const float delta = l == 0 ? delta0 : l == 1 ? delta1 : l == 2 ? delta2 : delta3;
            constant uint32_t * e = iq1s_grid_gpu + idx;
            const uint32_t ev = *e;
            int sumi = 0;
            int sumy = 0;
            for (int i = 0; i < 4; i++) {
                const int g0 = (int)((ev >> (8*i)) & 0xF);
                const int g1 = (int)((ev >> (8*i + 4)) & 0xF);
                sumi += g0 * (int)a_block[ib].qs[8*l + i] + g1 * (int)a_block[ib].qs[8*l + i + 4];
                sumy += (int)a_block[ib].qs[8*l + i] + (int)a_block[ib].qs[8*l + i + 4];
            }
            if (l < 2) { sumi0 += sumi; sumf0 += delta * (float)sumy; }
            else       { sumi1 += sumi; sumf1 += delta * (float)sumy; }
        }
        s += d_a * (((float)sumi0 + sumf0) * (float)ls1 + ((float)sumi1 + sumf1) * (float)ls2);
    }
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    sum += (float)scale.f16 * s;
}

// IQ4_NL: 32 elements per block, one scale, 4-bit non-linear codes
// (kvalues_iq4nl), 2 per byte (CPU vec_dot_iq4_nl_q8_1 reference).
static inline void kernel_moe_cache_mv_block_dot(
    device const block_iq4_nl * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    const float d_a = (float)a_block[0].d;
    int sumi = 0;
    for (int j = 0; j < 16; j++) {
        const uint8_t byte = w_block->qs[j];
        sumi += (int)kvalues_iq4nl_f[byte & 0xF] * (int)a_block[0].qs[j];
        sumi += (int)kvalues_iq4nl_f[byte >> 4] * (int)a_block[0].qs[j + 16];
    }
    sum += (float)w_block->d * d_a * (float)sumi;
}

// IQ4_XS: per 32-window, 16 qs bytes and a 6-bit scale (4 bits from scales_l,
// 2 from scales_h), offset by -32 (CPU reference).
static inline void kernel_moe_cache_mv_block_dot(
    device const block_iq4_xs * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    float s = 0.0f;
    for (int ib = 0; ib < 8; ib++) {
        const int ls = ((w_block->scales_l[ib >> 1] >> (4 * (ib & 1))) & 0xF) |
                       (((w_block->scales_h >> (2*ib)) & 3) << 4);
        const float dl = (float)(ls - 32);
        const float d_a = (float)a_block[ib].d;
        int sumi = 0;
        for (int j = 0; j < 16; j++) {
            const uint8_t byte = w_block->qs[16*ib + j];
            sumi += (int)kvalues_iq4nl_f[byte & 0xF] * (int)a_block[ib].qs[j];
            sumi += (int)kvalues_iq4nl_f[byte >> 4] * (int)a_block[ib].qs[j + 16];
        }
        s += d_a * dl * (float)sumi;
    }
    sum += (float)w_block->d * s;
}

// MXFP4: 32 elements, one E8M0 exponent, 2 E2M1 values per byte. The Metal
// kvalues_mxfp4_f table already holds the half-step values (CPU/CUDA
// reference: e8m0 * 0.5 * 2*E2M1 == e8m0 * E2M1).
static inline void kernel_moe_cache_mv_block_dot(
    device const block_mxfp4 * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    const float d = e8m0_to_fp32(w_block->e) * (float)a_block->d;
    int sumi = 0;
    for (int j = 0; j < 16; j++) {
        const uint8_t byte = w_block->qs[j];
        sumi += (int)kvalues_mxfp4_f[byte & 0xF] * (int)a_block->qs[j];
        sumi += (int)kvalues_mxfp4_f[byte >> 4] * (int)a_block->qs[j + 16];
    }
    sum += d * (float)sumi;
}

// NVFP4: 64 elements = 4 sub-blocks of 16, each with a UE4M3 scale; 2 E2M1
// values per byte (CPU vec_dot_nvfp4 reference packing).
static inline void kernel_moe_cache_mv_block_dot(
    device const block_nvfp4 * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    for (int s = 0; s < 4; s++) {
        const float d = nvfp4_ue4m3_to_fp32(w_block->d[s]) * (float)a_block[s >> 1].d;
        const int off = (s & 1) * 16;
        int sumi = 0;
        for (int j = 0; j < 8; j++) {
            const uint8_t byte = w_block->qs[8*s + j];
            sumi += (int)kvalues_mxfp4_f[byte & 0xF] * (int)a_block[s >> 1].qs[off + j];
            sumi += (int)kvalues_mxfp4_f[byte >> 4] * (int)a_block[s >> 1].qs[off + j + 8];
        }
        sum += d * (float)sumi;
    }
}

// Host binds all six scalars as one packed blob at buffer index 4
// (ggml_metal_encoder_set_bytes(enc, args, sizeof(args), 4)), so the kernel must
// take a single struct there. Six separate `constant int64_t &` parameters would
// claim indices 4..9, leaving every field after n_in unbound.
struct moe_cache_mv_args {
    int64_t n_in;
    int64_t n_out;
    int64_t expert_stride;
    int64_t row_stride;
    int64_t n_hits;
    int64_t padded_n_in;
};

template <typename block_t, short qk>
kernel void kernel_moe_cache_mv_generic(
    device const char * slab,
    device const int32_t * ids,
    device const block_q8_1 * act_q8,
    device float * dst,
    constant moe_cache_mv_args & args,
    uint id [[thread_position_in_grid]]
) {
    const int64_t n_in          = args.n_in;
    const int64_t n_out         = args.n_out;
    const int64_t expert_stride = args.expert_stride;
    const int64_t row_stride    = args.row_stride;
    const int64_t n_hits        = args.n_hits;
    const int64_t padded_n_in   = args.padded_n_in;
    const int64_t hit = (int64_t)id / n_out;
    const int64_t row = (int64_t)id - hit * n_out;

    if (hit >= n_hits) return;

    const int slot = ids[hit];
    if (slot < 0) {
        dst[hit * n_out + row] = 0.0f;
        return;
    }

    device const block_t * w_row = (device const block_t *)(slab + (int64_t)slot * expert_stride + row * row_stride);
    device const block_q8_1 * act = act_q8 + hit * (padded_n_in / QK8_1);

    const int nb = (int)(n_in / qk);
    float sum = 0.0f;
    for (int ib = 0; ib < nb; ib++) {
        // a 256-column K-type weight block consumes qk / QK8_1 act blocks
        kernel_moe_cache_mv_block_dot(&w_row[ib], &act[ib * (qk / QK8_1)], sum);
    }
    dst[hit * n_out + row] = sum;
}

// Per-type instantiations with host_name for runtime dispatch

template [[host_name("kernel_moe_cache_mv_q8_0_f32")]]
kernel void kernel_moe_cache_mv_generic<block_q8_0, QK8_0>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

template [[host_name("kernel_moe_cache_mv_q4_0_f32")]]
kernel void kernel_moe_cache_mv_generic<block_q4_0, QK4_0>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

template [[host_name("kernel_moe_cache_mv_q4_K_f32")]]
kernel void kernel_moe_cache_mv_generic<block_q4_K, QK_K>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

template [[host_name("kernel_moe_cache_mv_q6_K_f32")]]
kernel void kernel_moe_cache_mv_generic<block_q6_K, QK_K>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

template [[host_name("kernel_moe_cache_mv_q5_K_f32")]]
kernel void kernel_moe_cache_mv_generic<block_q5_K, QK_K>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

template [[host_name("kernel_moe_cache_mv_q1_0_f32")]]
kernel void kernel_moe_cache_mv_generic<block_q1_0, QK1_0>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

template [[host_name("kernel_moe_cache_mv_q2_0_f32")]]
kernel void kernel_moe_cache_mv_generic<block_q2_0, QK2_0>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

template [[host_name("kernel_moe_cache_mv_q4_1_f32")]]
kernel void kernel_moe_cache_mv_generic<block_q4_1, QK4_1>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

template [[host_name("kernel_moe_cache_mv_q5_0_f32")]]
kernel void kernel_moe_cache_mv_generic<block_q5_0, QK5_0>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

template [[host_name("kernel_moe_cache_mv_q5_1_f32")]]
kernel void kernel_moe_cache_mv_generic<block_q5_1, QK5_1>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

template [[host_name("kernel_moe_cache_mv_q2_K_f32")]]
kernel void kernel_moe_cache_mv_generic<block_q2_K, QK_K>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

template [[host_name("kernel_moe_cache_mv_q3_K_f32")]]
kernel void kernel_moe_cache_mv_generic<block_q3_K, QK_K>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

template [[host_name("kernel_moe_cache_mv_iq2_xxs_f32")]]
kernel void kernel_moe_cache_mv_generic<block_iq2_xxs, QK_K>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

template [[host_name("kernel_moe_cache_mv_iq2_xs_f32")]]
kernel void kernel_moe_cache_mv_generic<block_iq2_xs, QK_K>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

template [[host_name("kernel_moe_cache_mv_iq2_s_f32")]]
kernel void kernel_moe_cache_mv_generic<block_iq2_s, QK_K>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

template [[host_name("kernel_moe_cache_mv_iq3_xxs_f32")]]
kernel void kernel_moe_cache_mv_generic<block_iq3_xxs, QK_K>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

template [[host_name("kernel_moe_cache_mv_iq3_s_f32")]]
kernel void kernel_moe_cache_mv_generic<block_iq3_s, QK_K>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

template [[host_name("kernel_moe_cache_mv_iq1_s_f32")]]
kernel void kernel_moe_cache_mv_generic<block_iq1_s, QK_K>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

template [[host_name("kernel_moe_cache_mv_iq1_m_f32")]]
kernel void kernel_moe_cache_mv_generic<block_iq1_m, QK_K>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

template [[host_name("kernel_moe_cache_mv_iq4_nl_f32")]]
kernel void kernel_moe_cache_mv_generic<block_iq4_nl, QK4_NL>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

template [[host_name("kernel_moe_cache_mv_iq4_xs_f32")]]
kernel void kernel_moe_cache_mv_generic<block_iq4_xs, QK_K>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

template [[host_name("kernel_moe_cache_mv_mxfp4_f32")]]
kernel void kernel_moe_cache_mv_generic<block_mxfp4, QK_MXFP4>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

template [[host_name("kernel_moe_cache_mv_nvfp4_f32")]]
kernel void kernel_moe_cache_mv_generic<block_nvfp4, QK_NVFP4>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

/*
 * TurboQuant SYCL helpers for KV cache compression
 * Ported from ggml-cuda/turbo-quant.cuh
 * Based on: arXiv 2504.19874 (ICLR 2026)
 *
 * Implements constants and device functions for:
 *   GGML_TYPE_TURBO2_0 (2-bit PolarQuant)
 *   GGML_TYPE_TURBO3_0 (3-bit PolarQuant)
 *   GGML_TYPE_TURBO4_0 (4-bit PolarQuant)
 */

#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>
#include "common.hpp"
#include "../ggml-common.h"

#if !TURBO4_USE_4BIT
#error "SYCL TurboQuant currently assumes TURBO4_USE_4BIT=1 (4-bit PolarQuant turbo4_0 layout)"
#endif

// ---- 2-bit centroids (Lloyd-Max for N(0, 1/128)) ----

constexpr float TURBO_CENTROIDS_2BIT[4] = {
    -0.133462f, -0.039994f, 0.039994f, 0.133462f
};

constexpr float TURBO_MID_2BIT[3] = {
    -0.086728f, 0.0f, 0.086728f
};

// ---- 3-bit centroids (Lloyd-Max for N(0, 1/128)) ----

constexpr float TURBO_CENTROIDS_3BIT[8] = {
    -0.190207f, -0.118786f, -0.066822f, -0.021663f,
     0.021663f,  0.066822f,  0.118786f,  0.190207f
};

constexpr float TURBO_MID_3BIT[7] = {
    -0.154496f, -0.092804f, -0.044243f, 0.0f,
     0.044243f,  0.092804f,  0.154496f
};

// ---- 4-bit centroids (Lloyd-Max for N(0, 1/128)) ----

constexpr float TURBO_CENTROIDS_4BIT[16] = {
    -0.241529f, -0.182877f, -0.143016f, -0.111036f,
    -0.083292f, -0.058050f, -0.034299f, -0.011349f,
     0.011349f,  0.034299f,  0.058050f,  0.083292f,
     0.111036f,  0.143016f,  0.182877f,  0.241529f
};

constexpr float TURBO_MID_4BIT[15] = {
    -0.212203f, -0.162947f, -0.127026f, -0.097164f,
    -0.070671f, -0.046174f, -0.022824f,  0.000000f,
     0.022824f,  0.046174f,  0.070671f,  0.097164f,
     0.127026f,  0.162947f,  0.212203f
};

// ---- WHT sign arrays (seed=42, 64 elements) ----

static constexpr float TURBO_WHT_SIGNS1_64[64] = {
    -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f
};

static constexpr float TURBO_WHT_SIGNS2_64[64] = {
    1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f
};

// ---- WHT sign arrays (seed=42, 128 elements) ----

constexpr float TURBO_WHT_SIGNS1[128] = {
    -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f,
    -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f
};

constexpr float TURBO_WHT_SIGNS2[128] = {
    1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f,
    -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f
};

// ---- Nearest centroid functions ----

inline uint8_t turbo_nearest_centroid_2bit(float val) {
    if      (val < TURBO_MID_2BIT[0]) return 0;
    else if (val < TURBO_MID_2BIT[1]) return 1;
    else if (val < TURBO_MID_2BIT[2]) return 2;
    else                              return 3;
}

inline uint8_t turbo_nearest_centroid_3bit(float val) {
    if      (val < TURBO_MID_3BIT[0]) return 0;
    else if (val < TURBO_MID_3BIT[1]) return 1;
    else if (val < TURBO_MID_3BIT[2]) return 2;
    else if (val < TURBO_MID_3BIT[3]) return 3;
    else if (val < TURBO_MID_3BIT[4]) return 4;
    else if (val < TURBO_MID_3BIT[5]) return 5;
    else if (val < TURBO_MID_3BIT[6]) return 6;
    else                              return 7;
}

inline uint8_t turbo_nearest_centroid_4bit(float val) {
    if      (val < TURBO_MID_4BIT[ 0]) return  0;
    else if (val < TURBO_MID_4BIT[ 1]) return  1;
    else if (val < TURBO_MID_4BIT[ 2]) return  2;
    else if (val < TURBO_MID_4BIT[ 3]) return  3;
    else if (val < TURBO_MID_4BIT[ 4]) return  4;
    else if (val < TURBO_MID_4BIT[ 5]) return  5;
    else if (val < TURBO_MID_4BIT[ 6]) return  6;
    else if (val < TURBO_MID_4BIT[ 7]) return  7;
    else if (val < TURBO_MID_4BIT[ 8]) return  8;
    else if (val < TURBO_MID_4BIT[ 9]) return  9;
    else if (val < TURBO_MID_4BIT[10]) return 10;
    else if (val < TURBO_MID_4BIT[11]) return 11;
    else if (val < TURBO_MID_4BIT[12]) return 12;
    else if (val < TURBO_MID_4BIT[13]) return 13;
    else if (val < TURBO_MID_4BIT[14]) return 14;
    else                               return 15;
}

// ---- Inline dequant helpers ----

inline float turbo2_dequant_element(const block_turbo2_0 * __restrict__ x, int j, float norm) {
    uint8_t idx = (x->qs[j / 4] >> ((j % 4) * 2)) & 0x3;
    return TURBO_CENTROIDS_2BIT[idx] * norm;
}

inline float turbo3_dequant_element(const block_turbo3_0 * __restrict__ x, int j, float norm) {
    uint8_t low2 = (x->qs[j / 4] >> ((j % 4) * 2)) & 0x3;
    uint8_t hi1  = (x->signs[j / 8] >> (j % 8)) & 0x1;
    uint8_t idx  = low2 | (hi1 << 2);
    return TURBO_CENTROIDS_3BIT[idx] * norm;
}

inline float turbo4_dequant_element(const block_turbo4_0 * __restrict__ x, int j, float norm) {
    uint8_t idx = (x->qs[j / 2] >> ((j % 2) * 4)) & 0xF;
    return TURBO_CENTROIDS_4BIT[idx] * norm;
}


/*
 * TurboQuant CUDA kernels for KV cache compression
 * Based on: arXiv 2504.19874 (ICLR 2026)
 *
 * Implements GGML_TYPE_TURBO3_0 (3-bit PolarQuant, block size 32)
 * Constants, WHT rotation, quantize/dequantize device functions.
 */

#pragma once

#include "common.cuh"
#include "turbo-innerq.cuh"
#include <cstdlib>
#include <cmath>

// ---- Quantization ratios for dequantize_block template ----
#define QR_TURBO3 1  // Each dequantize call produces 2 consecutive elements (like q8_0)
#define QR_TURBO2 1  // Each dequantize call produces 2 consecutive elements (like q8_0)
#define QR_TURBO4 1  // Each dequantize call produces 2 consecutive elements (like q8_0)
#define QR_TQ6    1  // Each dequantize call produces 2 consecutive elements (like q8_0)
#define QR_TQ5    1  // Each dequantize call produces 2 consecutive elements (like q8_0)

// ---- 2-bit centroids (Lloyd-Max for N(0, 1/128)) ----

static __constant__ float TURBO_CENTROIDS_2BIT[4] = {
    -0.133462f, -0.039994f, 0.039994f, 0.133462f
};

static __constant__ float TURBO_MID_2BIT[3] = {
    -0.086728f, 0.0f, 0.086728f
};

// ---- 3-bit centroids (Lloyd-Max for N(0, 1/128)) ----

static __constant__ float TURBO_CENTROIDS_3BIT[8] = {
    -0.190207f, -0.118786f, -0.066822f, -0.021663f,
     0.021663f,  0.066822f,  0.118786f,  0.190207f
};

// ---- Midpoints for nearest centroid lookup ----

static __constant__ float TURBO_MID_3BIT[7] = {
    -0.154496f, -0.092804f, -0.044243f, 0.0f,
     0.044243f,  0.092804f,  0.154496f
};

// ---- WHT sign arrays (seed=42) ----

static __constant__ float TURBO_WHT_SIGNS1[128] = {
    -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f,
    -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f
};

static __constant__ float TURBO_WHT_SIGNS2[128] = {
    1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f,
    -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f
};

// ---- Packed sign bits, same signs as TURBO_WHT_SIGNS1/2 ----
//
// Element e is bit (e & 31) of word (e >> 5); a set bit means -1.0f.
// Reading SIGNS1[t] makes a warp touch 32 constant addresses, which the
// constant cache serializes. Packed, a lane owning 4 elements reads 1 word.
static __constant__ unsigned TURBO_WHT_SIGNBITS1[4] = {
    0xE46A0359u, 0x42F85949u, 0xA4C63BBBu, 0x49F250A9u
};

static __constant__ unsigned TURBO_WHT_SIGNBITS2[4] = {
    0x16ACEE90u, 0x3F628FDCu, 0xB7A5357Au, 0xBEA56562u
};

// ---- 64-element WHT sign arrays (first 64 of the 128-element arrays) ----

static __constant__ float TURBO_WHT_SIGNS1_64[64] = {
    -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f
};

static __constant__ float TURBO_WHT_SIGNS2_64[64] = {
    1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f
};

// ---- Fast Walsh-Hadamard Transform (in-place, normalized) ----
// O(n log n) = 896 ops for n=128

static __device__ __forceinline__ void turbo_fwht_128(float * x) {
    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j];
                float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
    const float inv_sqrt_128 = 0.08838834764831845f;
    for (int i = 0; i < 128; i++) {
        x[i] *= inv_sqrt_128;
    }
}

// ---- Fast Walsh-Hadamard Transform for 64-element groups ----
// O(n log n) = 384 ops for n=64

static __device__ __forceinline__ void turbo_fwht_64(float * x) {
    for (int h = 1; h < 64; h *= 2) {
        for (int i = 0; i < 64; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j];
                float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
    const float inv_sqrt_64 = 0.125f;
    for (int i = 0; i < 64; i++) {
        x[i] *= inv_sqrt_64;
    }
}

// ---- Forward rotation: signs1 → FWHT → signs2 ----

static __device__ __forceinline__ void turbo_rotate_forward(float * x) {
    for (int i = 0; i < 128; i++) x[i] *= TURBO_WHT_SIGNS1[i];
    turbo_fwht_128(x);
    for (int i = 0; i < 128; i++) x[i] *= TURBO_WHT_SIGNS2[i];
}

// ---- Forward rotation for 64-element groups ----

static __device__ __forceinline__ void turbo_rotate_forward_64(float * x) {
    for (int i = 0; i < 64; i++) x[i] *= TURBO_WHT_SIGNS1_64[i];
    turbo_fwht_64(x);
    for (int i = 0; i < 64; i++) x[i] *= TURBO_WHT_SIGNS2_64[i];
}

// ---- InnerQ per-channel equalization ----
// Equalizes K channel variances before WHT rotation to reduce quantization error.
// Enabled via TURBO_INNERQ=N env var (N = calibration token count).
// Math: <Q/s, s*K> = <Q, K> preserves dot products.
// INNERQ_MAX_CHANNELS is defined in turbo-innerq.cuh

static __device__ float d_innerq_scale[INNERQ_MAX_CHANNELS];
static __device__ float d_innerq_scale_inv[INNERQ_MAX_CHANNELS];
static __device__ float d_innerq_sq_accum[INNERQ_MAX_CHANNELS];
static __device__ int   d_innerq_count;
static __device__ int   d_innerq_active;       // 0 = scales are identity, 1 = scales applied
static __device__ int   d_innerq_calibrating;  // 1 = accumulating K² stats

static int  innerq_enabled       = 0;  // host: 0=off, 1=calibrating, 2=active
static int  innerq_target_tokens = 0;
static float innerq_strength     = 0.5f;
static bool  innerq_initialized  = false;

// Host: read TURBO_INNERQ env, start calibration if enabled
static void turbo_innerq_init(void) {
    if (innerq_initialized) return;
    innerq_initialized = true;

    const char * env = getenv("TURBO_INNERQ");
    if (!env || atoi(env) <= 0) {
        innerq_enabled = 0;
        return;
    }
    innerq_target_tokens = atoi(env);
    innerq_enabled = 1;  // calibrating

    const char * env_str = getenv("TURBO_INNERQ_STRENGTH");
    if (env_str) innerq_strength = atof(env_str);
    if (innerq_strength <= 0.0f || innerq_strength > 1.0f) innerq_strength = 0.5f;

    // Zero accumulators and set calibrating flag on device
    float zeros[INNERQ_MAX_CHANNELS] = {0};
    int zero = 0, one = 1;
    CUDA_CHECK(cudaMemcpyToSymbol(d_innerq_sq_accum, zeros, sizeof(zeros)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_innerq_count, &zero, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_innerq_active, &zero, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_innerq_calibrating, &one, sizeof(int)));

    GGML_LOG_INFO("%s: InnerQ calibration started (target=%d tokens, strength=%.2f)\n",
                   __func__, innerq_target_tokens, innerq_strength);
}

// Host: finalize calibration — compute scales, upload, activate
static void turbo_innerq_finalize(int group_size) {
    // Read accumulators from device
    float sq_accum[INNERQ_MAX_CHANNELS];
    int count = 0;
    CUDA_CHECK(cudaMemcpyFromSymbol(sq_accum, d_innerq_sq_accum, group_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpyFromSymbol(&count, d_innerq_count, sizeof(int)));

    if (count <= 0) {
        GGML_LOG_WARN("%s: InnerQ calibration got 0 tokens, disabling\n", __func__);
        innerq_enabled = 0;
        int zero = 0;
        CUDA_CHECK(cudaMemcpyToSymbol(d_innerq_calibrating, &zero, sizeof(int)));
        return;
    }

    // Compute per-channel RMS
    float rms[INNERQ_MAX_CHANNELS];
    float mean_rms = 0.0f;
    float max_ratio = 0.0f, min_ratio = 1e30f;
    for (int i = 0; i < group_size; i++) {
        rms[i] = sqrtf(sq_accum[i] / (float)count);
        mean_rms += rms[i];
    }
    mean_rms /= (float)group_size;

    // Compute scale[i] = (mean_rms / channel_rms[i])^strength, clamp to [0.5, 2.0]
    float scale[INNERQ_MAX_CHANNELS];
    float scale_inv[INNERQ_MAX_CHANNELS];
    for (int i = 0; i < group_size; i++) {
        float ratio = (rms[i] > 1e-10f) ? (mean_rms / rms[i]) : 1.0f;
        float s = powf(ratio, innerq_strength);
        if (s < 0.5f) s = 0.5f;
        if (s > 2.0f) s = 2.0f;
        scale[i] = s;
        scale_inv[i] = 1.0f / s;
        if (ratio > max_ratio) max_ratio = ratio;
        if (ratio < min_ratio) min_ratio = ratio;
    }

    // Auto-skip if max channel ratio < 1.2 (already balanced)
    if (max_ratio < 1.2f && min_ratio > (1.0f / 1.2f)) {
        GGML_LOG_INFO("%s: InnerQ auto-disabled (channels already balanced, max_ratio=%.3f)\n",
                       __func__, max_ratio);
        innerq_enabled = 0;
        int zero = 0;
        CUDA_CHECK(cudaMemcpyToSymbol(d_innerq_calibrating, &zero, sizeof(int)));
        return;
    }

    // Stop calibrating, upload scales, activate
    int zero = 0, one = 1;
    CUDA_CHECK(cudaMemcpyToSymbol(d_innerq_calibrating, &zero, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_innerq_scale, scale, group_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_innerq_scale_inv, scale_inv, group_size * sizeof(float)));
    CUDA_CHECK(cudaDeviceSynchronize());  // ensure scales are visible before activating
    CUDA_CHECK(cudaMemcpyToSymbol(d_innerq_active, &one, sizeof(int)));

    innerq_enabled = 2;  // active

    // Publish scale_inv to shared host state for cross-TU tensor update
    turbo_innerq_publish(scale_inv, group_size);

    GGML_LOG_INFO("%s: InnerQ finalized (%d tokens, max_ratio=%.3f, min_ratio=%.3f)\n",
                   __func__, count, max_ratio, min_ratio);
}

// Host: called before each set_rows kernel launch
static void turbo_innerq_check_finalize(int group_size, int64_t ne00) {
    if (!innerq_initialized) {
        turbo_innerq_init();
    }
    if (innerq_enabled == 0) return;

    // InnerQ only works when each WHT group = one head (group_size == head_dim).
    // For standard models: ne00 = n_heads * head_dim, group_size = head_dim → ne00 % group_size == 0, fine.
    // For non-standard models (head_dim > group_size, e.g. GLM 576 → 64-group):
    //   ne00 = head_dim (single head), group_size = 64, ne00/group_size = 9 groups per head → WRONG.
    // Detect: if ne00 / group_size doesn't divide evenly into standard head counts (1,2,4,8,16,32,64,128),
    // it's likely multi-group-per-head. Simpler check: group_size < 128 means head_dim > 128.
    const bool multi_group_per_head = (group_size < 128);  // 64-group → head_dim > 128, multi-group
    if (multi_group_per_head) {
        if (innerq_enabled == 1) {
            GGML_LOG_WARN("%s: InnerQ disabled (ne00=%lld != group_size=%d, multi-group heads)\n",
                           __func__, (long long)ne00, group_size);
            innerq_enabled = 0;
            int zero = 0;
            CUDA_CHECK(cudaMemcpyToSymbol(d_innerq_calibrating, &zero, sizeof(int)));
        }
        return;
    }

    // Check if calibration is complete
    if (innerq_enabled == 1) {
        int count = 0;
        CUDA_CHECK(cudaMemcpyFromSymbol(&count, d_innerq_count, sizeof(int)));
        if (count >= innerq_target_tokens) {
            turbo_innerq_finalize(group_size);
        }
    }
}

// Host: check if InnerQ is currently active (finalized)
static bool turbo_innerq_is_active(void) {
    return innerq_enabled == 2;
}

// ---- 4-bit centroids (Lloyd-Max for N(0, 1/128)) ----

static __constant__ float TURBO_CENTROIDS_4BIT[16] = {
    -0.241529f, -0.182877f, -0.143016f, -0.111036f,
    -0.083292f, -0.058050f, -0.034299f, -0.011349f,
     0.011349f,  0.034299f,  0.058050f,  0.083292f,
     0.111036f,  0.143016f,  0.182877f,  0.241529f
};

// ---- Midpoints for nearest 4-bit centroid lookup ----

static __constant__ float TURBO_MID_4BIT[15] = {
    -0.212203f, -0.162947f, -0.127026f, -0.097164f,
    -0.070671f, -0.046174f, -0.022824f,  0.000000f,
     0.022824f,  0.046174f,  0.070671f,  0.097164f,
     0.127026f,  0.162947f,  0.212203f
};

// ---- Nearest 4-bit centroid index ----

static __device__ __forceinline__ uint8_t turbo_nearest_centroid_4bit(float val) {
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

// ---- Per-block quantize for turbo4 (128 elements, expects already-rotated input) ----

static __device__ void quantize_f32_turbo4_0_block(const float * __restrict__ src,
                                                    block_turbo4_0 * __restrict__ dst) {
    for (int j = 0; j < QK_TURBO4 / 2; j++) dst->qs[j] = 0;

    for (int j = 0; j < QK_TURBO4; j++) {
        uint8_t idx = turbo_nearest_centroid_4bit(src[j]);
        dst->qs[j / 2] |= (idx & 0xF) << ((j % 2) * 4);
    }
}

// ---- Inline dequant helper: extract one float from turbo4 block ----

static __device__ __forceinline__ float turbo4_dequant_element(
        const block_turbo4_0 * __restrict__ x, int j, float norm) {
    uint8_t idx = (x->qs[j / 2] >> ((j % 2) * 4)) & 0xF;
    return TURBO_CENTROIDS_4BIT[idx] * norm;
}

// ---- TQ6: 64 centroids (Lloyd-Max for N(0, 1/128)), mirrored from ggml-turbo-quant.c ----

static __constant__ float TQ6_CENTROIDS[64] = {
    -0.330935f, -0.286417f, -0.257865f, -0.236198f,
    -0.218435f, -0.203203f, -0.189753f, -0.177626f,
    -0.166522f, -0.156230f, -0.146600f, -0.137517f,
    -0.128895f, -0.120663f, -0.112765f, -0.105157f,
    -0.097801f, -0.090663f, -0.083717f, -0.076940f,
    -0.070310f, -0.063809f, -0.057422f, -0.051133f,
    -0.044929f, -0.038798f, -0.032729f, -0.026710f,
    -0.020733f, -0.014787f, -0.008863f, -0.002953f,
     0.002953f,  0.008863f,  0.014787f,  0.020733f,
     0.026710f,  0.032729f,  0.038798f,  0.044929f,
     0.051133f,  0.057422f,  0.063809f,  0.070310f,
     0.076940f,  0.083717f,  0.090663f,  0.097801f,
     0.105157f,  0.112765f,  0.120663f,  0.128895f,
     0.137517f,  0.146600f,  0.156230f,  0.166522f,
     0.177626f,  0.189753f,  0.203203f,  0.218435f,
     0.236198f,  0.257865f,  0.286417f,  0.330935f
};

// ---- Midpoints for nearest 6-bit centroid lookup ----

static __constant__ float TQ6_MID[63] = {
    -0.308676f, -0.272141f, -0.247031f, -0.227316f,
    -0.210819f, -0.196478f, -0.183690f, -0.172074f,
    -0.161376f, -0.151415f, -0.142059f, -0.133206f,
    -0.124779f, -0.116714f, -0.108961f, -0.101479f,
    -0.094232f, -0.087190f, -0.080329f, -0.073625f,
    -0.067060f, -0.060616f, -0.054277f, -0.048031f,
    -0.041864f, -0.035763f, -0.029719f, -0.023722f,
    -0.017760f, -0.011825f, -0.005908f,  0.000000f,
     0.005908f,  0.011825f,  0.017760f,  0.023722f,
     0.029719f,  0.035763f,  0.041864f,  0.048031f,
     0.054277f,  0.060616f,  0.067060f,  0.073625f,
     0.080329f,  0.087190f,  0.094232f,  0.101479f,
     0.108961f,  0.116714f,  0.124779f,  0.133206f,
     0.142059f,  0.151415f,  0.161376f,  0.172074f,
     0.183690f,  0.196478f,  0.210819f,  0.227316f,
     0.247031f,  0.272141f,  0.308676f
};

// ---- Nearest 6-bit centroid index ----
//
// 6-step binary search over the midpoints, not the if/else ladder the 2/3/4-bit
// codebooks use: 63 sequential compares would cost more than the search.
// Same result as nearest_centroid_6bit in ggml-turbo-quant.c.

static __device__ __forceinline__ uint8_t tq6_nearest_centroid(float val) {
    int lo = 0;
    int hi = 63;
#pragma unroll
    for (int step = 0; step < 6; ++step) {
        const int mid = (lo + hi) / 2;
        if (val < TQ6_MID[mid]) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return (uint8_t) lo;
}

// ---- Per-block quantize for tq6 (128 elements, expects already-rotated input) ----

static __device__ void quantize_f32_tq6_0_block(const float * __restrict__ src,
                                                block_tq6_0 * __restrict__ dst) {
    for (int j = 0; j < QK_TQ6 / 2; j++) dst->qs[j] = 0;
    for (int j = 0; j < QK_TQ6 / 4; j++) dst->qh[j] = 0;

    for (int j = 0; j < QK_TQ6; j++) {
        uint8_t idx = tq6_nearest_centroid(src[j]);
        dst->qs[j / 2] |= ( idx       & 0xF) << ((j % 2) * 4);
        dst->qh[j / 4] |= ((idx >> 4) & 0x3) << ((j % 4) * 2);
    }
}

// ---- Inline dequant helper: extract one float from tq6 block ----

static __device__ __forceinline__ float tq6_dequant_element(
        const block_tq6_0 * __restrict__ x, int j, float norm) {
    uint8_t lo = (x->qs[j / 2] >> ((j % 2) * 4)) & 0xF;
    uint8_t hi = (x->qh[j / 4] >> ((j % 4) * 2)) & 0x3;
    return TQ6_CENTROIDS[lo | (hi << 4)] * norm;
}

// ---- TQ5: 32 centroids (Lloyd-Max for N(0, 1/128)), mirrored from ggml-turbo-quant.c ----

static __constant__ float TQ5_CENTROIDS[32] = {
    -0.288236f, -0.237892f, -0.204892f, -0.179348f,
    -0.158003f, -0.139353f, -0.122569f, -0.107141f,
    -0.092730f, -0.079097f, -0.066064f, -0.053491f,
    -0.041269f, -0.029304f, -0.017515f, -0.005827f,
     0.005827f,  0.017515f,  0.029304f,  0.041269f,
     0.053491f,  0.066064f,  0.079097f,  0.092730f,
     0.107141f,  0.122569f,  0.139353f,  0.158003f,
     0.179348f,  0.204892f,  0.237892f,  0.288236f
};

// ---- Midpoints for nearest 5-bit centroid lookup ----

static __constant__ float TQ5_MID[31] = {
    -0.263064f, -0.221392f, -0.192120f, -0.168675f,
    -0.148678f, -0.130961f, -0.114855f, -0.099935f,
    -0.085914f, -0.072580f, -0.059778f, -0.047380f,
    -0.035287f, -0.023410f, -0.011671f,  0.000000f,
     0.011671f,  0.023410f,  0.035287f,  0.047380f,
     0.059778f,  0.072580f,  0.085914f,  0.099935f,
     0.114855f,  0.130961f,  0.148678f,  0.168675f,
     0.192120f,  0.221392f,  0.263064f
};

// Same result as nearest_centroid_5bit in ggml-turbo-quant.c (5-step binary search).

static __device__ __forceinline__ uint8_t tq5_nearest_centroid(float val) {
    int lo = 0;
    int hi = 31;
#pragma unroll
    for (int step = 0; step < 5; ++step) {
        const int mid = (lo + hi) / 2;
        if (val < TQ5_MID[mid]) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return (uint8_t) lo;
}

// ---- TQ5 storage is sign-magnitude (see block_tq5_0): qs nibble = magnitude index m, qh bit = 1 if negative ----

static __device__ __forceinline__ uint8_t tq5_code_to_mag(uint8_t idx) { return idx >= 16 ? idx - 16 : 15 - idx; }
static __device__ __forceinline__ uint8_t tq5_code_to_neg(uint8_t idx) { return idx < 16; }
static __device__ __forceinline__ uint8_t tq5_sm_to_code(uint8_t mag, uint8_t neg) {
    return (uint8_t) ((mag ^ (neg * 0xF)) | ((neg ^ 1) << 4));   // neg ? 15-m : 16+m
}

// ---- Per-block quantize for tq5 (128 elements, expects already-rotated input) ----

static __device__ void quantize_f32_tq5_0_block(const float * __restrict__ src,
                                                block_tq5_0 * __restrict__ dst) {
    for (int j = 0; j < QK_TQ5 / 2; j++) dst->qs[j] = 0;
    for (int j = 0; j < QK_TQ5 / 8; j++) dst->qh[j] = 0;

    for (int j = 0; j < QK_TQ5; j++) {
        const uint8_t idx = tq5_nearest_centroid(src[j]);
        dst->qs[j / 2] |= tq5_code_to_mag(idx) << ((j % 2) * 4);
        dst->qh[j / 8] |= tq5_code_to_neg(idx) << (j % 8);
    }
}

// ---- Inline dequant helper: extract one float from tq5 block ----

static __device__ __forceinline__ float tq5_dequant_element(
        const block_tq5_0 * __restrict__ x, int j, float norm) {
    const uint8_t mag = (x->qs[j / 2] >> ((j % 2) * 4)) & 0xF;
    const uint8_t neg = (x->qh[j / 8] >> (j % 8)) & 0x1;
    return TQ5_CENTROIDS[tq5_sm_to_code(mag, neg)] * norm;
}

// ---- Nearest 3-bit centroid index ----

static __device__ __forceinline__ uint8_t turbo_nearest_centroid_3bit(float val) {
    if      (val < TURBO_MID_3BIT[0]) return 0;
    else if (val < TURBO_MID_3BIT[1]) return 1;
    else if (val < TURBO_MID_3BIT[2]) return 2;
    else if (val < TURBO_MID_3BIT[3]) return 3;
    else if (val < TURBO_MID_3BIT[4]) return 4;
    else if (val < TURBO_MID_3BIT[5]) return 5;
    else if (val < TURBO_MID_3BIT[6]) return 6;
    else                              return 7;
}

// ---- Per-block quantize (32 elements, expects already-rotated input) ----
// Used by set_rows after group-level WHT rotation

static __device__ void quantize_f32_turbo3_0_block(const float * __restrict__ src,
                                                    block_turbo3_0 * __restrict__ dst) {
    for (int j = 0; j < QK_TURBO3 / 4; j++) dst->qs[j] = 0;
    for (int j = 0; j < QK_TURBO3 / 8; j++) dst->signs[j] = 0;

    for (int j = 0; j < QK_TURBO3; j++) {
        uint8_t idx = turbo_nearest_centroid_3bit(src[j]);
        dst->qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
        if (idx & 0x4) {
            dst->signs[j / 8] |= (1 << (j % 8));
        }
    }
}

// ---- Inline dequant helper: extract one float from turbo3 block ----

static __device__ __forceinline__ float turbo3_dequant_element(
        const block_turbo3_0 * __restrict__ x, int j, float norm) {
    uint8_t low2 = (x->qs[j / 4] >> ((j % 4) * 2)) & 0x3;
    uint8_t hi1  = (x->signs[j / 8] >> (j % 8)) & 0x1;
    uint8_t idx  = low2 | (hi1 << 2);
    return TURBO_CENTROIDS_3BIT[idx] * norm;
}

// ---- Nearest 2-bit centroid index ----

static __device__ __forceinline__ uint8_t turbo_nearest_centroid_2bit(float val) {
    if      (val < TURBO_MID_2BIT[0]) return 0;
    else if (val < TURBO_MID_2BIT[1]) return 1;
    else if (val < TURBO_MID_2BIT[2]) return 2;
    else                              return 3;
}

// ---- Per-block quantize for turbo2 (32 elements, expects already-rotated input) ----

static __device__ void quantize_f32_turbo2_0_block(const float * __restrict__ src,
                                                    block_turbo2_0 * __restrict__ dst) {
    for (int j = 0; j < QK_TURBO2 / 4; j++) dst->qs[j] = 0;

    for (int j = 0; j < QK_TURBO2; j++) {
        uint8_t idx = turbo_nearest_centroid_2bit(src[j]);
        dst->qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
    }
}

// ---- Inline dequant helper: extract one float from turbo2 block ----

static __device__ __forceinline__ float turbo2_dequant_element(
        const block_turbo2_0 * __restrict__ x, int j, float norm) {
    uint8_t idx = (x->qs[j / 4] >> ((j % 4) * 2)) & 0x3;
    return TURBO_CENTROIDS_2BIT[idx] * norm;
}

// ============================================================================
// Weight compression types (TQ3_1S, TQ4_1S)
// These use N(0,1) centroids (NOT N(0,1/128) like KV cache types)
// and require inverse WHT (RHT) after centroid lookup.
// ============================================================================

#define QR_TQ4_1S 1  // dequantize produces 2 consecutive elements
#define QR_TQ3_1S 1

// ---- Weight centroids: Lloyd-Max for N(0,1) ----

static __constant__ float TQ4_CENTROIDS_WEIGHT[16] = {
    -2.732590f, -2.069017f, -1.618046f, -1.256231f,
    -0.942340f, -0.656759f, -0.388048f, -0.128395f,
     0.128395f,  0.388048f,  0.656759f,  0.942340f,
     1.256231f,  1.618046f,  2.069017f,  2.732590f
};

static __constant__ float TQ3_CENTROIDS_WEIGHT[8] = {
    -1.996684f, -1.291398f, -0.740341f, -0.247508f,
     0.230106f,  0.725222f,  1.277503f,  1.988943f
};

// ---- Sign array for weight WHT (golden ratio hash, 32 elements) ----

static __constant__ float TQ_WEIGHT_SIGNS[32] = {
    +1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f,
    -1.0f, +1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f
};

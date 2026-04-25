/*
 * TurboQuant: KV cache compression via PolarQuant + QJL
 * Based on: arXiv 2504.19874 (ICLR 2026)
 *
 * Implements GGML_TYPE_TURBO2_0 (2-bit), GGML_TYPE_TURBO3_0 (3-bit) and
 * GGML_TYPE_TURBO4_0 (4-bit) for use as --cache-type-k turboN in llama-server.
 */

#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <stdint.h>

/* Global: WHT group size for CPU quantize path (set by CPU SET_ROWS handler) */
int turbo3_cpu_wht_group_size = 0;

/* ---------- constants ---------- */

#define TURBO_SEED_ROTATION 42
#define TURBO_SEED_QJL      1042
#define TURBO_D             128  /* rotation group size = head_dim (independent of block size) */
#define TURBO_QJL_CONST     1.2533141373155003f  /* sqrt(pi/2) */

/* Optimal centroids from paper (scaled by 1/sqrt(d)) */
/* 1-bit: ±sqrt(2/(pi*d)) */
static const float CENTROIDS_1BIT[2] = { -0.070711f, 0.070711f };  /* for d=128 */

/* 2-bit: {±0.453, ±1.51} / sqrt(d) */
static const float CENTROIDS_2BIT[4] = { -0.133462f, -0.039994f, 0.039994f, 0.133462f };

/* 3-bit: Lloyd-Max for N(0, 1/128), pre-computed */
static const float CENTROIDS_3BIT[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

enum turbo_empvar_mode {
    TURBO_EMPVAR_MODE_DISABLED = 0,
    TURBO_EMPVAR_MODE_WHT_ONLY = 1,
};

static int   turbo_empvar_initialized = 0;
static int   turbo_empvar_mode = TURBO_EMPVAR_MODE_DISABLED;
enum { TURBO_EMPVAR_MAX_DIM = 1024 };
static int   turbo_empvar_k_dim = 0;
static int   turbo_empvar_v_dim = 0;
static float turbo_empvar_k[TURBO_EMPVAR_MAX_DIM];
static float turbo_empvar_v[TURBO_EMPVAR_MAX_DIM];

static void turbo_trim_ws(char ** begin, char ** end) {
    while (*begin < *end && (**begin == ' ' || **begin == '\t' || **begin == '\n' || **begin == '\r')) {
        ++(*begin);
    }
    while (*end > *begin && ((*(*end - 1) == ' ') || (*(*end - 1) == '\t') || (*(*end - 1) == '\n') || (*(*end - 1) == '\r'))) {
        --(*end);
    }
}

static int turbo_parse_float_list(const char * text, float * out, int max_n) {
    if (text == NULL || *text == '\0') {
        return 0;
    }
    int n = 0;
    const char * p = text;
    while (*p != '\0' && n < max_n) {
        char * end = NULL;
        float v = strtof(p, &end);
        if (end == p) {
            while (*p != '\0' && *p != ',' && *p != ';') {
                ++p;
            }
        } else {
            out[n++] = v;
            p = end;
        }
        while (*p == ',' || *p == ';' || *p == ' ' || *p == '\n' || *p == '\t' || *p == '\r') {
            ++p;
        }
    }
    return n;
}

static void turbo_empvar_load_from_env(void) {
    memset(turbo_empvar_k, 0, sizeof(turbo_empvar_k));
    memset(turbo_empvar_v, 0, sizeof(turbo_empvar_v));
    turbo_empvar_k_dim = 0;
    turbo_empvar_v_dim = 0;
    turbo_empvar_mode = TURBO_EMPVAR_MODE_DISABLED;

    // Keep the CPU reference path on its own env switch so the live Metal
    // runtime can use TURBO_EXPERIMENTAL_MODE without affecting model init.
    const char * mode = getenv("TURBO_CPU_EXPERIMENTAL_MODE");
    if (mode == NULL) {
        turbo_empvar_initialized = 1;
        return;
    }
    char * begin = (char *) mode;
    char * end = begin + strlen(mode);
    turbo_trim_ws(&begin, &end);
    if ((size_t) (end - begin) == strlen("wht_only_empvar") &&
            strncmp(begin, "wht_only_empvar", strlen("wht_only_empvar")) == 0) {
        turbo_empvar_mode = TURBO_EMPVAR_MODE_WHT_ONLY;
    } else {
        turbo_empvar_initialized = 1;
        return;
    }

    turbo_empvar_k_dim = turbo_parse_float_list(getenv("TURBO_CPU_EMPVAR_K"), turbo_empvar_k, TURBO_EMPVAR_MAX_DIM);
    turbo_empvar_v_dim = turbo_parse_float_list(getenv("TURBO_CPU_EMPVAR_V"), turbo_empvar_v, TURBO_EMPVAR_MAX_DIM);
    turbo_empvar_initialized = 1;
}

void ggml_turbo_empvar_reload_from_env(void) {
    turbo_empvar_initialized = 0;
    turbo_empvar_load_from_env();
}

void ggml_turbo_quant_set_context(int group_size, int kv_kind) {
    GGML_UNUSED(kv_kind);
    turbo3_cpu_wht_group_size = group_size;
    if (!turbo_empvar_initialized) {
        turbo_empvar_load_from_env();
    }
}

static const float * turbo_empvar_variances_for_group(int group_size, int group_idx, int kv_kind) {
    if (!turbo_empvar_initialized) {
        turbo_empvar_load_from_env();
    }
    if (turbo_empvar_mode != TURBO_EMPVAR_MODE_WHT_ONLY) {
        return NULL;
    }
    const float * table = NULL;
    int dim = 0;
    if (kv_kind == 1) {
        table = turbo_empvar_k;
        dim = turbo_empvar_k_dim;
    }
    if (kv_kind == 2) {
        table = turbo_empvar_v;
        dim = turbo_empvar_v_dim;
    }
    if (table == NULL || dim <= 0 || dim % group_size != 0) {
        return NULL;
    }
    const int n_profiles = dim / group_size;
    const int profile_idx = n_profiles > 1 ? (group_idx % n_profiles) : 0;
    return table + profile_idx * group_size;
}

static int nearest_centroid_scaled(float val, const float * base, int n_levels, float scale) {
    int best = 0;
    float best_dist = fabsf(val - base[0] * scale);
    for (int i = 1; i < n_levels; ++i) {
        const float dist = fabsf(val - base[i] * scale);
        if (dist < best_dist) {
            best = i;
            best_dist = dist;
        }
    }
    return best;
}

/* ---------- rotation matrix (lazy init) ---------- */

static float turbo_rotation[TURBO_D * TURBO_D];
static float turbo_rotation_t[TURBO_D * TURBO_D]; /* transpose */
static int   turbo_rotation_initialized = 0;

/* Simple LCG PRNG for deterministic rotation generation */
static uint64_t turbo_prng_state;

static void turbo_prng_seed(uint64_t seed) {
    turbo_prng_state = seed;
}

static double turbo_prng_normal(void) {
    /* Box-Muller transform from uniform LCG */
    turbo_prng_state = turbo_prng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u1 = (double)(turbo_prng_state >> 11) / (double)(1ULL << 53);
    if (u1 < 1e-15) u1 = 1e-15;
    turbo_prng_state = turbo_prng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u2 = (double)(turbo_prng_state >> 11) / (double)(1ULL << 53);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static void turbo_init_rotation(void) {
    if (turbo_rotation_initialized) return;

    const int d = TURBO_D;

    /* Generate random Gaussian matrix */
    turbo_prng_seed(TURBO_SEED_ROTATION);
    float G[TURBO_D * TURBO_D];
    for (int i = 0; i < d * d; i++) {
        G[i] = (float)turbo_prng_normal();
    }

    /* QR decomposition via modified Gram-Schmidt */
    /* Q stored column-major in turbo_rotation */
    memcpy(turbo_rotation, G, d * d * sizeof(float));

    for (int j = 0; j < d; j++) {
        /* Normalize column j */
        float norm = 0.0f;
        for (int i = 0; i < d; i++) {
            norm += turbo_rotation[i * d + j] * turbo_rotation[i * d + j];
        }
        norm = sqrtf(norm);
        if (norm > 1e-10f) {
            for (int i = 0; i < d; i++) {
                turbo_rotation[i * d + j] /= norm;
            }
        }

        /* Orthogonalize remaining columns against j */
        for (int k = j + 1; k < d; k++) {
            float dot = 0.0f;
            for (int i = 0; i < d; i++) {
                dot += turbo_rotation[i * d + j] * turbo_rotation[i * d + k];
            }
            for (int i = 0; i < d; i++) {
                turbo_rotation[i * d + k] -= dot * turbo_rotation[i * d + j];
            }
        }
    }

    /* Compute transpose */
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            turbo_rotation_t[i * d + j] = turbo_rotation[j * d + i];
        }
    }

    turbo_rotation_initialized = 1;
}

/* ---------- QJL projection matrix (lazy init, seed-based) ---------- */

static float turbo_qjl_matrix[TURBO_D * TURBO_D];
static float turbo_qjl_matrix_t[TURBO_D * TURBO_D];
static int   turbo_qjl_initialized = 0;

static void turbo_init_qjl(void) {
    if (turbo_qjl_initialized) return;

    const int d = TURBO_D;
    turbo_prng_seed(TURBO_SEED_QJL);

    for (int i = 0; i < d * d; i++) {
        turbo_qjl_matrix[i] = (float)turbo_prng_normal();
    }

    /* Transpose */
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            turbo_qjl_matrix_t[i * d + j] = turbo_qjl_matrix[j * d + i];
        }
    }

    turbo_qjl_initialized = 1;
}

/* ---------- helper: matrix-vector multiply ---------- */

static void matvec(const float * M, const float * x, float * y, int d) {
    /* y = M @ x, M is row-major d×d */
    for (int i = 0; i < d; i++) {
        float sum = 0.0f;
        for (int j = 0; j < d; j++) {
            sum += M[i * d + j] * x[j];
        }
        y[i] = sum;
    }
}

/* ---------- nearest centroid ---------- */

static int nearest_centroid_2bit(float val) {
    /* Binary search on midpoints: {-0.133, -0.040, 0.040, 0.133} */
    if (val < -0.086728f) return 0;       /* midpoint(-0.133, -0.040) */
    if (val <  0.000000f) return 1;       /* midpoint(-0.040, 0.040) */
    if (val <  0.086728f) return 2;       /* midpoint(0.040, 0.133) */
    return 3;
}

static int nearest_centroid_3bit(float val) {
    /* 8 centroids, find nearest via midpoints */
    if (val < -0.154259f) return 0;
    if (val < -0.091775f) return 1;
    if (val < -0.043589f) return 2;
    if (val <  0.000000f) return 3;
    if (val <  0.043589f) return 4;
    if (val <  0.091775f) return 5;
    if (val <  0.154259f) return 6;
    return 7;
}

static int nearest_centroid_4bit(float val) {
    /* 16 centroids, optimal for N(0, 1/sqrt(128)), find nearest via midpoints */
    if (val < -0.145560f) return 0;
    if (val < -0.103361f) return 1;
    if (val < -0.079142f) return 2;
    if (val < -0.060009f) return 3;
    if (val < -0.043430f) return 4;
    if (val < -0.028293f) return 5;
    if (val < -0.013963f) return 6;
    if (val <  0.000000f) return 7;
    if (val <  0.013963f) return 8;
    if (val <  0.028293f) return 9;
    if (val <  0.043430f) return 10;
    if (val <  0.060009f) return 11;
    if (val <  0.079142f) return 12;
    if (val <  0.103361f) return 13;
    if (val <  0.145560f) return 14;
    return 15;
}

/* ---------- WHT sign arrays (must match CUDA/Metal, seed=42) ---------- */

static const float turbo_cpu_s1[128] = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1
};

static const float turbo_cpu_s2[128] = {
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1
};

/* ---------- CPU forward WHT (in-place, group_size elements) ---------- */

static void turbo_cpu_fwht(float * x, int group_size) {
    const float * s1 = turbo_cpu_s1;
    const float * s2 = turbo_cpu_s2;
    const float inv_sqrt = (group_size == 128) ? 0.08838834764831845f : 0.125f;

    // signs1
    for (int i = 0; i < group_size; i++) x[i] *= s1[i];

    // butterfly stages
    for (int h = 1; h < group_size; h *= 2) {
        for (int i = 0; i < group_size; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }

    // normalize + signs2
    for (int i = 0; i < group_size; i++) x[i] *= inv_sqrt * s2[i];
}

/* ---------- TURBO3_0: 3-bit PolarQuant with WHT rotation ---------- */

static void quantize_row_turbo3_0_impl(const float * GGML_RESTRICT x, block_turbo3_0 * GGML_RESTRICT y, int64_t k, int kv_kind, int apply_wht) {
    assert(k % QK_TURBO3 == 0);

    // Read WHT group size from global (set by CPU SET_ROWS handler before each call).
    // Fallback: 128 if row is 128-aligned, else 64.
    extern int turbo3_cpu_wht_group_size;
    int group_size = turbo3_cpu_wht_group_size;
    if (group_size != 64 && group_size != 128) {
        group_size = (k % 128 == 0) ? 128 : 64;
    }
    if (k % group_size != 0) group_size = (group_size == 128) ? 64 : 128;
    assert(k % group_size == 0);

    const int n_groups = k / group_size;
    const int blocks_per_group = group_size / QK_TURBO3;
    const float sigma_ref = 1.0f / sqrtf((float) group_size);

    for (int g = 0; g < n_groups; g++) {
        const float * grp_src = x + g * group_size;
        block_turbo3_0 * grp_dst = y + g * blocks_per_group;
        const float * empvar = turbo_empvar_variances_for_group(group_size, g, kv_kind);
        const int use_empvar = (empvar != NULL);

        // 1. L2 norm over the group
        float norm_sq = 0.0f;
        float buf[128];  // max group_size
        for (int j = 0; j < group_size; j++) {
            buf[j] = grp_src[j];
            norm_sq += buf[j] * buf[j];
        }
        float grp_norm = sqrtf(norm_sq);
        float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

        // 2. Normalize
        for (int j = 0; j < group_size; j++) buf[j] *= inv_norm;

        // 3. Forward rotation. PCA mode rotates at graph level, so SET_ROWS must not WHT again.
        if (apply_wht) {
            turbo_cpu_fwht(buf, group_size);
        }

        // 4. Quantize + pack into sub-blocks
        float recon_sq = 0.0f;
        for (int b = 0; b < blocks_per_group; b++) {
            block_turbo3_0 * blk = &grp_dst[b];
            const int off = b * QK_TURBO3;

            memset(blk->qs, 0, QK_TURBO3 / 4);
            memset(blk->signs, 0, QK_TURBO3 / 8);

            for (int j = 0; j < QK_TURBO3; j++) {
                const int coord = off + j;
                int idx = nearest_centroid_3bit(buf[coord]);
                if (use_empvar) {
                    const float sigma = sqrtf(fmaxf(empvar[coord], 1e-6f));
                    const float scale = sigma / sigma_ref;
                    idx = nearest_centroid_scaled(buf[coord], CENTROIDS_3BIT, 8, scale);
                }
                blk->qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
                if (idx & 0x4) {
                    blk->signs[j / 8] |= (1 << (j % 8));
                }
                const float sigma = use_empvar ? sqrtf(fmaxf(empvar[coord], 1e-6f)) : sigma_ref;
                const float scale = sigma / sigma_ref;
                const float c = CENTROIDS_3BIT[idx] * scale;
                recon_sq += c * c;
            }
        }

        // 5. Corrected norm: grp_norm / recon_norm (matching CUDA kernel)
        float recon_norm = sqrtf(recon_sq);
        float corrected = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
        for (int b = 0; b < blocks_per_group; b++) {
            grp_dst[b].norm = GGML_FP32_TO_FP16(corrected);
        }
    }
}

void quantize_row_turbo3_0_ref(const float * GGML_RESTRICT x, block_turbo3_0 * GGML_RESTRICT y, int64_t k) {
    quantize_row_turbo3_0_impl(x, y, k, 0, 1);
}

void quantize_row_turbo3_empvar_k_ref(const float * GGML_RESTRICT x, block_turbo3_0 * GGML_RESTRICT y, int64_t k) {
    quantize_row_turbo3_0_impl(x, y, k, 1, 1);
}

void quantize_row_turbo3_empvar_v_ref(const float * GGML_RESTRICT x, block_turbo3_0 * GGML_RESTRICT y, int64_t k) {
    quantize_row_turbo3_0_impl(x, y, k, 2, 1);
}

void quantize_row_turbo3_pca_k_ref(const float * GGML_RESTRICT x, block_turbo3_0 * GGML_RESTRICT y, int64_t k) {
    quantize_row_turbo3_0_impl(x, y, k, 1, 0);
}

void quantize_row_turbo3_pca_v_ref(const float * GGML_RESTRICT x, block_turbo3_0 * GGML_RESTRICT y, int64_t k) {
    quantize_row_turbo3_0_impl(x, y, k, 2, 0);
}

static void dequantize_row_turbo3_0_impl(const block_turbo3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k, int kv_kind) {
    assert(k % QK_TURBO3 == 0);

    extern int turbo3_cpu_wht_group_size;
    int group_size = turbo3_cpu_wht_group_size;
    if (group_size != 64 && group_size != 128) {
        group_size = (k % 128 == 0) ? 128 : 64;
    }
    if (k % group_size != 0) group_size = (group_size == 128) ? 64 : 128;
    assert(k % group_size == 0);

    const int n_groups = k / group_size;
    const int blocks_per_group = group_size / QK_TURBO3;
    const float sigma_ref = 1.0f / sqrtf((float) group_size);

    for (int g = 0; g < n_groups; ++g) {
        const block_turbo3_0 * grp_src = x + g * blocks_per_group;
        const float * empvar = turbo_empvar_variances_for_group(group_size, g, kv_kind);
        const int use_empvar = (empvar != NULL);

        for (int b = 0; b < blocks_per_group; ++b) {
            const block_turbo3_0 * blk = &grp_src[b];
            const float norm = GGML_FP16_TO_FP32(blk->norm);
            const int off = g * group_size + b * QK_TURBO3;
            for (int j = 0; j < QK_TURBO3; ++j) {
                const int coord = b * QK_TURBO3 + j;
                uint8_t low2 = (blk->qs[j/4] >> ((j%4)*2)) & 0x3;
                uint8_t hi1 = (blk->signs[j/8] >> (j%8)) & 0x1;
                uint8_t idx = low2 | (hi1 << 2);
                const float sigma = use_empvar ? sqrtf(fmaxf(empvar[coord], 1e-6f)) : sigma_ref;
                const float scale = sigma / sigma_ref;
                y[off + j] = CENTROIDS_3BIT[idx] * scale * norm;
            }
        }
    }
}

void dequantize_row_turbo3_0(const block_turbo3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    dequantize_row_turbo3_0_impl(x, y, k, 0);
}

void dequantize_row_turbo3_empvar_k(const block_turbo3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    dequantize_row_turbo3_0_impl(x, y, k, 1);
}

void dequantize_row_turbo3_empvar_v(const block_turbo3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    dequantize_row_turbo3_0_impl(x, y, k, 2);
}

size_t quantize_turbo3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO3 == 0);

    size_t row_size = (n_per_row / QK_TURBO3) * sizeof(block_turbo3_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo3_0_ref(
            src + row * n_per_row,
            (block_turbo3_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

static size_t quantize_turbo3_empvar_impl(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                                          int64_t nrows, int64_t n_per_row, const float * imatrix, int kv_kind) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO3 == 0);

    size_t row_size = (n_per_row / QK_TURBO3) * sizeof(block_turbo3_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo3_0_impl(
            src + row * n_per_row,
            (block_turbo3_0 *)((char *)dst + row * row_size),
            n_per_row,
            kv_kind,
            1
        );
    }
    return nrows * row_size;
}

static size_t quantize_turbo3_pca_impl(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                                       int64_t nrows, int64_t n_per_row, const float * imatrix, int kv_kind) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO3 == 0);

    size_t row_size = (n_per_row / QK_TURBO3) * sizeof(block_turbo3_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo3_0_impl(
            src + row * n_per_row,
            (block_turbo3_0 *)((char *)dst + row * row_size),
            n_per_row,
            kv_kind,
            0
        );
    }
    return nrows * row_size;
}

size_t quantize_turbo3_pca_k(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                             int64_t nrows, int64_t n_per_row, const float * imatrix) {
    return quantize_turbo3_pca_impl(src, dst, nrows, n_per_row, imatrix, 1);
}

size_t quantize_turbo3_pca_v(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                             int64_t nrows, int64_t n_per_row, const float * imatrix) {
    return quantize_turbo3_pca_impl(src, dst, nrows, n_per_row, imatrix, 2);
}

size_t quantize_turbo3_empvar_k(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                                int64_t nrows, int64_t n_per_row, const float * imatrix) {
    return quantize_turbo3_empvar_impl(src, dst, nrows, n_per_row, imatrix, 1);
}

size_t quantize_turbo3_empvar_v(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                                int64_t nrows, int64_t n_per_row, const float * imatrix) {
    return quantize_turbo3_empvar_impl(src, dst, nrows, n_per_row, imatrix, 2);
}

/* ---------- TURBO2_0: 2-bit PolarQuant (no QJL) ---------- */

void quantize_row_turbo2_0_ref(const float * GGML_RESTRICT x, block_turbo2_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO2 == 0);

    extern int turbo3_cpu_wht_group_size;
    int group_size = turbo3_cpu_wht_group_size;
    if (group_size != 64 && group_size != 128) {
        group_size = (k % 128 == 0) ? 128 : 64;
    }
    if (k % group_size != 0) group_size = (group_size == 128) ? 64 : 128;
    assert(k % group_size == 0);

    const int n_groups = k / group_size;
    const int blocks_per_group = group_size / QK_TURBO2;

    for (int g = 0; g < n_groups; g++) {
        const float * grp_src = x + g * group_size;
        block_turbo2_0 * grp_dst = y + g * blocks_per_group;
        /* 1. L2 norm over the group */
        float norm_sq = 0.0f;
        float buf[128];
        for (int j = 0; j < group_size; j++) {
            buf[j] = grp_src[j];
            norm_sq += buf[j] * buf[j];
        }
        float grp_norm = sqrtf(norm_sq);
        float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

        /* 2. Normalize */
        for (int j = 0; j < group_size; j++) buf[j] *= inv_norm;

        /* 3. Forward WHT rotation */
        turbo_cpu_fwht(buf, group_size);

        /* 4. Quantize + pack into sub-blocks */
        float recon_sq = 0.0f;
        for (int b = 0; b < blocks_per_group; b++) {
            block_turbo2_0 * blk = &grp_dst[b];
            const int off = b * QK_TURBO2;

            memset(blk->qs, 0, QK_TURBO2 / 4);

            for (int j = 0; j < QK_TURBO2; j++) {
                const int coord = off + j;
                const int idx = nearest_centroid_2bit(buf[coord]);
                blk->qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
                recon_sq += CENTROIDS_2BIT[idx] * CENTROIDS_2BIT[idx];
            }
        }

        /* 5. Corrected norm */
        float recon_norm = sqrtf(recon_sq);
        float corrected = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
        for (int b = 0; b < blocks_per_group; b++) {
            grp_dst[b].norm = GGML_FP32_TO_FP16(corrected);
        }
    }
}

void dequantize_row_turbo2_0(const block_turbo2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO2 == 0);

    extern int turbo3_cpu_wht_group_size;
    int group_size = turbo3_cpu_wht_group_size;
    if (group_size != 64 && group_size != 128) {
        group_size = (k % 128 == 0) ? 128 : 64;
    }
    if (k % group_size != 0) group_size = (group_size == 128) ? 64 : 128;
    assert(k % group_size == 0);

    const int n_groups = k / group_size;
    const int blocks_per_group = group_size / QK_TURBO2;

    for (int g = 0; g < n_groups; ++g) {
        const block_turbo2_0 * grp_src = x + g * blocks_per_group;
        for (int b = 0; b < blocks_per_group; ++b) {
            const block_turbo2_0 * blk = &grp_src[b];
            const float norm = GGML_FP16_TO_FP32(blk->norm);
            const int off = g * group_size + b * QK_TURBO2;
            for (int j = 0; j < QK_TURBO2; ++j) {
                uint8_t idx = (blk->qs[j/4] >> ((j%4)*2)) & 0x3;
                y[off + j] = CENTROIDS_2BIT[idx] * norm;
            }
        }
    }
}

size_t quantize_turbo2_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO2 == 0);

    size_t row_size = (n_per_row / QK_TURBO2) * sizeof(block_turbo2_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo2_0_ref(
            src + row * n_per_row,
            (block_turbo2_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- TURBO4_0: 3-bit PolarQuant + 1-bit QJL ---------- */

static void quantize_row_turbo4_0_impl(const float * GGML_RESTRICT x, block_turbo4_0 * GGML_RESTRICT y, int64_t k, int kv_kind, int apply_wht) {
    turbo_init_rotation();
    turbo_init_qjl();

    assert(k % QK_TURBO4 == 0);
    const int nb = k / QK_TURBO4;
    const int d  = QK_TURBO4;

    for (int block = 0; block < nb; block++) {
        const float * src = x + block * d;

        /* Step 1: Extract norm */
        float norm_sq = 0.0f;
        for (int i = 0; i < d; i++) norm_sq += src[i] * src[i];
        float norm = sqrtf(norm_sq);

        /* Normalize */
        float normalized[TURBO_D];
        if (norm > 1e-10f) {
            const float inv = 1.0f / norm;
            for (int i = 0; i < d; i++) normalized[i] = src[i] * inv;
        } else {
            memset(normalized, 0, d * sizeof(float));
        }

        const float * empvar = turbo_empvar_variances_for_group(QK_TURBO4, block, kv_kind);
        const int use_empvar = (empvar != NULL);
        const float sigma_ref = 1.0f / sqrtf((float) QK_TURBO4);

        /* Step 2: Rotate. PCA mode rotates at graph level, so do not WHT again. */
        float rotated[TURBO_D];
        if (apply_wht) {
            matvec(turbo_rotation, normalized, rotated, d);
        } else {
            memcpy(rotated, normalized, d * sizeof(float));
        }

#if TURBO4_USE_4BIT
        /* Step 3: 4-bit quantization (16 centroids) */
        static const float CENTROIDS_4BIT[16] = {
            -0.173926f, -0.117195f, -0.089527f, -0.068756f,
            -0.051262f, -0.035597f, -0.020989f, -0.006938f,
             0.006938f,  0.020989f,  0.035597f,  0.051262f,
             0.068756f,  0.089527f,  0.117195f,  0.173926f
        };
        uint8_t indices[TURBO_D];
        for (int i = 0; i < d; i++) {
            if (use_empvar) {
                const float sigma = sqrtf(fmaxf(empvar[i], 1e-6f));
                const float scale = sigma / sigma_ref;
                indices[i] = (uint8_t) nearest_centroid_scaled(rotated[i], CENTROIDS_4BIT, 16, scale);
            } else {
                indices[i] = (uint8_t) nearest_centroid_4bit(rotated[i]);
            }
        }

        /* Norm correction */
        float recon_norm_sq = 0.0f;
        for (int i = 0; i < d; i++) {
            const float sigma = use_empvar ? sqrtf(fmaxf(empvar[i], 1e-6f)) : sigma_ref;
            const float scale = sigma / sigma_ref;
            const float c = CENTROIDS_4BIT[indices[i]] * scale;
            recon_norm_sq += c * c;
        }
        float recon_norm = sqrtf(recon_norm_sq);
        float corrected_norm = (recon_norm > 1e-10f) ? norm / recon_norm : norm;
        y[block].norm = GGML_FP32_TO_FP16(corrected_norm);
#else
        /* Step 3: 3-bit quantization (8 centroids) */
        uint8_t indices[TURBO_D];
        for (int i = 0; i < d; i++) {
            indices[i] = (uint8_t)nearest_centroid_3bit(rotated[i]);
        }

        /* Step 4: Residual */
        float reconstructed[TURBO_D];
        for (int i = 0; i < d; i++) {
            reconstructed[i] = CENTROIDS_3BIT[indices[i]];
        }
        float mse_recon[TURBO_D];
        matvec(turbo_rotation_t, reconstructed, mse_recon, d);

        float residual[TURBO_D];
        for (int i = 0; i < d; i++) {
            residual[i] = normalized[i] - mse_recon[i];
        }

        /* Step 5: QJL */
        float projected[TURBO_D];
        matvec(turbo_qjl_matrix, residual, projected, d);
#endif

        /* Pack */
#if !TURBO4_USE_4BIT
        y[block].norm  = GGML_FP32_TO_FP16(norm);
#endif

#if TURBO4_USE_4BIT
        /* 4-bit PolarQuant: nibble pack into qs[64] */
        memset(y[block].qs, 0, d / 2);
        for (int i = 0; i < d; i++) {
            y[block].qs[i / 2] |= (uint8_t)((indices[i] & 0xF) << ((i % 2) * 4));
        }
        y[block].rnorm = GGML_FP32_TO_FP16(0.0f);
#else
        /* Legacy 3-bit + QJL: pack 3-bit indices + QJL signs */
        memset(y[block].qs, 0, d * 3 / 8);
        for (int i = 0; i < d; i++) {
            int bit_offset = i * 3;
            int byte_idx   = bit_offset / 8;
            int bit_pos    = bit_offset % 8;
            uint16_t val   = (uint16_t)(indices[i] & 0x7);
            y[block].qs[byte_idx] |= (uint8_t)(val << bit_pos);
            if (bit_pos > 5 && byte_idx + 1 < d * 3 / 8) {
                y[block].qs[byte_idx + 1] |= (uint8_t)(val >> (8 - bit_pos));
            }
        }
        memset(y[block].signs, 0, d / 8);
        for (int i = 0; i < d; i++) {
            if (projected[i] >= 0.0f) {
                y[block].signs[i / 8] |= (1 << (i % 8));
            }
        }
#endif
    }
}

void quantize_row_turbo4_0_ref(const float * GGML_RESTRICT x, block_turbo4_0 * GGML_RESTRICT y, int64_t k) {
    quantize_row_turbo4_0_impl(x, y, k, 0, 1);
}

void quantize_row_turbo4_pca_k_ref(const float * GGML_RESTRICT x, block_turbo4_0 * GGML_RESTRICT y, int64_t k) {
    quantize_row_turbo4_0_impl(x, y, k, 1, 0);
}

void quantize_row_turbo4_pca_v_ref(const float * GGML_RESTRICT x, block_turbo4_0 * GGML_RESTRICT y, int64_t k) {
    quantize_row_turbo4_0_impl(x, y, k, 2, 0);
}

void dequantize_row_turbo4_0(const block_turbo4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    turbo_init_rotation();

    assert(k % QK_TURBO4 == 0);
    const int nb = k / QK_TURBO4;
    const int d  = QK_TURBO4;

#if TURBO4_USE_4BIT
    /* 4-bit PolarQuant: nibble unpack → centroid → inverse rotate → scale */
    /* TODO: add proper 4-bit centroid table to C code (currently only in Metal) */
    static const float CENTROIDS_4BIT[16] = {
        -0.173926f, -0.117195f, -0.089527f, -0.068756f,
        -0.051262f, -0.035597f, -0.020989f, -0.006938f,
         0.006938f,  0.020989f,  0.035597f,  0.051262f,
         0.068756f,  0.089527f,  0.117195f,  0.173926f
    };
    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);
        float rotated[QK_TURBO4];
        for (int i = 0; i < d; i++) {
            uint8_t idx = (x[block].qs[i / 2] >> ((i % 2) * 4)) & 0xF;
            rotated[i] = CENTROIDS_4BIT[idx];
        }
        float * dst = y + block * d;
        matvec(turbo_rotation_t, rotated, dst, d);
        for (int i = 0; i < d; i++) dst[i] *= norm;
    }
#else
    /* Legacy 3-bit + QJL dequant */
    turbo_init_qjl();
    for (int block = 0; block < nb; block++) {
        float norm  = GGML_FP16_TO_FP32(x[block].norm);

        uint8_t indices[TURBO_D];
        for (int i = 0; i < d; i++) {
            int bit_offset = i * 3;
            int byte_idx   = bit_offset / 8;
            int bit_pos    = bit_offset % 8;
            uint16_t raw   = (uint16_t)x[block].qs[byte_idx];
            if (byte_idx + 1 < d * 3 / 8) {
                raw |= (uint16_t)x[block].qs[byte_idx + 1] << 8;
            }
            indices[i] = (uint8_t)((raw >> bit_pos) & 0x7);
        }

        float signs[TURBO_D];
        for (int i = 0; i < d; i++) {
            signs[i] = (x[block].signs[i / 8] & (1 << (i % 8))) ? 1.0f : -1.0f;
        }

        float rnorm = GGML_FP16_TO_FP32(x[block].rnorm);
        const float qjl_scale = TURBO_QJL_CONST / (float)d * rnorm;

        float rotated_recon[TURBO_D];
        for (int i = 0; i < d; i++) {
            rotated_recon[i] = CENTROIDS_3BIT[indices[i]];
        }
        float mse_recon[TURBO_D];
        matvec(turbo_rotation_t, rotated_recon, mse_recon, d);

        float qjl_recon[TURBO_D];
        matvec(turbo_qjl_matrix_t, signs, qjl_recon, d);
        for (int i = 0; i < d; i++) {
            qjl_recon[i] *= qjl_scale;
        }

        float * dst = y + block * d;
        for (int i = 0; i < d; i++) {
            dst[i] = (mse_recon[i] + qjl_recon[i]) * norm;
        }
    }
#endif
}

size_t quantize_turbo4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO4 == 0);

    size_t row_size = (n_per_row / QK_TURBO4) * sizeof(block_turbo4_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo4_0_ref(
            src + row * n_per_row,
            (block_turbo4_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

static size_t quantize_turbo4_pca_impl(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                                       int64_t nrows, int64_t n_per_row, const float * imatrix, int kv_kind) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO4 == 0);

    size_t row_size = (n_per_row / QK_TURBO4) * sizeof(block_turbo4_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo4_0_impl(
            src + row * n_per_row,
            (block_turbo4_0 *)((char *)dst + row * row_size),
            n_per_row,
            kv_kind,
            0
        );
    }
    return nrows * row_size;
}

size_t quantize_turbo4_pca_k(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                             int64_t nrows, int64_t n_per_row, const float * imatrix) {
    return quantize_turbo4_pca_impl(src, dst, nrows, n_per_row, imatrix, 1);
}

size_t quantize_turbo4_pca_v(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                             int64_t nrows, int64_t n_per_row, const float * imatrix) {
    return quantize_turbo4_pca_impl(src, dst, nrows, n_per_row, imatrix, 2);
}

static const float CENTROIDS_4BIT_REF[16] = {
    -0.173926f, -0.117195f, -0.089527f, -0.068756f,
    -0.051262f, -0.035597f, -0.020989f, -0.006938f,
     0.006938f,  0.020989f,  0.035597f,  0.051262f,
     0.068756f,  0.089527f,  0.117195f,  0.173926f
};

static void quantize_row_turbo4333_pca_common(const float * GGML_RESTRICT x, block_turbo4333_pca_0 * GGML_RESTRICT y, int64_t k, int kv_kind) {
    turbo_init_rotation();

    assert(k % QK_TURBO4333_PCA == 0);
    const int nb = k / QK_TURBO4333_PCA;
    const int d  = QK_TURBO4333_PCA;
    const float sigma_ref = 1.0f / sqrtf((float) QK_TURBO4333_PCA);

    for (int block = 0; block < nb; ++block) {
        const float * src = x + block * d;
        const float * empvar = turbo_empvar_variances_for_group(QK_TURBO4333_PCA, block, kv_kind);

        float norm_sq = 0.0f;
        for (int i = 0; i < d; ++i) {
            norm_sq += src[i] * src[i];
        }
        const float norm = sqrtf(norm_sq);
        const float inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;
        float recon_norm_sq = 0.0f;

        memset(y[block].qs4, 0, sizeof(y[block].qs4));
        memset(y[block].qs3, 0, sizeof(y[block].qs3));
        memset(y[block].signs3, 0, sizeof(y[block].signs3));

        for (int i = 0; i < 32; ++i) {
            const float val = src[i] * inv_norm;
            float scale = 1.0f;
            if (empvar != NULL) {
                const float sigma = sqrtf(fmaxf(empvar[i], 1e-6f));
                scale = sigma / sigma_ref;
            }
            const uint8_t idx = (uint8_t) nearest_centroid_scaled(val, CENTROIDS_4BIT_REF, 16, scale);
            y[block].qs4[i / 2] |= (uint8_t) ((idx & 0xF) << ((i % 2) * 4));
            const float c = CENTROIDS_4BIT_REF[idx] * scale;
            recon_norm_sq += c * c;
        }

        for (int region = 0; region < 3; ++region) {
            const int base = 32 + region * 32;
            for (int i = 0; i < 32; ++i) {
                const float val = src[base + i] * inv_norm;
                float scale = 1.0f;
                if (empvar != NULL) {
                    const float sigma = sqrtf(fmaxf(empvar[base + i], 1e-6f));
                    scale = sigma / sigma_ref;
                }
                const uint8_t idx = (uint8_t) nearest_centroid_scaled(val, CENTROIDS_3BIT, 8, scale);
                y[block].qs3[region][i / 4] |= (uint8_t) ((idx & 0x3) << ((i % 4) * 2));
                if (idx & 0x4) {
                    y[block].signs3[region][i / 8] |= (uint8_t) (1u << (i % 8));
                }
                const float c = CENTROIDS_3BIT[idx] * scale;
                recon_norm_sq += c * c;
            }
        }

        const float recon_norm = sqrtf(recon_norm_sq);
        y[block].norm = GGML_FP32_TO_FP16((recon_norm > 1e-10f) ? norm / recon_norm : norm);
        y[block].pad  = GGML_FP32_TO_FP16(0.0f);
    }
}

static void quantize_row_turbo4333_pca_impl(const float * GGML_RESTRICT x, block_turbo4333_pca_0 * GGML_RESTRICT y, int64_t k) {
    quantize_row_turbo4333_pca_common(x, y, k, 1);
}

void quantize_row_turbo4333_pca_k_ref(const float * GGML_RESTRICT x, block_turbo4333_pca_0 * GGML_RESTRICT y, int64_t k) {
    quantize_row_turbo4333_pca_impl(x, y, k);
}

void quantize_row_turbo4333_pca_v_ref(const float * GGML_RESTRICT x, block_turbo4333_pca_0 * GGML_RESTRICT y, int64_t k) {
    quantize_row_turbo4333_pca_common(x, y, k, 2);
}

void dequantize_row_turbo4333_pca(const block_turbo4333_pca_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    turbo_init_rotation();

    assert(k % QK_TURBO4333_PCA == 0);
    const int nb = k / QK_TURBO4333_PCA;

    for (int block = 0; block < nb; ++block) {
        float rotated[QK_TURBO4333_PCA];
        const float norm = GGML_FP16_TO_FP32(x[block].norm);
        for (int i = 0; i < 32; ++i) {
            const uint8_t idx = (x[block].qs4[i / 2] >> ((i % 2) * 4)) & 0xF;
            rotated[i] = CENTROIDS_4BIT_REF[idx] * norm;
        }
        for (int region = 0; region < 3; ++region) {
            const int base = 32 + region * 32;
            for (int i = 0; i < 32; ++i) {
                const uint8_t low = (x[block].qs3[region][i / 4] >> ((i % 4) * 2)) & 0x3;
                const uint8_t hi  = ((x[block].signs3[region][i / 8] >> (i % 8)) & 1) << 2;
                rotated[base + i] = CENTROIDS_3BIT[low | hi] * norm;
            }
        }
        matvec(turbo_rotation_t, rotated, y + block * QK_TURBO4333_PCA, QK_TURBO4333_PCA);
    }
}

static size_t quantize_turbo4333_pca_impl(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                                          int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO4333_PCA == 0);

    size_t row_size = (n_per_row / QK_TURBO4333_PCA) * sizeof(block_turbo4333_pca_0);
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_row_turbo4333_pca_k_ref(
            src + row * n_per_row,
            (block_turbo4333_pca_0 *) ((char *) dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

size_t quantize_turbo4333_pca_k(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                                int64_t nrows, int64_t n_per_row, const float * imatrix) {
    return quantize_turbo4333_pca_impl(src, dst, nrows, n_per_row, imatrix);
}

size_t quantize_turbo4333_pca_v(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                                int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO4333_PCA == 0);

    size_t row_size = (n_per_row / QK_TURBO4333_PCA) * sizeof(block_turbo4333_pca_0);
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_row_turbo4333_pca_v_ref(
            src + row * n_per_row,
            (block_turbo4333_pca_0 *) ((char *) dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

static void quantize_row_turbo4322_pca_common(const float * GGML_RESTRICT x, block_turbo4322_pca_0 * GGML_RESTRICT y, int64_t k, int kv_kind) {
    turbo_init_rotation();

    assert(k % QK_TURBO4322_PCA == 0);
    const int nb = k / QK_TURBO4322_PCA;
    const int d  = QK_TURBO4322_PCA;
    const float sigma_ref = 1.0f / sqrtf((float) QK_TURBO4322_PCA);

    for (int block = 0; block < nb; ++block) {
        const float * src = x + block * d;
        const float * empvar = turbo_empvar_variances_for_group(QK_TURBO4322_PCA, block, kv_kind);

        float norm_sq = 0.0f;
        for (int i = 0; i < d; ++i) {
            norm_sq += src[i] * src[i];
        }
        const float norm = sqrtf(norm_sq);
        const float inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;
        float recon_norm_sq = 0.0f;

        memset(y[block].qs4, 0, sizeof(y[block].qs4));
        memset(y[block].qs3, 0, sizeof(y[block].qs3));
        memset(y[block].signs3, 0, sizeof(y[block].signs3));
        memset(y[block].qs2, 0, sizeof(y[block].qs2));

        for (int i = 0; i < 32; ++i) {
            const float val = src[i] * inv_norm;
            float scale = 1.0f;
            if (empvar != NULL) {
                const float sigma = sqrtf(fmaxf(empvar[i], 1e-6f));
                scale = sigma / sigma_ref;
            }
            const uint8_t idx = (uint8_t) nearest_centroid_scaled(val, CENTROIDS_4BIT_REF, 16, scale);
            y[block].qs4[i / 2] |= (uint8_t) ((idx & 0xF) << ((i % 2) * 4));
            recon_norm_sq += (CENTROIDS_4BIT_REF[idx] * scale) * (CENTROIDS_4BIT_REF[idx] * scale);
        }

        for (int i = 0; i < 32; ++i) {
            const int coord = 32 + i;
            const float val = src[coord] * inv_norm;
            float scale = 1.0f;
            if (empvar != NULL) {
                const float sigma = sqrtf(fmaxf(empvar[coord], 1e-6f));
                scale = sigma / sigma_ref;
            }
            const uint8_t idx = (uint8_t) nearest_centroid_scaled(val, CENTROIDS_3BIT, 8, scale);
            y[block].qs3[i / 4] |= (uint8_t) ((idx & 0x3) << ((i % 4) * 2));
            if (idx & 0x4) {
                y[block].signs3[i / 8] |= (uint8_t) (1u << (i % 8));
            }
            recon_norm_sq += (CENTROIDS_3BIT[idx] * scale) * (CENTROIDS_3BIT[idx] * scale);
        }

        for (int region = 0; region < 2; ++region) {
            const int base = 64 + region * 32;
            for (int i = 0; i < 32; ++i) {
                const float val = src[base + i] * inv_norm;
                float scale = 1.0f;
                if (empvar != NULL) {
                    const float sigma = sqrtf(fmaxf(empvar[base + i], 1e-6f));
                    scale = sigma / sigma_ref;
                }
                const uint8_t idx = (uint8_t) nearest_centroid_scaled(val, CENTROIDS_2BIT, 4, scale);
                y[block].qs2[region][i / 4] |= (uint8_t) ((idx & 0x3) << ((i % 4) * 2));
                recon_norm_sq += (CENTROIDS_2BIT[idx] * scale) * (CENTROIDS_2BIT[idx] * scale);
            }
        }

        const float recon_norm = sqrtf(recon_norm_sq);
        y[block].norm = GGML_FP32_TO_FP16((recon_norm > 1e-10f) ? norm / recon_norm : norm);
        y[block].pad  = GGML_FP32_TO_FP16(0.0f);
    }
}

void quantize_row_turbo4322_pca_k_ref(const float * GGML_RESTRICT x, block_turbo4322_pca_0 * GGML_RESTRICT y, int64_t k) {
    quantize_row_turbo4322_pca_common(x, y, k, 1);
}

void quantize_row_turbo4322_pca_v_ref(const float * GGML_RESTRICT x, block_turbo4322_pca_0 * GGML_RESTRICT y, int64_t k) {
    quantize_row_turbo4322_pca_common(x, y, k, 2);
}

void dequantize_row_turbo4322_pca(const block_turbo4322_pca_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    turbo_init_rotation();

    assert(k % QK_TURBO4322_PCA == 0);
    const int nb = k / QK_TURBO4322_PCA;

    for (int block = 0; block < nb; ++block) {
        float rotated[QK_TURBO4322_PCA];
        const float norm = GGML_FP16_TO_FP32(x[block].norm);
        for (int i = 0; i < 32; ++i) {
            const uint8_t idx = (x[block].qs4[i / 2] >> ((i % 2) * 4)) & 0xF;
            rotated[i] = CENTROIDS_4BIT_REF[idx] * norm;
        }
        for (int i = 0; i < 32; ++i) {
            const uint8_t low = (x[block].qs3[i / 4] >> ((i % 4) * 2)) & 0x3;
            const uint8_t hi  = ((x[block].signs3[i / 8] >> (i % 8)) & 1) << 2;
            rotated[32 + i] = CENTROIDS_3BIT[low | hi] * norm;
        }
        for (int region = 0; region < 2; ++region) {
            const int base = 64 + region * 32;
            for (int i = 0; i < 32; ++i) {
                const uint8_t idx = (x[block].qs2[region][i / 4] >> ((i % 4) * 2)) & 0x3;
                rotated[base + i] = CENTROIDS_2BIT[idx] * norm;
            }
        }
        matvec(turbo_rotation_t, rotated, y + block * QK_TURBO4322_PCA, QK_TURBO4322_PCA);
    }
}

size_t quantize_turbo4322_pca_k(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                                int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO4322_PCA == 0);

    size_t row_size = (n_per_row / QK_TURBO4322_PCA) * sizeof(block_turbo4322_pca_0);
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_row_turbo4322_pca_k_ref(src + row * n_per_row,
                                         (block_turbo4322_pca_0 *) ((char *) dst + row * row_size),
                                         n_per_row);
    }
    return nrows * row_size;
}

size_t quantize_turbo4322_pca_v(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                                int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO4322_PCA == 0);

    size_t row_size = (n_per_row / QK_TURBO4322_PCA) * sizeof(block_turbo4322_pca_0);
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_row_turbo4322_pca_v_ref(src + row * n_per_row,
                                         (block_turbo4322_pca_0 *) ((char *) dst + row * row_size),
                                         n_per_row);
    }
    return nrows * row_size;
}

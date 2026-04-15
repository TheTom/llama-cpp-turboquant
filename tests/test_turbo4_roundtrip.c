/*
 * Self-contained turbo4_0 (QJL mode) round-trip test.
 * Zero external dependencies — only standard C library.
 *
 * Build:  gcc -O0 -g test_turbo4_roundtrip.c -lm -o test_turbo4_roundtrip
 * Run:    ./test_turbo4_roundtrip
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <assert.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---- Minimal fp16 ---- */
typedef uint16_t ggml_half;

static ggml_half fp32_to_fp16(float f) {
    union { float f; uint32_t u; } v = { f };
    uint32_t b = v.u;
    uint16_t sign = (b >> 16) & 0x8000;
    int exp = ((b >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (b >> 13) & 0x3FF;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7C00;
    return sign | (exp << 10) | mant;
}

static float fp16_to_fp32(ggml_half h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0) {
        if (mant == 0) { union { uint32_t u; float f; } r = { sign }; return r.f; }
        while (!(mant & 0x400)) { mant <<= 1; exp--; }
        exp++; mant &= 0x3FF;
    } else if (exp == 31) {
        union { uint32_t u; float f; } r = { sign | 0x7F800000 | (mant << 13) };
        return r.f;
    }
    uint32_t bits = sign | ((exp + 112) << 23) | (mant << 13);
    union { uint32_t u; float f; } r = { bits };
    return r.f;
}

/* ---- Block definition ---- */
#define QK_TURBO4 128

typedef struct {
    ggml_half  norm;
    ggml_half  rnorm;
    uint8_t    qs[QK_TURBO4 * 3 / 8];  /* 48 bytes */
    uint8_t    signs[QK_TURBO4 / 8];   /* 16 bytes */
} block_turbo4_0;

/* ---- Constants ---- */
#define TURBO_SEED_ROTATION 42
#define TURBO_D             128
#define TURBO_QJL_CONST     1.2533141373155003f

static const float CENTROIDS_3BIT[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

/* ---- PRNG ---- */
static uint64_t turbo_prng_state;
static void turbo_prng_seed(uint64_t seed) { turbo_prng_state = seed; }
static double turbo_prng_normal(void) {
    turbo_prng_state = turbo_prng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u1 = (double)(turbo_prng_state >> 11) / (double)(1ULL << 53);
    if (u1 < 1e-15) u1 = 1e-15;
    turbo_prng_state = turbo_prng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u2 = (double)(turbo_prng_state >> 11) / (double)(1ULL << 53);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* ---- Orthogonal matrix via QR ---- */
static float turbo_rotation[TURBO_D * TURBO_D];
static float turbo_rotation_t[TURBO_D * TURBO_D];
static int   turbo_rotation_initialized = 0;

static void init_orthogonal(float *mat, float *mat_t, uint64_t seed) {
    const int d = TURBO_D;
    turbo_prng_seed(seed);
    float G[TURBO_D * TURBO_D];
    for (int i = 0; i < d * d; i++) G[i] = (float)turbo_prng_normal();
    memcpy(mat, G, d * d * sizeof(float));
    for (int j = 0; j < d; j++) {
        float norm = 0.0f;
        for (int i = 0; i < d; i++) norm += mat[i * d + j] * mat[i * d + j];
        norm = sqrtf(norm);
        if (norm > 1e-10f) for (int i = 0; i < d; i++) mat[i * d + j] /= norm;
        for (int k = j + 1; k < d; k++) {
            float dot = 0.0f;
            for (int i = 0; i < d; i++) dot += mat[i * d + j] * mat[i * d + k];
            for (int i = 0; i < d; i++) mat[i * d + k] -= dot * mat[i * d + j];
        }
    }
    for (int i = 0; i < d; i++)
        for (int j = 0; j < d; j++)
            mat_t[i * d + j] = mat[j * d + i];
}

static void turbo_init_rotation(void) {
    if (turbo_rotation_initialized) return;
    init_orthogonal(turbo_rotation, turbo_rotation_t, TURBO_SEED_ROTATION);
    turbo_rotation_initialized = 1;
}

/* QJL matrix — must use different seed (1042) from rotation (42) */
#define TURBO_SEED_QJL 1042
static float turbo_qjl_matrix[TURBO_D * TURBO_D];
static float turbo_qjl_matrix_t[TURBO_D * TURBO_D];
static int   turbo_qjl_initialized = 0;

static void turbo_init_qjl(void) {
    if (turbo_qjl_initialized) return;
    init_orthogonal(turbo_qjl_matrix, turbo_qjl_matrix_t, TURBO_SEED_QJL);
    turbo_qjl_initialized = 1;
}

/* ---- matvec ---- */
static void matvec(const float *M, const float *x, float *y, int d) {
    for (int i = 0; i < d; i++) {
        float sum = 0.0f;
        for (int j = 0; j < d; j++) sum += M[i * d + j] * x[j];
        y[i] = sum;
    }
}

/* ---- nearest centroid ---- */
static int nearest_centroid_3bit(float val) {
    if (val < -0.154259f) return 0;
    if (val < -0.091775f) return 1;
    if (val < -0.043589f) return 2;
    if (val <  0.000000f) return 3;
    if (val <  0.043589f) return 4;
    if (val <  0.091775f) return 5;
    if (val <  0.154259f) return 6;
    return 7;
}

/* ---- quantize ---- */
static void test_quantize(const float *x, block_turbo4_0 *y, int64_t k) {
    turbo_init_rotation();
    turbo_init_qjl();
    assert(k % QK_TURBO4 == 0);
    const int nb = k / QK_TURBO4, d = QK_TURBO4;

    for (int block = 0; block < nb; block++) {
        const float *src = x + block * d;
        float norm_sq = 0;
        for (int i = 0; i < d; i++) norm_sq += src[i] * src[i];
        float norm = sqrtf(norm_sq);

        float normalized[128];
        if (norm > 1e-10f) {
            float inv = 1.0f / norm;
            for (int i = 0; i < d; i++) normalized[i] = src[i] * inv;
        } else memset(normalized, 0, sizeof(normalized));

        float rotated[128];
        matvec(turbo_rotation, normalized, rotated, d);

        uint8_t indices[128];
        for (int i = 0; i < d; i++) indices[i] = nearest_centroid_3bit(rotated[i]);

        float reconstructed[128];
        for (int i = 0; i < d; i++) reconstructed[i] = CENTROIDS_3BIT[indices[i]];

        float mse_recon[128];
        matvec(turbo_rotation_t, reconstructed, mse_recon, d);

        float residual[128];
        for (int i = 0; i < d; i++) residual[i] = normalized[i] - mse_recon[i];

        float norm_rs = 0;
        for (int i = 0; i < d; i++) norm_rs += residual[i] * residual[i];
        norm_rs = sqrtf(norm_rs);

        float projected[128];
        matvec(turbo_qjl_matrix, residual, projected, d);

        y[block].norm  = fp32_to_fp16(norm);
        y[block].rnorm = fp32_to_fp16(norm_rs);

        memset(y[block].qs, 0, 48);
        for (int i = 0; i < d; i++) {
            int bit_offset = i * 3, byte_idx = bit_offset / 8, bit_pos = bit_offset % 8;
            uint16_t val = (uint16_t)(indices[i] & 0x7);
            y[block].qs[byte_idx] |= (uint8_t)(val << bit_pos);
            if (bit_pos > 5 && byte_idx + 1 < 48)
                y[block].qs[byte_idx + 1] |= (uint8_t)(val >> (8 - bit_pos));
        }

        memset(y[block].signs, 0, 16);
        for (int i = 0; i < d; i++)
            if (projected[i] >= 0.0f)
                y[block].signs[i / 8] |= (1 << (i % 8));
    }
}

/* ---- dequantize ---- */
static void test_dequantize(const block_turbo4_0 *x, float *y, int64_t k) {
    turbo_init_rotation();
    turbo_init_qjl();
    assert(k % QK_TURBO4 == 0);
    const int nb = k / QK_TURBO4, d = QK_TURBO4;

    for (int block = 0; block < nb; block++) {
        float norm = fp16_to_fp32(x[block].norm);

        uint8_t indices[128];
        for (int i = 0; i < d; i++) {
            int bit_offset = i * 3, byte_idx = bit_offset / 8, bit_pos = bit_offset % 8;
            uint16_t raw = (uint16_t)x[block].qs[byte_idx];
            if (byte_idx + 1 < 48) raw |= (uint16_t)x[block].qs[byte_idx + 1] << 8;
            indices[i] = (uint8_t)((raw >> bit_pos) & 0x7);
        }

        float signs[128];
        for (int i = 0; i < d; i++)
            signs[i] = (x[block].signs[i / 8] & (1 << (i % 8))) ? 1.0f : -1.0f;

        float rnorm = fp16_to_fp32(x[block].rnorm);
        float qjl_scale = TURBO_QJL_CONST / sqrtf((float)d) * rnorm;

        float rotated_recon[128];
        for (int i = 0; i < d; i++) rotated_recon[i] = CENTROIDS_3BIT[indices[i]];

        float mse_recon[128];
        matvec(turbo_rotation_t, rotated_recon, mse_recon, d);

        float qjl_recon[128];
        matvec(turbo_qjl_matrix_t, signs, qjl_recon, d);
        for (int i = 0; i < d; i++) qjl_recon[i] *= qjl_scale;

        float *dst = y + block * d;
        for (int i = 0; i < d; i++)
            dst[i] = (mse_recon[i] + qjl_recon[i]) * norm;
    }
}

/* ---- test runner ---- */
static float rand_normal_test(void) {
    float u1 = (float)rand() / RAND_MAX, u2 = (float)rand() / RAND_MAX;
    if (u1 < 1e-10f) u1 = 1e-10f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

static void run_test(const char *name, const float *input, int dim) {
    int nb = dim / QK_TURBO4;
    block_turbo4_0 *qbuf = calloc(nb, sizeof(block_turbo4_0));
    test_quantize(input, qbuf, dim);

    float *output = calloc(dim, sizeof(float));
    test_dequantize(qbuf, output, dim);

    float mse = 0, max_err = 0, in_sq = 0;
    for (int i = 0; i < dim; i++) {
        float d = output[i] - input[i];
        mse += d * d;
        if (fabsf(d) > max_err) max_err = fabsf(d);
        in_sq += input[i] * input[i];
    }
    mse /= dim;

    float dot_in = 0, dot_out = 0;
    for (int i = 0; i < dim; i++) { dot_in += input[i]*input[i]; dot_out += output[i]*input[i]; }

    printf("%-30s | MSE=%.6e | MaxErr=%.6f | <x,x>=%.4f | <x',x>=%.4f | ratio=%.4f\n",
           name, mse, max_err, dot_in, dot_out, dot_out/(dot_in+1e-10f));
    printf("  in[0:8] :");
    for (int i = 0; i < 8 && i < dim; i++) printf(" %9.5f", input[i]);
    printf(" ...\n  out[0:8]:");
    for (int i = 0; i < 8 && i < dim; i++) printf(" %9.5f", output[i]);
    printf(" ...\n");

    int n_nan=0, n_inf=0;
    for (int i = 0; i < dim; i++) { if(isnan(output[i]))n_nan++; if(isinf(output[i]))n_inf++; }
    if (n_nan||n_inf) printf("  *** %d NaN, %d Inf ***\n", n_nan, n_inf);

    float rel = mse / (in_sq/dim + 1e-10f);
    printf("  RelMSE=%.6f %s\n\n", rel, rel>1?"*** FAIL ***":rel>0.1?"*** WARN ***":"PASS");

    free(qbuf); free(output);
}

int main(void) {
    srand(12345);
    printf("=== turbo4_0 QJL round-trip test ===\n");
    printf("sizeof(block_turbo4_0) = %zu\n\n", sizeof(block_turbo4_0));

    { float x[128]={0}; x[0]=1; run_test("unit_e0", x, 128); }
    { float x[128]; for(int i=0;i<128;i++) x[i]=1.0f/sqrtf(128); run_test("uniform", x, 128); }
    { float x[128]; for(int i=0;i<128;i++) x[i]=rand_normal_test()*0.1f; run_test("gauss_0.1", x, 128); }
    { float x[128]; for(int i=0;i<128;i++) x[i]=rand_normal_test(); run_test("gauss_1.0", x, 128); }
    { float x[128]={0}; x[0]=5;x[32]=-3;x[64]=2;x[96]=-1; run_test("sparse", x, 128); }

    printf("--- dot product test ---\n");
    { float k[128],q[128];
      for(int i=0;i<128;i++){k[i]=rand_normal_test()*0.1f; q[i]=rand_normal_test()*0.1f;}
      block_turbo4_0 qb; memset(&qb,0,sizeof(qb));
      test_quantize(k,&qb,128);
      float kr[128]; test_dequantize(&qb,kr,128);
      float d0=0,d1=0;
      for(int i=0;i<128;i++){d0+=k[i]*q[i]; d1+=kr[i]*q[i];}
      printf("  <k,q>=%.6f  <k',q>=%.6f  err=%.6f  rel=%.2f%%\n\n",
             d0, d1, fabsf(d1-d0), fabsf(d1-d0)/(fabsf(d0)+1e-10f)*100);
    }
    return 0;
}
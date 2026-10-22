// Unit tests for quantization specific functions - quantize, dequantize and dot product

#include "ggml.h"
#include "ggml-cpu.h"

#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

constexpr float MAX_QUANTIZATION_REFERENCE_ERROR = 0.0001f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR = 0.002f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_BINARY = 0.025f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_TERNARY = 0.01f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_2BITS = 0.0075f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_3BITS = 0.0040f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_3BITS_XXS = 0.0050f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_FP4 = 0.0030f;
constexpr float MAX_DOT_PRODUCT_ERROR = 0.02f;
constexpr float MAX_DOT_PRODUCT_ERROR_LOWBIT = 0.04f;
constexpr float MAX_DOT_PRODUCT_ERROR_FP4 = 0.03f;
constexpr float MAX_DOT_PRODUCT_ERROR_BINARY = 0.40f;
constexpr float MAX_DOT_PRODUCT_ERROR_TERNARY = 0.15f;

static const char* RESULT_STR[] = {"ok", "FAILED"};


// Generate synthetic data
static void generate_data(float offset, size_t n, float * dst) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = 0.1 + 2*cosf(i + offset);
    }
}

// Calculate RMSE between two float arrays
static float array_rmse(const float * a1, const float * a2, size_t n) {
    double sum = 0;
    for (size_t i = 0; i < n; i++) {
        double diff = a1[i] - a2[i];
        sum += diff * diff;
    }
    return sqrtf(sum) / n;
}

// Total quantization error on test data
static float total_quantization_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data) {
    // For types whose vec_dot_type is GGML_TYPE_F32 (e.g. turbo quants), from_float writes
    // test_size*sizeof(float) bytes, which exceeds the legacy 2*test_size sizing.
    std::vector<uint8_t> tmp_q(std::max<size_t>(2*test_size, test_size * sizeof(float)));
    std::vector<float> tmp_out(test_size);

    qfns_cpu->from_float(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out.data(), test_size);
    return array_rmse(test_data, tmp_out.data(), test_size);
}

// Total quantization error on test data
static float reference_quantization_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data) {
    std::vector<uint8_t> tmp_q(std::max<size_t>(2*test_size, test_size * sizeof(float)));
    std::vector<float> tmp_out(test_size);
    std::vector<float> tmp_out_ref(test_size);

    // FIXME: why is done twice?
    qfns_cpu->from_float(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out.data(), test_size);

    qfns->from_float_ref(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out_ref.data(), test_size);

    return array_rmse(tmp_out.data(), tmp_out_ref.data(), test_size);
}

static float dot_product(const float * a1, const float * a2, size_t test_size) {
    double sum = 0;
    for (size_t i = 0; i < test_size; i++) {
        sum += a1[i] * a2[i];
    }
    return sum;
}

// Total dot product error
static float dot_product_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data1, const float * test_data2) {
    GGML_UNUSED(qfns);

    std::vector<uint8_t> tmp_q1(std::max<size_t>(2*test_size, test_size * sizeof(float)));
    std::vector<uint8_t> tmp_q2(std::max<size_t>(2*test_size, test_size * sizeof(float)));

    const auto * vdot = ggml_get_type_traits_cpu(qfns_cpu->vec_dot_type);

    qfns_cpu->from_float(test_data1, tmp_q1.data(), test_size);
    vdot->from_float(test_data2, tmp_q2.data(), test_size);

    float result = INFINITY;
    qfns_cpu->vec_dot(test_size, &result, 0, tmp_q1.data(), 0, tmp_q2.data(), 0, 1);

    const float dot_ref = dot_product(test_data1, test_data2, test_size);

    return fabsf(result - dot_ref) / test_size;
}

static int test_vec_dot_f32(bool verbose) {
    const auto * f32 = ggml_get_type_traits_cpu(GGML_TYPE_F32);
    int num_failed = 0;
    for (int n : {1, 2, 3, 5, 7, 8, 15, 16, 17, 31, 33, 63, 67, 127, 129, 193, 255, 1023}) {
        std::vector<float> a(n);
        std::vector<float> b(n);
        generate_data(0.0, n, a.data());
        generate_data(1.0, n, b.data());

        float result = 0.0f;
        f32->vec_dot(n, &result, 0, a.data(), 0, b.data(), 0, 1);
        const float ref = dot_product(a.data(), b.data(), n);
        const float error = fabsf(result - ref) / n;

        const bool failed = !(error < MAX_QUANTIZATION_REFERENCE_ERROR);
        num_failed += failed;
        if (failed || verbose) {
            printf(" f32 vec_dot n=%4d:                 %s (ref=%f got=%f err=%f)\n",
                   n, RESULT_STR[failed], ref, result, error);
        }
    }
    return num_failed;
}

static int test_vec_dot_q(bool verbose) {
    int num_failed = 0;

    const size_t test_size = 32 * 128;

    std::vector<float> test_data(test_size);
    std::vector<float> test_data2(test_size);

    generate_data(0.0, test_data.size(), test_data.data());
    generate_data(1.0, test_data2.size(), test_data2.data());

    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        ggml_type type = (ggml_type) i;
        const auto * qfns = ggml_get_type_traits(type);
        const auto * qfns_cpu = ggml_get_type_traits_cpu(type);

        // deprecated - skip
        if (qfns->blck_size == 0) {
            continue;
        }

        // TurboQuant KV-cache types (TURBO2_0/TURBO3_0/TURBO4_0/TURBO6_0/TURBO5_0) intentionally keep
        // their dequantized output in the WHT-rotated domain; the inverse WHT is
        // applied separately via GGML_OP_TURBO_WHT in the attention graph. They do
        // not round-trip through float space, so the total/reference/dot-product
        // error tests in this harness are not applicable.
        if (type == GGML_TYPE_TURBO2_0 || type == GGML_TYPE_TURBO3_0 || type == GGML_TYPE_TURBO4_0 ||
            type == GGML_TYPE_TURBO5_0 || type == GGML_TYPE_TURBO6_0) {
            printf("Testing %s (skipped: rotated-domain KV quant)\n", ggml_type_name(type));
            continue;
        }

        const ggml_type ei = (ggml_type)i;

        printf("Testing %s\n", ggml_type_name((ggml_type) i));
        ggml_quantize_init(ei);

        if (qfns_cpu->from_float && qfns->to_float) {
            const float total_error = total_quantization_error(qfns, qfns_cpu, test_size, test_data.data());
            const float max_quantization_error =
                type == GGML_TYPE_Q1_0    ? MAX_QUANTIZATION_TOTAL_ERROR_BINARY :
                type == GGML_TYPE_TQ1_0   ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_TQ2_0   ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_Q2_0    ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_Q2_K    ? MAX_QUANTIZATION_TOTAL_ERROR_2BITS :
                type == GGML_TYPE_IQ2_S   ? MAX_QUANTIZATION_TOTAL_ERROR_2BITS :
                type == GGML_TYPE_Q3_K    ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS :
                type == GGML_TYPE_IQ3_S   ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS :
                type == GGML_TYPE_IQ3_XXS ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS_XXS :
                type == GGML_TYPE_TQ3_1S  ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS :
                type == GGML_TYPE_NVFP4   ? MAX_QUANTIZATION_TOTAL_ERROR_FP4 : MAX_QUANTIZATION_TOTAL_ERROR;
            bool failed = !(total_error < max_quantization_error);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s absolute quantization error:    %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], total_error);
            }

            const float reference_error = reference_quantization_error(qfns, qfns_cpu, test_size, test_data.data());
            failed = !(reference_error < MAX_QUANTIZATION_REFERENCE_ERROR);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s reference implementation error: %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], reference_error);
            }

            const float vec_dot_error = dot_product_error(qfns, qfns_cpu, test_size, test_data.data(), test_data2.data());
            const float max_allowed_error = type == GGML_TYPE_Q2_K || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_XXS ||
                type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ2_S ||
                type == GGML_TYPE_TQ3_1S
                ? MAX_DOT_PRODUCT_ERROR_LOWBIT
                : type == GGML_TYPE_Q1_0
                ? MAX_DOT_PRODUCT_ERROR_BINARY
                : type == GGML_TYPE_TQ1_0 || type == GGML_TYPE_TQ2_0 || type == GGML_TYPE_Q2_0
                ? MAX_DOT_PRODUCT_ERROR_TERNARY
                : type == GGML_TYPE_NVFP4
                ? MAX_DOT_PRODUCT_ERROR_FP4
                : MAX_DOT_PRODUCT_ERROR;
            failed = !(vec_dot_error < max_allowed_error);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s dot product error:              %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], vec_dot_error);
            }
        }
    }

    return num_failed;
}

// The 3-bit index extraction of the TQ3_1S MMQ tile loader, transcribed from
// ggml/src/ggml-cuda/mmq-load-tiles.cuh (tq3_1s_qs_to_int8x4). The CUDA side
// picks one of four groups and returns four indices; this copy does the same so
// the test can compare it with the CPU codec's unpacking.
//
// Keep in sync with the CUDA loader: the test below fails if the two disagree.
static void tq3_1s_unpack_indices_cuda(const uint8_t * qs, int kqsx, int idx[4]) {
    const int g    = kqsx >> 1;        // group 0..3
    const int sub  = (kqsx & 1) * 4;   // first element within the group
    const uint8_t * qp = qs + 3*g;
    const uint32_t packed = (uint32_t) qp[0] | ((uint32_t) qp[1] << 8) | ((uint32_t) qp[2] << 16);
    idx[0] = (packed >> (3*sub + 0)) & 7;
    idx[1] = (packed >> (3*sub + 3)) & 7;
    idx[2] = (packed >> (3*sub + 6)) & 7;
    idx[3] = (packed >> (3*sub + 9)) & 7;
}

// The CPU codec's unpacking of one group of eight indices, transcribed from
// ggml/src/ggml-turbo-quant.c (dequantize_row_tq3_1s).
static void tq3_1s_unpack_indices_cpu(const uint8_t * qp, int idx[8]) {
    idx[0] =  qp[0]       & 7;
    idx[1] = (qp[0] >> 3) & 7;
    idx[2] = ((qp[0] >> 6) | (qp[1] << 2)) & 7;
    idx[3] = (qp[1] >> 1) & 7;
    idx[4] = (qp[1] >> 4) & 7;
    idx[5] = ((qp[1] >> 7) | (qp[2] << 1)) & 7;
    idx[6] = (qp[2] >> 2) & 7;
    idx[7] = (qp[2] >> 5) & 7;
}

// The TQ3_1S centroids and the int8 table the CUDA loader uses, transcribed
// from ggml/src/ggml-turbo-quant.c (TQ3_0_CENTROIDS) and
// ggml/src/ggml-common.h (kvalues_tq3).
static const float tq3_1s_centroids[8] = {
    -1.996684f, -1.291398f, -0.740341f, -0.247508f, 0.230106f, 0.725222f, 1.277503f, 1.988943f
};

static const int8_t tq3_1s_kvalues[8] = { -127, -82, -47, -16, 15, 46, 81, 127 };

// The loader must select the same index for every element the CPU codec reads,
// for every possible 3-byte group. Exhaustive over 2^24 words per group: this is
// the contract that lets the MMQ path stand in for the CPU codec.
//
// Each group gets a different word, so the group selection is covered too: a
// loader that read the wrong group would disagree with the CPU for three of the
// four groups.
static bool test_tq3_1s_mmq_unpack_matches_cpu() {
    static const uint32_t group_xor[4] = { 0x000000, 0x010203, 0x040506, 0x070809 };

    for (uint32_t w = 0; w < (1u << 24); w++) {
        uint8_t qs[12];
        for (int g = 0; g < 4; g++) {
            const uint32_t v = w ^ group_xor[g];
            qs[3*g + 0] = (v >>  0) & 0xff;
            qs[3*g + 1] = (v >>  8) & 0xff;
            qs[3*g + 2] = (v >> 16) & 0xff;
        }

        for (int g = 0; g < 4; g++) {
            int cpu[8];
            tq3_1s_unpack_indices_cpu(qs + 3*g, cpu);

            for (int half = 0; half < 2; half++) {
                int cuda[4];
                tq3_1s_unpack_indices_cuda(qs, 2*g + half, cuda);
                for (int j = 0; j < 4; j++) {
                    if (cuda[j] != cpu[4*half + j]) {
                        fprintf(stderr, "%s: word %06x group %d quad %d: cuda %d != cpu %d\n",
                                __func__, w, g, 2*g + half, cuda[j], cpu[4*half + j]);
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

// The int8 table the loader uses must be the CPU centroid table scaled to 127,
// and the rescale must bring it back: kvalues[i] * (max|c|/127) ~ c[i].
static bool test_tq3_1s_mmq_centroid_table() {
    const float max_abs = 1.996684f;   // max|CENTROID|, from TQ3_0_CENTROIDS
    const float rescale = max_abs / 127.0f;
    for (int i = 0; i < 8; i++) {
        const int expected = (int) roundf(tq3_1s_centroids[i] * 127.0f / max_abs);
        if (tq3_1s_kvalues[i] != expected) {
            fprintf(stderr, "%s: entry %d is %d, expected %d\n",
                    __func__, i, tq3_1s_kvalues[i], expected);
            return false;
        }
        const float reconstructed = tq3_1s_kvalues[i] * rescale;
        const float err = fabsf(reconstructed - tq3_1s_centroids[i]) / max_abs;
        if (err > 0.03f) {
            fprintf(stderr, "%s: entry %d reconstructs to %f, expected %f (%.1f%% of full scale)\n",
                    __func__, i, reconstructed, tq3_1s_centroids[i], 100*err);
            return false;
        }
    }
    return true;
}

int main(int argc, char * argv[]) {
    bool verbose = false;

    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];

        if (arg == "-v") {
            verbose = true;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }

    ggml_cpu_init();

    int num_failed = 0;

    num_failed += test_vec_dot_f32(verbose);
    num_failed += test_vec_dot_q(verbose);

    // The TQ3_1S MMQ tile loader (ggml/src/ggml-cuda/mmq-load-tiles.cuh) replaces the
    // CPU codec's unpacking with an equivalent bit extraction, so the two must agree.
    {
        const bool unpack_ok = test_tq3_1s_mmq_unpack_matches_cpu();
        const bool table_ok  = test_tq3_1s_mmq_centroid_table();
        if (!unpack_ok || !table_ok) {
            num_failed++;
        }
        if (!unpack_ok || !table_ok || verbose) {
            // RESULT_STR is indexed by the failure flag, not by a success bool
            printf("%5s TQ3_1S MMQ unpack matches the CPU codec: %s\n", RESULT_STR[!unpack_ok], RESULT_STR[!unpack_ok]);
            printf("%5s TQ3_1S MMQ centroid table:               %s\n", RESULT_STR[!table_ok],  RESULT_STR[!table_ok]);
        }
    }

    if (num_failed || verbose) {
        printf("%d tests failed\n", num_failed);
    }

    return num_failed > 0;
}

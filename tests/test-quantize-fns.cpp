// Unit tests for quantization specific functions - quantize, dequantize and dot product

#include "ggml.h"
#include "ggml-cpu.h"

// The TQ3_1S block layout and the int8 table the MMQ loader reads, taken from
// the real headers. The table is defined once per translation unit, so this
// gives the test the same values the CUDA loader uses.
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

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
// This is a check on the derivation, not on the loader: the copy can drift from
// the CUDA code without the test noticing. The loader is covered by
// test-backend-ops -o MUL_MAT -p type_a=tq3_1s with GGML_TQ_MMQ=1.
static void tq3_1s_unpack_indices_cuda(const uint8_t * qs, int kqsx, int idx[4]) {
    const int g    = kqsx >> 1;        // group 0..3
    const int sub  = (kqsx & 1) * 4;   // first element within the group
    const uint8_t * qp = qs + 3*g;
    const uint32_t packed = (uint32_t) qp[0] | ((uint32_t) qp[1] << 8) | ((uint32_t) qp[2] << 16);
    const uint32_t q = packed >> (3*sub);
    idx[0] =  q       & 7;
    idx[1] = (q >> 3) & 7;
    idx[2] = (q >> 6) & 7;
    idx[3] = (q >> 9) & 7;
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

// The int8 table the CUDA loader reads, taken from the real header rather than
// copied here, so an edit to it fails the test below.
static const int8_t * tq3_1s_kvalues = kvalues_tq3;

// The CPU codec's centroid for every index, recovered from the real decoder
// instead of copied from the source.
//
// The decoder applies the inverse RHT, so a block whose 32 elements all carry
// one index decodes to that centroid times a fixed vector. The vector does not
// depend on the index, so the decoded values are proportional to the centroids.
//
// The recovered values carry one common factor, which the caller removes by
// normalizing with the largest absolute centroid.
static bool tq3_1s_cpu_centroids(float centroid[8]) {
    const ggml_type_traits * traits = ggml_get_type_traits(GGML_TYPE_TQ3_1S);

    float decoded[8][32];
    for (int i = 0; i < 8; i++) {
        block_tq3_1s block = {};
        block.d0 = ggml_fp32_to_fp16(1.0f);
        block.d1 = ggml_fp32_to_fp16(1.0f);

        // every group holds the same index in all 8 of its slots
        for (int g = 0; g < 4; g++) {
            uint32_t word = 0;
            for (int k = 0; k < 8; k++) {
                word |= (uint32_t) i << (3*k);
            }
            block.qs[3*g + 0] = (uint8_t) (word >>  0);
            block.qs[3*g + 1] = (uint8_t) (word >>  8);
            block.qs[3*g + 2] = (uint8_t) (word >> 16);
        }
        traits->to_float(&block, decoded[i], 32);
    }

    // the largest component is the same element for every index, because the
    // recovered values differ only by a scale factor
    int best = 0;
    for (int j = 1; j < 32; j++) {
        if (fabsf(decoded[0][j]) > fabsf(decoded[0][best])) {
            best = j;
        }
    }
    if (decoded[0][best] == 0.0f) {
        return false;
    }

    // the codec's centroids ascend with the index, so take the sign that makes
    // the recovered values ascend too
    bool ascending = true;
    for (int i = 1; i < 8; i++) {
        if (decoded[i][best] < decoded[i-1][best]) {
            ascending = false;
            break;
        }
    }
    const float sign = ascending ? 1.0f : -1.0f;
    for (int i = 0; i < 8; i++) {
        centroid[i] = sign * decoded[i][best];
    }
    return true;
}

// The loader must select the same index for every element the CPU codec reads,
// for every possible 3-byte group. Exhaustive over 2^24 words per group.
//
// Each group gets a different word, so the group selection is covered too: a
// loader that read the wrong group would disagree with the CPU for three of the
// four groups.
//
// This checks the derivation above against the CPU codec, not the CUDA code.
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

// The int8 table the loader uses must be the CPU codec's centroid table scaled
// to int8, and the rescale must bring it back: kvalues[i] * (max|c|/127) ~ c[i].
static bool test_tq3_1s_mmq_centroid_table() {
    float centroid[8];
    if (!tq3_1s_cpu_centroids(centroid)) {
        fprintf(stderr, "%s: could not recover the CPU centroids\n", __func__);
        return false;
    }

    // the recovered centroids share one unknown factor, so normalize with the
    // largest absolute one: the table stores that one at 127 in absolute value
    float max_abs = 0.0f;
    for (int i = 0; i < 8; i++) {
        max_abs = fmaxf(max_abs, fabsf(centroid[i]));
    }
    if (max_abs <= 0.0f) {
        fprintf(stderr, "%s: the recovered centroids are all zero\n", __func__);
        return false;
    }

    for (int i = 0; i < 8; i++) {
        const int expected = (int) roundf(centroid[i] * 127.0f / max_abs);
        if (tq3_1s_kvalues[i] != expected) {
            fprintf(stderr, "%s: entry %d is %d, expected %d\n",
                    __func__, i, tq3_1s_kvalues[i], expected);
            return false;
        }
        const float reconstructed = tq3_1s_kvalues[i] * (max_abs / 127.0f);
        const float err = fabsf(reconstructed - centroid[i]) / max_abs;
        if (err > 0.03f) {
            fprintf(stderr, "%s: entry %d reconstructs to %f, expected %f (%.1f%% of full scale)\n",
                    __func__, i, reconstructed, centroid[i], 100*err);
            return false;
        }
    }
    return true;
}

// The TQ3_1S MMQ loader selects the four centroids of a quad with one byte permute over
// two constants holding the int8 table. The constants are transcribed from
// ggml/src/ggml-cuda/mmq-load-tiles.cuh (tq3_1s_qs_to_int8x4); the table they must match is
// the real kvalues_tq3, so an edit to the table fails this check, an edit to the loader's
// copy of the constants does not.
static const uint32_t tq3_1s_kv_lo = 0xF0D1AE81u;
static const uint32_t tq3_1s_kv_hi = 0x7F512E0Fu;

// v_perm_b32(s0, s1, sel): output byte k is byte sel[8k+7:8k] of the 8-byte value {s0, s1},
// s1 being bytes 0-3. Only selector values 0-7 are used, so the special values above 7 are
// not modelled.
static uint32_t emulate_v_perm_b32(uint32_t s0, uint32_t s1, uint32_t sel) {
    const uint64_t src = ((uint64_t) s0 << 32) | s1;
    uint32_t r = 0;
    for (int k = 0; k < 4; k++) {
        const uint32_t s = (sel >> (8*k)) & 0xff;
        r |= (uint32_t) ((src >> (8*s)) & 0xff) << (8*k);
    }
    return r;
}

// __byte_perm(x, y, s): output byte k is byte s[4k+2:4k] of the 8-byte value {y, x}, x being
// bytes 0-3. The sign-replicate mode (bit 3 of each nibble) is not used, so not modelled.
static uint32_t emulate_byte_perm(uint32_t x, uint32_t y, uint32_t s) {
    const uint64_t src = ((uint64_t) y << 32) | x;
    uint32_t r = 0;
    for (int k = 0; k < 4; k++) {
        const uint32_t n = (s >> (4*k)) & 0x7;
        r |= (uint32_t) ((src >> (8*n)) & 0xff) << (8*k);
    }
    return r;
}

// Both selector packings must give the four table entries for every possible quad of indices.
static bool test_tq3_1s_mmq_byte_permute() {
    for (uint32_t q = 0; q < 8*8*8*8; q++) {
        const uint32_t i0 = q & 7, i1 = (q >> 3) & 7, i2 = (q >> 6) & 7, i3 = (q >> 9) & 7;
        const uint32_t want = (uint32_t) (uint8_t) tq3_1s_kvalues[i0]
                            | ((uint32_t) (uint8_t) tq3_1s_kvalues[i1] << 8)
                            | ((uint32_t) (uint8_t) tq3_1s_kvalues[i2] << 16)
                            | ((uint32_t) (uint8_t) tq3_1s_kvalues[i3] << 24);
        const uint32_t hip  = emulate_v_perm_b32(tq3_1s_kv_hi, tq3_1s_kv_lo, i0 | (i1 << 8) | (i2 << 16) | (i3 << 24));
        const uint32_t cuda = emulate_byte_perm (tq3_1s_kv_lo, tq3_1s_kv_hi, i0 | (i1 << 4) | (i2 <<  8) | (i3 << 12));
        if (hip != want || cuda != want) {
            fprintf(stderr, "%s: indices %u %u %u %u: want %08x, v_perm form %08x, prmt form %08x\n",
                    __func__, i0, i1, i2, i3, want, hip, cuda);
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
    // The centroid table is the real one from ggml/src/ggml-common.h, checked against
    // the real CPU decoder.
    {
        const bool unpack_ok = test_tq3_1s_mmq_unpack_matches_cpu();
        const bool table_ok  = test_tq3_1s_mmq_centroid_table();
        const bool perm_ok   = test_tq3_1s_mmq_byte_permute();
        if (!unpack_ok || !table_ok || !perm_ok) {
            num_failed++;
        }
        if (!unpack_ok || !table_ok || !perm_ok || verbose) {
            // RESULT_STR is indexed by the failure flag, not by a success bool
            printf("%5s TQ3_1S MMQ index extraction matches the CPU codec: %s\n", RESULT_STR[!unpack_ok], RESULT_STR[!unpack_ok]);
            printf("%5s TQ3_1S MMQ centroid table matches the CPU codec:    %s\n", RESULT_STR[!table_ok],  RESULT_STR[!table_ok]);
            printf("%5s TQ3_1S MMQ byte-permute centroid lookup:            %s\n", RESULT_STR[!perm_ok],   RESULT_STR[!perm_ok]);
        }
    }

    if (num_failed || verbose) {
        printf("%d tests failed\n", num_failed);
    }

    return num_failed > 0;
}

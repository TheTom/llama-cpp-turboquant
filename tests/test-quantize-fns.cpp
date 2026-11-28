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
#include <algorithm>
#include <cmath>
#include <limits>
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
static void generate_data(float offset, size_t n, float * dst, float amplitude = 2.0f) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = 0.1 + amplitude*std::cos(i + offset);
    }
}

// Calculate RMSE between two float arrays
static float array_rmse(const float * a1, const float * a2, size_t n) {
    double sum = 0;
    for (size_t i = 0; i < n; i++) {
        double diff = a1[i] - a2[i];
        sum += diff * diff;
    }
    return std::sqrt(sum) / n;
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
static float dot_product_error(const ggml_type_traits_cpu * qfns_cpu, ggml_type src0_type, size_t test_size,
                               const float * test_data1, const float * test_data2,
                               const float * test_data3, const float * test_data4,
                               const int nrc) {
    const auto * vdot = ggml_get_type_traits_cpu(qfns_cpu->vec_dot_type);
    const size_t pad  = 64;
    const size_t bx   = ggml_row_size(src0_type, test_size) + pad;
    const size_t by   = ggml_row_size(qfns_cpu->vec_dot_type, test_size) + pad;

    std::vector<uint8_t> tmp_q1(bx * nrc);
    std::vector<uint8_t> tmp_q2(by * nrc);

    qfns_cpu->from_float(test_data1, tmp_q1.data(), test_size);
    vdot->from_float(test_data2, tmp_q2.data(), test_size);

    if (nrc == 1) {
        float result = INFINITY;
        qfns_cpu->vec_dot(test_size, &result, 0, tmp_q1.data(), 0, tmp_q2.data(), 0, 1);

        const float dot_ref = dot_product(test_data1, test_data2, test_size);
        return std::fabs(result - dot_ref) / test_size;
    }

    // nrc == 2: kernel computes a 2x2 dot product matrix
    // Output layout: s[0]=dot(vx0,vy0), s[1]=dot(vx1,vy0), s[bs]=dot(vx0,vy1), s[bs+1]=dot(vx1,vy1)
    // row and output strides are padded, same as in the mul_mat path
    qfns_cpu->from_float(test_data3, tmp_q1.data() + bx, test_size);
    vdot->from_float(test_data4, tmp_q2.data() + by, test_size);

    const size_t bs = 16;
    std::vector<float> result(bs + 2, INFINITY);
    qfns_cpu->vec_dot(test_size, result.data(), bs, tmp_q1.data(), bx, tmp_q2.data(), by, 2);

    const float ref00 = dot_product(test_data1, test_data2, test_size);
    const float ref10 = dot_product(test_data3, test_data2, test_size);
    const float ref01 = dot_product(test_data1, test_data4, test_size);
    const float ref11 = dot_product(test_data3, test_data4, test_size);

    const auto err = [test_size](float val, float ref) {
        const float e = std::fabs(val - ref) / test_size;
        return std::isfinite(e) ? e : INFINITY;
    };

    return std::max({err(result[0], ref00), err(result[1], ref10), err(result[bs], ref01), err(result[bs + 1], ref11)});
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
        const float error = std::fabs(result - ref) / n;

        const bool failed = !(error < MAX_QUANTIZATION_REFERENCE_ERROR);
        num_failed += failed;
        if (failed || verbose) {
            printf(" f32 vec_dot n=%4d:                 %s (ref=%f got=%f err=%f)\n",
                   n, RESULT_STR[failed], ref, result, error);
        }
    }
    return num_failed;
}

// Positive finite E4M3 values in encoding order, from OCP FP8 rev. 1.1, section 5.1.
static const float F8_E4M3_VALUES[] = {
    0.0f, 0.001953125f, 0.00390625f, 0.005859375f, 0.0078125f, 0.009765625f, 0.01171875f, 0.013671875f,
    0.015625f, 0.017578125f, 0.01953125f, 0.021484375f, 0.0234375f, 0.025390625f, 0.02734375f, 0.029296875f,
    0.03125f, 0.03515625f, 0.0390625f, 0.04296875f, 0.046875f, 0.05078125f, 0.0546875f, 0.05859375f,
    0.0625f, 0.0703125f, 0.078125f, 0.0859375f, 0.09375f, 0.1015625f, 0.109375f, 0.1171875f,
    0.125f, 0.140625f, 0.15625f, 0.171875f, 0.1875f, 0.203125f, 0.21875f, 0.234375f,
    0.25f, 0.28125f, 0.3125f, 0.34375f, 0.375f, 0.40625f, 0.4375f, 0.46875f,
    0.5f, 0.5625f, 0.625f, 0.6875f, 0.75f, 0.8125f, 0.875f, 0.9375f,
    1.0f, 1.125f, 1.25f, 1.375f, 1.5f, 1.625f, 1.75f, 1.875f,
    2.0f, 2.25f, 2.5f, 2.75f, 3.0f, 3.25f, 3.5f, 3.75f,
    4.0f, 4.5f, 5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f,
    8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
    16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f,
    32.0f, 36.0f, 40.0f, 44.0f, 48.0f, 52.0f, 56.0f, 60.0f,
    64.0f, 72.0f, 80.0f, 88.0f, 96.0f, 104.0f, 112.0f, 120.0f,
    128.0f, 144.0f, 160.0f, 176.0f, 192.0f, 208.0f, 224.0f, 240.0f,
    256.0f, 288.0f, 320.0f, 352.0f, 384.0f, 416.0f, 448.0f,
};

static uint8_t fp32_to_f8_e4m3_ref(float value) {
    if (std::isnan(value)) {
        return 0x7F;
    }
    const uint8_t sign = std::signbit(value) ? 0x80 : 0;
    const double magnitude = std::fabs((double) value);
    if (magnitude >= F8_E4M3_VALUES[126]) {
        return sign | 126;
    }
    int nearest = 0;
    double distance = magnitude;
    for (int i = 1; i < 127; ++i) {
        const double candidate = std::fabs(magnitude - F8_E4M3_VALUES[i]);
        if (candidate < distance || (candidate == distance && (i & 1) == 0)) {
            nearest = i;
            distance = candidate;
        }
    }
    return sign | (uint8_t) nearest;
}

static int test_f8_e4m3(bool verbose) {
    static_assert(sizeof(ggml_fp8_e4m3_t) == 1);

    const auto * traits = ggml_get_type_traits(GGML_TYPE_F8_E4M3);
    const auto * traits_cpu = ggml_get_type_traits_cpu(GGML_TYPE_F8_E4M3);
    std::vector<uint8_t> encoded(256);
    std::vector<uint8_t> roundtrip(256);
    std::vector<float> decoded(256);
    int num_failed = 0;

    const bool traits_failed = traits->blck_size != 1 || traits->type_size != sizeof(ggml_fp8_e4m3_t) || traits->is_quantized;
    num_failed += traits_failed;
    if (verbose || traits_failed) {
        printf("f8_e4m3 scalar type traits:          %s\n", RESULT_STR[traits_failed]);
    }

    for (int i = 0; i < 256; ++i) {
        encoded[i] = (uint8_t) i;
    }
    traits->to_float(encoded.data(), decoded.data(), decoded.size());
    traits_cpu->from_float(decoded.data(), roundtrip.data(), decoded.size());

    for (int i = 0; i < 256; ++i) {
        const bool is_nan = (i & 0x7F) == 0x7F;
        const float expected = is_nan ? NAN : std::copysign(F8_E4M3_VALUES[i & 0x7F], i & 0x80 ? -1.0f : 1.0f);
        const bool decode_failed = is_nan ? !std::isnan(decoded[i])
            : decoded[i] != expected || (expected == 0.0f && std::signbit(decoded[i]) != std::signbit(expected));
        const bool encode_failed = !is_nan && roundtrip[i] != encoded[i];
        if (decode_failed || encode_failed) {
            num_failed++;
            if (verbose) {
                printf("f8_e4m3 code 0x%02x failed: decoded=%f expected=%f roundtrip=0x%02x\n",
                        i, decoded[i], expected, roundtrip[i]);
            }
        }
    }

    std::vector<float> inputs;
    for (float value : F8_E4M3_VALUES) {
        inputs.push_back(value);
        inputs.push_back(-value);
    }
    for (int i = 0; i < 126; ++i) {
        const float midpoint = (F8_E4M3_VALUES[i] + F8_E4M3_VALUES[i + 1]) / 2.0f;
        for (float value : {std::nextafter(midpoint, -INFINITY), midpoint, std::nextafter(midpoint, INFINITY)}) {
            inputs.push_back(value);
            inputs.push_back(-value);
        }
    }
    for (float value : {std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::max(),
                       std::nextafter(448.0f, INFINITY), std::nextafter(464.0f, -INFINITY), 464.0f,
                       std::nextafter(464.0f, INFINITY), INFINITY, NAN}) {
        inputs.push_back(value);
        inputs.push_back(-value);
    }

    std::vector<uint8_t> quantized(inputs.size());
    std::vector<uint8_t> quantized_ref(inputs.size());
    traits_cpu->from_float(inputs.data(), quantized.data(), inputs.size());
    traits->from_float_ref(inputs.data(), quantized_ref.data(), inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        const uint8_t expected = fp32_to_f8_e4m3_ref(inputs[i]);
        if (quantized[i] != expected || quantized_ref[i] != expected) {
            num_failed++;
            if (verbose) {
                printf("f8_e4m3 input %a failed: encoded=0x%02x reference=0x%02x expected=0x%02x\n",
                       inputs[i], quantized[i], quantized_ref[i], expected);
            }
        }
    }

    for (int n : {0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 65, 129, 257}) {
        std::vector<uint8_t> x(n + 1);
        std::vector<float> y(n + 1);
        double expected = 0.0;
        for (int i = 0; i < n; ++i) {
            const uint8_t code = (uint8_t) (i % 254 + (i % 254 >= 127));
            x[i + 1] = code;
            y[i + 1] = (i % 7 - 3) * 0.125f;
            expected += std::copysign(F8_E4M3_VALUES[code & 0x7F], code & 0x80 ? -1.0f : 1.0f) * y[i + 1];
        }
        float actual;
        traits_cpu->vec_dot(n, &actual, 0, x.data() + 1, 0, y.data() + 1, 0, 1);
        const bool failed = std::fabs(actual - expected) > 1e-5 * std::max(1.0, std::fabs(expected));
        num_failed += failed;
        if (verbose || failed) {
            printf("f8_e4m3 dot n=%d:                  %s (ref=%f got=%f)\n", n, RESULT_STR[failed], expected, actual);
        }
    }

    for (uint8_t code : {0x7F, 0xFF}) {
        std::vector<uint8_t> x(65, 0);
        std::vector<float> y(65, 1.0f);
        x[0] = x.back() = code;
        float actual;
        traits_cpu->vec_dot(x.size(), &actual, 0, x.data(), 0, y.data(), 0, 1);
        num_failed += !std::isnan(actual);
    }

    if (verbose || num_failed) {
        printf("f8_e4m3 exhaustive conversion:      %s (%d failures)\n", RESULT_STR[num_failed != 0], num_failed);
    }
    return num_failed;
}

static int test_vec_dot_q(bool verbose) {
    int num_failed = 0;

    const size_t test_size = 32 * 128;

    std::vector<float> test_data(test_size);
    std::vector<float> test_data2(test_size);
    std::vector<float> test_data3(test_size);
    std::vector<float> test_data4(test_size);

    generate_data(0.0, test_data.size(), test_data.data());
    generate_data(1.0, test_data2.size(), test_data2.data());
    generate_data(3.0, test_data3.size(), test_data3.data(), 1.0f);
    generate_data(4.0, test_data4.size(), test_data4.data(), 1.5f);

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
                type == GGML_TYPE_TQ3_4S  ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS :
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

            const float vec_dot_error = dot_product_error(qfns_cpu, type, test_size, test_data.data(), test_data2.data(), nullptr, nullptr, 1);
            const float max_allowed_error = type == GGML_TYPE_Q2_K || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_XXS ||
                type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ2_S ||
                type == GGML_TYPE_TQ3_1S || type == GGML_TYPE_TQ3_4S
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

            // Test nrc=2 path for types that support it
            if (qfns_cpu->nrows == 2) {
                const float vec_dot_error_nrc2 = dot_product_error(qfns_cpu, type, test_size, test_data.data(), test_data2.data(), test_data3.data(), test_data4.data(), 2);
                failed = !(vec_dot_error_nrc2 < max_allowed_error);
                num_failed += failed;
                if (failed || verbose) {
                    printf("%5s dot product error (nrc=2):    %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], vec_dot_error_nrc2);
                }
            }
        }
    }

    return num_failed;
}

// In every group, all values with importance are equal, and the max (0) has none.
// The scale search in make_qkx3_quants then fits min == max and passes inf/nan to nearest_int (#29804).
static int test_quantize_imatrix_degenerate(bool verbose) {
    const int64_t n = 256;
    std::vector<float> x(n);
    std::vector<float> imatrix(n);
    std::vector<float> out(n);
    int num_failed = 0;

    printf("Testing degenerate imatrix:\n");
    for (ggml_type type : {GGML_TYPE_Q2_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1}) {
        const int64_t group = type == GGML_TYPE_Q2_K ? 16 : 32;
        for (int64_t i = 0; i < n; ++i) {
            const int64_t g = i / group;
            const int64_t p = i % group;
            const bool important = p % 3 == 1;
            x[i] = important ? -0.02f*(g + 1) : (p % 2 ? -1.0f : 0.0f);
            imatrix[i] = important ? 1.0f : 0.0f;
        }
        printf("  - %s\n", ggml_type_name(type));

        std::vector<uint8_t> q(ggml_row_size(type, n));
        ggml_quantize_init(type);
        ggml_quantize_chunk(type, x.data(), q.data(), 0, 1, n, imatrix.data());
        ggml_get_type_traits(type)->to_float(q.data(), out.data(), n);

        const bool failed = !std::all_of(out.begin(), out.end(), [](float v) { return std::isfinite(v); });
        num_failed += failed;
        if (failed || verbose) {
            printf("%5s imatrix degenerate groups:      %s\n", ggml_type_name(type), RESULT_STR[failed]);
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
// test-backend-ops -o MUL_MAT -p type_a=tq3_1s.
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
    num_failed += test_f8_e4m3(verbose);
    num_failed += test_vec_dot_q(verbose);
    num_failed += test_quantize_imatrix_degenerate(verbose);

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

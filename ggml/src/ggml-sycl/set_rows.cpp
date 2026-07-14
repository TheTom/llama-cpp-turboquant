#include "set_rows.hpp"
#include "cpy.hpp"

namespace utils {
template<typename T>
static constexpr bool is_arithmetic_v() {
    return std::is_arithmetic_v<T> || std::is_same_v<T, sycl::half>
#ifdef GGML_SYCL_HAS_BF16
        || std::is_same_v<T, sycl::ext::oneapi::bfloat16>
#endif
        ;
}
}

template<typename TIn, typename TOut>
static inline std::enable_if_t<utils::is_arithmetic_v<TIn>() && utils::is_arithmetic_v<TOut>(), void>
convert (const char* src, char* dst) {
    auto src_val = *reinterpret_cast<const TIn*>(src);
    auto dst_val = sycl::vec<TIn, 1>(src_val).template convert<TOut, sycl::rounding_mode::automatic>()[0];
   *reinterpret_cast<TOut*>(dst) = dst_val;
}

template <typename TIdx, typename blockType, int qk, cpy_kernel_t cpyblck>
static void set_rows_sycl_q(const char * __restrict__ src0_d,
                            const TIdx * __restrict__ src1_d,
                            blockType * __restrict__ dst_d,
                            // tensor dimensions src0 and src1
                            const int64_t ne00,
                            const int64_t ne01,
                            const int64_t ne02,
                            const int64_t ne03,
                            const int64_t ne10,
                            const int64_t ne11,
                            const int64_t ne12,
                            const int64_t ne13,
                            // strides for src0
                            const size_t  nb00,
                            const size_t  nb01,
                            const size_t  nb02,
                            const size_t  nb03,
                            // strides for src1
                            const size_t  nb10,
                            const size_t  nb11,
                            const size_t  nb12,
                            const size_t  nb13,
                            // strides for dst
                            const size_t  nb1,
                            const size_t  nb2,
                            const size_t  nb3,
                            queue_ptr     stream) {
    const int64_t total_blocks = (ne00 * ne01 * ne02 * ne03) / qk;
    constexpr int block_size   = 256;
    const int64_t grid_size    = ceil_div(total_blocks, block_size);

    stream->parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> item_ct1) {
        const int64_t i = item_ct1.get_global_linear_id();
        if (i >= total_blocks) {
            return;
        }
        const int64_t i_base      = i * qk;
        const int64_t i03         = i_base / (ne00 * ne01 * ne02);
        const int64_t rem1        = i_base - i03 * (ne00 * ne01 * ne02);
        const int64_t i02         = rem1 / (ne00 * ne01);
        const int64_t rem2        = rem1 - i02 * ne00 * ne01;
        const int64_t i01         = rem2 / ne00;
        const int64_t i00         = rem2 - i01 * ne00;
        const int64_t i12         = i03 % ne12;
        const int64_t i11         = i02 % ne11;
        const int64_t i10         = i01;
        const size_t  src_offset  = calculate_offset<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
        const char *  src_block   = src0_d + src_offset + i00 * sizeof(float);
        const size_t  src1_offset = calculate_offset<3>({ nb10, nb11, nb12 }, { i10, i11, i12 });
        const int64_t dst_row     = src1_d[src1_offset / sizeof(TIdx)];
        const size_t  dst_offset =
            calculate_offset<3>({ nb1, nb2, nb3 }, { dst_row, i02, i03 }) + (i00 / qk) * sizeof(blockType);
        char * dst_block = reinterpret_cast<char *>(reinterpret_cast<char *>(dst_d) + dst_offset);
        cpyblck(src_block, dst_block);
    });
    GGML_UNUSED(ne10);
    GGML_UNUSED(ne13);
    GGML_UNUSED(nb00);
    GGML_UNUSED(nb13);
}

template<typename TIn, typename TIdx, typename TOut>
static void k_set_rows(
        const char * __restrict__ src0, const TIdx * __restrict__ src1, char * __restrict__ dst,
        const int64_t ne00, const int64_t ne01, const int64_t ne02,
        const int64_t ne11, const int64_t ne12,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        const size_t src_type_size, const size_t dst_type_size,
        const int64_t total_elements,
        const sycl::nd_item<1> & item_ct1) {

    const int64_t i = item_ct1.get_global_linear_id();
    if (i >= total_elements) {
        return;
    }

    const int64_t i03 = i / (ne00 * ne01 * ne02);
    const int64_t i02 = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    const int64_t i01 = (i - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01) / ne00;
    const int64_t i00 = i - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01 - i01 * ne00;

    const int64_t i12 = i03 % ne12;
    const int64_t i11 = i02 % ne11;
    const int64_t i10 = i01;

    const int64_t dst_row = *(const TIdx *)((const char *)src1 + calculate_offset<3>({nb10, nb11, nb12}, {i10, i11, i12}));

    const char * src0_row = src0 + calculate_offset<3>({nb01, nb02, nb03}, {i01, i02, i03});
    const char * src_elem = src0_row + i00 * src_type_size;
    char * dst_row_ptr = dst + dst_row*nb1 + i02*nb2 + i03*nb3;
    char * dst_elem = dst_row_ptr + i00 * dst_type_size;

    convert<TIn, TOut>(src_elem, dst_elem);
}

template<typename TIn, typename TIdx, typename TOut>
static void set_rows_sycl(
        const char * src0_d, const TIdx * src1_d, char * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne11, const int64_t ne12, const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        const size_t src_type_size, const size_t dst_type_size,
        queue_ptr stream) {

    const int64_t total_elements = ne00 * ne01 * ne02 * ne03;

    constexpr int block_size = 64;
    const int64_t grid_size = ceil_div(total_elements, block_size);

    stream->parallel_for(
        sycl::nd_range<1>(grid_size * block_size, block_size),
        [=](sycl::nd_item<1> item_ct1) {
            k_set_rows<TIn, TIdx, TOut>(
                src0_d, src1_d, dst_d,
                ne00, ne01, ne02,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                src_type_size, dst_type_size,
                total_elements,
                item_ct1
            );
        }
    );
}

template<typename TIn, typename TIdx>
static void set_rows_sycl(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const char * src0_d = (const char *)src0->data;
    const TIdx * src1_d = (const TIdx *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    dpct::queue_ptr stream = ctx.stream();
    switch (dst->type) {
        case GGML_TYPE_F32:
            set_rows_sycl<TIn, TIdx, float>(
                src0_d, src1_d, (char *)dst->data,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                sizeof(TIn), sizeof(float),
                stream
            );
            break;
        case GGML_TYPE_F16:
            dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
            set_rows_sycl<TIn, TIdx, sycl::half>(
                src0_d, src1_d, (char *)dst->data,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                sizeof(TIn), sizeof(sycl::half),
                stream
            );
            break;
#ifdef GGML_SYCL_HAS_BF16
        case GGML_TYPE_BF16:
            set_rows_sycl<TIn, TIdx, sycl::ext::oneapi::bfloat16>(
                src0_d, src1_d, (char *)dst->data,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                sizeof(TIn), sizeof(sycl::ext::oneapi::bfloat16),
                stream
            );
            break;
#endif
        case GGML_TYPE_Q8_0:
            set_rows_sycl_q<TIdx, block_q8_0, QK8_0, cpy_blck_f32_q8_0>(src0_d, src1_d, (block_q8_0 *)dst->data, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q5_1:
            set_rows_sycl_q<TIdx, block_q5_1, QK5_1, cpy_blck_f32_q5_1>(src0_d, src1_d, (block_q5_1 *)dst->data, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q5_0:
            set_rows_sycl_q<TIdx, block_q5_0, QK5_0, cpy_blck_f32_q5_0>(src0_d, src1_d, (block_q5_0 *)dst->data, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q4_1:
            set_rows_sycl_q<TIdx, block_q4_1, QK4_1, cpy_blck_f32_q4_1>(src0_d, src1_d, (block_q4_1 *)dst->data, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q4_0:
            set_rows_sycl_q<TIdx, block_q4_0, QK4_0, cpy_blck_f32_q4_0>(src0_d, src1_d, (block_q4_0 *)dst->data, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            set_rows_sycl_q<TIdx, block_iq4_nl, QK4_NL, cpy_blck_f32_iq4_nl>(src0_d, src1_d, (block_iq4_nl *)dst->data, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            break;

        default:
            GGML_ABORT("Unsupported tensor type!");
            break;
    }
}

// =============================================================================
// TurboQuant SYCL SET_ROWS — WHT-based KV cache quantization kernels
// Ported from ggml/src/ggml-cuda/set-rows.cu
// =============================================================================

// ---- Centroid and midpoint tables (KV-cache types: N(0,1/128)) ----

static const float SYCL_TURBO_CENTROIDS_3BIT[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

static const float SYCL_TURBO_MID_3BIT[7] = {
    -0.154259f, -0.091775f, -0.043589f, 0.0f,
     0.043589f,  0.091775f,  0.154259f
};

static const float SYCL_TURBO_CENTROIDS_2BIT[4] = {
    -0.133462f, -0.039994f, 0.039994f, 0.133462f
};

static const float SYCL_TURBO_MID_2BIT[3] = {
    -0.086728f, 0.0f, 0.086728f
};

static const float SYCL_TURBO_CENTROIDS_4BIT[16] = {
    -0.173926f, -0.117195f, -0.089527f, -0.068756f,
    -0.051262f, -0.035597f, -0.020989f, -0.006938f,
     0.006938f,  0.020989f,  0.035597f,  0.051262f,
     0.068756f,  0.089527f,  0.117195f,  0.173926f
};

static const float SYCL_TURBO_MID_4BIT[15] = {
    -0.145561f, -0.103361f, -0.079142f, -0.060009f,
    -0.043430f, -0.028293f, -0.013964f,  0.000000f,
     0.013964f,  0.028293f,  0.043430f,  0.060009f,
     0.079142f,  0.103361f,  0.145561f
};

// ---- WHT sign arrays (seed=42, 128-element) ----

static const float SYCL_TURBO_WHT_SIGNS1[128] = {
    -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f,
    -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f
};

static const float SYCL_TURBO_WHT_SIGNS2[128] = {
    1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f,
    -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f
};

// ---- Nearest centroid lookup ----

static inline uint8_t turbo_nearest_centroid_3bit_sycl(float val) {
    if      (val < SYCL_TURBO_MID_3BIT[0]) return 0;
    else if (val < SYCL_TURBO_MID_3BIT[1]) return 1;
    else if (val < SYCL_TURBO_MID_3BIT[2]) return 2;
    else if (val < SYCL_TURBO_MID_3BIT[3]) return 3;
    else if (val < SYCL_TURBO_MID_3BIT[4]) return 4;
    else if (val < SYCL_TURBO_MID_3BIT[5]) return 5;
    else if (val < SYCL_TURBO_MID_3BIT[6]) return 6;
    else                                    return 7;
}

static inline uint8_t turbo_nearest_centroid_2bit_sycl(float val) {
    if      (val < SYCL_TURBO_MID_2BIT[0]) return 0;
    else if (val < SYCL_TURBO_MID_2BIT[1]) return 1;
    else if (val < SYCL_TURBO_MID_2BIT[2]) return 2;
    else                                    return 3;
}

static inline uint8_t turbo_nearest_centroid_4bit_sycl(float val) {
    if      (val < SYCL_TURBO_MID_4BIT[ 0]) return  0;
    else if (val < SYCL_TURBO_MID_4BIT[ 1]) return  1;
    else if (val < SYCL_TURBO_MID_4BIT[ 2]) return  2;
    else if (val < SYCL_TURBO_MID_4BIT[ 3]) return  3;
    else if (val < SYCL_TURBO_MID_4BIT[ 4]) return  4;
    else if (val < SYCL_TURBO_MID_4BIT[ 5]) return  5;
    else if (val < SYCL_TURBO_MID_4BIT[ 6]) return  6;
    else if (val < SYCL_TURBO_MID_4BIT[ 7]) return  7;
    else if (val < SYCL_TURBO_MID_4BIT[ 8]) return  8;
    else if (val < SYCL_TURBO_MID_4BIT[ 9]) return  9;
    else if (val < SYCL_TURBO_MID_4BIT[10]) return 10;
    else if (val < SYCL_TURBO_MID_4BIT[11]) return 11;
    else if (val < SYCL_TURBO_MID_4BIT[12]) return 12;
    else if (val < SYCL_TURBO_MID_4BIT[13]) return 13;
    else if (val < SYCL_TURBO_MID_4BIT[14]) return 14;
    else                                     return 15;
}

// ---- Helper: work-group L2 norm (reduction across GROUP_SIZE threads) ----
// Requires: s_x[GROUP_SIZE] in shared memory, s_accum[n_warps+1] in shared memory.
// Returns: sqrt(sum(s_x[j]^2)) broadcast to all work-items.
// Uses sub-group reduce + shared memory inter-warp accumulation.

template <int GROUP_SIZE>
static inline float turbo_group_norm_sq(
        float val_sq,
        const sycl::nd_item<1>& item,
        float * __restrict__ s_accum)
{
    constexpr int n_warps = GROUP_SIZE / WARP_SIZE;
    const int j = item.get_local_id(0);

    // Warp-level reduce
    float v2 = val_sq;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        v2 += dpct::permute_sub_group_by_xor(item.get_sub_group(), v2, offset);
    }
    if (j % WARP_SIZE == 0) {
        s_accum[j / WARP_SIZE] = v2;
    }
    sycl::group_barrier(item.get_group());

    // Thread 0 reduces warp results and broadcasts via s_accum[n_warps]
    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += s_accum[w];
        s_accum[n_warps] = total;
    }
    sycl::group_barrier(item.get_group());
    return s_accum[n_warps];
}

// ---- Turbo3 SET_ROWS kernel: GROUP_SIZE=128 (one block per WHT group) ----
// One work-group = one WHT group (128 threads, each handles one element).
// Uses shared memory for x[], per-warp accum, and qs/signs bit packing.

template <typename idx_t>
static void k_set_rows_turbo3_sycl(
        const float * __restrict__    src0,
        const idx_t * __restrict__    src1,
        block_turbo3_0 * __restrict__ dst,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne02,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t s1,
        const int64_t s2,
        const int64_t s3,
        const sycl::nd_item<1>&      item,
        uint8_t * __restrict__       local_mem)
{
    constexpr int GROUP_SIZE = QK_TURBO3;   // 128: one SYCL work-group per block
    constexpr int n_warps    = GROUP_SIZE / WARP_SIZE;

    // Carve shared memory
    float   * s_x     = reinterpret_cast<float *>(local_mem);
    float   * s_accum = s_x + GROUP_SIZE;               // [n_warps + 1]
    uint8_t * s_qs    = reinterpret_cast<uint8_t *>(s_accum + n_warps + 1);
    uint8_t * s_signs = s_qs + GROUP_SIZE;

    const int j = item.get_local_id(0);     // 0 .. 127

    // Each work-group corresponds to one block (= one WHT group of 128 elements)
    // Decode: which row (i01), which batch dim (i02, i03), and which block offset
    const int64_t n_blocks_per_row = ne00 / GROUP_SIZE;
    const int64_t g     = item.get_group(0);
    const int64_t i_blk = g % n_blocks_per_row;    // block index within row
    int64_t       tmp   = g / n_blocks_per_row;
    const int64_t i01   = tmp % ne01;
    tmp                 = tmp / ne01;
    const int64_t i02   = tmp % ne02;
    const int64_t i03   = tmp / ne02;

    const int64_t i10 = i01;
    const int64_t i11 = i02 % ne11;
    const int64_t i12 = i03 % ne12;

    const int64_t      dst_row  = src1[i10*s10 + i11*s11 + i12*s12];
    const float *      src_row  = src0 + i01*s01 + i02*s02 + i03*s03;
    block_turbo3_0 *   blk      = reinterpret_cast<block_turbo3_0 *>(
                                      reinterpret_cast<char *>(dst) + dst_row*s1 + i02*s2 + i03*s3)
                                  + i_blk;

    // Step 1: Load element j
    s_x[j] = src_row[i_blk * GROUP_SIZE + j];
    sycl::group_barrier(item.get_group());

    // Step 2: Parallel L2 norm
    const float s_norm_sq = turbo_group_norm_sq<GROUP_SIZE>(s_x[j] * s_x[j], item, s_accum);
    const float grp_norm  = sycl::sqrt(s_norm_sq);
    const float inv_norm  = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

    // Step 3: Normalize
    s_x[j] *= inv_norm;
    sycl::group_barrier(item.get_group());

    // Step 4: Forward WHT (signs1 → butterfly → signs2, normalized)
    s_x[j] *= SYCL_TURBO_WHT_SIGNS1[j];
    sycl::group_barrier(item.get_group());

#define TURBO3_WHT_STAGE(h) \
    if (j % (2*(h)) < (h)) { float a = s_x[j], b = s_x[j+(h)]; s_x[j] = a+b; s_x[j+(h)] = a-b; } \
    sycl::group_barrier(item.get_group());

    TURBO3_WHT_STAGE(1)
    TURBO3_WHT_STAGE(2)
    TURBO3_WHT_STAGE(4)
    TURBO3_WHT_STAGE(8)
    TURBO3_WHT_STAGE(16)
    TURBO3_WHT_STAGE(32)
    TURBO3_WHT_STAGE(64)
#undef TURBO3_WHT_STAGE

    constexpr float INV_SQRT_128 = 0.08838834764831845f;
    s_x[j] = s_x[j] * INV_SQRT_128 * SYCL_TURBO_WHT_SIGNS2[j];
    sycl::group_barrier(item.get_group());

    // Step 5: Quantize element j to 3-bit centroid
    const float   rv  = s_x[j];
    const uint8_t idx = turbo_nearest_centroid_3bit_sycl(rv);

    // Step 6: Pack qs (2 low bits) and signs (1 high bit) via shared memory
    s_qs[j]    = idx & 0x3;
    s_signs[j] = (idx >> 2) & 1;
    sycl::group_barrier(item.get_group());

    // Every 4th thread writes one qs byte (4 × 2-bit = 1 byte)
    if (j % 4 == 0) {
        blk->qs[j / 4] = s_qs[j] | (s_qs[j+1] << 2) | (s_qs[j+2] << 4) | (s_qs[j+3] << 6);
    }
    // Every 8th thread writes one signs byte (8 × 1-bit = 1 byte)
    if (j % 8 == 0) {
        blk->signs[j / 8] =
            s_signs[j]   | (s_signs[j+1] << 1) | (s_signs[j+2] << 2) | (s_signs[j+3] << 3) |
            (s_signs[j+4] << 4) | (s_signs[j+5] << 5) | (s_signs[j+6] << 6) | (s_signs[j+7] << 7);
    }

    // Step 7: Reconstruction norm
    const float   c    = SYCL_TURBO_CENTROIDS_3BIT[idx];
    const float s_recon_sq = turbo_group_norm_sq<GROUP_SIZE>(c * c, item, s_accum);
    const float recon_norm  = sycl::sqrt(s_recon_sq);
    const float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;

    // Step 8: Write corrected norm (thread 0 only)
    if (j == 0) {
        blk->norm = (ggml_half)corrected_norm;
    }
}

// ---- Turbo2 SET_ROWS kernel: identical to turbo3 minus the signs packing ----

template <typename idx_t>
static void k_set_rows_turbo2_sycl(
        const float * __restrict__    src0,
        const idx_t * __restrict__    src1,
        block_turbo2_0 * __restrict__ dst,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne02,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t s1,
        const int64_t s2,
        const int64_t s3,
        const sycl::nd_item<1>&      item,
        uint8_t * __restrict__       local_mem)
{
    constexpr int GROUP_SIZE = QK_TURBO2;   // 128
    constexpr int n_warps    = GROUP_SIZE / WARP_SIZE;

    float   * s_x     = reinterpret_cast<float *>(local_mem);
    float   * s_accum = s_x + GROUP_SIZE;
    uint8_t * s_qs    = reinterpret_cast<uint8_t *>(s_accum + n_warps + 1);

    const int j = item.get_local_id(0);

    const int64_t n_blocks_per_row = ne00 / GROUP_SIZE;
    const int64_t g     = item.get_group(0);
    const int64_t i_blk = g % n_blocks_per_row;
    int64_t       tmp   = g / n_blocks_per_row;
    const int64_t i01   = tmp % ne01;
    tmp                 = tmp / ne01;
    const int64_t i02   = tmp % ne02;
    const int64_t i03   = tmp / ne02;

    const int64_t i10 = i01;
    const int64_t i11 = i02 % ne11;
    const int64_t i12 = i03 % ne12;

    const int64_t      dst_row  = src1[i10*s10 + i11*s11 + i12*s12];
    const float *      src_row  = src0 + i01*s01 + i02*s02 + i03*s03;
    block_turbo2_0 *   blk      = reinterpret_cast<block_turbo2_0 *>(
                                      reinterpret_cast<char *>(dst) + dst_row*s1 + i02*s2 + i03*s3)
                                  + i_blk;

    s_x[j] = src_row[i_blk * GROUP_SIZE + j];
    sycl::group_barrier(item.get_group());

    const float s_norm_sq = turbo_group_norm_sq<GROUP_SIZE>(s_x[j] * s_x[j], item, s_accum);
    const float grp_norm  = sycl::sqrt(s_norm_sq);
    const float inv_norm  = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

    s_x[j] *= inv_norm;
    sycl::group_barrier(item.get_group());

    s_x[j] *= SYCL_TURBO_WHT_SIGNS1[j];
    sycl::group_barrier(item.get_group());

#define TURBO2_WHT_STAGE(h) \
    if (j % (2*(h)) < (h)) { float a = s_x[j], b = s_x[j+(h)]; s_x[j] = a+b; s_x[j+(h)] = a-b; } \
    sycl::group_barrier(item.get_group());

    TURBO2_WHT_STAGE(1)
    TURBO2_WHT_STAGE(2)
    TURBO2_WHT_STAGE(4)
    TURBO2_WHT_STAGE(8)
    TURBO2_WHT_STAGE(16)
    TURBO2_WHT_STAGE(32)
    TURBO2_WHT_STAGE(64)
#undef TURBO2_WHT_STAGE

    constexpr float INV_SQRT_128 = 0.08838834764831845f;
    s_x[j] = s_x[j] * INV_SQRT_128 * SYCL_TURBO_WHT_SIGNS2[j];
    sycl::group_barrier(item.get_group());

    const float   rv  = s_x[j];
    const uint8_t idx = turbo_nearest_centroid_2bit_sycl(rv);

    s_qs[j] = idx & 0x3;
    sycl::group_barrier(item.get_group());

    if (j % 4 == 0) {
        blk->qs[j / 4] = s_qs[j] | (s_qs[j+1] << 2) | (s_qs[j+2] << 4) | (s_qs[j+3] << 6);
    }

    const float   c    = SYCL_TURBO_CENTROIDS_2BIT[idx];
    const float s_recon_sq = turbo_group_norm_sq<GROUP_SIZE>(c * c, item, s_accum);
    const float recon_norm  = sycl::sqrt(s_recon_sq);
    const float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;

    if (j == 0) {
        blk->norm = (ggml_half)corrected_norm;
    }
}

// ---- Turbo4 SET_ROWS kernel: GROUP_SIZE=128, 4-bit nibble packing ----

template <typename idx_t>
static void k_set_rows_turbo4_sycl(
        const float * __restrict__    src0,
        const idx_t * __restrict__    src1,
        block_turbo4_0 * __restrict__ dst,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne02,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t s1,
        const int64_t s2,
        const int64_t s3,
        const sycl::nd_item<1>&      item,
        uint8_t * __restrict__       local_mem)
{
    constexpr int GROUP_SIZE = QK_TURBO4;   // 128
    constexpr int n_warps    = GROUP_SIZE / WARP_SIZE;

    float   * s_x     = reinterpret_cast<float *>(local_mem);
    float   * s_accum = s_x + GROUP_SIZE;
    uint8_t * s_qs    = reinterpret_cast<uint8_t *>(s_accum + n_warps + 1);

    const int j = item.get_local_id(0);

    const int64_t n_blocks_per_row = ne00 / GROUP_SIZE;
    const int64_t g     = item.get_group(0);
    const int64_t i_blk = g % n_blocks_per_row;
    int64_t       tmp   = g / n_blocks_per_row;
    const int64_t i01   = tmp % ne01;
    tmp                 = tmp / ne01;
    const int64_t i02   = tmp % ne02;
    const int64_t i03   = tmp / ne02;

    const int64_t i10 = i01;
    const int64_t i11 = i02 % ne11;
    const int64_t i12 = i03 % ne12;

    const int64_t      dst_row  = src1[i10*s10 + i11*s11 + i12*s12];
    const float *      src_row  = src0 + i01*s01 + i02*s02 + i03*s03;
    block_turbo4_0 *   blk      = reinterpret_cast<block_turbo4_0 *>(
                                      reinterpret_cast<char *>(dst) + dst_row*s1 + i02*s2 + i03*s3)
                                  + i_blk;

    s_x[j] = src_row[i_blk * GROUP_SIZE + j];
    sycl::group_barrier(item.get_group());

    const float s_norm_sq = turbo_group_norm_sq<GROUP_SIZE>(s_x[j] * s_x[j], item, s_accum);
    const float grp_norm  = sycl::sqrt(s_norm_sq);
    const float inv_norm  = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

    s_x[j] *= inv_norm;
    sycl::group_barrier(item.get_group());

    s_x[j] *= SYCL_TURBO_WHT_SIGNS1[j];
    sycl::group_barrier(item.get_group());

#define TURBO4_WHT_STAGE(h) \
    if (j % (2*(h)) < (h)) { float a = s_x[j], b = s_x[j+(h)]; s_x[j] = a+b; s_x[j+(h)] = a-b; } \
    sycl::group_barrier(item.get_group());

    TURBO4_WHT_STAGE(1)
    TURBO4_WHT_STAGE(2)
    TURBO4_WHT_STAGE(4)
    TURBO4_WHT_STAGE(8)
    TURBO4_WHT_STAGE(16)
    TURBO4_WHT_STAGE(32)
    TURBO4_WHT_STAGE(64)
#undef TURBO4_WHT_STAGE

    constexpr float INV_SQRT_128 = 0.08838834764831845f;
    s_x[j] = s_x[j] * INV_SQRT_128 * SYCL_TURBO_WHT_SIGNS2[j];
    sycl::group_barrier(item.get_group());

    const float   rv  = s_x[j];
    const uint8_t idx = turbo_nearest_centroid_4bit_sycl(rv);

    // Nibble pack: 2 elements per byte, 4 bits each
    s_qs[j] = idx & 0xF;
    sycl::group_barrier(item.get_group());

    if (j % 2 == 0) {
        blk->qs[j / 2] = s_qs[j] | (s_qs[j+1] << 4);
    }

    const float   c    = SYCL_TURBO_CENTROIDS_4BIT[idx];
    const float s_recon_sq = turbo_group_norm_sq<GROUP_SIZE>(c * c, item, s_accum);
    const float recon_norm  = sycl::sqrt(s_recon_sq);
    const float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;

    if (j == 0) {
        blk->norm  = (ggml_half)corrected_norm;
        blk->rnorm = (ggml_half)0.0f;
    }
}

// ---- Dispatch wrappers ----

template <typename idx_t>
static void set_rows_sycl_turbo3(
        ggml_backend_sycl_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst)
{
    const float *      src0_d = static_cast<const float *>(src0->data);
    const idx_t *      src1_d = static_cast<const idx_t *>(src1->data);
    block_turbo3_0 *   dst_d  = static_cast<block_turbo3_0 *>(dst->data);
    dpct::queue_ptr    stream  = ctx.stream();

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(ne00 % QK_TURBO3 == 0);

    const int64_t n_blocks_per_row = ne00 / QK_TURBO3;
    const int64_t ne_total         = n_blocks_per_row * ne01 * ne02 * ne03;
    if (ne_total == 0) return;

    const int64_t s01 = nb01 / sizeof(float);
    const int64_t s02 = nb02 / sizeof(float);
    const int64_t s03 = nb03 / sizeof(float);
    const int64_t s10 = nb10 / sizeof(idx_t);
    const int64_t s11 = nb11 / sizeof(idx_t);
    const int64_t s12 = nb12 / sizeof(idx_t);

    constexpr int  WG   = QK_TURBO3;  // work-group size = 128
    constexpr int  NW   = WG / WARP_SIZE;
    // Shared memory: s_x[128] + s_accum[NW+1] + s_qs[128] + s_signs[128]
    const size_t lm = WG * sizeof(float) + (NW + 1) * sizeof(float) + WG + WG;

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint8_t, 1> local_buf(sycl::range<1>(lm), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(ne_total * WG, WG),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                k_set_rows_turbo3_sycl<idx_t>(
                    src0_d, src1_d, dst_d,
                    ne00, ne01, ne02, ne11, ne12,
                    s01, s02, s03, s10, s11, s12,
                    nb1, nb2, nb3,
                    item,
                    local_buf.get_multi_ptr<sycl::access::decorated::no>().get());
            });
    });
}

template <typename idx_t>
static void set_rows_sycl_turbo2(
        ggml_backend_sycl_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst)
{
    const float *      src0_d = static_cast<const float *>(src0->data);
    const idx_t *      src1_d = static_cast<const idx_t *>(src1->data);
    block_turbo2_0 *   dst_d  = static_cast<block_turbo2_0 *>(dst->data);
    dpct::queue_ptr    stream  = ctx.stream();

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(ne00 % QK_TURBO2 == 0);

    const int64_t n_blocks_per_row = ne00 / QK_TURBO2;
    const int64_t ne_total         = n_blocks_per_row * ne01 * ne02 * ne03;
    if (ne_total == 0) return;

    const int64_t s01 = nb01 / sizeof(float);
    const int64_t s02 = nb02 / sizeof(float);
    const int64_t s03 = nb03 / sizeof(float);
    const int64_t s10 = nb10 / sizeof(idx_t);
    const int64_t s11 = nb11 / sizeof(idx_t);
    const int64_t s12 = nb12 / sizeof(idx_t);

    constexpr int  WG   = QK_TURBO2;
    constexpr int  NW   = WG / WARP_SIZE;
    // Shared memory: s_x[128] + s_accum[NW+1] + s_qs[128]
    const size_t lm = WG * sizeof(float) + (NW + 1) * sizeof(float) + WG;

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint8_t, 1> local_buf(sycl::range<1>(lm), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(ne_total * WG, WG),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                k_set_rows_turbo2_sycl<idx_t>(
                    src0_d, src1_d, dst_d,
                    ne00, ne01, ne02, ne11, ne12,
                    s01, s02, s03, s10, s11, s12,
                    nb1, nb2, nb3,
                    item,
                    local_buf.get_multi_ptr<sycl::access::decorated::no>().get());
            });
    });
}

template <typename idx_t>
static void set_rows_sycl_turbo4(
        ggml_backend_sycl_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst)
{
    const float *      src0_d = static_cast<const float *>(src0->data);
    const idx_t *      src1_d = static_cast<const idx_t *>(src1->data);
    block_turbo4_0 *   dst_d  = static_cast<block_turbo4_0 *>(dst->data);
    dpct::queue_ptr    stream  = ctx.stream();

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(ne00 % QK_TURBO4 == 0);

    const int64_t n_blocks_per_row = ne00 / QK_TURBO4;
    const int64_t ne_total         = n_blocks_per_row * ne01 * ne02 * ne03;
    if (ne_total == 0) return;

    const int64_t s01 = nb01 / sizeof(float);
    const int64_t s02 = nb02 / sizeof(float);
    const int64_t s03 = nb03 / sizeof(float);
    const int64_t s10 = nb10 / sizeof(idx_t);
    const int64_t s11 = nb11 / sizeof(idx_t);
    const int64_t s12 = nb12 / sizeof(idx_t);

    constexpr int  WG   = QK_TURBO4;
    constexpr int  NW   = WG / WARP_SIZE;
    // Shared memory: s_x[128] + s_accum[NW+1] + s_qs[128]
    const size_t lm = WG * sizeof(float) + (NW + 1) * sizeof(float) + WG;

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint8_t, 1> local_buf(sycl::range<1>(lm), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(ne_total * WG, WG),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                k_set_rows_turbo4_sycl<idx_t>(
                    src0_d, src1_d, dst_d,
                    ne00, ne01, ne02, ne11, ne12,
                    s01, s02, s03, s10, s11, s12,
                    nb1, nb2, nb3,
                    item,
                    local_buf.get_multi_ptr<sycl::access::decorated::no>().get());
            });
    });
}

void ggml_sycl_op_set_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_I64 || dst->src[1]->type == GGML_TYPE_I32);

    const bool is_turbo = (dst->type == GGML_TYPE_TURBO3_0 ||
                           dst->type == GGML_TYPE_TURBO2_0 ||
                           dst->type == GGML_TYPE_TURBO4_0);
    if (is_turbo) {
        auto dispatch_turbo = [&](auto idx_type) {
            using idx_t = decltype(idx_type);
            switch (dst->type) {
                case GGML_TYPE_TURBO3_0: set_rows_sycl_turbo3<idx_t>(ctx, src0, src1, dst); break;
                case GGML_TYPE_TURBO2_0: set_rows_sycl_turbo2<idx_t>(ctx, src0, src1, dst); break;
                case GGML_TYPE_TURBO4_0: set_rows_sycl_turbo4<idx_t>(ctx, src0, src1, dst); break;
                default: GGML_ABORT("unexpected turbo type"); break;
            }
        };
        if (src1->type == GGML_TYPE_I64) dispatch_turbo(int64_t{});
        else                              dispatch_turbo(int32_t{});
        return;
    }

    if (src1->type == GGML_TYPE_I64) {
        set_rows_sycl<float, int64_t>(ctx, src0, src1, dst);
    } else {
        set_rows_sycl<float, int32_t>(ctx, src0, src1, dst);
    }
}

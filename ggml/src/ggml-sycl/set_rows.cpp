#include "set_rows.hpp"
#include "cpy.hpp"
#include "turbo-quant.hpp"

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

// ============================================================
// TurboQuant SET_ROWS cooperative kernels (128 work-items each)
// Ported from ggml-cuda/set-rows.cu
// Sub-group reduction: 128/WARP_SIZE sub-groups, warp_accum[128/WARP_SIZE] SLM
// ============================================================

template <typename TIdx>
static void set_rows_sycl_turbo3(ggml_backend_sycl_context & ctx,
                                  const ggml_tensor * src0,
                                  const ggml_tensor * src1,
                                  ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS

    const float * src0_d = (const float *)src0->data;
    const TIdx  * src1_d = (const TIdx *)src1->data;
    block_turbo3_0 * dst_d = (block_turbo3_0 *)dst->data;

    const int64_t s01 = nb01 / sizeof(float);
    const int64_t s02 = nb02 / sizeof(float);
    const int64_t s03 = nb03 / sizeof(float);
    const int64_t s10 = nb10 / sizeof(TIdx);
    const int64_t s11 = nb11 / sizeof(TIdx);
    const int64_t s12 = nb12 / sizeof(TIdx);

    GGML_ASSERT(ne00 % 128 == 0);

    const int64_t n_groups_per_row = ne00 / 128;
    const int64_t n_groups = n_groups_per_row * ne01 * ne02 * ne03;

    if (n_groups == 0) return;

    dpct::queue_ptr stream = ctx.stream();

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> x(128, cgh);
        sycl::local_accessor<float, 1> warp_accum(128 / WARP_SIZE, cgh);
        sycl::local_accessor<uint8_t, 1> local_idx(128, cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(n_groups * 128, 128),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                const int j = item.get_local_id(0);
                const int64_t g = item.get_group(0);
                auto sg = item.get_sub_group();

                const int64_t i_grp = g % n_groups_per_row;
                int64_t tmp = g / n_groups_per_row;
                const int64_t i01l = tmp % ne01;
                tmp = tmp / ne01;
                const int64_t i02l = tmp % ne02;
                const int64_t i03l = tmp / ne02;

                const int64_t i10l = i01l;
                const int64_t i11l = i02l % ne11;
                const int64_t i12l = i03l % ne12;

                const int64_t dst_row = *(src1_d + i10l*s10 + i11l*s11 + i12l*s12);
                // dst_row is work-group-uniform (derived from group id only, not local id j) -- safe to early-return without barrier divergence
                if (dst_row < 0 || dst_row >= ne1) return;
                const float * src_row = src0_d + i01l*s01 + i02l*s02 + i03l*s03;
                block_turbo3_0 * blk = (block_turbo3_0 *)((char *)dst_d + dst_row*nb1 + i02l*nb2 + i03l*nb3) + i_grp;

                x[j] = src_row[i_grp * 128 + j];
                item.barrier(sycl::access::fence_space::local_space);

                float sg_sum = sycl::reduce_over_group(sg, x[j] * x[j], sycl::plus<float>());
                if (sg.get_local_id()[0] == 0) warp_accum[sg.get_group_id()[0]] = sg_sum;
                item.barrier(sycl::access::fence_space::local_space);
                float norm_sq = 0.0f;
                for (int i = 0; i < 128 / WARP_SIZE; i++) norm_sq += warp_accum[i];
                const float grp_norm = sycl::sqrt(norm_sq);
                const float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

                x[j] *= inv_norm;
                x[j] *= TURBO_WHT_SIGNS1[j];
                item.barrier(sycl::access::fence_space::local_space);

                for (int h = 1; h < 128; h *= 2) {
                    if (j % (2*h) < h) {
                        float a = x[j], b = x[j+h];
                        x[j] = a + b;
                        x[j+h] = a - b;
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                constexpr float inv_sqrt_128 = 0.08838834764831845f;
                x[j] = x[j] * inv_sqrt_128 * TURBO_WHT_SIGNS2[j];

                const uint8_t idx = turbo_nearest_centroid_3bit(x[j]);

                local_idx[j] = idx;
                item.barrier(sycl::access::fence_space::local_space);

                if (j % 4 == 0) {
                    blk->qs[j/4] = (local_idx[j] & 0x3)
                                  | ((local_idx[j+1] & 0x3) << 2)
                                  | ((local_idx[j+2] & 0x3) << 4)
                                  | ((local_idx[j+3] & 0x3) << 6);
                }
                if (j % 8 == 0) {
                    uint8_t byte = 0;
                    for (int k = 0; k < 8; k++)
                        byte |= ((local_idx[j+k] >> 2) & 1) << k;
                    blk->signs[j/8] = byte;
                }

                const float c = TURBO_CENTROIDS_3BIT[idx];
                // warp_accum reuse is safe: preceding barriers fully drain all reads before this write
                sg_sum = sycl::reduce_over_group(sg, c * c, sycl::plus<float>());
                if (sg.get_local_id()[0] == 0) warp_accum[sg.get_group_id()[0]] = sg_sum;
                item.barrier(sycl::access::fence_space::local_space);
                float recon_sq = 0.0f;
                for (int i = 0; i < 128 / WARP_SIZE; i++) recon_sq += warp_accum[i];
                const float recon_norm = sycl::sqrt(recon_sq);
                const float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;

                if (j == 0) blk->norm = sycl::half(corrected_norm);
            });
    });

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne13);
    GGML_UNUSED(nb00);
    GGML_UNUSED(nb13);
}

template <typename TIdx>
static void set_rows_sycl_turbo4(ggml_backend_sycl_context & ctx,
                                  const ggml_tensor * src0,
                                  const ggml_tensor * src1,
                                  ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS

    const float * src0_d = (const float *)src0->data;
    const TIdx  * src1_d = (const TIdx *)src1->data;
    block_turbo4_0 * dst_d = (block_turbo4_0 *)dst->data;

    const int64_t s01 = nb01 / sizeof(float);
    const int64_t s02 = nb02 / sizeof(float);
    const int64_t s03 = nb03 / sizeof(float);
    const int64_t s10 = nb10 / sizeof(TIdx);
    const int64_t s11 = nb11 / sizeof(TIdx);
    const int64_t s12 = nb12 / sizeof(TIdx);

    GGML_ASSERT(ne00 % 128 == 0);

    const int64_t n_blocks_per_row = ne00 / 128;
    const int64_t n_groups = n_blocks_per_row * ne01 * ne02 * ne03;

    if (n_groups == 0) return;

    dpct::queue_ptr stream = ctx.stream();

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> x(128, cgh);
        sycl::local_accessor<float, 1> warp_accum(128 / WARP_SIZE, cgh);
        sycl::local_accessor<uint8_t, 1> local_idx(128, cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(n_groups * 128, 128),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                const int j = item.get_local_id(0);
                const int64_t g = item.get_group(0);
                auto sg = item.get_sub_group();

                const int64_t i_blk = g % n_blocks_per_row;
                int64_t tmp = g / n_blocks_per_row;
                const int64_t i01l = tmp % ne01;
                tmp = tmp / ne01;
                const int64_t i02l = tmp % ne02;
                const int64_t i03l = tmp / ne02;

                const int64_t i10l = i01l;
                const int64_t i11l = i02l % ne11;
                const int64_t i12l = i03l % ne12;

                const int64_t dst_row = *(src1_d + i10l*s10 + i11l*s11 + i12l*s12);
                // dst_row is work-group-uniform (derived from group id only, not local id j) -- safe to early-return without barrier divergence
                if (dst_row < 0 || dst_row >= ne1) return;
                const float * src_row = src0_d + i01l*s01 + i02l*s02 + i03l*s03;
                block_turbo4_0 * blk = (block_turbo4_0 *)((char *)dst_d + dst_row*nb1 + i02l*nb2 + i03l*nb3) + i_blk;

                x[j] = src_row[i_blk * 128 + j];
                item.barrier(sycl::access::fence_space::local_space);

                float sg_sum = sycl::reduce_over_group(sg, x[j] * x[j], sycl::plus<float>());
                if (sg.get_local_id()[0] == 0) warp_accum[sg.get_group_id()[0]] = sg_sum;
                item.barrier(sycl::access::fence_space::local_space);
                float norm_sq = 0.0f;
                for (int i = 0; i < 128 / WARP_SIZE; i++) norm_sq += warp_accum[i];
                const float grp_norm = sycl::sqrt(norm_sq);
                const float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

                x[j] *= inv_norm;
                x[j] *= TURBO_WHT_SIGNS1[j];
                item.barrier(sycl::access::fence_space::local_space);

                for (int h = 1; h < 128; h *= 2) {
                    if (j % (2*h) < h) {
                        float a = x[j], b = x[j+h];
                        x[j] = a + b;
                        x[j+h] = a - b;
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                constexpr float inv_sqrt_128 = 0.08838834764831845f;
                x[j] = x[j] * inv_sqrt_128 * TURBO_WHT_SIGNS2[j];

                const uint8_t idx = turbo_nearest_centroid_4bit(x[j]);

                local_idx[j] = idx;
                item.barrier(sycl::access::fence_space::local_space);

                if (j % 2 == 0) {
                    blk->qs[j/2] = (local_idx[j] & 0xF) | ((local_idx[j+1] & 0xF) << 4);
                }

                const float c = TURBO_CENTROIDS_4BIT[idx];
                // warp_accum reuse is safe: preceding barriers fully drain all reads before this write
                sg_sum = sycl::reduce_over_group(sg, c * c, sycl::plus<float>());
                if (sg.get_local_id()[0] == 0) warp_accum[sg.get_group_id()[0]] = sg_sum;
                item.barrier(sycl::access::fence_space::local_space);
                float recon_sq = 0.0f;
                for (int i = 0; i < 128 / WARP_SIZE; i++) recon_sq += warp_accum[i];
                const float recon_norm = sycl::sqrt(recon_sq);
                const float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;

                if (j == 0) {
                    blk->norm = sycl::half(corrected_norm);
                }
            });
    });

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne13);
    GGML_UNUSED(nb00);
    GGML_UNUSED(nb13);
}

template <typename TIdx>
static void set_rows_sycl_turbo2(ggml_backend_sycl_context & ctx,
                                  const ggml_tensor * src0,
                                  const ggml_tensor * src1,
                                  ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS

    const float * src0_d = (const float *)src0->data;
    const TIdx  * src1_d = (const TIdx *)src1->data;
    block_turbo2_0 * dst_d = (block_turbo2_0 *)dst->data;

    const int64_t s01 = nb01 / sizeof(float);
    const int64_t s02 = nb02 / sizeof(float);
    const int64_t s03 = nb03 / sizeof(float);
    const int64_t s10 = nb10 / sizeof(TIdx);
    const int64_t s11 = nb11 / sizeof(TIdx);
    const int64_t s12 = nb12 / sizeof(TIdx);

    GGML_ASSERT(ne00 % 128 == 0);

    const int64_t n_groups_per_row = ne00 / 128;
    const int64_t n_groups = n_groups_per_row * ne01 * ne02 * ne03;

    if (n_groups == 0) return;

    dpct::queue_ptr stream = ctx.stream();

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> x(128, cgh);
        sycl::local_accessor<float, 1> warp_accum(128 / WARP_SIZE, cgh);
        sycl::local_accessor<uint8_t, 1> local_idx(128, cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(n_groups * 128, 128),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                const int j = item.get_local_id(0);
                const int64_t g = item.get_group(0);
                auto sg = item.get_sub_group();

                const int64_t i_grp = g % n_groups_per_row;
                int64_t tmp = g / n_groups_per_row;
                const int64_t i01l = tmp % ne01;
                tmp = tmp / ne01;
                const int64_t i02l = tmp % ne02;
                const int64_t i03l = tmp / ne02;

                const int64_t i10l = i01l;
                const int64_t i11l = i02l % ne11;
                const int64_t i12l = i03l % ne12;

                const int64_t dst_row = *(src1_d + i10l*s10 + i11l*s11 + i12l*s12);
                // dst_row is work-group-uniform (derived from group id only, not local id j) -- safe to early-return without barrier divergence
                if (dst_row < 0 || dst_row >= ne1) return;
                const float * src_row = src0_d + i01l*s01 + i02l*s02 + i03l*s03;
                block_turbo2_0 * blk = (block_turbo2_0 *)((char *)dst_d + dst_row*nb1 + i02l*nb2 + i03l*nb3) + i_grp;

                x[j] = src_row[i_grp * 128 + j];
                item.barrier(sycl::access::fence_space::local_space);

                float sg_sum = sycl::reduce_over_group(sg, x[j] * x[j], sycl::plus<float>());
                if (sg.get_local_id()[0] == 0) warp_accum[sg.get_group_id()[0]] = sg_sum;
                item.barrier(sycl::access::fence_space::local_space);
                float norm_sq = 0.0f;
                for (int i = 0; i < 128 / WARP_SIZE; i++) norm_sq += warp_accum[i];
                const float grp_norm = sycl::sqrt(norm_sq);
                const float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

                x[j] *= inv_norm;
                x[j] *= TURBO_WHT_SIGNS1[j];
                item.barrier(sycl::access::fence_space::local_space);

                for (int h = 1; h < 128; h *= 2) {
                    if (j % (2*h) < h) {
                        float a = x[j], b = x[j+h];
                        x[j] = a + b;
                        x[j+h] = a - b;
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                constexpr float inv_sqrt_128 = 0.08838834764831845f;
                x[j] = x[j] * inv_sqrt_128 * TURBO_WHT_SIGNS2[j];

                const uint8_t idx = turbo_nearest_centroid_2bit(x[j]);

                local_idx[j] = idx;
                item.barrier(sycl::access::fence_space::local_space);

                if (j % 4 == 0) {
                    blk->qs[j/4] = (local_idx[j] & 0x3)
                                  | ((local_idx[j+1] & 0x3) << 2)
                                  | ((local_idx[j+2] & 0x3) << 4)
                                  | ((local_idx[j+3] & 0x3) << 6);
                }

                const float c = TURBO_CENTROIDS_2BIT[idx];
                // warp_accum reuse is safe: preceding barriers fully drain all reads before this write
                sg_sum = sycl::reduce_over_group(sg, c * c, sycl::plus<float>());
                if (sg.get_local_id()[0] == 0) warp_accum[sg.get_group_id()[0]] = sg_sum;
                item.barrier(sycl::access::fence_space::local_space);
                float recon_sq = 0.0f;
                for (int i = 0; i < 128 / WARP_SIZE; i++) recon_sq += warp_accum[i];
                const float recon_norm = sycl::sqrt(recon_sq);
                const float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;

                if (j == 0) blk->norm = sycl::half(corrected_norm);
            });
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
        case GGML_TYPE_TURBO2_0:
            set_rows_sycl_turbo2<TIdx>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_TURBO3_0:
            set_rows_sycl_turbo3<TIdx>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_TURBO4_0:
            set_rows_sycl_turbo4<TIdx>(ctx, src0, src1, dst);
            break;

        default:
            GGML_ABORT("Unsupported tensor type!");
            break;
    }
}

void ggml_sycl_op_set_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->src[0]->nb[0] == sizeof(float));
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_I64 || dst->src[1]->type == GGML_TYPE_I32);

    if (src1->type == GGML_TYPE_I64) {
        set_rows_sycl<float, int64_t>(ctx, src0, src1, dst);
    } else {
        set_rows_sycl<float, int32_t>(ctx, src0, src1, dst);
    }
}

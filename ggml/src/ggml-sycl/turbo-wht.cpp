/*
 * TurboQuant SYCL Walsh-Hadamard Transform kernel
 * Ported from ggml-cuda/turbo-wht.cu
 *
 * One work-group per WHT group (128 or 64 threads).
 * Template on direction (0=forward, 1=inverse).
 * InnerQ scale_inv: wired up but always identity on SYCL - calibration is CUDA-only (llama-kv-cache.cpp, #ifdef GGML_USE_CUDA).
 */

#include <cstring>

#include "turbo-wht.hpp"
#include "turbo-quant.hpp"

template <int direction>
static void k_turbo_wht_f32_128(const float * __restrict__ src,
                                 float * __restrict__ dst,
                                 const float * __restrict__ scale_inv,
                                 int64_t n_groups,
                                 int64_t head_dim,
                                 int64_t groups_per_head,
                                 sycl::local_accessor<float, 1> shared,
                                 const sycl::nd_item<1> & item) {
    const int64_t g = item.get_group(0);
    if (g >= n_groups) return;

    const int t = item.get_local_id(0);

    const int64_t head_idx    = g / groups_per_head;
    const int64_t grp_in_head = g % groups_per_head;
    const int64_t base        = head_idx * head_dim + grp_in_head * 128;

    shared[t] = src[base + t];
    item.barrier(sycl::access::fence_space::local_space);

    if constexpr (direction == 0) {
        if (scale_inv != nullptr) {
            shared[t] *= scale_inv[t];
            item.barrier(sycl::access::fence_space::local_space);
        }
        shared[t] *= TURBO_WHT_SIGNS1[t];
    } else {
        shared[t] *= TURBO_WHT_SIGNS2[t];
    }
    item.barrier(sycl::access::fence_space::local_space);

#define WHT_STAGE(h) \
    if (t % (2*(h)) < (h)) { \
        float a = shared[t], b = shared[t+(h)]; \
        shared[t] = a + b; shared[t+(h)] = a - b; \
    } \
    item.barrier(sycl::access::fence_space::local_space);

    WHT_STAGE(1)
    WHT_STAGE(2)
    WHT_STAGE(4)
    WHT_STAGE(8)
    WHT_STAGE(16)
    WHT_STAGE(32)
    WHT_STAGE(64)
#undef WHT_STAGE

    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    float result;
    if constexpr (direction == 0) {
        result = shared[t] * inv_sqrt_128 * TURBO_WHT_SIGNS2[t];
    } else {
        result = shared[t] * inv_sqrt_128 * TURBO_WHT_SIGNS1[t];
        if (scale_inv != nullptr) {
            result *= scale_inv[t];
        }
    }

    dst[base + t] = result;
}

template <int direction>
static void k_turbo_wht_f32_64(const float * __restrict__ src,
                                float * __restrict__ dst,
                                const float * __restrict__ scale_inv,
                                int64_t n_groups,
                                int64_t head_dim,
                                int64_t groups_per_head,
                                sycl::local_accessor<float, 1> shared,
                                const sycl::nd_item<1> & item) {
    const int64_t g = item.get_group(0);
    if (g >= n_groups) return;

    const int t = item.get_local_id(0);

    const int64_t head_idx    = g / groups_per_head;
    const int64_t grp_in_head = g % groups_per_head;
    const int64_t base        = head_idx * head_dim + grp_in_head * 64;

    shared[t] = src[base + t];
    item.barrier(sycl::access::fence_space::local_space);

    if constexpr (direction == 0) {
        if (scale_inv != nullptr) {
            shared[t] *= scale_inv[t];
            item.barrier(sycl::access::fence_space::local_space);
        }
        shared[t] *= TURBO_WHT_SIGNS1_64[t];
    } else {
        shared[t] *= TURBO_WHT_SIGNS2_64[t];
    }
    item.barrier(sycl::access::fence_space::local_space);

#define WHT_STAGE(h) \
    if (t % (2*(h)) < (h)) { \
        float a = shared[t], b = shared[t+(h)]; \
        shared[t] = a + b; shared[t+(h)] = a - b; \
    } \
    item.barrier(sycl::access::fence_space::local_space);

    WHT_STAGE(1)
    WHT_STAGE(2)
    WHT_STAGE(4)
    WHT_STAGE(8)
    WHT_STAGE(16)
    WHT_STAGE(32)
#undef WHT_STAGE

    constexpr float inv_sqrt_64 = 0.125f;
    float result;
    if constexpr (direction == 0) {
        result = shared[t] * inv_sqrt_64 * TURBO_WHT_SIGNS2_64[t];
    } else {
        result = shared[t] * inv_sqrt_64 * TURBO_WHT_SIGNS1_64[t];
        if (scale_inv != nullptr) {
            result *= scale_inv[t];
        }
    }

    dst[base + t] = result;
}

static void k_turbo_wht_copy_tail(const float * __restrict__ src,
                                   float * __restrict__ dst,
                                   int64_t n_heads,
                                   int64_t head_dim,
                                   int64_t tail_offset,
                                   int tail_size,
                                   const sycl::nd_item<1> & item) {
    const int64_t i = item.get_global_linear_id();
    if (i >= n_heads * tail_size) return;

    const int64_t head_idx  = i / tail_size;
    const int64_t tail_elem = i % tail_size;
    const int64_t offset    = head_idx * head_dim + tail_offset + tail_elem;
    dst[offset] = src[offset];
}

void ggml_sycl_turbo_wht(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(ggml_is_contiguous(dst));

    int direction;
    int group_size;
    memcpy(&direction,  dst->op_params + 0,            sizeof(int));
    memcpy(&group_size, dst->op_params + sizeof(int),  sizeof(int));

    GGML_ASSERT(direction == 0 || direction == 1);

    const int64_t head_dim = src->ne[0];
    GGML_ASSERT(head_dim > 0);
    GGML_ASSERT(group_size == 64 || group_size == 128);
    GGML_ASSERT(head_dim % group_size == 0 || head_dim > group_size);

    const ggml_tensor * scale_tensor = dst->src[1];
    if (scale_tensor) {
        GGML_ASSERT(scale_tensor->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(scale_tensor));
        GGML_ASSERT(scale_tensor->ne[0] >= group_size);
    }
    const float * scale_inv_ptr = scale_tensor ? (const float *) scale_tensor->data : nullptr;

    const int64_t n_heads         = ggml_nelements(src) / head_dim;
    const int64_t groups_per_head = head_dim / group_size;
    const int     tail_size       = (int)(head_dim % group_size);
    const int64_t n_groups        = groups_per_head * n_heads;

    const float * src_ptr = (const float *) src->data;
    float       * dst_ptr = (float       *) dst->data;

    dpct::queue_ptr stream = ctx.stream();

    if (n_groups > 0) {
        if (group_size == 128) {
            sycl::range<1> global(n_groups * 128);
            sycl::range<1> local(128);

            if (direction == 0) {
                stream->submit([&](sycl::handler & cgh) {
                    sycl::local_accessor<float, 1> shared(128, cgh);
                    cgh.parallel_for(sycl::nd_range<1>(global, local),
                        [=](sycl::nd_item<1> item) {
                            k_turbo_wht_f32_128<0>(src_ptr, dst_ptr, scale_inv_ptr, n_groups, head_dim, groups_per_head, shared, item);
                        });
                });
            } else {
                stream->submit([&](sycl::handler & cgh) {
                    sycl::local_accessor<float, 1> shared(128, cgh);
                    cgh.parallel_for(sycl::nd_range<1>(global, local),
                        [=](sycl::nd_item<1> item) {
                            k_turbo_wht_f32_128<1>(src_ptr, dst_ptr, scale_inv_ptr, n_groups, head_dim, groups_per_head, shared, item);
                        });
                });
            }
        } else if (group_size == 64) {
            sycl::range<1> global(n_groups * 64);
            sycl::range<1> local(64);

            if (direction == 0) {
                stream->submit([&](sycl::handler & cgh) {
                    sycl::local_accessor<float, 1> shared(64, cgh);
                    cgh.parallel_for(sycl::nd_range<1>(global, local),
                        [=](sycl::nd_item<1> item) {
                            k_turbo_wht_f32_64<0>(src_ptr, dst_ptr, scale_inv_ptr, n_groups, head_dim, groups_per_head, shared, item);
                        });
                });
            } else {
                stream->submit([&](sycl::handler & cgh) {
                    sycl::local_accessor<float, 1> shared(64, cgh);
                    cgh.parallel_for(sycl::nd_range<1>(global, local),
                        [=](sycl::nd_item<1> item) {
                            k_turbo_wht_f32_64<1>(src_ptr, dst_ptr, scale_inv_ptr, n_groups, head_dim, groups_per_head, shared, item);
                        });
                });
            }
        } else {
            GGML_ABORT("SYCL TURBO_WHT: unsupported group_size");
        }
    }

    if (tail_size > 0) {
        const int64_t total_tail = n_heads * tail_size;
        constexpr int block_sz = 256;
        const int64_t grid_sz = (total_tail + block_sz - 1) / block_sz;

        stream->parallel_for(
            sycl::nd_range<1>(grid_sz * block_sz, block_sz),
            [=](sycl::nd_item<1> item) {
                k_turbo_wht_copy_tail(src_ptr, dst_ptr, n_heads, head_dim,
                                      groups_per_head * group_size, tail_size, item);
            });
    }
}

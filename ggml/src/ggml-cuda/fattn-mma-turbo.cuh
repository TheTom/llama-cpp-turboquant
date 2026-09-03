// Fused turbo4 (4-bit PolarQuant) MMA flash-attention DECODE launcher.
//
// This is the host-side case launcher for the GQA-packed MMA path with turbo4 KV.
// It reuses the f16 MMA device kernel (flash_attn_ext_f16 in fattn-mma-f16.cuh) but
// instantiates it with type_K/type_V = TURBO4_0 so the in-kernel load tiles dequantize
// raw turbo4 blocks straight into SRAM. Q is ALREADY rotated at the graph level
// (src/llama-graph.cpp) and the FA output is inverse-rotated there too — this path does
// NO inline FWHT and NO src swap (that would double-rotate Q).
//
// Differences vs ggml_cuda_flash_attn_ext_mma_f16_case:
//   * nstages is forced to 0 inside the kernel for turbo (synchronous dequant load), so
//     here we size shared memory for the 1-stage path.
//   * launch_fattn is called with need_f16_K = need_f16_V = false, so launch_fattn does
//     NOT pre-convert K/V to f16; the kernel receives the raw quantized bytes and the
//     true byte pitch nb11/nb21.

#pragma once

#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"

template <int DKQ, int DV, int ncols1, int ncols2, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_mma_turbo_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    constexpr int ncols = ncols1 * ncols2;

    const int  nthreads       = ggml_cuda_fattn_mma_get_nthreads      (DKQ, DV, ncols, cc);
    const int  nbatch_fa      = ggml_cuda_fattn_mma_get_nbatch_fa     (DKQ, DV, ncols, cc);
    const int  nbatch_K2      = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols, cc);
    const int  nbatch_V2      = ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols, cc);
    const int  nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols, cc);
    const bool Q_in_reg       = ggml_cuda_fattn_mma_get_Q_in_reg      (DKQ, DV, ncols, cc);

    // turbo path is always single-stage synchronous (nstages forced to 0 in the kernel).
    const int cols_per_warp = std::min(ncols, get_cols_per_warp(cc));
    const int warp_size_host = ggml_cuda_info().devices[ctx.device].warp_size;
    const int nwarps         = nthreads / warp_size_host;

    // turbo4 never aliases V onto K.
    constexpr bool V_is_K_view = false;

    // must match the swizzled tile stride flash_attn_ext_turbo{2,3,4}_load_tile write through
    // (fattn-mma-f16.cuh's turbo_store_h2 / bytes_rc), same helper as fattn-mma-f16.cuh:2287.
    const int stride_tile_K = ggml_cuda_fattn_smem_swizzle::tile_stride(nbatch_K2, cc);
    const int stride_tile_V = ggml_cuda_fattn_smem_swizzle::tile_stride(nbatch_V2, cc);
    const size_t nbytes_shared_KV_1stage = nbatch_fa            * std::max(stride_tile_K, stride_tile_V) * sizeof(half2);
    const size_t nbytes_shared_Q         = ncols                * (DKQ/2 + 4)                             * sizeof(half2);
    const size_t nbytes_shared_mask      = ncols1               * (nbatch_fa/2 + 4)                       * sizeof(half2);
    const size_t nbytes_shared_combine   = nwarps*cols_per_warp * (nbatch_combine + 4)                    * sizeof(half2);

    const size_t nbytes_shared_KV = nbytes_shared_KV_1stage;

    const size_t nbytes_shared_total = std::max(nbytes_shared_combine, Q_in_reg ?
        std::max(nbytes_shared_Q,  nbytes_shared_KV + nbytes_shared_mask) :
                 nbytes_shared_Q + nbytes_shared_KV + nbytes_shared_mask);

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

#if defined(GGML_USE_HIP)
    using fattn_kernel_ptr_t = const void*;
#else
    using fattn_kernel_ptr_t = fattn_kernel_t;
#endif // defined(GGML_USE_HIP)
    fattn_kernel_t fattn_kernel;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, /* use_sparse */ false, type_K, type_V>;

#if !defined(GGML_USE_MUSA)
        static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
        if (!shared_memory_limit_raised[id]) {
            CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
            shared_memory_limit_raised[id] = true;
        }
#endif // !defined(GGML_USE_MUSA)
    } else {
        constexpr bool use_logit_softcap = true;
        fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, /* use_sparse */ false, type_K, type_V>;

#if !defined(GGML_USE_MUSA)
        static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
        if (!shared_memory_limit_raised[id]) {
            CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
            shared_memory_limit_raised[id] = true;
        }
#endif // !defined(GGML_USE_MUSA)
    }

    // need_f16_K = need_f16_V = false: launch_fattn does NOT convert turbo bytes to f16;
    // the kernel receives raw quantized KV + the true byte pitch. stream_k = true.
    launch_fattn<DV, ncols1, ncols2>
        (ctx, dst, fattn_kernel, nwarps, nbytes_shared_total, nbatch_fa,
         /*need_f16_K=*/false, /*need_f16_V=*/false, /*stream_k=*/true, /*use_sparse=*/false, warp_size_host);
}


#define DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, ncols1, ncols2, tK, tV)                  \
    template void ggml_cuda_flash_attn_ext_mma_turbo_case                           \
    <DKQ, DV, ncols1, ncols2, tK, tV>(ggml_backend_cuda_context & ctx, ggml_tensor * dst)

// The reachable (ncols1, ncols2) set for Q->ne[1] in {1..4} with turing_mma_available
// is exactly: (1,8),(2,8),(4,8),(2,4),(4,4),(4,2),(8,1). Declare those externs only.
#define DECL_FATTN_MMA_TURBO_ALL(DKQ, DV, tK, tV)        \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, 1, 8, tK, tV); \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, 2, 8, tK, tV); \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, 4, 8, tK, tV); \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, 2, 4, tK, tV); \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, 4, 4, tK, tV); \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, 4, 2, tK, tV); \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, 8, 1, tK, tV); \

DECL_FATTN_MMA_TURBO_ALL(128, 128, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0);
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0);
DECL_FATTN_MMA_TURBO_ALL(128, 128, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0);
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0);
DECL_FATTN_MMA_TURBO_ALL(128, 128, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0);
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0);

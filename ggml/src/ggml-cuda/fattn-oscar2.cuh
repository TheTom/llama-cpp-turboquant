// RETIRED: flash_attn_ext_oscar2 — dedicated scalar flash-attention kernel for OSCAR2.
//
// P3 (2026-07-21): This scalar kernel is fully retired. ALL oscar2 decode (K==V,
// D in {128,256,512}) now goes through the MMA turbo path (fattn-mma-f16.cuh +
// fattn-mma-turbo.cuh), which loads oscar2 blocks into shared memory, dequants
// to f16 tiles on-the-fly via flash_attn_ext_oscar2_load_tile, then runs KQ/VKQ
// through tensor-core MMA instructions. Prefill and mixed-type oscar2 cases are
// handled by the VEC kernel (fattn-vec.cuh) for D<=256, or the TILE kernel
// (fattn-tile.cuh) for D>=512 via f16 pre-conversion in launch_fattn.
//
// Removed dispatch from fattn.cu:
//   - #include "fattn-oscar2.cuh"
//   - ggml_cuda_flash_attn_ext_oscar2() dispatcher function
//   - BEST_FATTN_KERNEL_OSCAR2 enum + switch case
//   - All 16 extern template instantiations of ggml_cuda_flash_attn_ext_oscar2_case
//   - BEST_FATTN_KERNEL_OSCAR2 return paths in best-kernel selector
// Replaced with:
//   - MMA turbo gate: always-entered for oscar2 K==V (bypasses GGML_TURBO_MMA_FUSED)
//   - Best-kernel selector: returns VEC for oscar2 D<=256, TILE for D>=512
//
// The flash_attn_ext_oscar2_load_tile() function in fattn-mma-f16.cuh remains
// active as the MMA turbo path's on-the-fly dequantizer for oscar2 blocks.

#include "common.cuh"
#include "fattn-common.cuh"

#if 0
// -- Original kernel code preserved below for reference --
// (All template instantiations removed from fattn.cu; this file is no longer
//  included anywhere and will not compile as-is.)
...
#endif

// This file has been hand-created (turbo MMA decode instances, pairs with a turbo5/turbo6 side). Do NOT run
// generate_cu_files.py over it - that script deletes all *.cu including the turbo VEC instances.

#include "../fattn-mma-f16.cuh"
#include "../fattn-mma-turbo.cuh"

#ifdef GGML_CUDA_FA_ALL_QUANTS
DECL_FATTN_MMA_TURBO_CASE(512, 512, 4, 8, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO6_0);
DECL_FATTN_MMA_TURBO_CASE(512, 512, 4, 8, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO5_0);
DECL_FATTN_MMA_TURBO_CASE(512, 512, 4, 8, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO6_0);
DECL_FATTN_MMA_TURBO_CASE(512, 512, 4, 8, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO5_0);
DECL_FATTN_MMA_TURBO_CASE(512, 512, 4, 8, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO6_0);
DECL_FATTN_MMA_TURBO_CASE(512, 512, 4, 8, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO5_0);
#endif // GGML_CUDA_FA_ALL_QUANTS

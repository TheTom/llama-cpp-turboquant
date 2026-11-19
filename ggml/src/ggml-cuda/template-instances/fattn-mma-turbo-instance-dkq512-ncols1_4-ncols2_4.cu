// This file has been hand-created (turbo MMA decode instance, head dim 512). Do NOT run generate_cu_files.py
// over it - that script deletes all *.cu including the turbo VEC instances.

#include "../fattn-mma-f16.cuh"
#include "../fattn-mma-turbo.cuh"

DECL_FATTN_MMA_TURBO_CASE(512, 512, 4, 4, GGML_TYPE_TURBO6_0, GGML_TYPE_TURBO5_0);
DECL_FATTN_MMA_TURBO_CASE(512, 512, 4, 4, GGML_TYPE_TURBO5_0, GGML_TYPE_TURBO4_0);
DECL_FATTN_MMA_TURBO_CASE(512, 512, 4, 4, GGML_TYPE_TURBO6_0, GGML_TYPE_TURBO6_0);
DECL_FATTN_MMA_TURBO_CASE(512, 512, 4, 4, GGML_TYPE_TURBO5_0, GGML_TYPE_TURBO5_0);

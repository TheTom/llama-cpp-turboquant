// Mixed KV: turbo6 K + f16 V
// turbo6/turbo5 blocks hold 128 values, so there is no head dim 64 instance.

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO6_0, GGML_TYPE_F16);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO6_0, GGML_TYPE_F16);

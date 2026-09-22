// Mixed KV: tq6 K + turbo3 V
// tq6/tq5 blocks hold 128 values, so there is no head dim 64 instance.

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(128, GGML_TYPE_TQ6_0, GGML_TYPE_TURBO3_0);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_TQ6_0, GGML_TYPE_TURBO3_0);

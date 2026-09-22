// Mixed KV: tq6 K + q8_0 V
// tq6/tq5 blocks hold 128 values, so there is no head dim 64 instance.

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(128, GGML_TYPE_TQ6_0, GGML_TYPE_Q8_0);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_TQ6_0, GGML_TYPE_Q8_0);

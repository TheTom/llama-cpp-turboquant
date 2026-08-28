#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst);

// Whether this (K, V) cache type pair can be used together for FlashAttention
// at all, independent of any particular tensor's shape - the same type-pair
// rules the full tensor-shape-aware dispatch above applies. Used by KV cache
// streaming's config validation as an early, type-only pre-check before any
// tensors exist to check shapes against.
bool ggml_cuda_fattn_kv_type_pair_supported(ggml_type type_k, ggml_type type_v);

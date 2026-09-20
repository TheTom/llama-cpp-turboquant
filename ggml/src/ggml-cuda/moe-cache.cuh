#pragma once

// MoE expert cache registration entry point, called from ggml_backend_cuda_reg().
// Registers the provider API with the backend registry (see ggml-backend-moe-cache.h).

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void ggml_cuda_moe_cache_register(void * reg);

// Surrender the device's cache VRAM under allocator pressure; returns bytes freed.
size_t ggml_moe_cache_trim(int device);

#ifdef __cplusplus
}
#endif

#pragma once

#include "ggml.h"

// TurboQuant InnerQ per-channel equalization — cross-TU shared state
// The host-side state lives in turbo-innerq.cu; device-side state is per-TU
// in turbo-quant.cuh (only set-rows.cu needs device access).

#define INNERQ_MAX_CHANNELS 128

// Host-side shared state (defined in turbo-innerq.cu)
extern bool  g_innerq_finalized;
GGML_API float g_innerq_scale_inv_host[INNERQ_MAX_CHANNELS];

// Called from set-rows.cu after InnerQ finalization to publish scale_inv
void turbo_innerq_publish(const float * scale_inv, int group_size);

// Called from llama-kv-cache.cpp (or equivalent) to check if tensor needs update
// Returns true if there are new scale_inv values to upload
GGML_API bool turbo_innerq_needs_tensor_update(void);

// Called after tensor update to clear the flag
GGML_API void turbo_innerq_mark_tensor_updated(void);

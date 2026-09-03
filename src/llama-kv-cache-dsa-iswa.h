#pragma once

#include "llama.h"
#include "ggml.h"

// DSA+ISWA combined KV cache header (upstream shim)
// This is a minimal placeholder to satisfy includes.
// Full implementation should be ported from upstream if needed.

struct llama_kv_cache_dsa_iswa : public llama_memory_i {
    // TODO: implement if DSA+ISWA is required
};

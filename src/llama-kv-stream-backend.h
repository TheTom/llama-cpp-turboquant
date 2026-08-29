#pragma once

#include "ggml.h"
#include "ggml-backend.h"

// Backend-reported capability for experimental block-granular KV cache
// streaming. A backend that implements no streaming support simply does not
// export the "ggml_backend_kv_stream_supported" proc address, in which case
// every field here reports false - this is the normal, expected state for
// any backend before its streaming execution path is wired in.
struct llama_kv_stream_backend_caps {
    bool streamable          = false; // the backend implements block KV streaming at all
    bool type_pair_supported = false; // ...and for the requested (K, V) cache type pair
};

// Resolves the backend's exported kv-stream capability symbols (if any) for
// `dev` and evaluates them for the given cache type pair. Safe to call with
// dev == nullptr, or a backend that exports neither symbol.
llama_kv_stream_backend_caps llama_kv_stream_backend_caps_query(
        ggml_backend_dev_t dev, ggml_type type_k, ggml_type type_v);

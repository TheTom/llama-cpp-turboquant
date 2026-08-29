#pragma once

#include <cstdint>
#include <string>

// Feature gate for experimental block-granular KV cache streaming: the
// authoritative KV cache lives in a pinned host buffer instead of the
// ordinary device buffer for streaming-eligible layers. Streaming is opt-in
// (stage_bytes == 0 disables it) and only valid on cache shapes this gate
// actually supports.
struct llama_kv_stream_config {
    uint64_t stage_bytes         = 0;
    uint64_t minimum_stage_bytes = 0;

    // True when the active memory is a plain unified llama_kv_cache (or the
    // attention component of a hybrid recurrent+attention cache) rather than
    // a specialized attention cache (DSA/DSV4 MLA-style, MSA indexer, SWA
    // dual-cache) this gate does not support.
    bool unified_kv_cache = false;
    bool context_default  = false;
    bool single_sequence  = false;
    bool flash_attention  = false;
    bool kv_offload       = false;

    // True when the active (K, V) cache type pair has a backend-reported
    // streamed FlashAttention implementation, for whichever types those are -
    // not restricted to one hardcoded pair.
    bool type_pair_supported = false;
};

struct llama_kv_stream_config_result {
    bool valid   = false;
    bool enabled = false;
    std::string error;
};

llama_kv_stream_config_result llama_kv_stream_config_validate(const llama_kv_stream_config & config);

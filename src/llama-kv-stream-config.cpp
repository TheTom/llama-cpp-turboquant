#include "llama-kv-stream-config.h"

llama_kv_stream_config_result llama_kv_stream_config_validate(const llama_kv_stream_config & config) {
    if (config.stage_bytes == 0) {
        return { true, false, {} };
    }

    auto invalid = [](const char * error) {
        return llama_kv_stream_config_result { false, false, error };
    };

    if (!config.unified_kv_cache) {
        return invalid("block KV streaming requires a standard unified KV cache "
                        "(unsupported on MLA/indexer or SWA dual-cache architectures)");
    }
    if (!config.single_sequence) {
        return invalid("block KV streaming requires exactly one sequence (-np 1)");
    }
    if (!config.flash_attention) {
        return invalid("block KV streaming requires Flash Attention");
    }
    if (!config.kv_offload) {
        return invalid("block KV streaming requires GPU KV offload");
    }
    if (!config.type_pair_supported) {
        return invalid("block KV streaming is not supported for this K/V cache type pair on the active backend");
    }
    if (config.minimum_stage_bytes == 0 || config.stage_bytes < config.minimum_stage_bytes) {
        return invalid("block KV streaming stage is too small for one 256-token cache page");
    }

    return { true, true, {} };
}

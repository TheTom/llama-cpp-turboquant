#include "llama-kv-stream-backend.h"

llama_kv_stream_backend_caps llama_kv_stream_backend_caps_query(
        ggml_backend_dev_t dev, ggml_type type_k, ggml_type type_v) {
    llama_kv_stream_backend_caps caps;

    if (!dev) {
        return caps;
    }

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (!reg) {
        return caps;
    }

    auto supported_fn = (ggml_backend_kv_stream_supported_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_kv_stream_supported");
    if (!supported_fn || !supported_fn(dev)) {
        return caps;
    }
    caps.streamable = true;

    auto type_pair_fn = (ggml_backend_kv_stream_type_pair_supported_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_kv_stream_type_pair_supported");
    caps.type_pair_supported = type_pair_fn && type_pair_fn(dev, type_k, type_v);

    return caps;
}

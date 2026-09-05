#pragma once

#include <string>
#include <vector>

// Partial FlashAttention result for one logical KV block. The numerator is
// deliberately unnormalized so independently computed blocks can be combined
// with the same max-rescaling used by the CUDA FlashAttention fixup kernel.
struct llama_kv_stream_softmax_part {
    float max_logit  = 0.0f;
    float normalizer = 0.0f;

    std::vector<float> numerator;
};

struct llama_kv_stream_softmax_result {
    bool valid = false;
    std::string error;

    float max_logit  = 0.0f;
    float normalizer = 0.0f;

    std::vector<float> numerator;
    std::vector<float> value;
};

llama_kv_stream_softmax_result llama_kv_stream_softmax_merge(
    const std::vector<llama_kv_stream_softmax_part> & parts);

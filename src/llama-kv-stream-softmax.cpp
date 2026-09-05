#include "llama-kv-stream-softmax.h"

#include <algorithm>
#include <cmath>

llama_kv_stream_softmax_result llama_kv_stream_softmax_merge(
        const std::vector<llama_kv_stream_softmax_part> & parts) {
    llama_kv_stream_softmax_result result;

    auto fail = [&](const char * message) {
        result.valid = false;
        result.error = message;
        result.numerator.clear();
        result.value.clear();
        return result;
    };

    if (parts.empty()) {
        return fail("softmax block merge requires at least one partial result");
    }

    const size_t value_width = parts.front().numerator.size();
    if (value_width == 0) {
        return fail("softmax block merge requires a non-empty value vector");
    }

    result.max_logit = parts.front().max_logit;

    for (const auto & part : parts) {
        if (part.numerator.size() != value_width) {
            return fail("softmax block partial results have different value dimensions");
        }

        if (!std::isfinite(part.max_logit) ||
            !std::isfinite(part.normalizer) || part.normalizer <= 0.0f) {
            return fail("softmax block partial result has invalid metadata");
        }

        if (!std::all_of(part.numerator.begin(), part.numerator.end(), [](float value) {
            return std::isfinite(value);
        })) {
            return fail("softmax block partial result has a non-finite numerator");
        }

        result.max_logit = std::max(result.max_logit, part.max_logit);
    }

    std::vector<double> numerator(value_width, 0.0);
    double normalizer = 0.0;

    for (const auto & part : parts) {
        const double scale = std::exp(double(part.max_logit) - result.max_logit);
        normalizer += scale*part.normalizer;

        for (size_t i = 0; i < value_width; ++i) {
            numerator[i] += scale*part.numerator[i];
        }
    }

    if (!std::isfinite(normalizer) || normalizer <= 0.0) {
        return fail("combined softmax normalizer is invalid");
    }

    result.normalizer = float(normalizer);
    result.numerator.resize(value_width);
    result.value.resize(value_width);

    for (size_t i = 0; i < value_width; ++i) {
        result.numerator[i] = float(numerator[i]);
        result.value[i] = float(numerator[i]/normalizer);
    }

    result.valid = true;
    return result;
}

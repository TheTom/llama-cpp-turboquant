#include "llama-kv-stream-softmax.h"
#include "testing.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

llama_kv_stream_softmax_part make_part(
        const std::vector<float> & logits,
        const std::vector<std::vector<float>> & values,
        size_t begin,
        size_t end) {
    llama_kv_stream_softmax_part result;
    result.max_logit = *std::max_element(logits.begin() + begin, logits.begin() + end);
    result.numerator.assign(values.front().size(), 0.0f);

    for (size_t i = begin; i < end; ++i) {
        const float weight = std::exp(logits[i] - result.max_logit);
        result.normalizer += weight;
        for (size_t j = 0; j < result.numerator.size(); ++j) {
            result.numerator[j] += weight*values[i][j];
        }
    }

    return result;
}

void assert_vector_near(testing & t, const std::vector<float> & expected, const std::vector<float> & actual, float tolerance) {
    if (!t.assert_equal("vector dimensions match", expected.size(), actual.size())) {
        return;
    }

    for (size_t i = 0; i < expected.size(); ++i) {
        t.assert_true("component is within tolerance", std::abs(expected[i] - actual[i]) <= tolerance);
    }
}

} // namespace

int main() {
    testing t;

    const std::vector<float> logits = { -1000.0f, -5.0f, 1.0f, 7.0f, 999.0f, 1000.0f, 3.0f };
    const std::vector<std::vector<float>> values = {
        {  1.0f,  2.0f,  3.0f },
        { -2.0f,  0.5f,  4.0f },
        {  8.0f, -1.0f,  0.0f },
        {  0.0f,  3.0f, -7.0f },
        { 11.0f, -9.0f,  2.0f },
        { 17.0f,  5.0f, -3.0f },
        { -4.0f, 12.0f,  6.0f },
    };

    t.test("partitioned merge matches monolithic stable softmax", [&](testing & t) {
        const auto monolithic = llama_kv_stream_softmax_merge({ make_part(logits, values, 0, logits.size()) });
        const auto partitioned = llama_kv_stream_softmax_merge({
            make_part(logits, values, 0, 2),
            make_part(logits, values, 2, 5),
            make_part(logits, values, 5, 7),
        });

        if (!t.assert_true("monolithic merge is valid", monolithic.valid) ||
            !t.assert_true("partitioned merge is valid", partitioned.valid)) {
            return;
        }

        assert_vector_near(t, monolithic.value, partitioned.value, 1e-5f);
        t.assert_true("global max is preserved", std::abs(partitioned.max_logit - 1000.0f) <= 1e-6f);
    });

    t.test("merge is invariant to block order", [&](testing & t) {
        const auto first  = make_part(logits, values, 0, 3);
        const auto second = make_part(logits, values, 3, 5);
        const auto third  = make_part(logits, values, 5, 7);

        const auto forward = llama_kv_stream_softmax_merge({ first, second, third });
        const auto reverse = llama_kv_stream_softmax_merge({ third, second, first });

        if (!t.assert_true("forward merge is valid", forward.valid) ||
            !t.assert_true("reverse merge is valid", reverse.valid)) {
            return;
        }

        assert_vector_near(t, forward.value, reverse.value, 1e-5f);
        assert_vector_near(t, forward.numerator, reverse.numerator, 1e-4f);
        t.assert_true("normalizer is invariant", std::abs(forward.normalizer - reverse.normalizer) <= 1e-6f);
    });

    t.test("invalid partial results are rejected", [](testing & t) {
        llama_kv_stream_softmax_part first;
        first.max_logit  = 1.0f;
        first.normalizer = 1.0f;
        first.numerator  = { 1.0f, 2.0f };

        auto second = first;
        second.numerator.push_back(3.0f);
        t.assert_true("dimension mismatch is rejected", !llama_kv_stream_softmax_merge({ first, second }).valid);

        second = first;
        second.normalizer = 0.0f;
        t.assert_true("zero normalizer is rejected", !llama_kv_stream_softmax_merge({ first, second }).valid);

        t.assert_true("empty merge is rejected", !llama_kv_stream_softmax_merge({}).valid);
    });

    return t.summary();
}

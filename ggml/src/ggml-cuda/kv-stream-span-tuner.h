#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

// Chooses between the ordinary coalesced streamed-attention kernel and a
// bounded-span pipeline using end-to-end decode graph timings. The tuner is
// reset whenever the resident/ring layout changes.
class ggml_cuda_kv_stream_span_tuner {
public:
    explicit ggml_cuda_kv_stream_span_tuner(
            uint32_t trial_samples = 16,
            double minimum_relative_gain = 0.005,
            uint32_t warmup_samples = 1) :
        trial_samples_(std::max<uint32_t>(trial_samples, 1)),
        minimum_relative_gain_(
                std::isfinite(minimum_relative_gain) ?
                    std::clamp(minimum_relative_gain, 0.0, 0.999) : 0.005),
        warmup_samples_(warmup_samples) {
    }

    void reset() {
        bounded_ = false;
        selected_ = false;
        unbounded_samples_ = 0;
        bounded_samples_ = 0;
        unbounded_warmups_ = 0;
        bounded_warmups_ = 0;
        unbounded_ms_ = 0.0;
        bounded_ms_ = 0.0;
    }

    void observe(double elapsed_ms, bool streamed, bool sample_was_bounded) {
        if (selected_ || !streamed || !std::isfinite(elapsed_ms) || elapsed_ms <= 0.0) {
            return;
        }

        uint32_t & warmups = sample_was_bounded ? bounded_warmups_ : unbounded_warmups_;
        if (warmups < warmup_samples_) {
            ++warmups;
            return;
        }

        if (sample_was_bounded) {
            bounded_ms_ += elapsed_ms;
            ++bounded_samples_;
        } else {
            unbounded_ms_ += elapsed_ms;
            ++unbounded_samples_;
        }

        if (unbounded_samples_ < trial_samples_) {
            bounded_ = false;
            return;
        }

        if (bounded_samples_ < trial_samples_) {
            bounded_ = true;
            return;
        }

        const double unbounded_average_ms = unbounded_ms_ / unbounded_samples_;
        const double bounded_average_ms = bounded_ms_ / bounded_samples_;
        bounded_ = bounded_average_ms < unbounded_average_ms * (1.0 - minimum_relative_gain_);
        selected_ = true;
    }

    bool use_bounded() const { return bounded_; }
    bool selected() const { return selected_; }
    double unbounded_average_ms() const {
        return unbounded_samples_ == 0 ? 0.0 : unbounded_ms_ / unbounded_samples_;
    }
    double bounded_average_ms() const {
        return bounded_samples_ == 0 ? 0.0 : bounded_ms_ / bounded_samples_;
    }

private:
    uint32_t trial_samples_ = 16;
    double minimum_relative_gain_ = 0.005;
    uint32_t warmup_samples_ = 1;
    uint32_t unbounded_warmups_ = 0;
    uint32_t bounded_warmups_ = 0;
    bool bounded_ = false;
    bool selected_ = false;
    uint32_t unbounded_samples_ = 0;
    uint32_t bounded_samples_ = 0;
    double unbounded_ms_ = 0.0;
    double bounded_ms_ = 0.0;
};

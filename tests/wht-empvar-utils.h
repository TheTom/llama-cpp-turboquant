#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace wht_empvar {

static constexpr float kEps = 1e-8f;

// Matches the fixed TurboQuant WHT sign convention in the local runtime.
static constexpr float kTurboWhtS1[128] = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1
};

static constexpr float kTurboWhtS2[128] = {
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1
};

inline bool is_power_of_two(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

inline float l2_norm(const std::vector<float> & x) {
    double sumsq = 0.0;
    for (float v : x) {
        sumsq += double(v) * double(v);
    }
    return float(std::sqrt(sumsq));
}

inline std::vector<float> normalize(const std::vector<float> & x) {
    std::vector<float> out = x;
    const float norm = l2_norm(x);
    if (norm <= kEps) {
        return out;
    }
    for (float & v : out) {
        v /= norm;
    }
    return out;
}

inline std::vector<float> pad_right(const std::vector<float> & x, int target_dim) {
    if ((int) x.size() >= target_dim) {
        return x;
    }
    std::vector<float> out(target_dim, 0.0f);
    std::copy(x.begin(), x.end(), out.begin());
    return out;
}

inline void fwht_inplace(std::vector<float> & x, int n) {
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += 2 * len) {
            for (int j = 0; j < len; ++j) {
                const float a = x[i + j];
                const float b = x[i + j + len];
                x[i + j]       = a + b;
                x[i + j + len] = a - b;
            }
        }
    }
    const float scale = 1.0f / std::sqrt(float(n));
    for (int i = 0; i < n; ++i) {
        x[i] *= scale;
    }
}

inline std::vector<float> turbo_rotate_forward(const std::vector<float> & x) {
    const int n = int(x.size());
    if (!(n == 64 || n == 128)) {
        throw std::runtime_error("turbo_rotate_forward requires 64 or 128 dims");
    }
    std::vector<float> y = x;
    for (int i = 0; i < n; ++i) {
        y[i] *= kTurboWhtS1[i];
    }
    fwht_inplace(y, n);
    for (int i = 0; i < n; ++i) {
        y[i] *= kTurboWhtS2[i];
    }
    return y;
}

inline std::vector<float> turbo_rotate_inverse(const std::vector<float> & x) {
    const int n = int(x.size());
    if (!(n == 64 || n == 128)) {
        throw std::runtime_error("turbo_rotate_inverse requires 64 or 128 dims");
    }
    std::vector<float> y = x;
    for (int i = 0; i < n; ++i) {
        y[i] *= kTurboWhtS2[i];
    }
    fwht_inplace(y, n);
    for (int i = 0; i < n; ++i) {
        y[i] *= kTurboWhtS1[i];
    }
    return y;
}

inline std::vector<float> gaussian_samples(float sigma, int bits) {
    const int n_samples = 1 << 16;
    std::mt19937 rng(0xC0DEC0DEu ^ uint32_t(bits * 7919) ^ uint32_t(std::llround(sigma * 1000000.0f)));
    std::normal_distribution<float> dist(0.0f, std::max(sigma, 1e-6f));
    std::vector<float> samples(n_samples);
    for (float & v : samples) {
        v = dist(rng);
    }
    std::sort(samples.begin(), samples.end());
    return samples;
}

inline std::vector<float> solve_lloyd_max_for_sigma(float sigma, int bits) {
    if (bits <= 0) {
        return {0.0f};
    }
    if (bits == 1) {
        const float c = sigma * std::sqrt(2.0f / float(M_PI));
        return {-c, c};
    }
    if (bits == 2) {
        return {-1.51f * sigma, -0.453f * sigma, 0.453f * sigma, 1.51f * sigma};
    }

    const int n_levels = 1 << bits;
    const std::vector<float> samples = gaussian_samples(sigma, bits);
    std::vector<float> centroids(n_levels);
    for (int i = 0; i < n_levels; ++i) {
        const size_t idx = std::min<size_t>(samples.size() - 1, (samples.size() * (2 * i + 1)) / (2 * n_levels));
        centroids[i] = samples[idx];
    }

    for (int iter = 0; iter < 100; ++iter) {
        std::vector<float> bounds(n_levels - 1);
        for (int i = 0; i < n_levels - 1; ++i) {
            bounds[i] = 0.5f * (centroids[i] + centroids[i + 1]);
        }

        std::vector<double> sum(n_levels, 0.0);
        std::vector<size_t> count(n_levels, 0);
        int bucket = 0;
        for (float v : samples) {
            while (bucket + 1 < n_levels && v > bounds[bucket]) {
                ++bucket;
            }
            sum[bucket] += v;
            count[bucket] += 1;
        }

        float max_delta = 0.0f;
        for (int i = 0; i < n_levels; ++i) {
            if (count[i] == 0) {
                continue;
            }
            const float updated = float(sum[i] / double(count[i]));
            max_delta = std::max(max_delta, std::fabs(updated - centroids[i]));
            centroids[i] = updated;
        }
        if (max_delta < 1e-6f) {
            break;
        }
    }

    return centroids;
}

inline std::vector<float> solve_fixed_codebook(int bits, int dim) {
    if (bits <= 0) {
        return {0.0f};
    }
    if (bits == 1) {
        const float c = std::sqrt(2.0f / (float(M_PI) * float(dim)));
        return {-c, c};
    }
    if (bits == 2) {
        const float scale = 1.0f / std::sqrt(float(dim));
        return {-1.51f * scale, -0.453f * scale, 0.453f * scale, 1.51f * scale};
    }
    return solve_lloyd_max_for_sigma(1.0f / std::sqrt(float(dim)), bits);
}

inline float quantize_nearest(float x, const std::vector<float> & codebook) {
    float best = codebook.front();
    float best_dist = std::fabs(x - best);
    for (float c : codebook) {
        const float dist = std::fabs(x - c);
        if (dist < best_dist) {
            best = c;
            best_dist = dist;
        }
    }
    return best;
}

struct CalibrationStats {
    std::vector<float> variances;
    float mean_variance = 0.0f;
    float std_variance = 0.0f;
    float l2_to_uniform = 0.0f;
};

struct ChunkedCalibrationStats {
    int runtime_dim = 0;
    int target_dim = 0;
    std::vector<CalibrationStats> chunks;
    std::vector<float> variances;
    float mean_variance = 0.0f;
    float std_variance = 0.0f;
    float l2_to_uniform = 0.0f;
};

inline CalibrationStats empirical_variances(const std::vector<std::vector<float>> & samples) {
    if (samples.empty()) {
        throw std::runtime_error("empirical_variances requires non-empty samples");
    }
    const int dim = int(samples.front().size());
    std::vector<double> sumsq(dim, 0.0);
    size_t n = 0;
    for (const auto & sample : samples) {
        const auto rotated = turbo_rotate_forward(normalize(sample));
        for (int i = 0; i < dim; ++i) {
            sumsq[i] += double(rotated[i]) * double(rotated[i]);
        }
        ++n;
    }

    CalibrationStats out;
    out.variances.resize(dim);
    double mean = 0.0;
    for (int i = 0; i < dim; ++i) {
        out.variances[i] = float(sumsq[i] / double(n));
        mean += out.variances[i];
    }
    mean /= dim;
    out.mean_variance = float(mean);

    double var = 0.0;
    double l2 = 0.0;
    const double uniform = 1.0 / double(dim);
    for (float v : out.variances) {
        const double dv = double(v) - mean;
        var += dv * dv;
        const double du = double(v) - uniform;
        l2 += du * du;
    }
    out.std_variance = float(std::sqrt(var / dim));
    out.l2_to_uniform = float(std::sqrt(l2));
    return out;
}

inline std::vector<std::vector<float>> split_rows_for_runtime_groups(
        const std::vector<std::vector<float>> & rows,
        int runtime_dim) {
    std::vector<std::vector<float>> out;
    for (const auto & row : rows) {
        if ((int) row.size() <= runtime_dim) {
            out.push_back(pad_right(row, runtime_dim));
            continue;
        }
        if ((int) row.size() % runtime_dim != 0) {
            throw std::runtime_error("row width is not divisible by runtime_dim");
        }
        for (int off = 0; off < (int) row.size(); off += runtime_dim) {
            out.emplace_back(row.begin() + off, row.begin() + off + runtime_dim);
        }
    }
    return out;
}

inline ChunkedCalibrationStats empirical_variances_chunked(
        const std::vector<std::vector<float>> & samples,
        int runtime_dim) {
    if (samples.empty()) {
        throw std::runtime_error("empirical_variances_chunked requires non-empty samples");
    }
    const int target_dim = (int) samples.front().size();
    const int n_chunks = std::max(1, (target_dim + runtime_dim - 1) / runtime_dim);

    ChunkedCalibrationStats out;
    out.runtime_dim = runtime_dim;
    out.target_dim = target_dim;
    out.chunks.resize(n_chunks);

    std::vector<std::vector<std::vector<float>>> grouped(n_chunks);
    for (const auto & sample : samples) {
        if ((int) sample.size() > runtime_dim && (int) sample.size() % runtime_dim != 0) {
            throw std::runtime_error("sample width is not divisible by runtime_dim");
        }
        const int sample_chunks = std::max(1, (int) sample.size() / runtime_dim);
        for (int chunk = 0; chunk < sample_chunks; ++chunk) {
            const int off = chunk * runtime_dim;
            grouped[chunk].emplace_back(sample.begin() + off, sample.begin() + off + runtime_dim);
        }
    }

    out.variances.reserve(n_chunks * runtime_dim);
    for (int chunk = 0; chunk < n_chunks; ++chunk) {
        out.chunks[chunk] = empirical_variances(grouped[chunk]);
        out.variances.insert(
                out.variances.end(),
                out.chunks[chunk].variances.begin(),
                out.chunks[chunk].variances.end());
    }

    const int total_dim = (int) out.variances.size();
    double mean = 0.0;
    for (float v : out.variances) {
        mean += v;
    }
    mean /= total_dim;
    out.mean_variance = float(mean);

    double var = 0.0;
    double l2 = 0.0;
    const double uniform = 1.0 / double(runtime_dim);
    for (float v : out.variances) {
        const double dv = double(v) - mean;
        var += dv * dv;
        const double du = double(v) - uniform;
        l2 += du * du;
    }
    out.std_variance = float(std::sqrt(var / total_dim));
    out.l2_to_uniform = float(std::sqrt(l2));
    return out;
}

struct ReconstructionMetrics {
    double mean_cosine = 0.0;
    double mean_mse = 0.0;
};

inline std::vector<std::vector<float>> reconstruct_dataset_cropped(
        const std::vector<std::vector<float>> & samples,
        const std::vector<std::vector<float>> & codebooks,
        bool renormalize_rotated,
        int target_dim);

struct AttentionMetrics {
    double mean_output_cosine = 0.0;
    double mean_score_cosine = 0.0;
    double mean_prob_kl = 0.0;
};

inline ReconstructionMetrics evaluate_reconstruction(
        const std::vector<std::vector<float>> & eval_samples,
        const std::vector<std::vector<float>> & codebooks,
        bool renormalize_rotated) {
    if (eval_samples.empty()) {
        throw std::runtime_error("evaluate_reconstruction requires non-empty samples");
    }
    const int dim = int(eval_samples.front().size());
    ReconstructionMetrics out;

    for (const auto & sample : eval_samples) {
        const float norm = std::max(l2_norm(sample), kEps);
        auto rotated = turbo_rotate_forward(normalize(sample));
        for (int i = 0; i < dim; ++i) {
            rotated[i] = quantize_nearest(rotated[i], codebooks[i]);
        }
        if (renormalize_rotated) {
            const float qr_norm = std::max(l2_norm(rotated), kEps);
            for (float & v : rotated) {
                v /= qr_norm;
            }
        }
        auto recon = turbo_rotate_inverse(rotated);
        for (float & v : recon) {
            v *= norm;
        }

        double dot = 0.0;
        double xx = 0.0;
        double yy = 0.0;
        double mse = 0.0;
        for (int i = 0; i < dim; ++i) {
            dot += double(sample[i]) * double(recon[i]);
            xx += double(sample[i]) * double(sample[i]);
            yy += double(recon[i]) * double(recon[i]);
            const double err = double(sample[i]) - double(recon[i]);
            mse += err * err;
        }
        const double denom = std::sqrt(std::max(xx * yy, 1e-20));
        out.mean_cosine += denom > 0.0 ? dot / denom : 0.0;
        out.mean_mse += mse / dim;
    }

    out.mean_cosine /= eval_samples.size();
    out.mean_mse /= eval_samples.size();
    return out;
}

inline std::vector<std::vector<float>> reconstruct_dataset_chunked(
        const std::vector<std::vector<float>> & samples,
        const std::vector<std::vector<std::vector<float>>> & chunk_codebooks,
        bool renormalize_rotated) {
    if (samples.empty()) {
        throw std::runtime_error("reconstruct_dataset_chunked requires non-empty samples");
    }
    if (chunk_codebooks.empty()) {
        throw std::runtime_error("reconstruct_dataset_chunked requires non-empty chunk codebooks");
    }
    const int runtime_dim = (int) chunk_codebooks.front().size();
    std::vector<std::vector<float>> out(samples.size());
    for (size_t row = 0; row < samples.size(); ++row) {
        const auto & sample = samples[row];
        if ((int) sample.size() > runtime_dim && (int) sample.size() % runtime_dim != 0) {
            throw std::runtime_error("sample width is not divisible by runtime_dim");
        }
        const int n_chunks = std::max(1, (int) sample.size() / runtime_dim);
        std::vector<float> recon_full;
        recon_full.reserve(n_chunks * runtime_dim);
        for (int chunk = 0; chunk < n_chunks; ++chunk) {
            const auto & books = chunk_codebooks[std::min<int>(chunk, (int) chunk_codebooks.size() - 1)];
            const int off = chunk * runtime_dim;
            std::vector<float> part(sample.begin() + off, sample.begin() + off + runtime_dim);
            const float norm = std::max(l2_norm(part), kEps);
            auto rotated = turbo_rotate_forward(normalize(part));
            for (int i = 0; i < runtime_dim; ++i) {
                rotated[i] = quantize_nearest(rotated[i], books[i]);
            }
            if (renormalize_rotated) {
                const float qr_norm = std::max(l2_norm(rotated), kEps);
                for (float & v : rotated) {
                    v /= qr_norm;
                }
            }
            auto recon = turbo_rotate_inverse(rotated);
            for (float & v : recon) {
                v *= norm;
            }
            recon_full.insert(recon_full.end(), recon.begin(), recon.end());
        }
        recon_full.resize(sample.size());
        out[row] = std::move(recon_full);
    }
    return out;
}

inline ReconstructionMetrics evaluate_reconstruction_chunked(
        const std::vector<std::vector<float>> & eval_samples,
        const std::vector<std::vector<std::vector<float>>> & chunk_codebooks,
        bool renormalize_rotated) {
    if (eval_samples.empty()) {
        throw std::runtime_error("evaluate_reconstruction_chunked requires non-empty samples");
    }
    const auto recon = reconstruct_dataset_chunked(eval_samples, chunk_codebooks, renormalize_rotated);
    ReconstructionMetrics out;

    for (size_t row = 0; row < eval_samples.size(); ++row) {
        const auto & sample = eval_samples[row];
        const auto & sample_recon = recon[row];
        double dot = 0.0;
        double xx = 0.0;
        double yy = 0.0;
        double mse = 0.0;
        for (size_t i = 0; i < sample.size(); ++i) {
            dot += double(sample[i]) * double(sample_recon[i]);
            xx += double(sample[i]) * double(sample[i]);
            yy += double(sample_recon[i]) * double(sample_recon[i]);
            const double err = double(sample[i]) - double(sample_recon[i]);
            mse += err * err;
        }
        const double denom = std::sqrt(std::max(xx * yy, 1e-20));
        out.mean_cosine += denom > 0.0 ? dot / denom : 0.0;
        out.mean_mse += mse / sample.size();
    }

    out.mean_cosine /= eval_samples.size();
    out.mean_mse /= eval_samples.size();
    return out;
}

inline ReconstructionMetrics evaluate_reconstruction_cropped(
        const std::vector<std::vector<float>> & eval_samples,
        const std::vector<std::vector<float>> & codebooks,
        bool renormalize_rotated,
        int target_dim) {
    const auto recon = reconstruct_dataset_cropped(eval_samples, codebooks, renormalize_rotated, target_dim);
    ReconstructionMetrics out;

    for (size_t row = 0; row < eval_samples.size(); ++row) {
        const auto & sample = eval_samples[row];
        const auto & sample_recon = recon[row];
        double dot = 0.0;
        double xx = 0.0;
        double yy = 0.0;
        double mse = 0.0;
        for (int i = 0; i < target_dim; ++i) {
            dot += double(sample[i]) * double(sample_recon[i]);
            xx += double(sample[i]) * double(sample[i]);
            yy += double(sample_recon[i]) * double(sample_recon[i]);
            const double err = double(sample[i]) - double(sample_recon[i]);
            mse += err * err;
        }
        const double denom = std::sqrt(std::max(xx * yy, 1e-20));
        out.mean_cosine += denom > 0.0 ? dot / denom : 0.0;
        out.mean_mse += mse / target_dim;
    }

    out.mean_cosine /= eval_samples.size();
    out.mean_mse /= eval_samples.size();
    return out;
}

inline std::vector<std::vector<float>> reconstruct_dataset(
        const std::vector<std::vector<float>> & samples,
        const std::vector<std::vector<float>> & codebooks,
        bool renormalize_rotated) {
    if (samples.empty()) {
        throw std::runtime_error("reconstruct_dataset requires non-empty samples");
    }
    const int dim = int(samples.front().size());
    std::vector<std::vector<float>> out(samples.size(), std::vector<float>(dim));
    for (size_t row = 0; row < samples.size(); ++row) {
        const auto & sample = samples[row];
        const float norm = std::max(l2_norm(sample), kEps);
        auto rotated = turbo_rotate_forward(normalize(sample));
        for (int i = 0; i < dim; ++i) {
            rotated[i] = quantize_nearest(rotated[i], codebooks[i]);
        }
        if (renormalize_rotated) {
            const float qr_norm = std::max(l2_norm(rotated), kEps);
            for (float & v : rotated) {
                v /= qr_norm;
            }
        }
        auto recon = turbo_rotate_inverse(rotated);
        for (float & v : recon) {
            v *= norm;
        }
        out[row] = std::move(recon);
    }
    return out;
}

inline std::vector<std::vector<float>> reconstruct_dataset_cropped(
        const std::vector<std::vector<float>> & samples,
        const std::vector<std::vector<float>> & codebooks,
        bool renormalize_rotated,
        int target_dim) {
    std::vector<std::vector<float>> padded;
    padded.reserve(samples.size());
    for (const auto & sample : samples) {
        padded.push_back(pad_right(sample, (int) codebooks.size()));
    }
    auto recon = reconstruct_dataset(padded, codebooks, renormalize_rotated);
    for (auto & row : recon) {
        row.resize(target_dim);
    }
    return recon;
}

inline std::vector<std::vector<float>> score_matrix_proxy_qeqk(const std::vector<std::vector<float>> & k) {
    if (k.empty()) {
        throw std::runtime_error("score_matrix_proxy_qeqk requires non-empty samples");
    }
    const int n = int(k.size());
    const int dim = int(k.front().size());
    const float inv_sqrt_d = 1.0f / std::sqrt(float(dim));
    std::vector<std::vector<float>> scores(n, std::vector<float>(n, 0.0f));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double dot = 0.0;
            for (int d = 0; d < dim; ++d) {
                dot += double(k[i][d]) * double(k[j][d]);
            }
            scores[i][j] = float(dot * inv_sqrt_d);
        }
    }
    return scores;
}

inline std::vector<float> softmax_row(const std::vector<float> & row) {
    std::vector<float> out(row.size());
    float max_v = -std::numeric_limits<float>::infinity();
    for (float v : row) {
        max_v = std::max(max_v, v);
    }
    double sum = 0.0;
    for (size_t i = 0; i < row.size(); ++i) {
        out[i] = std::exp(row[i] - max_v);
        sum += out[i];
    }
    const float inv = float(1.0 / std::max(sum, 1e-20));
    for (float & v : out) {
        v *= inv;
    }
    return out;
}

inline std::vector<std::vector<float>> attention_output_proxy_qeqk(
        const std::vector<std::vector<float>> & k,
        const std::vector<std::vector<float>> & v) {
    if (k.size() != v.size()) {
        throw std::runtime_error("attention_output_proxy_qeqk requires matching K/V sample counts");
    }
    if (k.empty()) {
        throw std::runtime_error("attention_output_proxy_qeqk requires non-empty samples");
    }
    const int n = int(k.size());
    const int dim = int(v.front().size());
    const auto scores = score_matrix_proxy_qeqk(k);
    std::vector<std::vector<float>> out(n, std::vector<float>(dim, 0.0f));
    for (int i = 0; i < n; ++i) {
        const auto probs = softmax_row(scores[i]);
        for (int j = 0; j < n; ++j) {
            for (int d = 0; d < dim; ++d) {
                out[i][d] += probs[j] * v[j][d];
            }
        }
    }
    return out;
}

inline double mean_cosine_rows(
        const std::vector<std::vector<float>> & a,
        const std::vector<std::vector<float>> & b) {
    if (a.size() != b.size() || a.empty()) {
        throw std::runtime_error("mean_cosine_rows requires matching non-empty inputs");
    }
    double total = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double dot = 0.0;
        double aa = 0.0;
        double bb = 0.0;
        for (size_t d = 0; d < a[i].size(); ++d) {
            dot += double(a[i][d]) * double(b[i][d]);
            aa += double(a[i][d]) * double(a[i][d]);
            bb += double(b[i][d]) * double(b[i][d]);
        }
        const double denom = std::sqrt(std::max(aa * bb, 1e-20));
        total += denom > 0.0 ? dot / denom : 0.0;
    }
    return total / a.size();
}

inline AttentionMetrics evaluate_attention_proxy_qeqk(
        const std::vector<std::vector<std::vector<float>>> & k_chunks,
        const std::vector<std::vector<std::vector<float>>> & v_chunks,
        const std::vector<std::vector<float>> & k_codebooks,
        const std::vector<std::vector<float>> & v_codebooks,
        bool renormalize_rotated) {
    if (k_chunks.size() != v_chunks.size() || k_chunks.empty()) {
        throw std::runtime_error("evaluate_attention_proxy_qeqk requires matching non-empty chunk inputs");
    }

    AttentionMetrics out;
    const double eps = 1e-8;
    for (size_t chunk_idx = 0; chunk_idx < k_chunks.size(); ++chunk_idx) {
        const auto & k = k_chunks[chunk_idx];
        const auto & v = v_chunks[chunk_idx];
        const auto k_hat = reconstruct_dataset(k, k_codebooks, renormalize_rotated);
        const auto v_hat = reconstruct_dataset(v, v_codebooks, renormalize_rotated);

        const auto scores_ref = score_matrix_proxy_qeqk(k);
        const auto scores_hat = score_matrix_proxy_qeqk(k_hat);
        const auto out_ref = attention_output_proxy_qeqk(k, v);
        const auto out_hat = attention_output_proxy_qeqk(k_hat, v_hat);

        out.mean_output_cosine += mean_cosine_rows(out_ref, out_hat);
        out.mean_score_cosine += mean_cosine_rows(scores_ref, scores_hat);

        double chunk_kl = 0.0;
        for (size_t row = 0; row < scores_ref.size(); ++row) {
            const auto p = softmax_row(scores_ref[row]);
            const auto q = softmax_row(scores_hat[row]);
            double row_kl = 0.0;
            for (size_t col = 0; col < p.size(); ++col) {
                const double pp = std::max<double>(p[col], eps);
                const double qq = std::max<double>(q[col], eps);
                row_kl += pp * (std::log(pp) - std::log(qq));
            }
            chunk_kl += row_kl;
        }
        out.mean_prob_kl += chunk_kl / scores_ref.size();
    }

    const double inv = 1.0 / k_chunks.size();
    out.mean_output_cosine *= inv;
    out.mean_score_cosine *= inv;
    out.mean_prob_kl *= inv;
    return out;
}

inline AttentionMetrics evaluate_attention_proxy_qeqk_cropped(
        const std::vector<std::vector<std::vector<float>>> & k_chunks,
        const std::vector<std::vector<std::vector<float>>> & v_chunks,
        const std::vector<std::vector<float>> & k_codebooks,
        const std::vector<std::vector<float>> & v_codebooks,
        bool renormalize_rotated,
        int k_target_dim,
        int v_target_dim) {
    if (k_chunks.size() != v_chunks.size() || k_chunks.empty()) {
        throw std::runtime_error("evaluate_attention_proxy_qeqk_cropped requires matching non-empty chunk inputs");
    }

    AttentionMetrics out;
    const double eps = 1e-8;
    for (size_t chunk_idx = 0; chunk_idx < k_chunks.size(); ++chunk_idx) {
        const auto & k = k_chunks[chunk_idx];
        const auto & v = v_chunks[chunk_idx];
        const auto k_hat = reconstruct_dataset_cropped(k, k_codebooks, renormalize_rotated, k_target_dim);
        const auto v_hat = reconstruct_dataset_cropped(v, v_codebooks, renormalize_rotated, v_target_dim);

        const auto scores_ref = score_matrix_proxy_qeqk(k);
        const auto scores_hat = score_matrix_proxy_qeqk(k_hat);
        const auto out_ref = attention_output_proxy_qeqk(k, v);
        const auto out_hat = attention_output_proxy_qeqk(k_hat, v_hat);

        out.mean_output_cosine += mean_cosine_rows(out_ref, out_hat);
        out.mean_score_cosine += mean_cosine_rows(scores_ref, scores_hat);

        double chunk_kl = 0.0;
        for (size_t row = 0; row < scores_ref.size(); ++row) {
            const auto p = softmax_row(scores_ref[row]);
            const auto q = softmax_row(scores_hat[row]);
            double row_kl = 0.0;
            for (size_t col = 0; col < p.size(); ++col) {
                const double pp = std::max<double>(p[col], eps);
                const double qq = std::max<double>(q[col], eps);
                row_kl += pp * (std::log(pp) - std::log(qq));
            }
            chunk_kl += row_kl;
        }
        out.mean_prob_kl += chunk_kl / scores_ref.size();
    }

    const double inv = 1.0 / k_chunks.size();
    out.mean_output_cosine *= inv;
    out.mean_score_cosine *= inv;
    out.mean_prob_kl *= inv;
    return out;
}

inline std::vector<std::vector<float>> build_empirical_codebooks(const std::vector<float> & variances, int bits) {
    std::vector<std::vector<float>> codebooks;
    codebooks.reserve(variances.size());
    for (float var : variances) {
        codebooks.push_back(solve_lloyd_max_for_sigma(std::sqrt(std::max(var, 1e-6f)), bits));
    }
    return codebooks;
}

inline std::vector<std::vector<float>> build_fixed_codebooks(int dim, int bits) {
    const std::vector<float> shared = solve_fixed_codebook(bits, dim);
    return std::vector<std::vector<float>>(dim, shared);
}

inline std::vector<std::vector<float>> make_gaussian_dataset(int n, int dim, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<std::vector<float>> out(n, std::vector<float>(dim));
    for (auto & row : out) {
        for (float & v : row) {
            v = dist(rng);
        }
    }
    return out;
}

} // namespace wht_empvar

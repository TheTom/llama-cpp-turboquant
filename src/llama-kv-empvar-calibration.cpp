#include "llama-kv-empvar-calibration.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <stdexcept>

namespace {

static constexpr float LLAMA_KV_EMPVAR_EPS = 1e-8f;
static constexpr int LLAMA_KV_CALIB_GROUP = 128;

// Matches the fixed TurboQuant WHT sign convention used by the local runtime.
static constexpr float LLAMA_KV_EMPVAR_WHT_S1[128] = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1
};

static constexpr float LLAMA_KV_EMPVAR_WHT_S2[128] = {
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1
};

static void fwht_128(float * x) {
    for (int len = 1; len < 128; len <<= 1) {
        for (int i = 0; i < 128; i += 2 * len) {
            for (int j = 0; j < len; ++j) {
                const float a = x[i + j];
                const float b = x[i + j + len];
                x[i + j]       = a + b;
                x[i + j + len] = a - b;
            }
        }
    }

    const float scale = 1.0f / std::sqrt(128.0f);
    for (int i = 0; i < 128; ++i) {
        x[i] *= scale;
    }
}

static void rotate_group_128(float * x) {
    for (int i = 0; i < 128; ++i) {
        x[i] *= LLAMA_KV_EMPVAR_WHT_S1[i];
    }
    fwht_128(x);
    for (int i = 0; i < 128; ++i) {
        x[i] *= LLAMA_KV_EMPVAR_WHT_S2[i];
    }
}

static int pad_to_group(int n, int group) {
    return ((n + group - 1) / group) * group;
}

static float compute_padded_group_norm(const float * row, int offset, int head_dim, int n) {
    double sumsq = 0.0;
    for (int i = 0; i < n; ++i) {
        const int coord = offset + i;
        const double v = coord < head_dim ? row[coord] : 0.0;
        sumsq += v * v;
    }
    return float(std::sqrt(sumsq));
}

static llama_kv_empvar_chunk_stats make_chunk_stats(
        const std::vector<double> & sumsq,
        uint64_t n_rows,
        int offset,
        int wht_group) {
    llama_kv_empvar_chunk_stats out;
    out.variances.resize(wht_group);

    if (n_rows == 0) {
        return out;
    }

    double mean = 0.0;
    for (int i = 0; i < wht_group; ++i) {
        out.variances[i] = float(sumsq[offset + i] / double(n_rows));
        mean += out.variances[i];
    }
    mean /= double(wht_group);
    out.mean_variance = float(mean);

    double var = 0.0;
    double l2 = 0.0;
    const double uniform = 1.0 / double(wht_group);
    for (float v : out.variances) {
        const double dv = double(v) - mean;
        const double du = double(v) - uniform;
        var += dv * dv;
        l2 += du * du;
    }
    out.std_variance = float(std::sqrt(var / double(wht_group)));
    out.l2_to_uniform = float(std::sqrt(l2));
    return out;
}

static void jacobi_eigen_128(
        const std::vector<double> & cov,
        std::vector<float> & eigvals,
        std::vector<float> & eigvecs_row_major) {
    constexpr int n = LLAMA_KV_CALIB_GROUP;
    std::vector<double> a = cov;
    std::vector<double> v(n * n, 0.0);
    for (int i = 0; i < n; ++i) {
        v[i * n + i] = 1.0;
    }

    for (int iter = 0; iter < 64 * n * n; ++iter) {
        int p = 0;
        int q = 1;
        double max_off = 0.0;
        for (int r = 0; r < n; ++r) {
            for (int c = r + 1; c < n; ++c) {
                const double cur = std::fabs(a[r * n + c]);
                if (cur > max_off) {
                    max_off = cur;
                    p = r;
                    q = c;
                }
            }
        }
        if (max_off < 1e-12) {
            break;
        }

        const double app = a[p * n + p];
        const double aqq = a[q * n + q];
        const double apq = a[p * n + q];
        const double phi = 0.5 * std::atan2(2.0 * apq, aqq - app);
        const double c = std::cos(phi);
        const double s = std::sin(phi);

        for (int k = 0; k < n; ++k) {
            const double akp = a[k * n + p];
            const double akq = a[k * n + q];
            a[k * n + p] = c * akp - s * akq;
            a[k * n + q] = s * akp + c * akq;
        }
        for (int k = 0; k < n; ++k) {
            const double apk = a[p * n + k];
            const double aqk = a[q * n + k];
            a[p * n + k] = c * apk - s * aqk;
            a[q * n + k] = s * apk + c * aqk;
        }
        a[p * n + q] = 0.0;
        a[q * n + p] = 0.0;

        for (int k = 0; k < n; ++k) {
            const double vkp = v[k * n + p];
            const double vkq = v[k * n + q];
            v[k * n + p] = c * vkp - s * vkq;
            v[k * n + q] = s * vkp + c * vkq;
        }
    }

    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
        return a[lhs * n + lhs] > a[rhs * n + rhs];
    });

    eigvals.assign(n, 0.0f);
    eigvecs_row_major.assign(n * n, 0.0f);
    for (int dst = 0; dst < n; ++dst) {
        const int src = order[dst];
        eigvals[dst] = float(std::max(0.0, a[src * n + src]));
        for (int r = 0; r < n; ++r) {
            eigvecs_row_major[r * n + dst] = float(v[r * n + src]);
        }
    }
}

static float make_rotation_t_and_orthogonality(
        const std::vector<float> & rotation,
        std::vector<float> & rotation_t) {
    constexpr int n = LLAMA_KV_CALIB_GROUP;
    rotation_t.assign(n * n, 0.0f);
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c) {
            rotation_t[c * n + r] = rotation[r * n + c];
        }
    }

    double err2 = 0.0;
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c) {
            double dot = 0.0;
            for (int k = 0; k < n; ++k) {
                dot += (double) rotation[k * n + r] * (double) rotation[k * n + c];
            }
            const double target = r == c ? 1.0 : 0.0;
            const double diff = dot - target;
            err2 += diff * diff;
        }
    }
    return float(std::sqrt(err2));
}

static llama_kv_pca_group_result make_pca_group_result(
        const std::vector<double> & cov,
        uint64_t n_rows,
        int offset,
        int wht_group) {
    llama_kv_pca_group_result out;
    out.offset = offset;
    out.n_rows = n_rows;
    out.variances.assign(wht_group, 0.0f);
    out.rotation.assign(wht_group * wht_group, 0.0f);
    out.rotation_t.assign(wht_group * wht_group, 0.0f);

    if (n_rows == 0) {
        for (int i = 0; i < wht_group; ++i) {
            out.rotation[i * wht_group + i] = 1.0f;
            out.rotation_t[i * wht_group + i] = 1.0f;
        }
        return out;
    }

    std::vector<double> cov_avg(wht_group * wht_group);
    const double inv_n = 1.0 / (double) n_rows;
    for (int i = 0; i < wht_group * wht_group; ++i) {
        cov_avg[i] = cov[i] * inv_n;
    }

    jacobi_eigen_128(cov_avg, out.variances, out.rotation);
    out.orthogonality_l2 = make_rotation_t_and_orthogonality(out.rotation, out.rotation_t);

    double sum = 0.0;
    for (float v : out.variances) {
        sum += v;
    }
    out.variance_sum = float(sum);
    return out;
}

} // namespace

llama_kv_empvar_calibration::mode_t llama_kv_empvar_calibration::mode_from_string(const std::string & mode) {
    if (mode.empty() || mode == "wht_only_empvar") {
        return mode_t::WHT_ONLY_EMPVAR;
    }
    if (mode == "turbo3_pca") {
        return mode_t::TURBO3_PCA;
    }
    if (mode == "turbo4_pca") {
        return mode_t::TURBO4_PCA;
    }
    if (mode == "turbo4333_pca") {
        return mode_t::TURBO4333_PCA;
    }
    if (mode == "turbo4322_pca") {
        return mode_t::TURBO4322_PCA;
    }
    if (mode.size() > strlen("turbo_pca") &&
            mode.rfind("turbo", 0) == 0 &&
            mode.size() >= 4 &&
            mode.compare(mode.size() - 4, 4, "_pca") == 0) {
        return mode_t::TURBO3_PCA;
    }
    throw std::runtime_error("unknown KV calibration mode: " + mode);
}

const char * llama_kv_empvar_calibration::mode_to_string(mode_t mode) {
    switch (mode) {
        case mode_t::WHT_ONLY_EMPVAR: return "wht_only_empvar";
        case mode_t::TURBO3_PCA:      return "turbo3_pca";
        case mode_t::TURBO4_PCA:      return "turbo4_pca";
        case mode_t::TURBO4333_PCA:   return "turbo4333_pca";
        case mode_t::TURBO4322_PCA:   return "turbo4322_pca";
    }
    return "wht_only_empvar";
}

llama_kv_empvar_calibration::llama_kv_empvar_calibration(int wht_group)
        : llama_kv_empvar_calibration(wht_group, mode_t::WHT_ONLY_EMPVAR) {
}

llama_kv_empvar_calibration::llama_kv_empvar_calibration(int wht_group, mode_t mode)
        : wht_group_(wht_group), mode_(mode) {
    if (wht_group_ != 128) {
        throw std::runtime_error("llama_kv_empvar_calibration currently supports only 128-d WHT groups");
    }
}

void llama_kv_empvar_calibration::observe_row_impl(accum_t & accum, const float * row, int head_dim, int wht_group, mode_t mode) {
    if (row == nullptr) {
        throw std::runtime_error("llama_kv_empvar_calibration received null row");
    }
    if (head_dim <= 0) {
        throw std::runtime_error("llama_kv_empvar_calibration requires a positive head_dim");
    }

    const int padded_head_dim = pad_to_group(head_dim, wht_group);

    if (accum.head_dim == 0) {
        accum.head_dim = padded_head_dim;
        accum.sumsq.assign(padded_head_dim, 0.0);
        if (mode == mode_t::TURBO3_PCA || mode == mode_t::TURBO4_PCA || mode == mode_t::TURBO4333_PCA || mode == mode_t::TURBO4322_PCA) {
            const int n_groups = padded_head_dim / wht_group;
            accum.pca_cov.assign(n_groups * wht_group * wht_group, 0.0);
            accum.pca_group_rows.assign(n_groups, 0);
        }
    } else if (accum.head_dim != padded_head_dim) {
        throw std::runtime_error("llama_kv_empvar_calibration observed mismatched head dimensions");
    }

    float tmp[128];
    for (int offset = 0; offset < padded_head_dim; offset += wht_group) {
        const int group = offset / wht_group;
        const float norm = compute_padded_group_norm(row, offset, head_dim, wht_group);
        const bool has_signal = norm > LLAMA_KV_EMPVAR_EPS;
        if (has_signal) {
            const float inv_norm = 1.0f / norm;
            for (int i = 0; i < wht_group; ++i) {
                const int coord = offset + i;
                tmp[i] = coord < head_dim ? row[coord] * inv_norm : 0.0f;
            }
        } else {
            for (int i = 0; i < wht_group; ++i) {
                tmp[i] = 0.0f;
            }
        }

        if (mode == mode_t::WHT_ONLY_EMPVAR) {
            rotate_group_128(tmp);
        }

        for (int i = 0; i < wht_group; ++i) {
            const double v = tmp[i];
            accum.sumsq[offset + i] += v * v;
        }

        if ((mode == mode_t::TURBO3_PCA || mode == mode_t::TURBO4_PCA || mode == mode_t::TURBO4333_PCA || mode == mode_t::TURBO4322_PCA) && has_signal) {
            double * cov = accum.pca_cov.data() + group * wht_group * wht_group;
            for (int r = 0; r < wht_group; ++r) {
                const double vr = tmp[r];
                double * cov_row = cov + r * wht_group;
                for (int c = 0; c < wht_group; ++c) {
                    cov_row[c] += vr * (double) tmp[c];
                }
            }
            accum.pca_group_rows[group]++;
        }
    }

    accum.n_rows++;
}

void llama_kv_empvar_calibration::observe_key_row(const float * row, int head_dim) {
    observe_row_impl(key_, row, head_dim, wht_group_, mode_);
}

void llama_kv_empvar_calibration::observe_value_row(const float * row, int head_dim) {
    observe_row_impl(value_, row, head_dim, wht_group_, mode_);
}

void llama_kv_empvar_calibration::observe_key_rows(const std::vector<std::vector<float>> & rows) {
    for (const auto & row : rows) {
        observe_key_row(row.data(), (int) row.size());
    }
}

void llama_kv_empvar_calibration::observe_value_rows(const std::vector<std::vector<float>> & rows) {
    for (const auto & row : rows) {
        observe_value_row(row.data(), (int) row.size());
    }
}

llama_kv_empvar_side_result llama_kv_empvar_calibration::finalize_impl(const accum_t & accum, int wht_group, mode_t mode) {
    llama_kv_empvar_side_result out;
    out.head_dim = accum.head_dim;
    out.wht_group = wht_group;
    out.n_rows = accum.n_rows;

    if (accum.head_dim == 0) {
        return out;
    }
    if (accum.n_rows == 0) {
        out.variances.assign(accum.head_dim, 0.0f);
        out.chunks.resize(accum.head_dim / wht_group);
        return out;
    }

    out.variances.resize(accum.head_dim);
    out.chunks.reserve(accum.head_dim / wht_group);

    double mean = 0.0;
    for (int i = 0; i < accum.head_dim; ++i) {
        out.variances[i] = float(accum.sumsq[i] / double(accum.n_rows));
        mean += out.variances[i];
    }
    mean /= double(accum.head_dim);
    out.mean_variance = float(mean);

    double var = 0.0;
    double l2 = 0.0;
    const double uniform = 1.0 / double(wht_group);
    for (float v : out.variances) {
        const double dv = double(v) - mean;
        const double du = double(v) - uniform;
        var += dv * dv;
        l2 += du * du;
    }
    out.std_variance = float(std::sqrt(var / double(accum.head_dim)));
    out.l2_to_uniform = float(std::sqrt(l2));

    for (int offset = 0; offset < accum.head_dim; offset += wht_group) {
        out.chunks.push_back(make_chunk_stats(accum.sumsq, accum.n_rows, offset, wht_group));
    }

    if (mode == mode_t::TURBO3_PCA || mode == mode_t::TURBO4_PCA || mode == mode_t::TURBO4333_PCA || mode == mode_t::TURBO4322_PCA) {
        const int n_groups = accum.head_dim / wht_group;
        out.pca_groups.reserve(n_groups);
        for (int group = 0; group < n_groups; ++group) {
            const int offset = group * wht_group;
            const double * cov_src = accum.pca_cov.data() + group * wht_group * wht_group;
            std::vector<double> cov(cov_src, cov_src + wht_group * wht_group);
            out.pca_groups.push_back(make_pca_group_result(cov, accum.pca_group_rows[group], offset, wht_group));
        }
    }

    return out;
}

llama_kv_empvar_side_result llama_kv_empvar_calibration::finalize_keys() const {
    return finalize_impl(key_, wht_group_, mode_);
}

llama_kv_empvar_side_result llama_kv_empvar_calibration::finalize_values() const {
    return finalize_impl(value_, wht_group_, mode_);
}

int llama_kv_empvar_calibration::wht_group() const {
    return wht_group_;
}

llama_kv_empvar_calibration::mode_t llama_kv_empvar_calibration::mode() const {
    return mode_;
}

namespace {

static void write_float_array(std::ostream & out, const std::vector<float> & values, int indent) {
    const std::string pad(indent, ' ');
    out << "[";
    if (!values.empty()) {
        out << "\n";
        for (size_t i = 0; i < values.size(); ++i) {
            out << pad << "  " << std::setprecision(9) << values[i];
            if (i + 1 != values.size()) {
                out << ",";
            }
            out << "\n";
        }
        out << pad;
    }
    out << "]";
}

static void write_pca_groups(
        std::ostream & out,
        const std::vector<llama_kv_pca_group_result> & groups,
        int indent) {
    const std::string pad(indent, ' ');
    out << "[";
    if (!groups.empty()) {
        out << "\n";
        for (size_t i = 0; i < groups.size(); ++i) {
            const auto & group = groups[i];
            out << pad << "  {\n";
            out << pad << "    \"offset\": " << group.offset << ",\n";
            out << pad << "    \"n_rows\": " << group.n_rows << ",\n";
            out << pad << "    \"variance_sum\": " << std::setprecision(9) << group.variance_sum << ",\n";
            out << pad << "    \"orthogonality_l2\": " << std::setprecision(9) << group.orthogonality_l2 << ",\n";
            out << pad << "    \"variances\": ";
            write_float_array(out, group.variances, indent + 4);
            out << ",\n";
            out << pad << "    \"rotation\": ";
            write_float_array(out, group.rotation, indent + 4);
            out << ",\n";
            out << pad << "    \"rotation_t\": ";
            write_float_array(out, group.rotation_t, indent + 4);
            out << "\n";
            out << pad << "  }";
            if (i + 1 != groups.size()) {
                out << ",";
            }
            out << "\n";
        }
        out << pad;
    }
    out << "]";
}

static void write_side(
        std::ostream & out,
        const char * name,
        const llama_kv_empvar_side_result & side,
        int indent,
        llama_kv_empvar_calibration::mode_t mode) {
    const std::string pad(indent, ' ');
    out << pad << "\"" << name << "\": {\n";
    out << pad << "  \"head_dim\": " << side.head_dim << ",\n";
    out << pad << "  \"wht_group\": " << side.wht_group << ",\n";
    out << pad << "  \"n_rows\": " << side.n_rows << ",\n";
    out << pad << "  \"mean_variance\": " << std::setprecision(9) << side.mean_variance << ",\n";
    out << pad << "  \"std_variance\": " << std::setprecision(9) << side.std_variance << ",\n";
    out << pad << "  \"l2_to_uniform\": " << std::setprecision(9) << side.l2_to_uniform << ",\n";
    out << pad << "  \"variances\": ";
    write_float_array(out, side.variances, indent + 2);
    out << ",\n";
    if (mode == llama_kv_empvar_calibration::mode_t::TURBO3_PCA || mode == llama_kv_empvar_calibration::mode_t::TURBO4_PCA || mode == llama_kv_empvar_calibration::mode_t::TURBO4333_PCA || mode == llama_kv_empvar_calibration::mode_t::TURBO4322_PCA) {
        out << pad << "  \"groups\": ";
        write_pca_groups(out, side.pca_groups, indent + 2);
        out << ",\n";
    }
    out << pad << "  \"chunks\": [";
    if (!side.chunks.empty()) {
        out << "\n";
        for (size_t i = 0; i < side.chunks.size(); ++i) {
            const auto & chunk = side.chunks[i];
            out << pad << "    {\n";
            out << pad << "      \"chunk_index\": " << i << ",\n";
            out << pad << "      \"mean_variance\": " << std::setprecision(9) << chunk.mean_variance << ",\n";
            out << pad << "      \"std_variance\": " << std::setprecision(9) << chunk.std_variance << ",\n";
            out << pad << "      \"l2_to_uniform\": " << std::setprecision(9) << chunk.l2_to_uniform << ",\n";
            out << pad << "      \"variances\": ";
            write_float_array(out, chunk.variances, indent + 6);
            out << "\n";
            out << pad << "    }";
            if (i + 1 != side.chunks.size()) {
                out << ",";
            }
            out << "\n";
        }
        out << pad << "  ";
    }
    out << "]\n";
    out << pad << "}";
}

} // namespace

void llama_kv_empvar_write_json(
        const std::string & path,
        llama_kv_empvar_calibration::mode_t mode,
        const std::string & model_hash,
        const llama_kv_empvar_side_result & keys,
        const llama_kv_empvar_side_result & values) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open empvar calibration output: " + path);
    }

    out << "{\n";
    out << "  \"version\": " << (mode == llama_kv_empvar_calibration::mode_t::TURBO3_PCA || mode == llama_kv_empvar_calibration::mode_t::TURBO4_PCA || mode == llama_kv_empvar_calibration::mode_t::TURBO4333_PCA || mode == llama_kv_empvar_calibration::mode_t::TURBO4322_PCA ? 2 : 1) << ",\n";
    out << "  \"mode\": \"" << llama_kv_empvar_calibration::mode_to_string(mode) << "\",\n";
    out << "  \"model_hash\": \"" << model_hash << "\",\n";
    out << "  \"group_dim\": " << keys.wht_group << ",\n";
    out << "  \"head_dim\": " << std::max(keys.head_dim, values.head_dim) << ",\n";
    write_side(out, "keys", keys, 2, mode);
    out << ",\n";
    write_side(out, "values", values, 2, mode);
    out << "\n}\n";
}

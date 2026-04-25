#include "wht-empvar-utils.h"

#include "../src/llama-kv-empvar-calibration.h"
#include "../ggml/src/ggml-quants.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace wht_empvar;

static void require(bool cond, const char * msg) {
    if (!cond) {
        std::fprintf(stderr, "test-wht-empvar: %s\n", msg);
        std::exit(1);
    }
}

static float max_abs_diff_slice(const std::vector<float> & a, size_t a0, const std::vector<float> & b, size_t b0, size_t n) {
    float out = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        out = std::max(out, std::fabs(a[a0 + i] - b[b0 + i]));
    }
    return out;
}

int main() {
    {
        auto data = make_gaussian_dataset(8, 64, 7);
        auto x = normalize(data.front());
        auto rot = turbo_rotate_forward(x);
        auto inv = turbo_rotate_inverse(rot);
        double max_err = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            max_err = std::max(max_err, std::fabs(double(x[i]) - double(inv[i])));
        }
        require(max_err < 1e-4, "WHT roundtrip mismatch");
    }

    {
        auto calib = make_gaussian_dataset(512, 64, 11);
        const auto stats_a = empirical_variances(calib);
        const auto stats_b = empirical_variances(calib);
        require(stats_a.variances.size() == 64, "wrong variance size");
        for (float v : stats_a.variances) {
            require(std::isfinite(v) && v >= 0.0f, "invalid empirical variance");
        }
        for (size_t i = 0; i < stats_a.variances.size(); ++i) {
            require(std::fabs(stats_a.variances[i] - stats_b.variances[i]) < 1e-8f, "non-deterministic calibration");
        }
    }

    {
        auto calib = make_gaussian_dataset(4096, 64, 19);
        const auto stats = empirical_variances(calib);
        require(std::fabs(stats.mean_variance - (1.0f / 64.0f)) < 0.01f, "mean variance not close to 1/d");
    }

    {
        auto full = make_gaussian_dataset(4096, 64, 23);
        std::vector<std::vector<float>> small(full.begin(), full.begin() + 64);
        std::vector<std::vector<float>> medium(full.begin(), full.begin() + 1024);
        const auto ref = empirical_variances(full);
        const auto s = empirical_variances(small);
        const auto m = empirical_variances(medium);
        require(m.l2_to_uniform <= s.l2_to_uniform + 0.1f, "larger calibration should not be less stable");
        require(std::fabs(m.mean_variance - ref.mean_variance) <= std::fabs(s.mean_variance - ref.mean_variance) + 0.01f,
                "larger calibration should move toward stable regime");
    }

    {
        auto calib = make_gaussian_dataset(512, 64, 31);
        auto eval = make_gaussian_dataset(256, 64, 37);
        const auto stats = empirical_variances(calib);
        const auto emp_books = build_empirical_codebooks(stats.variances, 2);
        const auto fixed_books = build_fixed_codebooks(64, 2);
        const auto emp = evaluate_reconstruction(eval, emp_books, false);
        const auto fixed = evaluate_reconstruction(eval, fixed_books, true);
        require(std::isfinite(emp.mean_cosine) && std::isfinite(fixed.mean_cosine), "invalid reconstruction metric");
    }

    {
        auto chunk_a = make_gaussian_dataset(32, 64, 41);
        auto chunk_b = make_gaussian_dataset(32, 64, 43);
        std::vector<std::vector<std::vector<float>>> k_chunks = {chunk_a, chunk_b};
        std::vector<std::vector<std::vector<float>>> v_chunks = {chunk_b, chunk_a};
        const auto stats_k = empirical_variances(make_gaussian_dataset(256, 64, 47));
        const auto stats_v = empirical_variances(make_gaussian_dataset(256, 64, 53));
        const auto emp_k = build_empirical_codebooks(stats_k.variances, 2);
        const auto emp_v = build_empirical_codebooks(stats_v.variances, 2);
        const auto fix_k = build_fixed_codebooks(64, 2);
        const auto fix_v = build_fixed_codebooks(64, 2);
        const auto emp = evaluate_attention_proxy_qeqk(k_chunks, v_chunks, emp_k, emp_v, false);
        const auto fix = evaluate_attention_proxy_qeqk(k_chunks, v_chunks, fix_k, fix_v, true);
        require(std::isfinite(emp.mean_output_cosine) && std::isfinite(fix.mean_output_cosine), "invalid attention cosine");
        require(std::isfinite(emp.mean_prob_kl) && emp.mean_prob_kl >= 0.0, "invalid attention KL");
    }

    {
        std::vector<float> x(128);
        for (int i = 0; i < 128; ++i) {
            x[i] = std::sin(float(i) * 0.1f) + 0.05f * float(i % 3);
        }

        block_turbo3_0 baseline[128 / QK_TURBO3];
        block_turbo3_0 baseline_repeat[128 / QK_TURBO3];
        block_turbo3_0 empirical[128 / QK_TURBO3];

        unsetenv("TURBO_CPU_EXPERIMENTAL_MODE");
        unsetenv("TURBO_CPU_EMPVAR_K");
        unsetenv("TURBO_CPU_EMPVAR_V");
        ggml_turbo_empvar_reload_from_env();
        ggml_turbo_quant_set_context(128, 1);
        quantize_row_turbo3_0_ref(x.data(), baseline, 128);
        ggml_turbo_quant_set_context(128, 1);
        quantize_row_turbo3_0_ref(x.data(), baseline_repeat, 128);
        require(std::memcmp(baseline, baseline_repeat, sizeof(baseline)) == 0, "baseline turbo3 quantization changed");

        std::string vars;
        for (int i = 0; i < 128; ++i) {
            const float v = (i < 64) ? 0.5f / 128.0f : 1.5f / 128.0f;
            vars += std::to_string(v);
            if (i + 1 != 128) {
                vars += ",";
            }
        }
        setenv("TURBO_CPU_EXPERIMENTAL_MODE", "wht_only_empvar", 1);
        setenv("TURBO_CPU_EMPVAR_K", vars.c_str(), 1);
        ggml_turbo_empvar_reload_from_env();
        ggml_turbo_quant_set_context(128, 1);
        quantize_row_turbo3_empvar_k_ref(x.data(), empirical, 128);
        require(std::memcmp(baseline, empirical, sizeof(baseline)) != 0, "empirical mode did not affect turbo3 quantization");
    }

    {
        auto calib256 = make_gaussian_dataset(512, 256, 59);
        const auto chunked = empirical_variances_chunked(calib256, 128);
        llama_kv_empvar_calibration calibration(128);
        calibration.observe_key_rows(calib256);
        const auto collected = calibration.finalize_keys();
        require(chunked.variances.size() == 256, "expected two 128-d chunk profiles");
        require(chunked.chunks.size() == 2, "expected two chunk stats for 256-d rows");
        require(collected.variances.size() == 256, "collector produced wrong 256-d profile size");
        require(collected.chunks.size() == 2, "collector produced wrong number of 128-d groups");
        require(collected.n_rows == calib256.size(), "collector row count mismatch");
        for (size_t i = 0; i < chunked.variances.size(); ++i) {
            require(std::fabs(chunked.variances[i] - collected.variances[i]) < 1e-6f,
                    "collector did not match reference chunked calibration");
        }
        require(std::fabs(chunked.mean_variance - collected.mean_variance) < 1e-6f,
                "collector mean variance mismatch");
        require(std::fabs(chunked.std_variance - collected.std_variance) < 1e-6f,
                "collector std variance mismatch");
        require(std::fabs(chunked.l2_to_uniform - collected.l2_to_uniform) < 1e-6f,
                "collector l2_to_uniform mismatch");

        std::vector<float> x(256);
        for (int i = 0; i < 128; ++i) {
            x[i] = std::sin(float(i) * 0.11f);
            x[i + 128] = x[i];
        }

        block_turbo3_0 repeated[256 / QK_TURBO3];
        block_turbo3_0 split[256 / QK_TURBO3];

        std::string repeated_vars;
        std::string split_vars;
        for (int i = 0; i < 128; ++i) {
            const float lo = 0.6f / 128.0f;
            const float hi = 1.4f / 128.0f;
            repeated_vars += std::to_string(lo);
            split_vars += std::to_string(lo);
            split_vars += ",";
            split_vars += std::to_string(hi);
            if (i + 1 != 128) {
                repeated_vars += ",";
                split_vars += ",";
            }
        }

        setenv("TURBO_CPU_EXPERIMENTAL_MODE", "wht_only_empvar", 1);
        setenv("TURBO_CPU_EMPVAR_K", repeated_vars.c_str(), 1);
        ggml_turbo_empvar_reload_from_env();
        ggml_turbo_quant_set_context(128, 1);
        quantize_row_turbo3_empvar_k_ref(x.data(), repeated, 256);

        setenv("TURBO_CPU_EMPVAR_K", split_vars.c_str(), 1);
        ggml_turbo_empvar_reload_from_env();
        ggml_turbo_quant_set_context(128, 1);
        quantize_row_turbo3_empvar_k_ref(x.data(), split, 256);

        std::vector<float> repeated_deq(256);
        std::vector<float> split_deq(256);
        setenv("TURBO_CPU_EMPVAR_K", repeated_vars.c_str(), 1);
        ggml_turbo_empvar_reload_from_env();
        ggml_turbo_quant_set_context(128, 1);
        dequantize_row_turbo3_empvar_k(repeated, repeated_deq.data(), 256);
        setenv("TURBO_CPU_EMPVAR_K", split_vars.c_str(), 1);
        ggml_turbo_empvar_reload_from_env();
        ggml_turbo_quant_set_context(128, 1);
        dequantize_row_turbo3_empvar_k(split, split_deq.data(), 256);

        require(std::memcmp(repeated, split, sizeof(split)) != 0, "split chunk profiles did not affect 256-d quantization");
        require(std::memcmp(repeated, repeated + (128 / QK_TURBO3), sizeof(block_turbo3_0) * (128 / QK_TURBO3)) == 0,
                "repeated profile should quantize identical 128-d halves identically");
        require(max_abs_diff_slice(repeated_deq, 0, repeated_deq, 128, 128) < 1e-6f,
                "repeated profile should reconstruct identical 128-d halves identically");
        require(max_abs_diff_slice(repeated_deq, 0, split_deq, 0, 256) > 1e-5f,
                "split profile should change 256-d reconstruction");
    }

    {
        auto calib256 = make_gaussian_dataset(512, 256, 67);
        llama_kv_empvar_calibration calibration(128, llama_kv_empvar_calibration::mode_t::TURBO3_PCA);
        calibration.observe_key_rows(calib256);
        const auto collected = calibration.finalize_keys();
        require(collected.pca_groups.size() == 2, "PCA collector should emit one group per 128-d chunk");
        for (const auto & group : collected.pca_groups) {
            require(group.variances.size() == 128, "PCA group has wrong variance size");
            require(group.rotation.size() == 128 * 128, "PCA group has wrong rotation size");
            require(group.rotation_t.size() == 128 * 128, "PCA group has wrong rotation_t size");
            require(group.n_rows == calib256.size(), "PCA group row count mismatch");
            require(std::fabs(group.variance_sum - 1.0f) < 1e-4f, "PCA eigenvalue sum should be close to 1");
            require(group.orthogonality_l2 < 1e-3f, "PCA rotation is not orthogonal enough");
            for (size_t i = 1; i < group.variances.size(); ++i) {
                require(group.variances[i - 1] + 1e-7f >= group.variances[i], "PCA variances are not sorted");
            }
            for (float v : group.variances) {
                require(std::isfinite(v) && v >= 0.0f, "invalid PCA variance");
            }
        }
    }

    std::printf("test-wht-empvar: ok\n");
    return 0;
}

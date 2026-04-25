#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct llama_kv_empvar_chunk_stats {
    std::vector<float> variances;
    float mean_variance = 0.0f;
    float std_variance = 0.0f;
    float l2_to_uniform = 0.0f;
};

struct llama_kv_pca_group_result {
    int offset = 0;
    uint64_t n_rows = 0;

    std::vector<float> variances;
    std::vector<float> rotation;   // U, row-major: rows are coordinates, columns are eigenvectors
    std::vector<float> rotation_t; // U^T, row-major

    float variance_sum = 0.0f;
    float orthogonality_l2 = 0.0f;
};

struct llama_kv_empvar_side_result {
    int head_dim = 0;
    int wht_group = 128;
    uint64_t n_rows = 0;

    std::vector<float> variances;
    std::vector<llama_kv_empvar_chunk_stats> chunks;
    std::vector<llama_kv_pca_group_result> pca_groups;

    float mean_variance = 0.0f;
    float std_variance = 0.0f;
    float l2_to_uniform = 0.0f;
};

class llama_kv_empvar_calibration {
public:
    enum class mode_t {
        WHT_ONLY_EMPVAR,
        TURBO3_PCA,
        TURBO4_PCA,
        TURBO4333_PCA,
        TURBO4322_PCA,
    };

    static mode_t mode_from_string(const std::string & mode);
    static const char * mode_to_string(mode_t mode);

    explicit llama_kv_empvar_calibration(int wht_group = 128);
    llama_kv_empvar_calibration(int wht_group, mode_t mode);

    void observe_key_row(const float * row, int head_dim);
    void observe_value_row(const float * row, int head_dim);

    void observe_key_rows(const std::vector<std::vector<float>> & rows);
    void observe_value_rows(const std::vector<std::vector<float>> & rows);

    llama_kv_empvar_side_result finalize_keys() const;
    llama_kv_empvar_side_result finalize_values() const;

    int wht_group() const;
    mode_t mode() const;

private:
    struct accum_t {
        int head_dim = 0;
        uint64_t n_rows = 0;
        std::vector<double> sumsq;
        std::vector<double> pca_cov;
        std::vector<uint64_t> pca_group_rows;
    };

    int wht_group_ = 128;
    mode_t mode_ = mode_t::WHT_ONLY_EMPVAR;
    accum_t key_;
    accum_t value_;

    static void observe_row_impl(accum_t & accum, const float * row, int head_dim, int wht_group, mode_t mode);
    static llama_kv_empvar_side_result finalize_impl(const accum_t & accum, int wht_group, mode_t mode);
};

void llama_kv_empvar_write_json(
        const std::string & path,
        llama_kv_empvar_calibration::mode_t mode,
        const std::string & model_hash,
        const llama_kv_empvar_side_result & keys,
        const llama_kv_empvar_side_result & values);

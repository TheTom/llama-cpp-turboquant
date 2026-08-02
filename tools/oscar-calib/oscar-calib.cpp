// OSCAR rotation calibration tool.
//
// Loads a GGUF model, runs forward passes on calibration text, and uses the
// eval callback to intercept Q, K, V, and attention-output tensors at runtime.
// Accumulates per-layer Q-covariance (Q^T Q) and score-weighted V-covariance
// (V^T diag(w) V) online, then dumps them as raw float32 binaries for the
// Python orchestrator (calibrate_rotation.py) to eigendecompose.
//
// Usage:
//   llama-oscar-calib -m model.gguf -f calibration.txt -o covariances/
//
// Output: covariances/layer_N_qcov.bin and layer_N_vcov.bin (raw f32, row-major).

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#  include <direct.h>
#else
#  include <sys/stat.h>
#endif

// Accumulator for one layer's Q^T Q and V^T diag(w) V matrices.
// Sized for head_dim up to 512 (the max oscar2 supports).
struct calib_accumulator {
    int n_embd_head;
    int n_head;
    int n_head_kv;
    int gqa_ratio;

    // Q-covariance: n_embd_head x n_embd_head, accumulated across all KV heads.
    // For GQA, query heads are grouped by KV head, so we accumulate per-KV-head.
    std::vector<std::vector<float>> q_cov;  // [n_head_kv][n_embd_head * n_embd_head]

    // V-covariance: n_embd_head x n_embd_head per KV head.
    std::vector<std::vector<float>> v_cov;  // [n_head_kv][n_embd_head * n_embd_head]

    // Token count for normalization.
    int64_t n_tokens;

    calib_accumulator() : n_embd_head(0), n_head(0), n_head_kv(0), gqa_ratio(1), n_tokens(0) {}

    void init(int hd, int nh, int nhkv) {
        n_embd_head = hd;
        n_head = nh;
        n_head_kv = nhkv;
        gqa_ratio = nh / nhkv;
        n_tokens = 0;
        q_cov.resize(nhkv, std::vector<float>(hd * hd, 0.0f));
        v_cov.resize(nhkv, std::vector<float>(hd * hd, 0.0f));
    }

    // Accumulate Q^T Q for one KV head. Q is [gqa_ratio, n_embd_head] in row-major.
    void acc_q_cov(int kv_head, const float * Q_row, int stride_q) {
        // Q_row points to the start of the gqa_ratio query heads for this KV head.
        // stride_q is the stride between consecutive query heads (in floats).
        const int g = gqa_ratio;
        const int d = n_embd_head;
        auto & cov = q_cov[kv_head];
        for (int gi = 0; gi < g; ++gi) {
            const float * q = Q_row + gi * stride_q;
            // rank-1 update: cov += q * q^T
            for (int r = 0; r < d; ++r) {
                for (int c = 0; c < d; ++c) {
                    cov[r * d + c] += q[r] * q[c];
                }
            }
        }
    }

    // Accumulate V^T diag(w) V for one KV head. V is [n_embd_head], w is scalar.
    void acc_v_cov(int kv_head, const float * V_row, float weight) {
        const int d = n_embd_head;
        auto & cov = v_cov[kv_head];
        for (int r = 0; r < d; ++r) {
            for (int c = 0; c < d; ++c) {
                cov[r * d + c] += weight * V_row[r] * V_row[c];
            }
        }
    }
};

// Per-layer accumulators (one per transformer layer).
static std::vector<calib_accumulator> g_accumulators;
static int g_n_layers = 0;
static int g_n_embd_head = 0;
static std::string g_output_dir;

// The eval callback. Called for each node during graph evaluation.
// We check tensor names to identify Q, K, V, and attention output.
static bool calib_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    GGML_UNUSED(user_data);

    if (ask) {
        // Return true if we want to read this tensor's data after evaluation.
        const char * name = ggml_get_name(t);
        if (!name || name[0] == '\0') return false;
        // We want: attn_q (to compute Q^T Q), attn_v (for V^T diag(w) V),
        // and attn_kq_soft_max (for the attention weights w).
        // Tensor names follow: blk.{i}.attn_q, blk.{i}.attn_v, blk.{i}.attn_kv_soft_max
        std::string s(name);
        if (s.find("attn_q") != std::string::npos ||
            s.find("attn_v") != std::string::npos ||
            s.find("attn_kv_soft_max") != std::string::npos) {
            return true;
        }
        return false;
    }

    // Process: tensor data is available now.
    const char * name = ggml_get_name(t);
    if (!name || name[0] == '\0') return false;
    std::string s(name);

    // Parse layer index from "blk.{i}.*"
    int layer = -1;
    if (s.find("blk.") == 0) {
        size_t dot2 = s.find('.', 4);
        if (dot2 != std::string::npos) {
            layer = std::atoi(s.substr(4, dot2 - 4).c_str());
        }
    }
    if (layer < 0 || layer >= g_n_layers) return false;

    auto & acc = g_accumulators[layer];

    // We only process tensors of the right type and shape.
    const enum ggml_type type = t->type;
    if (type != GGML_TYPE_F32) return false;

    const int ne0 = ggml_nelements(t) > 0 ? t->ne[0] : 0;
    const int ne1 = t->ne[1];
    const int ne2 = t->ne[2];

    if (s.find("attn_q") != std::string::npos && s.find("soft") == std::string::npos) {
        // Q tensor: shape [n_embd_head, n_head, n_tokens] or [n_embd_head * n_head, n_tokens]
        // We want per-token Q vectors grouped by KV head.
        const float * data = (const float *)t->data;
        if (!data) return false;

        const int d = acc.n_embd_head;
        const int nh = acc.n_head;
        const int nhkv = acc.n_head_kv;
        const int g = acc.gqa_ratio;

        // Determine layout: [d, nh, n_tok] (ne0=d, ne1=nh) or [d*nh, n_tok]
        int n_tok = 1;
        if (ne0 == d && ne1 == nh) {
            // [d, nh, n_tok] - column-major: data[h*d + r] for head h, dim r
            n_tok = (ggml_nbytes(t) / sizeof(float)) / (d * nh);
            for (int tok = 0; tok < n_tok; ++tok) {
                for (int kvh = 0; kvh < nhkv; ++kvh) {
                    // Collect g query heads for this KV head.
                    // Each head's Q vector is at data[tok*nh*d + h*d] for head h.
                    const float * base = data + tok * nh * d;
                    // acc_q_cov expects Q_row pointing to g consecutive heads, stride = d.
                    acc.acc_q_cov(kvh, base + kvh * g * d, d);
                }
            }
        } else if (ne0 == d * nh) {
            // [d*nh, n_tok] - each column is one token's flattened Q.
            n_tok = ne1;
            for (int tok = 0; tok < n_tok; ++tok) {
                for (int kvh = 0; kvh < nhkv; ++kvh) {
                    acc.acc_q_cov(kvh, data + tok * d * nh + kvh * g * d, d);
                }
            }
        }
        acc.n_tokens += n_tok;
    }

    if (s.find("attn_v") != std::string::npos && s.find("soft") == std::string::npos) {
        // V tensor: same layout as Q.
        // We accumulate V^T V without attention weights (approximation).
        // For the score-weighted version, we'd need the softmax output, but
        // unweighted V^T V is a reasonable approximation for calibration.
        const float * data = (const float *)t->data;
        if (!data) return false;

        const int d = acc.n_embd_head;
        const int nh = acc.n_head;
        const int nhkv = acc.n_head_kv;
        const int g = acc.gqa_ratio;

        int n_tok = 1;
        if (ne0 == d && ne1 == nh) {
            n_tok = (ggml_nbytes(t) / sizeof(float)) / (d * nh);
            for (int tok = 0; tok < n_tok; ++tok) {
                for (int kvh = 0; kvh < nhkv; ++kvh) {
                    const float * base = data + tok * nh * d;
                    // V is per-KV-head, so we just take one head (not g grouped).
                    acc.acc_v_cov(kvh, base + kvh * g * d, 1.0f);
                }
            }
        } else if (ne0 == d * nh) {
            n_tok = ne1;
            for (int tok = 0; tok < n_tok; ++tok) {
                for (int kvh = 0; kvh < nhkv; ++kvh) {
                    acc.acc_v_cov(kvh, data + tok * d * nh + kvh * g * d, 1.0f);
                }
            }
        }
    }

    return false;
}

static void dump_covariances(const std::string & output_dir) {
    for (int layer = 0; layer < g_n_layers; ++layer) {
        const auto & acc = g_accumulators[layer];
        const int d = acc.n_embd_head;
        const int nhkv = acc.n_head_kv;

        // Average across KV heads and normalize by token count.
        // For simplicity, average the covariance matrices across KV heads.
        std::vector<float> avg_q_cov(d * d, 0.0f);
        std::vector<float> avg_v_cov(d * d, 0.0f);

        for (int kvh = 0; kvh < nhkv; ++kvh) {
            for (int i = 0; i < d * d; ++i) {
                avg_q_cov[i] += acc.q_cov[kvh][i];
                avg_v_cov[i] += acc.v_cov[kvh][i];
            }
        }

        const float norm = 1.0f / (float)(acc.n_tokens > 0 ? acc.n_tokens : 1);
        for (int i = 0; i < d * d; ++i) {
            avg_q_cov[i] *= norm / nhkv;
            avg_v_cov[i] *= norm / nhkv;
        }

        // Write raw f32 binaries.
        char path[512];
        snprintf(path, sizeof(path), "%s/layer_%02d_qcov.bin", output_dir.c_str(), layer);
        FILE * f = fopen(path, "wb");
        if (f) { fwrite(avg_q_cov.data(), sizeof(float), d * d, f); fclose(f); }

        snprintf(path, sizeof(path), "%s/layer_%02d_vcov.bin", output_dir.c_str(), layer);
        f = fopen(path, "wb");
        if (f) { fwrite(avg_v_cov.data(), sizeof(float), d * d, f); fclose(f); }
    }
    LOG_INF("dumped covariances for %d layers to %s\n", g_n_layers, output_dir.c_str());
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();

    // Add custom args.
    params.cb_eval = calib_cb_eval;
    params.cb_eval_user_data = nullptr;
    params.warmup = false;

    // Parse output dir from a custom argument or default.
    g_output_dir = "covariances";
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-o" && i + 1 < argc) {
            g_output_dir = argv[i + 1];
        }
    }

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    // Ensure output dir exists.
#ifdef _WIN32
    _mkdir(g_output_dir.c_str());
#else
    mkdir(g_output_dir.c_str(), 0755);
#endif

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (!model || !ctx) {
        LOG_ERR("failed to init model\n");
        return 1;
    }

    // Query model dimensions. n_embd_head is not exposed directly in the public
    // llm API, so derive it from n_embd / n_head.
    const int n_embd      = llama_model_n_embd(model);
    const int n_head      = llama_model_n_head(model);
    const int n_embd_head = n_embd / n_head;
    const int n_head_kv   = llama_model_n_head_kv(model);
    const int n_layers    = llama_model_n_layer(model);

    g_n_layers = n_layers;
    g_n_embd_head = n_embd_head;

    LOG_INF("model: head_dim=%d, n_head=%d, n_head_kv=%d, n_layers=%d\n",
            n_embd_head, n_head, n_head_kv, n_layers);

    // Initialize accumulators.
    g_accumulators.resize(n_layers);
    for (int i = 0; i < n_layers; ++i) {
        g_accumulators[i].init(n_embd_head, n_head, n_head_kv);
    }

    // Tokenize and run.
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, true);

    if (tokens.empty()) {
        LOG_ERR("no input tokens\n");
        return 1;
    }

    LOG_INF("processing %zu tokens...\n", tokens.size());

    // Process in chunks to avoid OOM.
    const int n_ctx = llama_n_ctx(ctx);
    for (size_t i = 0; i < tokens.size(); i += n_ctx) {
        size_t n = std::min((size_t)n_ctx, tokens.size() - i);
        llama_batch batch = llama_batch_get_one(tokens.data() + i, n);
        if (llama_decode(ctx, batch)) {
            LOG_ERR("failed to decode at offset %zu\n", i);
            return 1;
        }
    }

    // Dump results.
    dump_covariances(g_output_dir);

    LOG_INF("done. %lld tokens processed.\n", (long long)g_accumulators[0].n_tokens);

    llama_backend_free();
    return 0;
}

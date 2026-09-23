// TriAttention calibration generator (native GGUF path)
//
// Produces a version-3 .triattention calibration file: the same base layout
// documented in src/llama-triattention.h (magic/header/per-head rotary
// stats), extended with per-head "content" stats for the non-rotary tail on
// partial-RoPE models (e.g. Qwen3.8, where rotary_dim=64 < model_head_dim=256).
//
// Unlike docs/TRIATTENTION.md's documented Python/HF path, this collects Q
// statistics by hooking ggml graph evaluation (the same cb_eval mechanism
// tools/imatrix and the experiment/triattention-integration branch use)
// during a forward pass of the already-loaded GGUF model, so it needs no
// separate fp16/bf16 HF checkpoint.
//
// Pre-RoPE Q tensors are identified by name, matching how llama-graph.cpp
// tags them: "Qcur-N" (2D, standard architectures) or "Qcur_normed-N" (3D,
// fused-QKV / hybrid architectures with a Q-norm before RoPE).

#include "llama-triattention.h"

#include "common.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

struct head_key {
    int32_t layer;
    int32_t head;
    bool operator<(const head_key & o) const {
        return layer != o.layer ? layer < o.layer : head < o.head;
    }
};

struct head_accum {
    std::vector<double> sum_real;   // [freq_count]
    std::vector<double> sum_imag;   // [freq_count]
    std::vector<double> sum_abs;    // [freq_count]
    std::vector<double> sum_content;     // [content_dim]
    std::vector<double> sum_content_abs; // [content_dim]
    int64_t n = 0;
};

struct calib_collector {
    uint32_t head_dim   = 0;  // full model head width
    uint32_t rotary_dim  = 0;
    uint32_t freq_count = 0;
    uint32_t content_dim = 0;
    uint32_t n_heads    = 0;

    std::map<head_key, head_accum> accum;
    std::vector<float> staging;  // reused host staging buffer for non-host tensors
    int64_t total_q_calls = 0;
};

static bool triattention_calib_cb(ggml_tensor * t, bool ask, void * user_data) {
    auto * self = (calib_collector *)user_data;

    const char * name = ggml_get_name(t);
    if (!name) return false;

    int layer_idx = -1;
    bool path_b = false;
    if (strncmp(name, "Qcur_normed-", 12) == 0) {
        layer_idx = atoi(name + 12);
        path_b = true;
    } else if (strncmp(name, "Qcur-", 5) == 0) {
        layer_idx = atoi(name + 5);
        path_b = false;
    } else {
        return false;
    }

    const int64_t hd = (int64_t)self->head_dim;
    const int64_t nh = (int64_t)self->n_heads;

    if (path_b) {
        if (t->ne[0] != hd || t->ne[1] != nh) return false;
        if (t->op == GGML_OP_ROPE) return false;
    } else {
        if (t->ne[0] != hd * nh) return false;
        if (t->op == GGML_OP_ROPE) return false;
    }

    if (ask) return true;

    // Tensors aren't always host-dereferenceable even when computed on the
    // CPU backend (buffer allocation strategy varies) -- copy to a host
    // staging buffer first, same as tools/imatrix does for its src1 tensors.
    const size_t nbytes = ggml_nbytes(t);
    std::vector<float> & staging = self->staging;
    const float * data;
    if (ggml_backend_buffer_is_host(t->buffer)) {
        data = (const float *)t->data;
    } else {
        staging.resize(nbytes / sizeof(float));
        ggml_backend_tensor_get(t, staging.data(), 0, nbytes);
        data = staging.data();
    }
    if (!data) return true;

    int32_t n_tokens;
    size_t head_stride_elems;
    if (path_b) {
        n_tokens = (int32_t)t->ne[2];
        head_stride_elems = t->nb[1] / sizeof(float);
    } else {
        n_tokens = (int32_t)t->ne[1];
        head_stride_elems = (size_t)self->head_dim;
    }
    const size_t token_stride = path_b
        ? head_stride_elems * (size_t)self->n_heads
        : (size_t)self->head_dim * (size_t)self->n_heads;

    const uint32_t fc = self->freq_count;
    const uint32_t rd = self->rotary_dim;
    const uint32_t cd = self->content_dim;

    for (int32_t tok = 0; tok < n_tokens; tok++) {
        for (uint32_t h = 0; h < self->n_heads; h++) {
            const float * q = data + (size_t)tok * token_stride + (size_t)h * head_stride_elems;

            head_key key{layer_idx, (int32_t)h};
            auto & acc = self->accum[key];
            if (acc.sum_real.empty()) {
                acc.sum_real.assign(fc, 0.0);
                acc.sum_imag.assign(fc, 0.0);
                acc.sum_abs.assign(fc, 0.0);
                acc.sum_content.assign(cd, 0.0);
                acc.sum_content_abs.assign(cd, 0.0);
            }

            for (uint32_t f = 0; f < fc; f++) {
                const float qr = q[f];
                const float qi = q[fc + f];
                acc.sum_real[f] += qr;
                acc.sum_imag[f] += qi;
                acc.sum_abs[f]  += sqrtf(qr * qr + qi * qi);
            }
            for (uint32_t d = 0; d < cd; d++) {
                const float qv = q[rd + d];
                acc.sum_content[d]     += qv;
                acc.sum_content_abs[d] += fabsf(qv);
            }
            acc.n++;
        }
    }

    self->total_q_calls += n_tokens;
    return true;
}

static void print_usage(const char * prog) {
    fprintf(stderr,
        "usage: %s -m MODEL.gguf -o OUTPUT.triattention --rotary-dim N --head-dim N --rope-theta F\n"
        "          [--rope-style 0|1] [--n-tokens N] [--text FILE] [-ngl N] [--ctx-size N]\n\n"
        "  -m, --model PATH       GGUF model to calibrate against (required)\n"
        "  -o, --output PATH      output .triattention file (required)\n"
        "  --rotary-dim N         RoPE-rotated width of the K/Q head (required)\n"
        "  --head-dim N           full model K/Q head width (required; == rotary-dim for full RoPE)\n"
        "  --rope-theta F         RoPE base theta (required)\n"
        "  --rope-style N         0=half (default), 1=interleaved\n"
        "  --n-tokens N           calibration tokens to process (default 4096)\n"
        "  --text FILE            calibration text file (default: built-in filler text)\n"
        "  -ngl N                 GPU layers to offload (default 99)\n"
        "  --ctx-size N           context size for the calibration pass (default n-tokens + 512)\n",
        prog);
}

static const char * k_default_filler =
    "The history of long-context language models is a story of incremental "
    "engineering victories against quadratic attention cost. Researchers "
    "explored sparse attention patterns, linear attention approximations, "
    "recurrent state compression, and retrieval augmentation as ways to "
    "sidestep the problem without abandoning the transformer block entirely. "
    "Key-value cache compression and eviction techniques trade a small "
    "amount of quality for a large amount of memory headroom, letting a "
    "single GPU serve longer contexts or more concurrent sequences. ";

int main(int argc, char ** argv) {
    std::string model_path, output_path, text_path;
    uint32_t rotary_dim = 0, head_dim = 0, rope_style = 0;
    double rope_theta = 0.0;
    int32_t n_tokens_target = 4096;
    int32_t n_gpu_layers = 99;
    int32_t ctx_size = 0;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        auto next = [&](const char * flag) -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", flag); exit(1); }
            return argv[++i];
        };
        if (arg == "-m" || arg == "--model") model_path = next(arg.c_str());
        else if (arg == "-o" || arg == "--output") output_path = next(arg.c_str());
        else if (arg == "--rotary-dim") rotary_dim = (uint32_t)std::stoul(next(arg.c_str()));
        else if (arg == "--head-dim") head_dim = (uint32_t)std::stoul(next(arg.c_str()));
        else if (arg == "--rope-theta") rope_theta = std::stod(next(arg.c_str()));
        else if (arg == "--rope-style") rope_style = (uint32_t)std::stoul(next(arg.c_str()));
        else if (arg == "--n-tokens") n_tokens_target = std::stoi(next(arg.c_str()));
        else if (arg == "--text") text_path = next(arg.c_str());
        else if (arg == "-ngl" || arg == "--n-gpu-layers") n_gpu_layers = std::stoi(next(arg.c_str()));
        else if (arg == "--ctx-size") ctx_size = std::stoi(next(arg.c_str()));
        else if (arg == "-h" || arg == "--help") { print_usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown arg: %s\n", arg.c_str()); print_usage(argv[0]); return 1; }
    }

    if (model_path.empty() || output_path.empty() || rotary_dim == 0 || head_dim == 0 || rope_theta == 0.0) {
        print_usage(argv[0]);
        return 1;
    }
    if (rotary_dim > head_dim || rotary_dim % 2 != 0) {
        fprintf(stderr, "invalid rotary-dim=%u for head-dim=%u\n", rotary_dim, head_dim);
        return 1;
    }
    if (ctx_size == 0) ctx_size = n_tokens_target + 512;

    std::string text;
    if (!text_path.empty()) {
        FILE * f = fopen(text_path.c_str(), "rb");
        if (!f) { fprintf(stderr, "cannot open --text file: %s\n", text_path.c_str()); return 1; }
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        std::string chunk;
        chunk.resize(sz);
        if (fread(&chunk[0], 1, sz, f) != (size_t)sz) { fclose(f); fprintf(stderr, "read error\n"); return 1; }
        fclose(f);
        // Tile short --text input up to n_tokens_target, same as the default
        // filler, so token count is comparable across different --text runs.
        while ((int32_t)text.size() < n_tokens_target * 6) {
            text += chunk;
        }
    } else {
        while ((int32_t)text.size() < n_tokens_target * 6) {
            text += k_default_filler;
        }
    }

    calib_collector collector;
    collector.head_dim    = head_dim;
    collector.rotary_dim   = rotary_dim;
    collector.freq_count  = rotary_dim / 2;
    collector.content_dim = head_dim - rotary_dim;

    common_params params;
    params.model.path      = model_path;
    params.n_gpu_layers    = n_gpu_layers;
    params.n_ctx           = ctx_size;
    params.n_batch         = std::min(ctx_size, 2048);
    params.warmup          = false;
    params.cb_eval          = triattention_calib_cb;
    params.cb_eval_user_data = &collector;

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();
    if (!model || !ctx) {
        fprintf(stderr, "failed to load model: %s\n", model_path.c_str());
        return 1;
    }

    collector.n_heads = (uint32_t)llama_model_n_head(model);
    const uint32_t n_layers    = (uint32_t)llama_model_n_layer(model);
    const uint32_t n_head_kv   = (uint32_t)llama_model_n_head_kv(model);

    std::vector<llama_token> tokens = common_tokenize(ctx, text, true);
    if ((int32_t)tokens.size() > n_tokens_target) {
        tokens.resize(n_tokens_target);
    }
    fprintf(stderr, "triattention-calibrate: %zu tokens, head_dim=%u, rotary_dim=%u, content_dim=%u, "
                     "n_layers=%u, n_heads=%u, n_kv_heads=%u\n",
            tokens.size(), head_dim, rotary_dim, collector.content_dim, n_layers, collector.n_heads, n_head_kv);

    const int32_t n_batch = params.n_batch;
    llama_batch batch = llama_batch_init(n_batch, 0, 1);

    for (size_t start = 0; start < tokens.size(); start += n_batch) {
        size_t end = std::min(start + (size_t)n_batch, tokens.size());
        common_batch_clear(batch);
        for (size_t i = start; i < end; i++) {
            common_batch_add(batch, tokens[i], (llama_pos)i, {0}, false);
        }
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "llama_decode failed at token %zu\n", start);
            return 1;
        }
        fprintf(stderr, "triattention-calibrate: processed %zu/%zu tokens (%lld Q samples so far)\n",
                end, tokens.size(), (long long)collector.total_q_calls);
    }
    llama_batch_free(batch);

    if (collector.accum.empty()) {
        fprintf(stderr, "ERROR: no Qcur-N / Qcur_normed-N tensors matched -- calibration collected nothing. "
                         "Check --head-dim against the model's actual per-head width.\n");
        return 1;
    }

    // ---- Write the .triattention v3 file ----
    FILE * out = fopen(output_path.c_str(), "wb");
    if (!out) {
        fprintf(stderr, "cannot open output: %s\n", output_path.c_str());
        return 1;
    }

    const uint32_t magic   = TRIATTENTION_MAGIC;
    const uint32_t version = TRIATTENTION_VERSION_CONTENT;
    const uint32_t n_sampled = (uint32_t)collector.accum.size();
    const uint32_t fc = collector.freq_count;
    const uint32_t cd = collector.content_dim;
    const double   theta = rope_theta;
    const std::string model_name = "triattention-calibrate-native";
    const uint32_t name_len = (uint32_t)model_name.size() + 1;

    fwrite(&magic,       sizeof(uint32_t), 1, out);
    fwrite(&version,     sizeof(uint32_t), 1, out);
    fwrite(&rotary_dim,  sizeof(uint32_t), 1, out);
    fwrite(&n_layers,    sizeof(uint32_t), 1, out);
    fwrite(&collector.n_heads, sizeof(uint32_t), 1, out);
    fwrite(&n_head_kv,   sizeof(uint32_t), 1, out);
    fwrite(&theta,       sizeof(double),   1, out);
    fwrite(&rope_style,  sizeof(uint32_t), 1, out);
    fwrite(&n_sampled,   sizeof(uint32_t), 1, out);
    fwrite(&fc,          sizeof(uint32_t), 1, out);
    fwrite(&name_len,    sizeof(uint32_t), 1, out);
    fwrite(model_name.c_str(), 1, name_len, out);
    fwrite(&cd,          sizeof(uint32_t), 1, out);

    for (auto & [key, acc] : collector.accum) {
        const uint32_t layer_u = (uint32_t)key.layer;
        const uint32_t head_u  = (uint32_t)key.head;
        fwrite(&layer_u, sizeof(uint32_t), 1, out);
        fwrite(&head_u,  sizeof(uint32_t), 1, out);

        std::vector<float> q_mean_real(fc), q_mean_imag(fc), q_abs_mean(fc), r_f(fc);
        const double inv_n = acc.n > 0 ? 1.0 / (double)acc.n : 0.0;
        for (uint32_t f = 0; f < fc; f++) {
            q_mean_real[f] = (float)(acc.sum_real[f] * inv_n);
            q_mean_imag[f] = (float)(acc.sum_imag[f] * inv_n);
            q_abs_mean[f]  = (float)(acc.sum_abs[f]  * inv_n);
            float mag = sqrtf(q_mean_real[f] * q_mean_real[f] + q_mean_imag[f] * q_mean_imag[f]);
            r_f[f] = q_abs_mean[f] > 1e-12f ? mag / q_abs_mean[f] : 0.0f;
        }
        fwrite(q_mean_real.data(), sizeof(float), fc, out);
        fwrite(q_mean_imag.data(), sizeof(float), fc, out);
        fwrite(q_abs_mean.data(),  sizeof(float), fc, out);
        fwrite(r_f.data(),         sizeof(float), fc, out);

        if (cd > 0) {
            std::vector<float> content_q_mean(cd), content_q_abs_mean(cd);
            for (uint32_t d = 0; d < cd; d++) {
                content_q_mean[d]     = (float)(acc.sum_content[d]     * inv_n);
                content_q_abs_mean[d] = (float)(acc.sum_content_abs[d] * inv_n);
            }
            fwrite(content_q_mean.data(),     sizeof(float), cd, out);
            fwrite(content_q_abs_mean.data(), sizeof(float), cd, out);
        }
    }

    fclose(out);
    fprintf(stderr, "triattention-calibrate: wrote %s (n_sampled=%u, content_dim=%u)\n",
            output_path.c_str(), n_sampled, cd);
    return 0;
}

// affine-calib: collect per-(layer, channel) means of the post-RoPE K and raw V
// activations (Kcur-<il> / Vcur-<il>) over a calibration text, and write a
// .afft table for the TorQuant affine tap (LLAMA_AFFINE_TAP).
//
// The K means are subtracted from keys before the quantized KV-cache write
// (exact under softmax: the induced score shift is constant across cache
// positions). The V means are subtracted at write and restored with one
// vector-add on the attention output (attention weights sum to 1).
//
// Usage:
//   llama-affine-calib -m model.gguf -f wiki.train.raw -o model.afft [-c 512] [--chunks 32]
//
// Table format (.afft, little-endian):
//   u32 magic 'AFFT' (0x54464641), u32 version = 1, u32 n_layer
//   per layer: u32 n_k, u32 n_v, f32 mu_k[n_k], f32 mu_v[n_v]

#include "common.h"
#include "arg.h"
#include "log.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

struct chan_acc {
    std::vector<double> sum;
    std::vector<double> sumsq;
    int64_t count = 0;
};

static std::map<int, chan_acc> g_acc_k;
static std::map<int, chan_acc> g_acc_v;
static std::mutex g_mutex;
static std::vector<uint8_t> g_buf;

static void accumulate(std::map<int, chan_acc> & acc_map, int il, const struct ggml_tensor * t) {
    // layouts: 3D (n_embd_head, n_head_kv, n_tokens) or 2D (n_embd_gqa, n_tokens)
    int64_t n_chan   = 0;
    int64_t n_tokens = 0;
    if (t->ne[2] > 1) {
        n_chan   = t->ne[0]*t->ne[1];
        n_tokens = t->ne[2];
    } else {
        n_chan   = t->ne[0];
        n_tokens = t->ne[1];
    }

    const float * data = nullptr;
    if (ggml_backend_buffer_is_host(t->buffer) && ggml_is_contiguous(t)) {
        data = (const float *) t->data;
    } else {
        g_buf.resize(ggml_nbytes(t));
        ggml_backend_tensor_get(t, g_buf.data(), 0, ggml_nbytes(t));
        data = (const float *) g_buf.data();
    }

    auto & acc = acc_map[il];
    if (acc.sum.empty()) {
        acc.sum.assign(n_chan, 0.0);
        acc.sumsq.assign(n_chan, 0.0);
    }
    if ((int64_t) acc.sum.size() != n_chan) {
        return; // shape changed mid-run - should not happen
    }

    for (int64_t tok = 0; tok < n_tokens; ++tok) {
        const float * row = data + tok*n_chan;
        for (int64_t c = 0; c < n_chan; ++c) {
            acc.sum[c]   += row[c];
            acc.sumsq[c] += (double) row[c]*row[c];
        }
    }
    acc.count += n_tokens;
}

static bool collect_means(struct ggml_tensor * t, bool ask, void * user_data) {
    GGML_UNUSED(user_data);

    const char * name = t->name;

    bool is_k = strncmp(name, "Kcur-", 5) == 0;
    bool is_v = strncmp(name, "Vcur-", 5) == 0;

    if (ask) {
        if (!is_k && !is_v) {
            return false;
        }
        if (t->type != GGML_TYPE_F32) {
            return false;
        }
        // skip tiny batches: keeps the 2D/3D layout disambiguation safe
        const int64_t n_tokens = t->ne[2] > 1 ? t->ne[2] : t->ne[1];
        return n_tokens >= 16;
    }

    if (!is_k && !is_v) {
        return true;
    }

    const int il = atoi(name + 5);

    std::lock_guard<std::mutex> lock(g_mutex);
    accumulate(is_k ? g_acc_k : g_acc_v, il, t);

    return true;
}

static void print_usage(int, char ** argv) {
    LOG("\nusage: %s -m model.gguf -f calib-text.raw -o table.afft [-c 512] [--chunks N]\n\n", argv[0]);
}

int main(int argc, char ** argv) {
    common_params params;

    params.n_ctx     = 512;
    params.escape    = false;
    params.out_file  = "affine-tap.afft";

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_IMATRIX, print_usage)) {
        return 1;
    }

    common_init();

    params.n_batch = std::min(params.n_batch, params.n_ctx);

    params.cb_eval           = collect_means;
    params.cb_eval_user_data = nullptr;
    params.warmup            = false;

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to load model\n", __func__);
        return 1;
    }

    const int n_ctx = llama_n_ctx(ctx);

    LOG_INF("%s: tokenizing calibration data\n", __func__);
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true, params.parse_special);

    const int n_chunk_max = (int) (tokens.size() / n_ctx);
    int n_chunk = params.n_chunks < 0 ? n_chunk_max : std::min(params.n_chunks, n_chunk_max);

    if (n_chunk == 0) {
        LOG_ERR("%s: calibration data too small (%zu tokens, need >= %d)\n", __func__, tokens.size(), n_ctx);
        return 1;
    }

    LOG_INF("%s: collecting K/V means over %d chunks of %d tokens\n", __func__, n_chunk, n_ctx);

    llama_batch batch = llama_batch_init(n_ctx, 0, 1);

    for (int i = 0; i < n_chunk; ++i) {
        llama_memory_clear(llama_get_memory(ctx), true);

        common_batch_clear(batch);
        for (int j = 0; j < n_ctx; ++j) {
            common_batch_add(batch, tokens[i*n_ctx + j], j, {0}, false);
        }

        if (llama_decode(ctx, batch)) {
            LOG_ERR("%s: llama_decode failed on chunk %d\n", __func__, i);
            llama_batch_free(batch);
            return 1;
        }

        LOG_INF("%s: chunk %d/%d done\n", __func__, i + 1, n_chunk);
    }

    llama_batch_free(batch);

    // write the table
    int n_layer = 0;
    for (const auto & [il, _] : g_acc_k) { n_layer = std::max(n_layer, il + 1); }
    for (const auto & [il, _] : g_acc_v) { n_layer = std::max(n_layer, il + 1); }

    if (n_layer == 0) {
        LOG_ERR("%s: no Kcur/Vcur activations were captured - wrong model arch for this hook?\n", __func__);
        return 1;
    }

    FILE * f = fopen(params.out_file.c_str(), "wb");
    if (!f) {
        LOG_ERR("%s: failed to open '%s' for writing\n", __func__, params.out_file.c_str());
        return 1;
    }

    auto write_u32 = [&](uint32_t v) { fwrite(&v, sizeof(v), 1, f); };

    write_u32(0x54464641); // 'AFFT'
    write_u32(1);
    write_u32((uint32_t) n_layer);

    for (int il = 0; il < n_layer; ++il) {
        std::vector<float> mu_k;
        std::vector<float> mu_v;

        if (auto it = g_acc_k.find(il); it != g_acc_k.end() && it->second.count > 0) {
            mu_k.resize(it->second.sum.size());
            for (size_t c = 0; c < mu_k.size(); ++c) {
                mu_k[c] = (float) (it->second.sum[c] / (double) it->second.count);
            }
        }
        if (auto it = g_acc_v.find(il); it != g_acc_v.end() && it->second.count > 0) {
            mu_v.resize(it->second.sum.size());
            for (size_t c = 0; c < mu_v.size(); ++c) {
                mu_v[c] = (float) (it->second.sum[c] / (double) it->second.count);
            }
        }

        write_u32((uint32_t) mu_k.size());
        write_u32((uint32_t) mu_v.size());
        if (!mu_k.empty()) { fwrite(mu_k.data(), sizeof(float), mu_k.size(), f); }
        if (!mu_v.empty()) { fwrite(mu_v.data(), sizeof(float), mu_v.size(), f); }

        // DC energy fraction per side: ||mu||^2 / sum_c E[x_c^2], and the
        // equivalent wasted-bits bound eq_bits = -0.5*log2(1 - frac)
        auto dc_stats = [&](const std::map<int, chan_acc> & m, std::vector<float> & mu) -> std::pair<double,double> {
            auto it = m.find(il);
            if (it == m.end() || it->second.count == 0 || mu.empty()) {
                return {0.0, 0.0};
            }
            double e2 = 0.0, m2 = 0.0;
            for (size_t c = 0; c < mu.size(); ++c) {
                e2 += it->second.sumsq[c] / (double) it->second.count;
                m2 += (double) mu[c]*mu[c];
            }
            const double frac = e2 > 0.0 ? m2/e2 : 0.0;
            const double bits = frac < 1.0 ? -0.5*log2(1.0 - frac) : INFINITY;
            return {frac, bits};
        };
        const auto [kfrac, kbits] = dc_stats(g_acc_k, mu_k);
        const auto [vfrac, vbits] = dc_stats(g_acc_v, mu_v);

        LOG_INF("%s: layer %3d: n_k = %5zu, n_v = %5zu, tokens = %lld | DC K: %6.4f (%.3f bits)  DC V: %6.4f (%.3f bits)\n",
                __func__, il, mu_k.size(), mu_v.size(),
                (long long) (g_acc_k.count(il) ? g_acc_k[il].count : 0),
                kfrac, kbits, vfrac, vbits);
    }

    fclose(f);
    LOG_INF("%s: wrote affine tap table to '%s' (%d layers)\n", __func__, params.out_file.c_str(), n_layer);

    llama_backend_free();

    return 0;
}

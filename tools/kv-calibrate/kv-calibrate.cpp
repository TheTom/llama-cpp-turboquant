#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <clocale>
#include <cstring>
#include <string>
#include <vector>

static std::vector<char *> translate_args(int argc, char ** argv, std::vector<std::string> & storage) {
    storage.clear();
    storage.reserve(argc + 8);
    storage.emplace_back(argv[0]);
    storage.emplace_back("--kv-empvar-calibrate");

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--mode") {
            storage.emplace_back("--kv-calibration-mode");
        } else if (arg == "-o" || arg == "--output") {
            storage.emplace_back("--kv-empvar-calibrate-out");
        } else {
            storage.emplace_back(arg);
        }
    }

    std::vector<char *> out;
    out.reserve(storage.size());
    for (auto & s : storage) {
        out.push_back(s.data());
    }
    return out;
}

static int run_kv_calibration(llama_context * ctx, common_params & params, int32_t n_ctx) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    const auto t_tok_start = std::chrono::high_resolution_clock::now();
    LOG_INF("%s: tokenizing calibration input\n", __func__);
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true);
    const auto t_tok_end = std::chrono::high_resolution_clock::now();
    LOG_INF("%s: tokenization took %.3f ms\n", __func__,
            1e-3 * std::chrono::duration_cast<std::chrono::microseconds>(t_tok_end - t_tok_start).count());

    if ((int) tokens.size() < n_ctx) {
        LOG_ERR("%s: need at least %d tokens, got %zu\n", __func__, n_ctx, tokens.size());
        return 1;
    }

    const int n_chunk_max = (int) tokens.size() / n_ctx;
    const int n_chunk = params.n_chunks < 0 ? n_chunk_max : std::min(params.n_chunks, n_chunk_max);
    if (n_chunk <= 0) {
        LOG_ERR("%s: no calibration chunks available\n", __func__);
        return 1;
    }

    const int n_batch = params.n_batch;
    const int n_seq = std::max(1, n_batch / n_ctx);
    const int num_batches = (n_ctx + n_batch - 1) / n_batch;

    GGML_ASSERT(n_batch < n_ctx || n_batch % n_ctx == 0);
    GGML_ASSERT(params.n_ctx == n_seq * n_ctx);

    llama_batch batch = llama_batch_init(std::min(n_batch, n_ctx * n_seq), 0, 1);

    LOG_INF("%s: calibrating over %d chunks, n_ctx=%d, batch_size=%d, n_seq=%d, mode=%s, output=%s\n",
            __func__, n_chunk, n_ctx, n_batch, n_seq,
            params.kv_calibration_mode.c_str(), params.kv_empvar_calibration_out.c_str());

    const auto t_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_chunk; i += n_seq) {
        const int start = i * n_ctx;
        const int end = start + n_ctx;
        const int n_seq_batch = std::min(n_seq, n_chunk - i);

        llama_memory_clear(llama_get_memory(ctx), true);

        for (int j = 0; j < num_batches; ++j) {
            const int batch_start = start + j * n_batch;
            const int batch_size = std::min(end - batch_start, n_batch);

            batch.n_tokens = 0;
            for (int seq = 0; seq < n_seq_batch; ++seq) {
                const int seq_start = batch_start + seq * n_ctx;
                const auto token_org = tokens[seq_start];
                if (add_bos && j == 0) {
                    tokens[seq_start] = llama_vocab_bos(vocab);
                }

                for (int k = 0; k < batch_size; ++k) {
                    const int idx = seq * n_ctx + k;
                    batch.token   [idx]    = tokens[seq_start + k];
                    batch.pos     [idx]    = j * n_batch + k;
                    batch.n_seq_id[idx]    = 1;
                    batch.seq_id  [idx][0] = seq;
                    batch.logits  [idx]    = 0;
                }
                batch.n_tokens += batch_size;
                tokens[seq_start] = token_org;
            }

            if (llama_decode(ctx, batch)) {
                LOG_ERR("%s: llama_decode failed\n", __func__);
                llama_batch_free(batch);
                return 1;
            }
        }

        LOG("[%d/%d]", std::min(i + n_seq, n_chunk), n_chunk);
    }
    LOG("\n");
    llama_synchronize(ctx);

    const auto t_end = std::chrono::high_resolution_clock::now();
    LOG_INF("%s: calibration decode took %.3f sec\n", __func__, std::chrono::duration<float>(t_end - t_start).count());

    llama_batch_free(batch);
    return 0;
}

int main(int argc, char ** argv) {
    setlocale(LC_ALL, "");

    std::vector<std::string> arg_storage;
    std::vector<char *> translated = translate_args(argc, argv, arg_storage);

    common_params params;
    params.n_ctx = 512;
    params.escape = false;
    params.no_perf = true;
    params.warmup = false;

    if (!common_params_parse((int) translated.size(), translated.data(), params, LLAMA_EXAMPLE_PERPLEXITY)) {
        return 1;
    }

    if (params.kv_empvar_calibration_out.empty()) {
        LOG_ERR("%s: output path required; use -o FILE or --output FILE\n", __func__);
        return 1;
    }

    if (params.kv_calibration_mode.empty()) {
        LOG_ERR("%s: calibration mode must be non-empty\n", __func__);
        return 1;
    }

    const int32_t n_ctx = params.n_ctx;
    if (n_ctx <= 0) {
        LOG_ERR("%s: --ctx-size must be > 0\n", __func__);
        return 1;
    }

    params.n_parallel = std::max(1, params.n_batch / n_ctx);
    params.n_ctx = params.n_parallel * n_ctx;
    params.n_batch = std::min(params.n_batch, params.n_ctx);
    params.cache_type_k = GGML_TYPE_F32;
    params.cache_type_v = GGML_TYPE_F32;

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    llama_context * ctx = llama_init->context();
    if (llama_init->model() == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to initialize model/context\n", __func__);
        llama_backend_free();
        return 1;
    }

    const int ret = run_kv_calibration(ctx, params, n_ctx);

    llama_backend_free();
    return ret;
}

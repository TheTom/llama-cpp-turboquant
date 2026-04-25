#include "wht-empvar-utils.h"

#include "../src/llama-context.h"
#include "../src/llama-kv-empvar-calibration.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-kv-cache-iswa.h"
#include "../src/llama-memory-hybrid.h"
#include "../src/llama-memory-hybrid-iswa.h"
#include "../src/llama-model.h"

#include "common.h"
#include "ggml.h"
#include "nlohmann/json.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

using json = nlohmann::json;
using namespace wht_empvar;

namespace {

struct bench_params {
    std::string model;
    std::string text_file = "wikitext-2-raw/wiki.test.raw";
    int seq_len = 128;
    int calib_seqs = 4;
    int eval_seqs = 4;
    int max_layers = 2;
    int max_heads = 2;
    int bits = 2;
    std::string output;
};

static constexpr int kTurboRuntimeWhtGroup = 128;

bool parse_args(int argc, char ** argv, bench_params & p) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                std::exit(1);
            }
            return argv[++i];
        };
        if (arg == "--model") {
            p.model = need("--model");
        } else if (arg == "--text-file") {
            p.text_file = need("--text-file");
        } else if (arg == "--seq-len") {
            p.seq_len = std::atoi(need("--seq-len"));
        } else if (arg == "--calib-seqs") {
            p.calib_seqs = std::atoi(need("--calib-seqs"));
        } else if (arg == "--eval-seqs") {
            p.eval_seqs = std::atoi(need("--eval-seqs"));
        } else if (arg == "--max-layers") {
            p.max_layers = std::atoi(need("--max-layers"));
        } else if (arg == "--max-heads") {
            p.max_heads = std::atoi(need("--max-heads"));
        } else if (arg == "--bits") {
            p.bits = std::atoi(need("--bits"));
        } else if (arg == "--output") {
            p.output = need("--output");
        } else if (arg == "--help") {
            return false;
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", arg.c_str());
            return false;
        }
    }
    return !p.model.empty();
}

void usage() {
    std::fprintf(stderr,
        "usage: bench-wht-empvar --model MODEL.gguf [--text-file FILE] [--seq-len N] [--calib-seqs N]\n"
        "                         [--eval-seqs N] [--max-layers N] [--max-heads N] [--bits N] [--output FILE]\n");
}

std::string read_file(const std::string & path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open " + path);
    }
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

std::vector<std::vector<llama_token>> make_token_chunks(
        const llama_vocab * vocab,
        const std::string & text,
        int seq_len,
        int n_chunks) {
    auto toks = common_tokenize(vocab, text, true, true);
    std::vector<std::vector<llama_token>> chunks;
    for (int i = 0; i + seq_len <= (int) toks.size() && (int) chunks.size() < n_chunks; i += seq_len) {
        chunks.emplace_back(toks.begin() + i, toks.begin() + i + seq_len);
    }
    return chunks;
}

llama_batch batch_from_tokens(const std::vector<llama_token> & toks) {
    return llama_batch_get_one(const_cast<llama_token *>(toks.data()), (int32_t) toks.size());
}

const llama_kv_cache * resolve_kv_cache_for_layer(
        const ::llama_context * ctx,
        int layer) {
    auto * mem = ctx->get_memory();
    if (auto * kv = dynamic_cast<llama_kv_cache *>(mem)) {
        return kv;
    }
    if (auto * kv_iswa = dynamic_cast<llama_kv_cache_iswa *>(mem)) {
        return ctx->get_model().hparams.is_swa(layer) ? kv_iswa->get_swa() : kv_iswa->get_base();
    }
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem)) {
        return hybrid->get_mem_attn();
    }
    if (auto * hybrid_iswa = dynamic_cast<llama_memory_hybrid_iswa *>(mem)) {
        auto * attn = hybrid_iswa->get_mem_attn();
        return ctx->get_model().hparams.is_swa(layer) ? attn->get_swa() : attn->get_base();
    }
    throw std::runtime_error("unsupported llama memory type for KV extraction");
}

std::vector<std::vector<float>> extract_vectors(
        llama_context * ctx_public,
        int layer,
        int head,
        int n_tokens,
        bool values) {
    auto * ctx = reinterpret_cast<::llama_context *>(ctx_public);
    auto * mem = resolve_kv_cache_for_layer(ctx, layer);

    llama_kv_cache::slot_info sinfo;
    sinfo.s0 = 0;
    sinfo.s1 = 0;
    sinfo.strm = {0};
    sinfo.idxs = {{}};
    sinfo.idxs[0].reserve(n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        sinfo.idxs[0].push_back(i);
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 1024 * 64,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * gctx = ggml_init(params);
    if (!gctx) {
        throw std::runtime_error("failed to init ggml context");
    }

    ggml_tensor * t = values ? mem->get_v(gctx, layer, n_tokens, sinfo) : mem->get_k(gctx, layer, n_tokens, sinfo);
    const bool v_trans = values && t->ne[0] == n_tokens && t->ne[2] > 0;
    const int dim = v_trans ? (int) t->ne[2] : (int) t->ne[0];
    std::vector<std::vector<float>> out(n_tokens, std::vector<float>(dim));
    for (int tok = 0; tok < n_tokens; ++tok) {
        for (int d = 0; d < dim; ++d) {
            char * base = static_cast<char *>(t->data);
            const size_t off = v_trans
                    ? size_t(tok) * t->nb[0] + size_t(head) * t->nb[1] + size_t(d) * t->nb[2]
                    : size_t(d) * t->nb[0] + size_t(head) * t->nb[1] + size_t(tok) * t->nb[2];
            out[tok][d] = *reinterpret_cast<float *>(base + off);
        }
    }
    ggml_free(gctx);
    return out;
}

std::vector<std::vector<std::vector<float>>> split_chunks_for_runtime_groups(
        const std::vector<std::vector<std::vector<float>>> & chunks,
        int runtime_dim) {
    std::vector<std::vector<std::vector<float>>> out;
    out.reserve(chunks.size());
    for (const auto & chunk : chunks) {
        out.push_back(split_rows_for_runtime_groups(chunk, runtime_dim));
    }
    return out;
}

struct group_dataset {
    std::vector<std::vector<float>> calib_keys;
    std::vector<std::vector<float>> calib_values;
    std::vector<std::vector<std::vector<float>>> eval_key_chunks;
    std::vector<std::vector<std::vector<float>>> eval_value_chunks;
    double calib_decode_ms = 0.0;
    double eval_decode_ms = 0.0;
};

group_dataset collect_group_dataset(
        llama_context * ctx,
        const std::vector<std::vector<llama_token>> & calib_chunks,
        const std::vector<std::vector<llama_token>> & eval_chunks,
        int layer,
        int head) {
    group_dataset out;

    for (const auto & chunk : calib_chunks) {
        const auto t0 = std::chrono::steady_clock::now();
        llama_memory_clear(llama_get_memory(ctx), true);
        if (llama_decode(ctx, batch_from_tokens(chunk)) != 0) {
            throw std::runtime_error("llama_decode failed during calibration");
        }
        const auto t1 = std::chrono::steady_clock::now();
        out.calib_decode_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

        auto key_rows = extract_vectors(ctx, layer, head, (int) chunk.size(), false);
        auto value_rows = extract_vectors(ctx, layer, head, (int) chunk.size(), true);
        out.calib_keys.insert(out.calib_keys.end(), key_rows.begin(), key_rows.end());
        out.calib_values.insert(out.calib_values.end(), value_rows.begin(), value_rows.end());
    }

    for (const auto & chunk : eval_chunks) {
        const auto t0 = std::chrono::steady_clock::now();
        llama_memory_clear(llama_get_memory(ctx), true);
        if (llama_decode(ctx, batch_from_tokens(chunk)) != 0) {
            throw std::runtime_error("llama_decode failed during evaluation");
        }
        const auto t1 = std::chrono::steady_clock::now();
        out.eval_decode_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

        out.eval_key_chunks.push_back(extract_vectors(ctx, layer, head, (int) chunk.size(), false));
        out.eval_value_chunks.push_back(extract_vectors(ctx, layer, head, (int) chunk.size(), true));
    }

    return out;
}

json summarize_one_side(
        const std::vector<std::vector<float>> & calib_vectors,
        const std::vector<std::vector<std::vector<float>>> & eval_chunks,
        int bits,
        int runtime_dim) {
    if (calib_vectors.empty()) {
        throw std::runtime_error("summarize_one_side requires non-empty calibration data");
    }
    std::vector<std::vector<float>> flat_eval;
    for (const auto & chunk : eval_chunks) {
        flat_eval.insert(flat_eval.end(), chunk.begin(), chunk.end());
    }

    llama_kv_empvar_calibration calibration(runtime_dim);
    calibration.observe_key_rows(calib_vectors);
    const auto stats = calibration.finalize_keys();
    std::vector<std::vector<std::vector<float>>> emp_books;
    std::vector<std::vector<std::vector<float>>> fixed_books;
    emp_books.reserve(stats.chunks.size());
    fixed_books.reserve(stats.chunks.size());
    for (const auto & chunk_stats : stats.chunks) {
        emp_books.push_back(build_empirical_codebooks(chunk_stats.variances, bits));
        fixed_books.push_back(build_fixed_codebooks(runtime_dim, bits));
    }

    const auto emp_metrics = evaluate_reconstruction_chunked(flat_eval, emp_books, false);
    const auto fixed_metrics = evaluate_reconstruction_chunked(flat_eval, fixed_books, true);
    json chunk_stats = json::array();
    for (size_t i = 0; i < stats.chunks.size(); ++i) {
        chunk_stats.push_back({
            {"chunk_index", (int) i},
            {"mean_variance", stats.chunks[i].mean_variance},
            {"std_variance", stats.chunks[i].std_variance},
            {"l2_to_uniform", stats.chunks[i].l2_to_uniform},
            {"variances", stats.chunks[i].variances},
        });
    }

    json out;
    out["runtime_dim"] = runtime_dim;
    out["target_dim"] = stats.head_dim;
    out["n_chunks_per_head"] = stats.chunks.size();
    out["n_calib_vectors"] = stats.n_rows;
    out["n_eval_vectors"] = flat_eval.size();
    out["mean_variance"] = stats.mean_variance;
    out["std_variance"] = stats.std_variance;
    out["l2_to_uniform"] = stats.l2_to_uniform;
    out["variances"] = stats.variances;
    out["chunk_variances"] = chunk_stats;
    out["empirical"] = {
        {"mean_cosine", emp_metrics.mean_cosine},
        {"mean_mse", emp_metrics.mean_mse},
    };
    out["fixed_baseline"] = {
        {"mean_cosine", fixed_metrics.mean_cosine},
        {"mean_mse", fixed_metrics.mean_mse},
    };
    return out;
}

json summarize_attention_proxy(
        const group_dataset & data,
        int bits,
        int runtime_dim) {
    GGML_UNUSED(data);
    GGML_UNUSED(bits);
    GGML_UNUSED(runtime_dim);
    return {
        {"skipped", true},
        {"reason", "attention proxy disabled for mixed-width runtime grouping"},
    };
}

} // namespace

int main(int argc, char ** argv) {
    bench_params params;
    if (!parse_args(argc, argv, params)) {
        usage();
        return params.model.empty() ? 1 : 0;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(params.model.c_str(), mparams);
    if (!model) {
        std::fprintf(stderr, "failed to load model: %s\n", params.model.c_str());
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = params.seq_len;
    cparams.n_batch = params.seq_len;
    cparams.n_ubatch = params.seq_len;
    cparams.type_k = GGML_TYPE_F32;
    cparams.type_v = GGML_TYPE_F32;
    cparams.n_threads = 4;
    cparams.n_threads_batch = 4;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "failed to init context\n");
        llama_model_free(model);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::string text = read_file(params.text_file);
    auto chunks = make_token_chunks(vocab, text, params.seq_len, params.calib_seqs + params.eval_seqs);
    if ((int) chunks.size() < params.calib_seqs + params.eval_seqs) {
        std::fprintf(stderr, "not enough token chunks in %s\n", params.text_file.c_str());
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    std::vector<std::vector<llama_token>> calib_chunks(chunks.begin(), chunks.begin() + params.calib_seqs);
    std::vector<std::vector<llama_token>> eval_chunks(chunks.begin() + params.calib_seqs, chunks.begin() + params.calib_seqs + params.eval_seqs);

    auto * ctx_internal = reinterpret_cast<::llama_context *>(ctx);
    const auto & hp = ctx_internal->get_model().hparams;
    const int n_layers = std::min<int>(params.max_layers, hp.n_layer);
    const int n_heads = std::min<int>(params.max_heads, hp.n_head_kv(0));

    json out;
    out["model"] = params.model;
    out["text_file"] = params.text_file;
    out["seq_len"] = params.seq_len;
    out["calib_seqs"] = params.calib_seqs;
    out["eval_seqs"] = params.eval_seqs;
    out["bits"] = params.bits;
    out["results"] = json::array();

    for (int layer = 0; layer < n_layers; ++layer) {
        for (int head = 0; head < n_heads; ++head) {
            const auto t0 = std::chrono::steady_clock::now();
            const auto data = collect_group_dataset(ctx, calib_chunks, eval_chunks, layer, head);
            const auto t1 = std::chrono::steady_clock::now();

            json item;
            item["layer"] = layer;
            item["head"] = head;
            item["keys"] = summarize_one_side(data.calib_keys, data.eval_key_chunks, params.bits, kTurboRuntimeWhtGroup);
            item["values"] = summarize_one_side(data.calib_values, data.eval_value_chunks, params.bits, kTurboRuntimeWhtGroup);
            item["attention_proxy_qeqk"] = summarize_attention_proxy(data, params.bits, kTurboRuntimeWhtGroup);
            item["timing_ms"] = {
                {"calibration_decode", data.calib_decode_ms},
                {"evaluation_decode", data.eval_decode_ms},
                {"total_group", std::chrono::duration<double, std::milli>(t1 - t0).count()},
            };
            out["results"].push_back(item);
        }
    }

    if (!params.output.empty()) {
        std::ofstream f(params.output);
        f << out.dump(2) << "\n";
    } else {
        std::cout << out.dump(2) << "\n";
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

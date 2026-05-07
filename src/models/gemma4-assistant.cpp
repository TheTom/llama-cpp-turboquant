#include "models.h"

#include <cmath>

static llm_graph_params graph_params_for_mtp(llm_graph_params p, const llama_model & mtp_model) {
    p.arch    = mtp_model.arch;
    p.hparams = mtp_model.hparams;
    p.gtype   = LLM_GRAPH_TYPE_MTP;
    return p;
}

// Last layer in [range_start, range_end) whose attention type matches want_swa.
static int32_t gemma4_mtp_kv_layer_last_in_range(
        const llama_hparams & tgt, int32_t range_start, int32_t range_end, bool want_swa) {
    int32_t best = -1;
    if (range_start < 0) {
        range_start = 0;
    }
    if (range_end > (int32_t) tgt.n_layer) {
        range_end = (int32_t) tgt.n_layer;
    }
    for (int32_t il = range_start; il < range_end; ++il) {
        if (tgt.is_swa((uint32_t) il) == want_swa) {
            best = il;
        }
    }
    return best;
}

// Build a single MTP step: token embedding (from target) + h_prev -> N transformer blocks ->
// (post-projected backbone hidden, vocabulary logits, argmax token id).
//
// Used by both the single-step path (n_mtp_steps==1) and the chained path
// (n_mtp_steps>1, called n_mtp_steps times within the same graph).
//
// `pos_step` must be a scalar I32 tensor [1] holding the absolute target position
// for this step's RoPE (the cross-attn mask is shared across steps because the
// target's KV cache only contains positions ≤ attn_pos and all step positions
// are > attn_pos, so causal/SWA admit all target cells uniformly).
//
// `out_logits`: F32 [n_vocab, 1] (full row for ordered embeddings too — required for greedy match with verify).
// `out_argmax` (I32 [1]): greedy token id on-device.
static void gemma4_mtp_build_one_step(
        const llm_graph_context & gctx,
        const llama_model & target,
        const llama_model & mtp,
        llm_graph_input_attn_kv_iswa * inp_attn,
        ggml_tensor * tok_step,            // I32 [1]
        ggml_tensor * h_step,              // F32 [n_bb, 1]
        ggml_tensor * pos_step,            // I32 [1]
        ggml_tensor ** out_logits,         // F32 [n_vocab, 1]
        ggml_tensor ** out_h_post,         // F32 [n_bb, 1]
        ggml_tensor ** out_argmax) {       // I32 [1]
    auto * ctx0    = gctx.ctx0;
    auto * gf      = gctx.gf;
    const auto & hparams = gctx.hparams;
    const auto & cparams = gctx.cparams;
    const int    n_layer = (int) gctx.n_layer;
    const int    n_tokens     = (int) gctx.n_tokens;
    const int    n_ctx_orig   = (int) gctx.n_ctx_orig;
    const int    rope_type    = gctx.rope_type;
    const float  ext_factor   = gctx.ext_factor;
    const float  attn_factor  = gctx.attn_factor;
    const float  beta_fast    = gctx.beta_fast;
    const float  beta_slow    = gctx.beta_slow;
    auto         cb = [&](ggml_tensor * t, const char * name, int il) { gctx.cb(t, name, il); };

    const float   tok_scale = sqrtf((float) target.hparams.n_embd);

    ggml_tensor * tok_e = ggml_get_rows(ctx0, target.tok_embd, tok_step);
    cb(tok_e, "mtp_tgt_tok_embd", -1);

    // Gemma 4 scales token embeddings by sqrt(n_embd) at the input pipeline (gemma4-iswa.cpp).
    // Use target n_embd so Edge / non-Edge targets match the main forward.
    tok_e = ggml_scale(ctx0, tok_e, tok_scale);
    cb(tok_e, "mtp_tgt_tok_embd_scaled", -1);

    ggml_tensor * inp_cat = ggml_concat(ctx0, tok_e, h_step, 0);
    cb(inp_cat, "mtp_concat", -1);

    ggml_tensor * inpL = gctx.build_lora_mm(mtp.mtp_pre_projection, inp_cat);
    cb(inpL, "mtp_pre_proj_out", -1);

    ggml_build_forward_expand(gf, inpL);

    ggml_tensor * cur = nullptr;

    for (int il = 0; il < n_layer; ++il) {
        const int64_t n_embd_head = hparams.n_embd_head_k(il);
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_v(il));

        const int64_t n_head = hparams.n_head(il);

        const float freq_base_l  = mtp.get_rope_freq_base(cparams, il);
        const float freq_scale_l = mtp.get_rope_freq_scale(cparams, il);
        const int   n_rot_l      = hparams.n_rot(il);

        cur = gctx.build_norm(inpL, mtp.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_tensor * freq_factors = nullptr;
        if (!hparams.is_swa(il)) {
            freq_factors = mtp.layers[il].rope_freqs;
        }

        ggml_tensor * Qcur = gctx.build_lora_mm(mtp.layers[il].wq, cur);
        cb(Qcur, "Qcur", il);

        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);

        Qcur = gctx.build_norm(Qcur, mtp.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
        cb(Qcur, "Qcur_normed", il);

        Qcur = ggml_rope_ext(ctx0, Qcur, pos_step, freq_factors, n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                             ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Qcur, "Qcur_pos", il);

        const bool read_swa = hparams.is_swa(il);

        const int32_t n_tgt = (int32_t) target.hparams.n_layer;

        // Per HF Gemma4AssistantForCausalLM (transformers main): MTP cross-attention reads
        // ONE shared KV per attention type from the target — the LAST layer of that type.
        //   ref: src/transformers/models/gemma4_assistant/modeling_gemma4_assistant.py
        //   shared_kv_states = {"full_attention": (K, V), "sliding_attention": (K, V)}
        const int32_t il_kv = gemma4_mtp_kv_layer_last_in_range(target.hparams, 0, n_tgt, read_swa);

        GGML_ASSERT(il_kv >= 0 && "Gemma4 MTP: target has no layer matching MTP attention type (SWA/full)");

        // Per HF Gemma4: even when target's attention_k_eq_v is True (so v_proj is None and
        // Vcur is created from Kcur source), the V cache slot is still WRITTEN with the
        // rms-norm-without-scale, non-rotated tensor. Therefore for cross-attention we must
        // ALWAYS fetch V from the cache — not reuse the post-RoPE K tensor.
        const bool use_k_as_v = false;

        const int64_t kv_embd_head_v = target.hparams.n_embd_head_v(il_kv);
        const int64_t kv_n_head_v    = target.hparams.n_head_kv(il_kv);

        cur = gctx.build_attn_mtp(inp_attn, mtp.layers[il].wo, nullptr, Qcur, nullptr, nullptr, nullptr,
                hparams.f_attention_scale, il, il_kv, read_swa, kv_embd_head_v, kv_n_head_v, use_k_as_v);

        cur = gctx.build_norm(cur, mtp.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        ggml_tensor * attn_out = ggml_add(ctx0, cur, inpL);
        cb(attn_out, "attn_out", il);

        GGML_ASSERT(mtp.layers[il].ffn_gate_inp == nullptr && "gemma4_assistant MTP does not support MoE FFN");

        cur = gctx.build_norm(attn_out, mtp.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = gctx.build_ffn(cur,
                mtp.layers[il].ffn_up,   nullptr, nullptr,
                mtp.layers[il].ffn_gate, nullptr, nullptr,
                mtp.layers[il].ffn_down, nullptr, nullptr,
                nullptr,
                LLM_FFN_GELU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = gctx.build_norm(cur, mtp.layers[il].ffn_post_norm, nullptr, LLM_NORM_RMS, -1);
        cb(cur, "ffn_post_norm", il);

        cur = ggml_add(ctx0, cur, attn_out);

        if (mtp.layers[il].out_scale) {
            cur = ggml_mul(ctx0, cur, mtp.layers[il].out_scale);
            cb(cur, "out_scaled", il);
        }

        cur = gctx.build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;

    cur = gctx.build_norm(cur, mtp.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    ggml_tensor * h_inner = cur;

    ggml_tensor * backbone = gctx.build_lora_mm(mtp.mtp_post_projection, h_inner);
    cb(backbone, "mtp_post_proj_out", -1);

    const int64_t n_vocab      = mtp.tok_embd->ne[1];
    const int64_t n_tokens_mtp = h_inner->ne[1];

    if (mtp.hparams.use_ordered_embeddings) {
        // Centroid-routed LM head (HF Gemma4AssistantMaskedEmbedder): scatter candidate logits into a full
        // [n_vocab] row then argmax — matches masked full-vocab greedy (sparse-only argmax broke server accept).
        GGML_ASSERT(mtp.mtp_centroids != nullptr && mtp.mtp_token_ordering != nullptr);
        GGML_ASSERT(n_tokens_mtp == 1 && "ordered embeddings MTP expects a single token column");
        const uint32_t n_c   = mtp.hparams.n_centroids;
        const uint32_t top_k = mtp.hparams.centroid_top_k;
        GGML_ASSERT(n_c > 0 && top_k > 0 && (int64_t) top_k <= (int64_t) n_c);
        GGML_ASSERT(n_vocab % (int64_t) n_c == 0);
        const int64_t vsc = n_vocab / (int64_t) n_c;

        ggml_tensor * centroid_logits = gctx.build_lora_mm(mtp.mtp_centroids, h_inner);
        cb(centroid_logits, "mtp_centroid_logits", -1);

        ggml_tensor * topk_idx = ggml_top_k(ctx0, centroid_logits, (int) top_k);
        cb(topk_idx, "mtp_centroid_topk_idx", -1);

        const size_t ordering_nb1 = ggml_row_size(GGML_TYPE_I32, vsc);
        ggml_tensor * ordering = ggml_view_2d(
                ctx0, mtp.mtp_token_ordering, vsc, (int64_t) n_c, ordering_nb1, 0);
        cb(ordering, "mtp_token_ordering_view", -1);

        ggml_tensor * sel_ids = ggml_get_rows(ctx0, ordering, topk_idx);
        cb(sel_ids, "mtp_selected_token_ids", -1);

        const int64_t n_sel = (int64_t) top_k * vsc * n_tokens_mtp;
        ggml_tensor * flat_ids = ggml_reshape_1d(ctx0, sel_ids, n_sel);
        cb(flat_ids, "mtp_selected_token_ids_flat", -1);

        ggml_tensor * sel_emb = ggml_get_rows(ctx0, mtp.tok_embd, flat_ids);
        cb(sel_emb, "mtp_selected_embd", -1);

        ggml_tensor * sel_logits = gctx.build_lora_mm(sel_emb, h_inner);
        cb(sel_logits, "mtp_selected_logits", -1);
        ggml_tensor * sel_logits_f32 = ggml_cast(ctx0, sel_logits, GGML_TYPE_F32);

        ggml_tensor * logits_full = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_vocab, n_tokens_mtp);
        logits_full = ggml_fill_inplace(ctx0, logits_full, -1e30f);
        cb(logits_full, "mtp_logits_masked_base", -1);

        ggml_tensor * scatter_dst = ggml_cont_2d(ctx0, logits_full, 1, n_vocab * n_tokens_mtp);
        ggml_tensor * scatter_src = ggml_cont_2d(ctx0, sel_logits_f32, 1, n_sel);
        cur = ggml_set_rows(ctx0, scatter_dst, scatter_src, flat_ids);
        cb(cur, "mtp_logits_scatter_view", -1);
        cur = ggml_reshape_2d(ctx0, cur, n_vocab, n_tokens_mtp);
        cb(cur, "mtp_logits_full", -1);
    } else {
        cur = gctx.build_lora_mm(mtp.tok_embd, h_inner);
        cb(cur, "result_output_dense", -1);
    }

    if (hparams.f_final_logit_softcapping) {
        cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
        cur = ggml_tanh(ctx0, cur);
        cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);
    }

    cb(cur, "result_output", -1);

    // Greedy argmax on-device: I32 [1] token index into the vocabulary row.
    ggml_tensor * arg = ggml_argmax(ctx0, cur);
    cb(arg, "result_argmax", -1);

    *out_logits = cur;
    *out_h_post = backbone;
    *out_argmax = arg;
}

llm_build_gemma4_mtp::llm_build_gemma4_mtp(
        const llama_model & target_model,
        const llama_model & mtp_model,
        const llm_graph_params & params) :
        llm_graph_context(graph_params_for_mtp(params, mtp_model)),
        target(target_model),
        mtp(mtp_model) {
    const int64_t n_bb = mtp.hparams.n_embd_backbone;
    GGML_ASSERT(n_bb > 0);
    GGML_ASSERT(mtp.mtp_pre_projection != nullptr && mtp.mtp_post_projection != nullptr);

    // Single-step MTP build. Async pipeline (see plan async-mtp-pipeline) calls this once
    // per draft step on a dedicated worker thread + dedicated ggml_backend_sched.
    ggml_tensor * inp_tok = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_input(inp_tok);
    cb(inp_tok, "mtp_inp_last_token", -1);

    ggml_tensor * inp_h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_bb, 1);
    ggml_set_input(inp_h);
    cb(inp_h, "mtp_inp_h_prev", -1);

    {
        auto inp_wrap = std::make_unique<llm_graph_input_mtp>();
        inp_wrap->inp_last_token = inp_tok;
        inp_wrap->inp_h_prev     = inp_h;
        res->add_input(std::move(inp_wrap));
    }

    ggml_tensor * inp_pos = build_inp_pos();
    auto *        inp_attn = build_attn_inp_kv_iswa();

    ggml_tensor * logits = nullptr;
    ggml_tensor * h_post = nullptr;
    ggml_tensor * arg    = nullptr;
    gemma4_mtp_build_one_step(*this, target, mtp, inp_attn,
            inp_tok, inp_h, inp_pos, &logits, &h_post, &arg);

    res->t_embd   = h_post;
    res->t_logits = logits;
    res->t_argmax = arg;

    ggml_build_forward_expand(gf, arg);
    ggml_build_forward_expand(gf, h_post);
}

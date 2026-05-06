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
    GGML_ASSERT(!mtp.hparams.use_ordered_embeddings && "ordered embeddings (centroid head) not implemented in MTP graph yet");

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

    ggml_tensor * tok_e = ggml_get_rows(ctx0, target.tok_embd, inp_tok);
    cb(tok_e, "mtp_tgt_tok_embd", -1);

    // Gemma 4 scales token embeddings by sqrt(hidden_size) at the input pipeline
    // (see gemma4-iswa.cpp). The MTP head was trained on the same scaled embeddings
    // before the pre_projection, so apply the matching scale here.
    tok_e = ggml_scale(ctx0, tok_e, sqrtf((float) n_bb));
    cb(tok_e, "mtp_tgt_tok_embd_scaled", -1);

    ggml_tensor * inp_cat = ggml_concat(ctx0, tok_e, inp_h, 0);
    cb(inp_cat, "mtp_concat", -1);

    ggml_tensor * inpL = build_lora_mm(mtp.mtp_pre_projection, inp_cat);
    cb(inpL, "mtp_pre_proj_out", -1);

    ggml_build_forward_expand(gf, inpL);

    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv_iswa();

    ggml_tensor * cur = nullptr;

    for (int il = 0; il < n_layer; ++il) {
        const int64_t n_embd_head = hparams.n_embd_head_k(il);
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_v(il));

        const int64_t n_head = hparams.n_head(il);

        const float freq_base_l  = mtp.get_rope_freq_base(cparams, il);
        const float freq_scale_l = mtp.get_rope_freq_scale(cparams, il);
        const int   n_rot_l      = hparams.n_rot(il);

        cur = build_norm(inpL, mtp.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_tensor * freq_factors = nullptr;
        if (!hparams.is_swa(il)) {
            freq_factors = mtp.layers[il].rope_freqs;
        }

        ggml_tensor * Qcur = build_lora_mm(mtp.layers[il].wq, cur);
        cb(Qcur, "Qcur", il);

        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);

        Qcur = build_norm(Qcur, mtp.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
        cb(Qcur, "Qcur_normed", il);

        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, freq_factors, n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
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

        cur = build_attn_mtp(inp_attn, mtp.layers[il].wo, nullptr, Qcur, nullptr, nullptr, nullptr,
                hparams.f_attention_scale, il, il_kv, read_swa, kv_embd_head_v, kv_n_head_v, use_k_as_v);

        cur = build_norm(cur, mtp.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        ggml_tensor * attn_out = ggml_add(ctx0, cur, inpL);
        cb(attn_out, "attn_out", il);

        GGML_ASSERT(mtp.layers[il].ffn_gate_inp == nullptr && "gemma4_assistant MTP does not support MoE FFN");

        cur = build_norm(attn_out, mtp.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                mtp.layers[il].ffn_up,   nullptr, nullptr,
                mtp.layers[il].ffn_gate, nullptr, nullptr,
                mtp.layers[il].ffn_down, nullptr, nullptr,
                nullptr,
                LLM_FFN_GELU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = build_norm(cur, mtp.layers[il].ffn_post_norm, nullptr, LLM_NORM_RMS, -1);
        cb(cur, "ffn_post_norm", il);

        cur = ggml_add(ctx0, cur, attn_out);

        if (mtp.layers[il].out_scale) {
            cur = ggml_mul(ctx0, cur, mtp.layers[il].out_scale);
            cb(cur, "out_scaled", il);
        }

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur, mtp.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    ggml_tensor * h_inner = cur;

    ggml_tensor * backbone = build_lora_mm(mtp.mtp_post_projection, h_inner);
    cb(backbone, "mtp_post_proj_out", -1);
    res->t_embd = backbone;

    cur = build_lora_mm(mtp.tok_embd, h_inner);

    if (hparams.f_final_logit_softcapping) {
        cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
        cur = ggml_tanh(ctx0, cur);
        cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);
    }

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

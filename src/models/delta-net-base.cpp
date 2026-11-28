#include "models.h"

#include "llama-impl.h"
#include "llama-memory-recurrent.h"

llm_build_delta_net_base::llm_build_delta_net_base(const llm_graph_params & params) : llm_graph_context(params) {}

std::pair<ggml_tensor *, ggml_tensor *> llm_build_delta_net_base::build_delta_net(
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * g,
        ggml_tensor * b,
        ggml_tensor * s) {
    const int64_t S_k      = q->ne[0];
    const int64_t H_k      = q->ne[1];
    const int64_t n_tokens = q->ne[2];
    const int64_t n_seqs   = q->ne[3];

    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];

    GGML_ASSERT(S_k == S_v);
    GGML_ASSERT(H_v % H_k == 0);

    GGML_ASSERT(q->ne[0] == S_k && q->ne[1] == H_k && q->ne[2] == n_tokens && q->ne[3] == n_seqs);
    GGML_ASSERT(k->ne[0] == S_k && k->ne[1] == H_k && k->ne[2] == n_tokens && k->ne[3] == n_seqs);
    GGML_ASSERT(v->ne[0] == S_v && v->ne[1] == H_v && v->ne[2] == n_tokens && v->ne[3] == n_seqs);

    GGML_ASSERT(g->ne[0] == 1   || g->ne[0] == S_v);
    GGML_ASSERT(                   g->ne[1] == H_v && g->ne[2] == n_tokens && g->ne[3] == n_seqs);
    GGML_ASSERT(b->ne[0] == 1   && b->ne[1] == H_v && b->ne[2] == n_tokens && b->ne[3] == n_seqs);
    GGML_ASSERT(s->ne[0] == S_v && s->ne[1] == S_v && s->ne[2] == H_v      && s->ne[3] == n_seqs);

    // K=1: output carries the final state only. state s is 4D [S_v, S_v, H_v, n_seqs].
    ggml_tensor * result = ggml_gated_delta_net(ctx0, q, k, v, g, b, s, /*K=*/1, /*emit_mode=*/0);
    ggml_tensor * output = ggml_view_4d(ctx0, result,
            S_v, H_v, n_tokens, n_seqs,
            ggml_row_size(result->type, S_v),
            ggml_row_size(result->type, S_v * H_v),
            ggml_row_size(result->type, S_v * H_v * n_tokens), 0);

    ggml_tensor * new_state = ggml_view_4d(ctx0, result,
            S_v, S_v, H_v, n_seqs,
            ggml_row_size(result->type, S_v),
            ggml_row_size(result->type, S_v * S_v),
            ggml_row_size(result->type, S_v * S_v * H_v),
            ggml_row_size(result->type, S_v * H_v * n_tokens * n_seqs));

    return {output, new_state};
}

ggml_tensor * llm_build_delta_net_base::build_conv_state(
        llm_graph_input_rs * inp,
        ggml_tensor *        conv_states_all,
        ggml_tensor *        qkv_mixed,
        int64_t              conv_kernel_size,
        int64_t              conv_channels,
        int                  il) {
    const int64_t n_seqs = ubatch.n_seqs;

    // gdn_replay rewinds the recurrent state by replaying ingredients, so the rollback depth is
    // recorded in the memory context instead of the rs index. The conv state has no replay path:
    // it keeps its (1 + n_rs_seq) rollback groups and is selected by the rs index. With the rs
    // index pinned at 0 the conv window would still contain the rejected draft tokens, so pick
    // the group that matches the recorded rollback depth.
    ggml_tensor * conv_src = conv_states_all;

    const uint32_t conv_rollback = inp->mctx->get_replay_len();
    if (conv_rollback > 0) {
        GGML_ASSERT((uint32_t) conv_states_all->ne[1] >= (conv_rollback + 1) * inp->mctx->get_size());
        conv_src = ggml_view_2d(ctx0, conv_states_all,
                conv_states_all->ne[0], inp->mctx->get_size(),
                conv_states_all->nb[1],
                (size_t) conv_rollback * inp->mctx->get_size() * conv_states_all->nb[1]);
    }

    ggml_tensor * conv_states = build_rs(inp, conv_src, hparams.n_embd_r(), n_seqs);
    cb(conv_states, "conv_states", il);

    conv_states = ggml_reshape_3d(ctx0, conv_states, conv_kernel_size - 1, conv_channels, n_seqs);
    cb(conv_states, "conv_states_reshaped", il);

    qkv_mixed = ggml_transpose(ctx0, qkv_mixed);
    cb(qkv_mixed, "qkv_mixed_transposed", il);

    ggml_tensor * conv_input = ggml_concat(ctx0, conv_states, qkv_mixed, 0);
    cb(conv_input, "conv_input", il);

    build_conv_state_store(inp, conv_states_all, conv_input, conv_kernel_size - 1, 0);

    return conv_input;
}

ggml_tensor * llm_build_delta_net_base::build_recurrent_attn(
        llm_graph_input_rs * inp,
        ggml_tensor *        ssm_states_all,
        ggml_tensor *        q,
        ggml_tensor *        k,
        ggml_tensor *        v,
        ggml_tensor *        g,
        ggml_tensor *        b,
        ggml_tensor *        s,
        int                  il) {
    const auto * mctx_cur   = inp->mctx;
    const auto   kv_head    = mctx_cur->get_head();
    const uint32_t mem_size = mctx_cur->get_size();

    const int64_t S_v          = s->ne[0];
    const int64_t H_v          = s->ne[2];
    const int64_t n_seqs       = s->ne[3];
    const int64_t n_seq_tokens = q->ne[2];

    const bool keep = cparams.n_rs_seq > 0;

    if (!keep) {
        auto attn_out = build_delta_net(q, k, v, g, b, s);
        ggml_tensor * output    = attn_out.first;
        ggml_tensor * new_state = attn_out.second;
        cb(output, "attn_output", il);
        cb(new_state, "new_state", il);

        ggml_build_forward_expand(gf,
                ggml_cpy(ctx0, new_state,
                    ggml_view_2d(ctx0, ssm_states_all, hparams.n_embd_s(), n_seqs, ssm_states_all->nb[1],
                        kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all))));

        return output;
    }

    ggml_tensor * ingr_all = mctx_cur->get_ingr_l(il);
    const bool gdn_replay = ingr_all != nullptr;

    if (!gdn_replay) {
        const int64_t D = S_v * S_v * H_v;
        const int64_t K = cparams.n_rs_seq + 1;

        // state s is 4D [S_v, S_v, H_v, n_seqs]; K snapshot slots are written into the output.
        ggml_tensor * gdn_out = ggml_gated_delta_net(ctx0, q, k, v, g, b, s, K, /*emit_mode=*/0);
        const int64_t attn_score_elems    = S_v * H_v * n_seq_tokens * n_seqs;
        const int64_t state_size_per_snap = S_v * S_v * H_v * n_seqs;

        ggml_tensor * output = ggml_view_4d(ctx0, gdn_out,
            S_v, H_v, n_seq_tokens, n_seqs,
            ggml_row_size(gdn_out->type, S_v),
            ggml_row_size(gdn_out->type, S_v * H_v),
            ggml_row_size(gdn_out->type, S_v * H_v * n_seq_tokens),
            0);
        cb(output, "attn_output", il);

        const size_t row_size = hparams.n_embd_s() * ggml_element_size(ssm_states_all);

        // op writes the last min(n_seq_tokens, K) snapshots; trailing slots are left unwritten
        const int64_t n_written = std::min<int64_t>(n_seq_tokens, K);

        // write the produced snapshots into the recurrent cache (snapshot slot i -> rollback group i)
        ggml_tensor * src = ggml_view_3d(ctx0, gdn_out,
            D, n_seqs, n_written,
            ggml_row_size(gdn_out->type, D),
            ggml_row_size(gdn_out->type, state_size_per_snap),
            ggml_row_size(gdn_out->type, attn_score_elems));

        ggml_tensor * dst = ggml_view_3d(ctx0, ssm_states_all,
            D, n_seqs, n_written,
            ssm_states_all->nb[1],
            (size_t) mem_size * row_size,
            (size_t) kv_head * row_size);

        ggml_build_forward_expand(gf, ggml_cpy(ctx0, src, dst));

        return output;
    }

    // --- DRC phase 2: gdn_replay path ---
    //
    // s holds the OPTIMISTIC final state (as if every token in the last decode's batch got
    // accepted). Delta-net's rank-1 update has no inverse: once a rollback is discovered, s
    // already holds the (now known wrong) state that ran past the rejected tail. Replaying
    // ingredients only moves the state forward, so the checkpoint to replay from must predate
    // the whole uncertain trailing window -- that is s_ckpt, the state as of n_rs_seq tokens
    // before the end of the last decode (the window's oldest edge).
    ggml_tensor * ckpt_all = mctx_cur->get_s_ckpt_l(il);
    GGML_ASSERT(ckpt_all != nullptr);

    ggml_tensor * s_ckpt = build_rs(inp, ckpt_all, hparams.n_embd_s(), n_seqs);
    s_ckpt = ggml_reshape_4d(ctx0, s_ckpt, S_v, S_v, H_v, n_seqs);

    const uint32_t n_rs_seq   = cparams.n_rs_seq;
    const uint32_t replay_len = mctx_cur->get_replay_len();
    GGML_ASSERT(replay_len <= n_rs_seq);

    ggml_tensor * base_state = s; // no rollback pending: the optimistic state is already correct

    if (replay_len > 0) {
        // Replay the accepted (n_rs_seq - replay_len) ingredients forward from s_ckpt in ONE
        // batched call. This works because emit_mode=1 stores ingredients in CHRONOLOGICAL order
        // (slot 0 = oldest of the K retained tokens), so the accepted prefix is exactly slots
        // [0, m) -- a plain forward-order view, no reversal needed.
        const uint32_t m             = n_rs_seq - replay_len;
        const bool     kda           = (g->ne[0] == S_v);
        const size_t   ingr_elemsize = ggml_element_size(ingr_all);
        const size_t   ingr_row_size = (size_t) hparams.n_embd_s_ingredient() * ingr_elemsize;
        const size_t   slot0_off     = (size_t) kv_head * ingr_row_size; // slot 0 = oldest accepted

        // ggml_gated_delta_net requires g/beta (and, transitively via q_dummy, q) to be fully
        // contiguous; these are strided views into the packed per-head ingredient ring, so
        // materialize them.
        ggml_tensor * k_batch = ggml_cont(ctx0, ggml_view_4d(ctx0, ingr_all, S_v, H_v, m, n_seqs,
            4 * S_v * ingr_elemsize, mem_size * ingr_row_size, ingr_all->nb[1],
            slot0_off));
        ggml_tensor * v_batch = ggml_cont(ctx0, ggml_view_4d(ctx0, ingr_all, S_v, H_v, m, n_seqs,
            4 * S_v * ingr_elemsize, mem_size * ingr_row_size, ingr_all->nb[1],
            slot0_off + S_v * ingr_elemsize));
        ggml_tensor * g_batch = ggml_cont(ctx0, ggml_view_4d(ctx0, ingr_all, kda ? S_v : 1, H_v, m, n_seqs,
            4 * S_v * ingr_elemsize, mem_size * ingr_row_size, ingr_all->nb[1],
            slot0_off + 2 * S_v * ingr_elemsize));
        ggml_tensor * beta_batch = ggml_cont(ctx0, ggml_view_4d(ctx0, ingr_all, 1, H_v, m, n_seqs,
            4 * S_v * ingr_elemsize, mem_size * ingr_row_size, ingr_all->nb[1],
            slot0_off + 3 * S_v * ingr_elemsize));
        // q does not affect the state, only the (discarded) attention output
        ggml_tensor * q_dummy = ggml_scale(ctx0, k_batch, 0.0f);

        ggml_tensor * replay_out = ggml_gated_delta_net(ctx0, q_dummy, k_batch, v_batch, g_batch, beta_batch, s_ckpt, /*K=*/1, /*emit_mode=*/0);
        base_state = ggml_view_4d(ctx0, replay_out, S_v, S_v, H_v, n_seqs,
            ggml_row_size(replay_out->type, S_v),
            ggml_row_size(replay_out->type, S_v * S_v),
            ggml_row_size(replay_out->type, S_v * S_v * H_v),
            (S_v * H_v * m * n_seqs) * ggml_element_size(replay_out));
    }

    // main call: emit_mode=1 records ingredients (+ the trailing final-state block) instead of
    // K full snapshots.
    const int64_t K = n_rs_seq;
    ggml_tensor * gdn_out = ggml_gated_delta_net(ctx0, q, k, v, g, b, base_state, K, /*emit_mode=*/1);

    const int64_t attn_score_elems   = S_v * H_v * n_seq_tokens * n_seqs;
    const int64_t ingr_size_per_snap = 4 * S_v * H_v * n_seqs;
    const int64_t ingr_elems_total   = K * ingr_size_per_snap;
    const int64_t n_written          = std::min<int64_t>(n_seq_tokens, K);

    ggml_tensor * output = ggml_view_4d(ctx0, gdn_out,
        S_v, H_v, n_seq_tokens, n_seqs,
        ggml_row_size(gdn_out->type, S_v),
        ggml_row_size(gdn_out->type, S_v * H_v),
        ggml_row_size(gdn_out->type, S_v * H_v * n_seq_tokens),
        0);
    cb(output, "attn_output", il);

    const size_t ingr_row_size_bytes  = (size_t) hparams.n_embd_s_ingredient() * ggml_element_size(ingr_all);
    const size_t state_row_size_bytes = (size_t) hparams.n_embd_s() * ggml_element_size(ssm_states_all);

    // scatter the K produced ingredient slots into the ring (slot i -> rollback group i, same
    // most-recent-first convention as the old K-snapshot scatter).
    ggml_tensor * ingr_src = ggml_view_3d(ctx0, gdn_out,
        4 * S_v * H_v, n_seqs, n_written,
        ggml_row_size(gdn_out->type, 4 * S_v * H_v),
        ggml_row_size(gdn_out->type, ingr_size_per_snap),
        ggml_row_size(gdn_out->type, attn_score_elems));

    ggml_tensor * ingr_dst = ggml_view_3d(ctx0, ingr_all,
        4 * S_v * H_v, n_seqs, n_written,
        ingr_all->nb[1],
        (size_t) mem_size * ingr_row_size_bytes,
        (size_t) kv_head * ingr_row_size_bytes);
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, ingr_src, ingr_dst));

    // scatter the trailing final-state block into s (single row, no widening).
    ggml_tensor * final_state = ggml_view_2d(ctx0, gdn_out, S_v * H_v, S_v * n_seqs,
        ggml_row_size(gdn_out->type, S_v * H_v),
        (attn_score_elems + ingr_elems_total) * ggml_element_size(gdn_out));
    ggml_tensor * final_dst = ggml_view_2d(ctx0, ssm_states_all, hparams.n_embd_s(), n_seqs,
        ssm_states_all->nb[1], (size_t) kv_head * state_row_size_bytes);
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0, ggml_reshape_2d(ctx0, final_state, hparams.n_embd_s(), n_seqs), final_dst));

    // update the oldest-edge-of-window checkpoint for the NEXT decode's replay base.
    //
    // The verify batch is always one token longer than the retained window (n_draft + 1 vs
    // n_rs_seq = n_draft), so this branch fires on every decode. The main call above captures
    // the needed state as a fixed-cost trailing block (ggml.h's emit_mode == 1 contract), so no
    // second kernel launch is needed.
    if (n_seq_tokens > (int64_t) n_rs_seq) {
        ggml_tensor * new_ckpt = ggml_view_2d(ctx0, gdn_out, S_v * H_v, S_v * n_seqs,
            ggml_row_size(gdn_out->type, S_v * H_v),
            (attn_score_elems + ingr_elems_total + S_v * S_v * H_v * n_seqs) * ggml_element_size(gdn_out));
        ggml_tensor * ckpt_dst = ggml_view_2d(ctx0, ckpt_all, hparams.n_embd_s(), n_seqs,
            ckpt_all->nb[1], (size_t) kv_head * state_row_size_bytes);
        ggml_build_forward_expand(gf,
            ggml_cpy(ctx0, ggml_reshape_2d(ctx0, new_ckpt, hparams.n_embd_s(), n_seqs), ckpt_dst));
    } else {
        // batch shorter than the retained window: base_state is the state at (end - n_seq_tokens),
        // i.e. INSIDE the uncertain window, so a later rollback deeper than n_seq_tokens would
        // replay from a too-recent checkpoint. Unreachable with the current verify-batch shape
        // (n_draft + 1 > n_draft == n_rs_seq); the warning catches it if that ever changes.
        if (n_seq_tokens < (int64_t) n_rs_seq) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                LLAMA_LOG_WARN("%s: gdn_replay: checkpoint taken with n_seq_tokens (%d) < n_rs_seq (%u); "
                               "a rollback deeper than %d tokens would be incorrect\n",
                               __func__, (int) n_seq_tokens, n_rs_seq, (int) n_seq_tokens);
            }
        }
        ggml_tensor * ckpt_dst = ggml_view_2d(ctx0, ckpt_all, hparams.n_embd_s(), n_seqs,
            ckpt_all->nb[1], (size_t) kv_head * state_row_size_bytes);
        ggml_build_forward_expand(gf,
            ggml_cpy(ctx0, ggml_reshape_2d(ctx0, base_state, hparams.n_embd_s(), n_seqs), ckpt_dst));
    }

    return output;
}

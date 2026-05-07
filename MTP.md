# Gemma 4 MTP — Multi-Token Prediction speculative decoding

> Scope: this document covers the MTP (Multi-Token Prediction) speculative
> decoding path added to this fork on top of `llama.cpp`, currently specialised
> to **Gemma 4** targets (`gemma4`) paired with the official **Gemma 4
> assistant** drafter (`gemma4_assistant`).

It is a self-contained reference: how the feature is built (model, graph, KV,
context, scheduler), what server-loop integration looks like, what knobs the
operator has, what the recent design decisions were, and where the throughput
numbers came from.

For the public/user-facing section about CLI flags and `--spec-type mtp`, see
also `docs/speculative.md`.

---

## 1. What MTP is here

Gemma 4 ships an "assistant" model — a small transformer head that consumes the
**target's last hidden state** (backbone output `h_prev`) plus the last sampled
**token id** and predicts the next token in one forward step. Chained across
`B - 1` steps inside one MTP graph, it produces a draft block of length `B - 1`
that is then verified by the target in a single batched decode.

Conceptually it is a draft-model speculation, but with three crucial twists:

1. **Single context.** The assistant is **not** a second `llama_context`. Its
   weights live next to the target in `llama_model::mtp_assistant`. There is no
   second tokenizer, no second KV cache, no second sampler.
2. **Cross-attention into the target's KV.** Each MTP layer reads `K/V` from
   the **last layer of the matching attention type** (full / sliding) of the
   target's KV cache (`llama_kv_cache_iswa::init_mtp` → `mtp_slot_info`). No
   draft-side KV is allocated.
3. **Shared `h_prev` from the target.** The assistant ingests the target's
   per-token backbone hidden state (`embeddings_ith`) for the **last accepted**
   position. Embeddings must therefore stay enabled on the target context
   (`llama_set_embeddings(ctx_tgt, true)`).

This makes MTP much cheaper than a normal "small draft model" approach: there
is essentially no draft KV, no second model orchestration, and the per-step
graph is tiny (4 transformer blocks for 26B/31B; centroid LM head for E2B/E4B).

---

## 2. Components and where they live

| Concern | File(s) |
|---|---|
| MTP graph (per-step build) | `src/models/gemma4-assistant.cpp` |
| Model arch + tensor types | `src/llama-arch.cpp/.h`, `src/llama-model.cpp` |
| GGUF tensor mapping | `gguf-py/gguf/tensor_mapping.py`, `gguf-py/gguf/constants.py`, `convert_hf_to_gguf.py` |
| Loading assistant into target | `src/llama.cpp::llama_model_load_mtp_from_file` |
| MTP scheduler + worker + APIs | `src/llama-context.cpp/.h` (`sched_mtp`, `mtp_worker_loop`, `decode_mtp_*`) |
| KV cross-attention helpers | `src/llama-kv-cache.cpp`, `src/llama-kv-cache-iswa.cpp` (`init_mtp`) |
| Speculative driver / overlap | `common/speculative.cpp` (`common_speculative_state_mtp`) |
| Server integration | `tools/server/server-context.cpp` |
| Public C API | `include/llama.h` (`llama_decode_mtp_async/_wait`, `llama_model_load_mtp_from_file`, `llama_model_mtp_n_embd_backbone`, …) |
| Verification helper | `scripts/verify-gemma4-assistant-gguf.py` |
| Run scripts | `scripts/run-gemma4-{,e2b-,e4b-,31b-}mtp-server.sh`, `scripts/quantize-gemma4-edge-assistant-mtp.sh` |
| Tests | `tests/test-speculative-mtp.cpp` |
| Tracing | `LLAMA_MTP_ACC_TRACE` (NDJSON in `common/speculative.cpp`) |

---

## 3. Model side: assistant, centroid LM head, GGUF layout

The assistant is loaded from its own GGUF and **attached** to a target model:

```c
int32_t llama_model_load_mtp_from_file(
        struct llama_model      * model,
        const char              * path_assistant,
        struct llama_model_params mparams);
```

After load, the target carries:

- `model.mtp_assistant` — a fully-loaded `llama_model` with arch
  `gemma4_assistant` (4 transformer blocks, the pre/post backbone projections,
  optional centroid head).
- `hparams.n_embd_backbone` — must equal target's backbone hidden size; this is
  asserted at load and re-checked at draft time (`n_bb` in
  `decode_mtp_run`).

CLI surface (`common/arg.cpp`, `common/common.cpp`):

- `--mtp-head <path>` (preferred) and `--model-draft / -md` (back-compat alias)
  — feed the same `mparams_dft.path` field.
- `--draft-block-size <B>` — head proposes `B - 1` tokens per round.
- `--gpu-layers-draft / -ngld`, `-ctkd / -ctvd` — placement and KV typing for
  the **assistant** weights when offloaded.

### Centroid / ordered-embeddings LM head (E2B / E4B)

For Edge models (`use_ordered_embeddings = true` in HF config), the LM head is
not the dense tied embedding but a **MaskedEmbedder**:

1. `centroid_logits = mul_mat(mtp.centroids, h)` → `[n_centroids]`.
2. `top_k(centroid_logits, centroid_intermediate_top_k)` → `top_k` centroid ids
   (I32, on-device).
3. `mtp.token_ordering` is viewed as `[vsc, n_centroids]`
   (`vsc = n_vocab / n_centroids`); each centroid column lists `vsc` candidate
   token ids. `get_rows` gathers the candidate ids for the chosen centroids.
4. `get_rows(token_embd, ids)` then `mul_mat(·, h)` produces sparse logits over
   only those candidates.
5. We **scatter** them into a full `[n_vocab]` row pre-filled with `-1e30`
   (`ggml_fill_inplace + ggml_set_rows`). The full-vocab row is what the
   verifier expects — sparse-only argmax broke server accept (rare-token edge
   cases) and so was reverted.

GGUF layout for the centroid head (see `convert_hf_to_gguf.py` and
`docs/development/gemma4-assistant-tensor-inventory.md`):

| Tensor | Stored type | On-disk shape | Notes |
|---|---|---|---|
| `mtp.centroids.weight` | F16/F32 (or quant) | `[n_embd, n_centroids]` after GGUF dim packing | numpy is `[n_centroids, n_embd]`; written as-is so loader sees `mul_mat`-compatible shape |
| `mtp.token_ordering.weight` | **I32** (kept integer end-to-end) | `[n_vocab]` | must not be quantized — the converter explicitly preserves I32 |
| `mtp.pre_projection.weight` / `mtp.post_projection.weight` | as model | `[2*n_embd, n_embd]` / `[n_embd, n_embd_backbone]` | concatenates `[token_embd, h_prev]` then projects back |

The verifier `scripts/verify-gemma4-assistant-gguf.py` enforces these shape /
dtype invariants and is run automatically by every `run-gemma4-*-mtp-server.sh`
script (skip with `VERIFY_ASSISTANT_GGUF=0`).

---

## 4. Per-step MTP graph (`gemma4-assistant.cpp`)

`llm_build_gemma4_mtp` builds a single-token, single-sequence graph (`n_tokens
= 1`, `n_seqs = 1`, `n_outputs = 1`):

1. Inputs (registered as `llm_graph_input_mtp`):
   - `inp_last_token : I32 [1]`
   - `inp_h_prev     : F32 [n_embd_backbone, 1]`
   - `inp_pos`       (standard `build_inp_pos`)
   - `inp_attn`      (`build_attn_inp_kv_iswa`)
2. Token embedding from the **target's** `tok_embd`, then scaled by
   `sqrt(n_embd)` to mirror Gemma 4's input scaling.
3. `concat([tok_e, h_prev], axis=0) → mtp.pre_projection` (collapses the
   `2 * n_embd` channel back to `n_embd`).
4. 4 transformer blocks (`mtp.layers[il]`):
   - RMSNorm → Q proj → Q-norm → RoPE.
   - **Cross-attention** via `build_attn_mtp`: queries from MTP, K/V fetched
     from the target's KV cache at `il_kv = last layer in target with the same
     attention type` (SWA / full).
   - Per HF Gemma 4 quirk (`attention_k_eq_v: true`): even when V was derived
     from K, the V slot is **written** with rms-norm-without-scale and
     un-rotated, so cross-attn must always read V from cache (not reuse the
     post-RoPE K). This is encoded as `use_k_as_v = false`.
   - Standard residual + post-norm + GELU FFN + post-FFN norm + per-layer
     `out_scale` (if present) + `build_cvec`.
5. Final RMSNorm → `mtp.post_projection` produces the **next-step `h_prev`**
   (this is what the host stitches between steps).
6. LM head — dense (tied) **or** centroid-routed for ordered embeddings.
7. Optional `f_final_logit_softcapping`.
8. **In-graph greedy argmax**: `ggml_argmax(cur)` → `I32 [1]`. The final result
   exposes three tensors via `llm_graph_result`:
   - `t_embd`   = `h_post`   (next `h_prev`)
   - `t_logits` = full-vocab row (kept for diagnostic / `out_logits` API)
   - `t_argmax` = greedy token id

The host reads `t_argmax` (4 bytes) per step instead of pulling the full
F32 `[n_vocab]` row across a backend boundary and running CPU argmax. On
Gemma 4 + Q4_K_XL this alone delivered ~+2-3% throughput
(109.5 → 112.5 tps at n=128; 95.8 → 97.8 tps at n=512), with bit-identical
greedy drafts. The full row is still computed in-graph and is fetched on
demand by passing `out_logits != NULL` to the synchronous
`llama_decode_mtp(...)` API (legacy / diagnostic path), which transparently
falls back to `decode_mtp_sync`.

---

## 5. KV sharing — what is read, what is appended

The MTP step does **not** allocate or write its own KV. It reads the target's
KV by:

- `kv_iswa->init_mtp(seq_id, ub)` — produces a memory context whose attention
  inputs (`build_attn_inp_kv_iswa`) wire the cross-attn to the target slot for
  `seq_id`, with a mask that admits all positions `≤ attn_pos`.
- `attn_pos` is taken from `llama_memory_seq_pos_max(mem, seq_id)` immediately
  before submission (post-`seq_rm`). All `n_steps` step positions chosen for
  RoPE are strictly **`> attn_pos`**, so the causal/SWA mask uniformly admits
  every target cell — that is why a single mask suffices for the whole chained
  draft.

KV-safety contract for asynchronous draft work (see Section 7):

- `decode_mtp_async` snapshots `h_prev` and `attn_pos` at submit time.
- The target may **append** at positions `> attn_pos` between submit and
  `_wait` (this is what the verify decode does), but it must not evict, rewrite
  or `seq_rm` cells at positions `≤ attn_pos` until `_wait` returns.
- The current append-only KV cache satisfies this. `common_speculative_cancel`
  is invoked at the few server-loop points that *do* mutate KV destructively
  (request stop / release; new request `seq_rm`; spec-disabled iterations).

---

## 6. `llama_context` plumbing — `sched_mtp`, worker, and APIs

The async pipeline lives in `src/llama-context.cpp/.h`. New members on the
context (Phase C of the original plan):

- `sched_mtp` — a **dedicated** `ggml_backend_sched` for MTP. Created lazily
  by `ensure_sched_mtp()` and reserved with a single-token MTP graph (the MTP
  graph is invariant in size: `n_tokens = 1`, `n_seqs = 1`, `n_outputs = 1`,
  so one reserve covers all subsequent calls).
- `gf_res_prev_mtp` — a **separate** `llm_graph_result` cache. This is the key
  reason MTP graph reuse survives target-decode resets, and is the single
  largest win of the async refactor.
- A worker thread (`mtp_worker`), `std::mutex` + 2 condition variables
  (`mtp_cv_request`, `mtp_cv_response`), and request/response slots
  (`std::optional<mtp_request>`, `bool mtp_in_flight`, `std::optional<mtp_response>`).
- `backend_cfg_mu` — guards shared backend reconfiguration
  (`set_threadpool_fn`, `set_n_threads_fns`) so the worker's
  `graph_compute_mtp` cannot race the main thread's `graph_compute`. The lock
  is held only across cheap setters; the actual `graph_compute_async` calls run
  unlocked so target verify and MTP encode can interleave on each scheduler.

Public C APIs (`include/llama.h`):

```c
LLAMA_API int32_t llama_decode_mtp_async(
        struct llama_context * ctx,
        llama_seq_id  seq_id,
        llama_pos     attn_pos,
        llama_token   last_token,
        const float * h_prev,
        int32_t       n_steps);

LLAMA_API int32_t llama_decode_mtp_wait(
        struct llama_context * ctx,
        llama_token * out_drafts,
        float       * out_h_prev_last);

// Backward-compatible synchronous facade. If out_logits != NULL falls back to
// decode_mtp_sync (per-step logits captured in-thread).
LLAMA_API int32_t llama_decode_mtp(
        struct llama_context * ctx,
        llama_seq_id seq_id, llama_pos attn_pos,
        llama_token last_token, float * h_prev,
        int32_t n_steps,
        llama_token * out_drafts, float * out_logits, float * out_h_prev_last);
```

Contract:

- At most **one in-flight request per context**. `_async` while a previous
  request has not been `_wait`ed returns `-7`.
- `h_prev` is *copied* into the request → caller may free / reuse immediately.
- Drafts are written into `out_drafts[0..n_steps-1]`; the last `h_prev` is
  optionally copied into `out_h_prev_last`.

Worker loop (`mtp_worker_loop`): waits on `mtp_pending`, runs
`decode_mtp_run` (the per-step chain on `sched_mtp`), publishes
`mtp_completed`, and notifies. `decode_mtp_run` per step:

1. Build `llama_ubatch` `{token=last_token, embd=h, pos=attn_pos+1+k, output=0}`.
2. `kv_iswa->init_mtp(seq_id, ub)` → memory context.
3. `process_ubatch_mtp` → reuse cached graph if `can_reuse(gparams)` else
   rebuild + alloc.
4. `graph_compute_mtp` → `sched_mtp.graph_compute_async` → synchronize.
5. Read `t_argmax` (4 bytes) → `last_token = drafts[k]`; read `t_embd` →
   `h` (next `h_prev`).

On context destruction the worker is signalled via `mtp_worker_stop`, woken,
and joined before tearing down `sched_mtp`.

---

## 7. Speculative driver — pipeline depth-2

The host driver lives in `common/speculative.cpp ::
common_speculative_state_mtp`. Its job is to translate the server's
"draft / accept" loop into the right `_async / _wait` calls and to enforce the
KV-safety contract.

### Depth-2 overlap

The server normally goes:

```
  loop:
    drafts          = common_speculative_draft(...)        # produce drafts
    target_decode(...)                                     # verify drafts
    n_acc, sampled  = sample_and_accept_n(...)
    common_speculative_accept(spec, n_acc)
    seq_rm(...);  update slot.sampled / batch.dft index
```

Depth-2 inserts a `prepare_next` at the **end** of the iteration, after
`accept` and `seq_rm`:

```
    common_speculative_prepare_next(spec, slot.sampled)    # async submit
```

…which calls `llama_decode_mtp_async(...)` for the *next* round using:

- `attn_pos = seq_pos_max(seq_id)` (post `seq_rm`),
- the real sampled token `slot.sampled` (no optimistic guess),
- `h_prev = embeddings_ith(h_idx)` snapshotted right after sample/accept (see
  Section 8 for `h_idx`).

Then on the next iteration, `common_speculative_draft` checks
`has_pending`. If pending and `pending_n_steps == n_steps`, it goes "lazy":

```
    llama_decode_mtp_wait(...)   # blocks only on whatever is left of MTP
```

This **overlaps MTP draft compute with everything that happens between
`accept` and the next `draft`**: token I/O, OAI streaming, slot bookkeeping,
batching, the next prefill if any. The benefit is real because the MTP graph,
while small, is not free — it is `B - 1` sequential single-token decodes
through 4 layers + cross-attn + LM head.

When `n_steps` changes between iterations (e.g. on the last iteration of a
request), or when the target is about to mutate KV destructively, the driver
**drains** the in-flight request (`mtp_drain_pending_discard`) to keep the
`_async`/`_wait` invariant intact.

The depth-2 path can be A/B-tested at runtime by exporting
`LLAMA_PIPELINE_DEPTH2=0`, which turns `prepare_next` into a no-op and
restores depth-1 (sync `_async + _wait` inside `draft`).

### Drain points (server-side)

`tools/server/server-context.cpp` invokes `common_speculative_cancel` in three
places:

1. When the iteration **skips** speculative decoding (`n_remaining == 1`,
   `n_min` not satisfied, etc.) — otherwise the worker would compute against
   KV that is about to change in the upcoming target_decode (we observed Metal
   command-buffer status 3 on turbo3 KV before this guard).
2. After `send_final_response` / `slot.release` — the next request will
   `seq_rm` and overwrite cells the worker is still reading.
3. In `common_speculative_begin` (new prompt) — the previous generation's
   in-flight MTP must not bleed into the next prompt.

---

## 8. The `h_idx` correction

A subtle correctness issue: `embeddings_ith(-1)` returns the **last batch
output**, which after partial draft acceptance is the hidden state of a
**rejected** draft (computed for the wrong input). Feeding that as `h_prev`
collapses acceptance.

Fix: after `sample_and_accept_n` the server sets

```cpp
common_speculative_set_h_idx(slot.spec, slot.i_batch_dft[ids.size() - 1]);
```

i.e. it points the next MTP draft at the batch index of the **last accepted
token**. This is honored both in the sync `draft` path and in
`prepare_next` (Section 7).

---

## 9. Adaptive skip-streak

There are workloads (numbers, code, rare tokens deep in long generations) where
the MTP head is consistently wrong. Drafting still costs ~10 ms with no
accepted tokens. The driver detects this:

- `prev_n_acc_drafts` snapshot at the end of each `draft`.
- Increment `zero_accept_streak` when `n_acc_drafts` did not move since the
  previous call; reset on any non-empty accept.
- After `LLAMA_MTP_SKIP_STREAK_THRESHOLD` consecutive zero-accepts (1..32),
  return an empty draft for one batch (server falls back to a single-token
  verify), reset the streak, and let the next batch re-arm.
- `skip_streak_last_draft` prevents threshold=1 from oscillating into a
  permanent skip.

Off by default (env unset / `0`). The `MTP_PRESET=throughput` Edge presets do
**not** enable it either — turn it on per-deployment when the workload
warrants.

---

## 10. Diagnostic NDJSON tracer (`LLAMA_MTP_ACC_TRACE`)

Set `LLAMA_MTP_ACC_TRACE=1` (stderr) or `LLAMA_MTP_ACC_TRACE=/path/to.ndjson`
(append) to enable the `mtp_acc_tracer` in `common/speculative.cpp`. Off by
default at zero overhead (enabled-only L2 reduction over `n_bb` per draft).

Two events per iteration, paired by `iter`:

- `mtp_draft` — `iter`, `path` (`sync` / `lazy` / `skip-streak` / `skip-nsteps`),
  `seq_id`, `id_last`, `h_idx`, `attn_pos`, `n_steps`, `h_l2` (L2 norm of
  `h_prev`), and `drafts[]`.
- `mtp_accept` — `iter`, `n_accepted`, `n_drafted_prev`.

This is the recommended tool for any acceptance-rate debugging: per-position
acceptance, h_prev stability, `h_idx` selection bias, and depth-2 lazy/sync
distribution all fall out by joining on `iter`.

---

## 11. Operating it — scripts and presets

### Pre-built assistant GGUFs

Official Gemma 4 assistant heads, converted with this fork's
`convert_hf_to_gguf.py` (preserves I32 `mtp.token_ordering` for the
centroid-head Edge variants), are published as a Hugging Face collection:

> [AtomicChat / Gemma 4 Assistant GGUF](https://huggingface.co/collections/AtomicChat/gemma-4-assistant-gguf)
> — F16 / Q8_0 / Q5_K_M / **Q4_K_M** / Q4_K_S quantizations.

| Target | Assistant repo | Recommended quant |
|---|---|---|
| Gemma 4 E2B | [`AtomicChat/gemma-4-E2B-it-assistant-GGUF`](https://huggingface.co/AtomicChat/gemma-4-E2B-it-assistant-GGUF) | **Q4_K_M** |
| Gemma 4 E4B | [`AtomicChat/gemma-4-E4B-it-assistant-GGUF`](https://huggingface.co/AtomicChat/gemma-4-E4B-it-assistant-GGUF) | **Q4_K_M** |
| Gemma 4 26B-A4B | [`AtomicChat/gemma-4-26B-A4B-it-assistant-GGUF`](https://huggingface.co/AtomicChat/gemma-4-26B-A4B-it-assistant-GGUF) | **Q4_K_M** / Q4_K_S |
| Gemma 4 31B | [`AtomicChat/gemma-4-31B-it-assistant-GGUF`](https://huggingface.co/AtomicChat/gemma-4-31B-it-assistant-GGUF) | **Q4_K_M** / Q4_K_S |

Q4_K_M is the recommended default: throughput is identical to F16 in the
matrix bench (the head is small enough that bandwidth, not weight precision,
dominates), while VRAM/RAM footprint is ~4× lower. Drop to F16 only if you
are debugging an acceptance regression that you suspect is quant-related; the
verifier `scripts/verify-gemma4-assistant-gguf.py` will refuse to load a
malformed assistant GGUF in either case.

The repo helpers prefer a quantized assistant under `.scratch/` when one
exists (`gemma-{e2b,e4b,…}-assistant-mtp-Q4_K_M.gguf`) and fall back to F16
otherwise. Override with `DRAFT_GGUF=…` or pass `--mtp-head` directly.

### Run scripts

Helper scripts live under `scripts/`:

| Script | Target | Notes |
|---|---|---|
| `run-gemma4-mtp-server.sh` | gemma 4 26B | dense LM head; `MTP_PRESET` not used |
| `run-gemma4-31b-mtp-server.sh` | gemma 4 31B | dense LM head |
| `run-gemma4-e2b-mtp-server.sh` | gemma 4 E2B | centroid head; `MTP_PRESET` aware |
| `run-gemma4-e4b-mtp-server.sh` | gemma 4 E4B | centroid head; `MTP_PRESET` aware |
| `run-gemma4-server-turbo.sh` | dense baselines, no MTP | TurboQuant KV demo |
| `quantize-gemma4-edge-assistant-mtp.sh` | quantizer for E2B/E4B assistant | preserves I32 ordering |

Edge presets (`MTP_PRESET`):

| Preset | `--draft-block-size` (B) | `--draft-max` |
|---|---:|---:|
| `throughput` | 2 | 6 |
| `lift`       | 3 | 8 |
| `balanced`   | 3 | 8 |
| `quality`    | 4 | 16 |

Override directly with `DRAFT_BLOCK_SIZE`, `DRAFT_MAX`,
`LLAMA_MTP_SKIP_STREAK_THRESHOLD`. KV typing is taken from `CTK / CTV / CTKD /
CTVD` — both target and assistant inherit the same default (`turbo3`).

The bench harness `.scratch/bench-matrix.sh` runs the matrix
`{model} × {f16-base, turbo3-base, f16-mtp, turbo3-mtp} × {short=128, long=512}
× 3 runs` against `/v1/chat/completions` with `temperature=0`,
`cache_prompt=false` and `stream=false`, and reports median tps + mean
draft-accept rate.

---

## 12. Latest matrix benchmark (`.scratch/bench-logs/matrix-q4chat.log`)

Run on 2026-05-07. Q4_K_S assistant heads, draft-block defaults from each
script (`B = 3` for the dense scripts at the time of this run). `accept` is
`draft_n_accepted / draft_n` averaged over 3 runs; `tps` is the median.

### Bench host

| Component | Value |
|---|---|
| Machine | MacBook Pro (`Mac16,5`, MX313LL/A) |
| SoC | Apple **M4 Max** — 16 CPU cores (12P + 4E), 40-core GPU |
| Unified memory | 48 GB LPDDR5 |
| OS | macOS 26.3.1 (build 25D2128), Darwin 25.3.0 |
| llama.cpp backend | Metal (full GPU offload: `-ngl 99 -ngld 99`, `-fa on`) |
| Server | local `llama-server` over `127.0.0.1:8080` |
| Client | `python3 urllib` → `/v1/chat/completions`, `temperature=0`, `cache_prompt=false`, `stream=false` |
| Driver | `.scratch/bench-matrix.sh` (3 runs/cell, median tps, mean accept) |

Single-slot configuration (`--parallel 1 -np 1 --cont-batching`); no other
heavy GPU/CPU workloads were running on the host during the matrix sweep.

| model | mode | short tps (n=128) | long tps (n=512) | short accept | long accept |
|---|---|---:|---:|---:|---:|
| gemma-26B | f16-base    | 81.54 | 83.06 | — | — |
| gemma-26B | turbo3-base | 53.81 | 53.89 | — | — |
| gemma-26B | f16-mtp     | **109.49** | **95.75** | 85.9% | 68.9% |
| gemma-26B | turbo3-mtp  | 81.91 | 72.17 | 82.3% | 67.9% |
| gemma-31B | f16-base    | 14.15 | 15.20 | — | — |
| gemma-31B | turbo3-base | 15.79 | 14.82 | — | — |
| gemma-31B | f16-mtp     | **20.24** | **17.30** | 88.0% | 74.6% |
| gemma-31B | turbo3-mtp  | 18.67 | 15.68 | 87.0% | 70.8% |

Key observations:

- **f16 MTP**: +34 % short / +15 % long over baseline on 26B; +43 % short /
  +14 % long on 31B. Acceptance is dominated by short, "essay-y" prompts; long
  drafts hit the natural ceiling once content drifts into less predictable
  spans.
- **turbo3 MTP**: +52 % short / +34 % long over the turbo3 baseline on 26B
  (turbo3 baseline is slower than f16 because the gemma-26B target is
  compute-bound at `f16` and bandwidth-helped by turbo3 only when
  memory-bound; that asymmetry is not specific to MTP).
- **31B base inversion** (`turbo3-base 15.79 > f16-base 14.15` short): 31B is
  bandwidth-bound on this rig, so turbo3 KV beats f16 on the short cell. MTP
  still adds value on top of either KV typing.
- **Accept short > accept long** is consistent across the matrix: as decode
  drifts away from boilerplate phrasing, the assistant's drafts become less
  reliable and `B - 1` chained steps amplify the rejection.

### How we got here (history within this branch)

The matrix logs in `.scratch/bench-logs/` show the optimisation journey for the
gemma-26B `f16-mtp` short-prompt cell:

| Log (mtime, `ls -lt`) | Short tps | Long tps | Short accept | What changed |
|---|---:|---:|---:|---|
| `matrix-run2.log`   (01:26) | 70.89 | 76.79 | 55.5% | early async pipeline, sync wrapper |
| `matrix-old.log`    (01:41) | 61.88 | 63.98 | 50.0% | depth-1 sync MTP, `h_idx=-1` regression |
| `matrix-q4chat.log` (02:02) | **109.49** | 95.75 | **85.9%** | depth-2 + in-graph argmax + correct `h_idx` |
| `matrix-c-prime.log` (02:50, partial) | 112.30 | 96.69 | 85.9% | identical config, additional run sample |

The big jump (~62 → ~109 tps short) came from three independent fixes
landing together:

1. **`h_idx` correction** so MTP feeds the *accepted* hidden state instead of a
   rejected draft's output (acceptance jumps from ~50% to ~86%).
2. **Pipeline depth-2 overlap** so MTP work overlaps post-accept bookkeeping
   (steady ~+8% throughput at fixed accept).
3. **In-graph argmax** so the host transfers 4 bytes instead of `n_vocab × 4 B`
   per step (~+2-3% on top).

---

## 13. Trade-offs and gotchas

These are the non-obvious failure / regression modes you should keep in mind
when changing or extending this code.

**Embeddings on the target context.** MTP is meaningless without
`llama_set_embeddings(ctx_tgt, true)`. The server wires this conditionally per
batch (`need_embeddings = need_embd() || mtp_active`). If a future code path
flips embeddings off mid-generation, MTP will silently degrade to drafting
against zero `h_prev`.

**`h_idx` after partial accept.** Forgetting to call
`common_speculative_set_h_idx` after `sample_and_accept_n` regresses accept
rate to ~50 % on the same workload (matrix-old vs matrix-q4chat). Any new code
path that produces drafts must restore the correct batch index of the *last
accepted* token, not `-1`.

**KV append-only invariant.** Async MTP correctness depends on `attn_pos` cells
remaining stable until `_wait`. Any new operation that rewrites KV in place
(eviction, sliding-window compaction, retroactive `seq_rm` past `attn_pos`)
must call `common_speculative_cancel` first. The server has three explicit
drain points (Section 7); reuse them rather than inventing a fourth contract.

**Single in-flight request per context.** This is intentional — multiplexing
MTP across slots requires a sched-per-slot or a request queue with its own
graph-cache. Today a second `_async` returns `-7` and `prepare_next` is a
no-op when one is in flight. With `--parallel > 1` slots run on the same
context: the MTP overlap currently benefits only the slot whose `prepare_next`
won the race; the others fall back to sync. Lifting this is non-trivial
(graph-cache, scheduler, KV snapshot all need per-slot identity).

**`draft_block_size` vs. `draft_max`.** `draft_block_size` is the **MTP head's
block** (head emits `B - 1` tokens). `draft_max` is the standard llama.cpp
upper bound on draft length the server will accept. For Edge centroid heads
(heavier per-step), small `B` (2-3) usually wins; for the dense 26B/31B,
`B = 3` is the current sweet spot in the matrix bench.

**Centroid-head `top_k` cost.** Edge MTP runs `top_k` over `n_centroids` and a
routed `get_rows` per draft step. Greedy still materialises the full-vocab row
(masked-fill + scatter) so verify-side argmax stays consistent.
`use_ordered_embeddings` has measurably higher per-step cost than the dense
head; budget `B = 2` (`MTP_PRESET=throughput`) by default on Edge. The Edge
matrix cell is not yet in `matrix-q4chat.log` (the script lists `gemma-E4B`
in `MODELS`, but the row is not present — the GGUF was missing on the bench
host that day).

**Skip-streak hysteresis.** With `LLAMA_MTP_SKIP_STREAK_THRESHOLD=1` and
without `skip_streak_last_draft`, the driver would skip every other batch
forever as soon as one zero-accept happened. Keep that guard.

**Backend reconfiguration races.** `set_n_threads` / `set_threadpool` are
process-global on a backend. The `backend_cfg_mu` window in `graph_compute` /
`graph_compute_mtp` is intentionally tiny (only the setters, never the
`graph_compute_async` itself). Lengthening that critical section will block
the worker on every target step and erase the depth-2 win.

**Vocab compatibility for MTP is laxer than for `--spec-type draft`.** Target
chat templates own stop / EOS tokens; the MTP head only predicts next-token
ids. `common_speculative_are_compatible_mtp` therefore checks `vocab_type`,
size (within `SPEC_VOCAB_MAX_SIZE_DIFFERENCE`) and per-token text equality
from id ≥ 5, but **skips** bos/eos/add_bos/add_eos checks. Don't reuse this
loosened check for non-MTP draft pairings.

**Optimistic last token (future work).** Submitting `prepare_next` with a
guess of the next sampled token before sample/accept could hide one extra
`llama_decode` on hits. On misses we'd waste the entire MTP block. Not landed
— would need a clear measurement that hit-rate is high enough to justify the
miss cost on this workload.

---

## 14. Quick reference

```sh
# 26B + Q4 assistant, MTP on TurboQuant3 KV (matches matrix-q4chat row).
scripts/run-gemma4-mtp-server.sh

# E4B, throughput preset (B=2, max=6), centroid head, optional skip-streak.
LLAMA_MTP_SKIP_STREAK_THRESHOLD=4 \
MTP_PRESET=throughput \
scripts/run-gemma4-e4b-mtp-server.sh

# A/B-test depth-2 overlap vs sync at the same model/config:
LLAMA_PIPELINE_DEPTH2=0 scripts/run-gemma4-mtp-server.sh

# NDJSON acceptance trace to a file.
LLAMA_MTP_ACC_TRACE=/tmp/mtp.ndjson scripts/run-gemma4-mtp-server.sh

# Re-run the matrix bench (median over 3 runs per cell).
bash .scratch/bench-matrix.sh | tee .scratch/bench-logs/matrix-$(date +%H%M).log
```

Environment knobs:

| Var | Default | Effect |
|---|---|---|
| `LLAMA_PIPELINE_DEPTH2` | unset (on) | `=0` disables depth-2 overlap; falls back to sync `_async + _wait` inside `draft`. |
| `LLAMA_MTP_SKIP_STREAK_THRESHOLD` | unset / `0` (off) | `1..32` enables zero-accept skip streak. |
| `LLAMA_MTP_ACC_TRACE` | unset (off) | `1` → stderr; any other value → file path (append). |
| `LLAMA_GRAPH_REUSE_DISABLE` | unset (off) | Disables `llm_graph_result::can_reuse`. Useful when changing the MTP graph; disastrous for throughput. |

Public API entry points:

```c
llama_model_load_mtp_from_file(model, path, mparams);
llama_model_has_mtp_assistant(model);
llama_model_get_mtp_assistant(model);
llama_model_mtp_n_embd_backbone(model);

llama_decode_mtp_async(ctx, seq_id, attn_pos, last_token, h_prev, n_steps);
llama_decode_mtp_wait (ctx, out_drafts, out_h_prev_last);
llama_decode_mtp     (ctx, ..., out_logits, ...);   // sync facade
```

Driver entry points (`common/speculative.h`):

```c
common_speculative_init / _free
common_speculative_set_seq_id        // server slot -> target seq id
common_speculative_set_h_idx         // last accepted batch idx after accept_n
common_speculative_begin             // per-prompt; drains stale MTP
common_speculative_draft             // emits drafts (lazy-waits depth-2)
common_speculative_accept            // updates stats; emits trace
common_speculative_prepare_next      // depth-2: async submit for next round
common_speculative_cancel            // drain in-flight MTP
common_speculative_print_stats
```

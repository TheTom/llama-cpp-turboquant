# Pipeline depth-2: pure MTP overlap (implementation notes)

## Summary

`llama-server` overlaps MTP draft compute with per-iteration post-accept work by
splitting `llama_decode_mtp_async` / `llama_decode_mtp_wait` across server loop
iterations:

1. End of iteration (after `common_speculative_accept`, prompt rollback, `slot.sampled` update, `llama_memory_seq_rm`): `common_speculative_prepare_next(slot.spec, slot.sampled)`.
2. Start of next iteration: `common_speculative_draft` calls `llama_decode_mtp_wait` first when a request is pending.

No optimistic token is used; drafts always match the last accepted target token.

## KV safety (Phase 0)

- `decode_mtp_async` snapshots `h_prev` and uses `attn_pos = seq_pos_max` after `seq_rm`.
- MTP memory context uses `llama_kv_cache_iswa::init_mtp` / `mtp_slot_info` (assistant read path).
- Target `llama_decode` only appends at positions strictly after the rolled-back window; cells at positions `≤ attn_pos` are not rewritten before `_wait`.

## API

- `void common_speculative_prepare_next(common_speculative * spec, llama_token id_last);` — no-op for non-MTP implementations.
- Virtual `prepare_next` on `common_speculative_state` (default empty).

## Bench decision gate

Plan target: **≥ +15%** median tps vs baseline on f16-mtp short prompt (3 runs), via `.scratch/bench-mtp.sh`.

**Status**: not executed in this workspace (no GGUF fixtures under `.scratch/`). Run locally after `scripts/run-gemma4-mtp-server.sh` (or your paths), then:

```bash
bash .scratch/bench-mtp.sh 127.0.0.1:8080 256
```

Compare accept rate (`draft_n` / `draft_n_accepted`) to ensure parity with pre-change runs.

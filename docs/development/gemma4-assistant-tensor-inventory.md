# Gemma 4 Assistant (MTP drafter) — tensor inventory (Stage 0)

This document records the expected tensor layout for `Gemma4AssistantForCausalLM` / `model_type: gemma4_assistant`, cross-referenced with the MLX-VLM reference implementation (PR #1112).

## HF config highlights (`google/gemma-4-26B-A4B-it-assistant`)

- `architectures`: `["Gemma4AssistantForCausalLM"]`
- `model_type`: `gemma4_assistant`
- `backbone_hidden_size`: 2816 (must match paired target Gemma 4 backbone)
- `use_ordered_embeddings`: typically `false` for 26B-A4B / 31B (dense tied LM head); `true` for E2B/E4B (centroid `MaskedEmbedder`)
- `num_centroids`: 2048 (when ordered embeddings)
- `centroid_intermediate_top_k`: 32
- Nested `text_config`: 4 layers, `layer_types` e.g. three `sliding_attention` + one `full_attention`, `attention_k_eq_v: true`, `vocab_size: 262144`, `hidden_size: 1024`, etc.

## Expected safetensors names (HF / MLX)

| Logical component | Typical HF / checkpoint name | GGUF name (this fork) |
|-------------------|------------------------------|------------------------|
| Pre-projection | `pre_projection.weight` | `mtp.pre_projection.weight` |
| Post-projection | `post_projection.weight` | `mtp.post_projection.weight` |
| Token embeddings (inner) | `model.embed_tokens.weight` | `token_embd.weight` |
| Final norm | `model.norm.weight` | `output_norm.weight` |
| Per-layer | `model.layers.{i}.*` | `blk.{i}.*` (via `tensor_mapping.py`) |
| Centroid head (E2B/E4B) | `masked_embedding.centroids.weight` | `mtp.centroids.weight` |
| Token ordering | `masked_embedding.token_ordering` | `mtp.token_ordering` (I32) |
| LM head (if untied) | `lm_head.weight` | `output.weight` |

## MLX reference files

- `mlx_vlm/speculative/drafters/gemma4_assistant/gemma4_assistant.py` — forward, `draft_block`, `sanitize`
- `mlx_vlm/speculative/drafters/gemma4_assistant/masked_embedder.py` — sparse centroid LM head
- `mlx_vlm/speculative/drafters/gemma4_assistant/masks.py` — SWA / full masks

## Notes for llama.cpp port

1. MTP consumes **target K/V** for the last sliding-attention and last full-attention layers; the assistant has **no independent KV** in the reference design. The initial C++ integration uses the assistant as a loadable arch with standard KV for bring-up; full KV sharing is wired through `--spec-type mtp` and remains dependent on target/draft pairing and cache layout.
2. `attention_k_eq_v: true` maps to missing `blk.*.attn_v.weight` (V taken from K), matching Gemma 4 handling in `gemma4-iswa.cpp`.
3. Greedy parity with the target requires byte-identical drafting; validate with `tests/test-speculative-mtp` when model paths are available.

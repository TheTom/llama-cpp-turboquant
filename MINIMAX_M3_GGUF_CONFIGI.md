# MiniMax-M3 GGUF Config-I — runbook (prepped, run later)

This branch (`tom/m3-gguf`) = upstream catch-up + PR #24523 (preliminary M3
support, text-only/dense) + an MXFP8 dequant patch. Goal: produce a Config-I
mixed-precision GGUF of MiniMax-M3 from the local MXFP8 checkpoint.

Status: **prepped, not run.** Waiting until the MLX Config-I upload finishes
before touching disk. No f16 needed (corrected): convert MXFP8 straight to Q8_0.

## What's done
- `74e5c6ffa` cherry-pick of #24523 (`conversion/minimax.py`, `src/models/minimax-m3.cpp`, arch reg) — applied clean onto the modern catch-up base, builds green on M5.
- `232af7d92` MXFP8 dequant in `conversion/base.py`: handles `quant_method == "mxfp8"` with U8 UE8M0 `[1,32]` scales (`weight * 2^(scale-127)`). Stock convert only handled DeepSeek-style float-block `fp8`.

## Source
`~/models/MiniMax-M3-MXFP8` (413 GB), `architectures = [MiniMaxM3SparseForConditionalGeneration]`, weights F8_E4M3, scales U8 `weight_scale_inv` (E8M0 [1,32]). lm_head/embed/router gates are BF16/full-precision already (Config-I-friendly).

## Disk plan (fits on the Mac only with staged deletes; ~900 GB free avoids deletes)
1. MLX upload finishes -> delete local `~/models/MiniMax-M3-ConfigI` (167 GB) -> ~487 GB free
2. convert MXFP8 -> Q8_0 GGUF (~454 GB)
3. delete the MXFP8 source (413 GB; re-downloadable from HF) -> ~446 GB free
4. llama-quantize Q8_0 -> Config-I (~167 GB)

## Commands
```bash
# 1. MXFP8 -> Q8_0 GGUF (one pass, no f16). --fp8-as-q8 stores dequant'd FP8 as Q8_0.
python3 convert_hf_to_gguf.py ~/models/MiniMax-M3-MXFP8 \
    --outfile ~/models/MiniMax-M3-Q8_0.gguf --outtype q8_0 --fp8-as-q8

# 2. Config-I quantize from the Q8_0 GGUF (tensor-type overrides).
#    Base = Q2_K (experts gate/up dominate ~97%), override the protected tensors up.
#    M3: 60 layers, boundary = first 2 (0,1) + last 2 (58,59).
./build-metal/bin/llama-quantize \
    --tensor-type ffn_down_exps=Q3_K \
    --tensor-type attn_q=Q4_K --tensor-type attn_k=Q4_K \
    --tensor-type attn_v=Q4_K --tensor-type attn_output=Q4_K \
    --tensor-type ffn_gate_inp=F16 \
    --tensor-type "blk.0.=Q8_0" --tensor-type "blk.1.=Q8_0" \
    --tensor-type "blk.58.=Q8_0" --tensor-type "blk.59.=Q8_0" \
    --output-tensor-type Q8_0 --token-embedding-type Q8_0 \
    ~/models/MiniMax-M3-Q8_0.gguf ~/models/MiniMax-M3-ConfigI.gguf Q2_K
```

## Verify before trusting the override regex
`--tensor-type name=TYPE` matches by tensor-name substring; with overlapping
patterns (boundary `blk.0.` vs component `attn_q`) confirm precedence on a
`--dry-run` first:
```bash
./build-metal/bin/llama-quantize --dry-run ~/models/MiniMax-M3-Q8_0.gguf Q2_K  # lists tensors+types
```
If precedence is wrong, emit an exact per-tensor `--tensor-type-file` from the
dry-run tensor list (classify each by the Config-I policy in
`~/dev/minimax-config-i/convert_m3.py`).

## Config-I policy (same as the MLX version)
| component (GGUF name) | bits | ggml type |
|---|---|---|
| expert gate/up (ffn_gate_exps, ffn_up_exps) | 2 | Q2_K |
| expert down (ffn_down_exps) | 3 | Q3_K |
| attention (attn_q/k/v/output) | 4 | Q4_K |
| boundary layers (blk.0/1/58/59, all) | 8 | Q8_0 |
| router (ffn_gate_inp) | f16 | F16 |
| embeddings + lm_head (token_embd, output) | 8 | Q8_0 |

Caveats: #24523 M3 support is text-only, dense-attention (sparse-attn, vision,
MTP dropped) — matches the MLX Config-I limitations.

#!/usr/bin/env python3
"""Verify Gemma 4 MTP assistant GGUF matches llama.cpp (token_embd smaller dim == embedding_length KV). Exit 1 on mismatch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", type=Path)
    args = ap.parse_args()
    gguf_py = Path(__file__).resolve().parent.parent / "gguf-py"
    sys.path.insert(0, str(gguf_py))
    from gguf import GGUFReader  # type: ignore

    r = GGUFReader(str(args.gguf))
    exp_emb: int | None = None
    arch: str | None = None
    for fk, field in r.fields.items():
        key = fk.decode("utf-8") if isinstance(fk, bytes) else str(fk)
        if key.endswith("general.architecture") or key.endswith("architecture") and "general." in key:
            try:
                arch = bytes(field.parts[-1]).decode("utf-8", errors="replace")
            except Exception:
                arch = str(field.parts[-1])
        if key == "gemma4_assistant.embedding_length" or key.endswith(".embedding_length") and "gemma4_assistant" in key:
            try:
                exp_emb = int(field.parts[-1][0])
            except Exception:
                pass
    emb_shape = None
    for t in r.tensors:
        nm = t.name.decode("utf-8") if isinstance(t.name, bytes) else str(t.name)
        if nm == "token_embd.weight":
            emb_shape = tuple(int(x) for x in t.shape)
            break

    if arch and "assistant" not in arch:
        print(f"warn: unexpected arch {arch!r}", file=sys.stderr)
    if not emb_shape:
        print("error: missing token_embd.weight", file=sys.stderr)
        return 1
    if len(emb_shape) != 2:
        print(f"error: token_embd.weight rank {len(emb_shape)}", file=sys.stderr)
        return 1
    lo, hi = min(emb_shape), max(emb_shape)
    if exp_emb is not None and lo != exp_emb:
        print(
            f"error: token_embd min_dim={lo} != gemma4_assistant.embedding_length={exp_emb} (full shape {emb_shape})",
            file=sys.stderr,
        )
        print(
            "hint: re-run convert_hf_to_gguf.py from HF assistant dir, or rebuild from current convert script.",
            file=sys.stderr,
        )
        return 1
    print(f"ok: token_embd.weight shape={emb_shape} embedding_length_kv={exp_emb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

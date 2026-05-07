#!/usr/bin/env python3
"""Verify Gemma 4 MTP assistant GGUF matches llama.cpp (token_embd smaller dim == embedding_length KV). Exit 1 on mismatch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _field_last_part(field: object):
    parts = getattr(field, "parts", None)
    if not parts:
        return None
    last = parts[-1]
    try:
        return last[0]
    except (IndexError, TypeError, KeyError):
        return last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", type=Path)
    args = ap.parse_args()
    gguf_py = Path(__file__).resolve().parent.parent / "gguf-py"
    sys.path.insert(0, str(gguf_py))
    from gguf import GGMLQuantizationType, GGUFReader  # type: ignore

    r = GGUFReader(str(args.gguf))
    exp_emb: int | None = None
    arch: str | None = None
    use_ordered_embeddings: bool | None = None
    n_centroids_kv: int | None = None
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
        if key == "gemma4_assistant.use_ordered_embeddings":
            v = _field_last_part(field)
            use_ordered_embeddings = bool(v)
        if key == "gemma4_assistant.n_centroids":
            v = _field_last_part(field)
            try:
                n_centroids_kv = int(v)
            except (TypeError, ValueError):
                n_centroids_kv = None
    emb_shape = None
    centroids_shape: tuple[int, ...] | None = None
    ordering_shape: tuple[int, ...] | None = None
    ordering_type: int | None = None
    for t in r.tensors:
        nm = t.name.decode("utf-8") if isinstance(t.name, bytes) else str(t.name)
        if nm == "token_embd.weight":
            emb_shape = tuple(int(x) for x in t.shape)
        elif nm == "mtp.centroids.weight":
            centroids_shape = tuple(int(x) for x in t.shape)
        elif nm == "mtp.token_ordering.weight":
            ordering_shape = tuple(int(x) for x in t.shape)
            ordering_type = int(t.tensor_type)

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

    if use_ordered_embeddings:
        if not centroids_shape or len(centroids_shape) != 2:
            print("error: use_ordered_embeddings=true but mtp.centroids.weight missing or not rank-2", file=sys.stderr)
            return 1
        if not ordering_shape or len(ordering_shape) != 1:
            print("error: use_ordered_embeddings=true but mtp.token_ordering.weight missing or not rank-1", file=sys.stderr)
            return 1
        if ordering_type != int(GGMLQuantizationType.I32):
            print(
                f"error: mtp.token_ordering.weight must be I32 (got tensor_type={ordering_type})",
                file=sys.stderr,
            )
            return 1
        # Read actual routing indices and reject zero-filled broken conversions.
        ordering_tensor = None
        for t in r.tensors:
            nm = t.name.decode("utf-8") if isinstance(t.name, bytes) else str(t.name)
            if nm == "mtp.token_ordering.weight":
                ordering_tensor = t
                break
        if ordering_tensor is None:
            print("error: missing mtp.token_ordering.weight tensor payload", file=sys.stderr)
            return 1
        ordering_data = ordering_tensor.data
        if ordering_data.size == 0:
            print("error: mtp.token_ordering.weight is empty", file=sys.stderr)
            return 1
        ord_min = int(ordering_data.min())
        ord_max = int(ordering_data.max())
        if ord_min == 0 and ord_max == 0:
            print(
                "error: mtp.token_ordering.weight is all zeros (broken conversion); rerun convert_hf_to_gguf.py from HF source",
                file=sys.stderr,
            )
            return 1
        n_embd_min = min(emb_shape)
        n_vocab = max(emb_shape)
        ce0, ce1 = centroids_shape[0], centroids_shape[1]
        if ce0 != n_embd_min:
            print(
                f"error: mtp.centroids.weight dim[0]={ce0} != token_embd n_embd={n_embd_min} (from {emb_shape})",
                file=sys.stderr,
            )
            return 1
        n_vocab_order = int(ordering_shape[0])
        if n_vocab_order != n_vocab:
            print(
                f"error: token_ordering length {n_vocab_order} != vocab dim of token_embd {n_vocab}",
                file=sys.stderr,
            )
            return 1
        if n_centroids_kv is not None and int(ce1) != int(n_centroids_kv):
            print(
                f"error: mtp.centroids.weight dim[1]={ce1} != gemma4_assistant.n_centroids KV {n_centroids_kv}",
                file=sys.stderr,
            )
            return 1
        if n_vocab % int(ce1) != 0:
            print(f"error: vocab_size {n_vocab} not divisible by n_centroids {ce1}", file=sys.stderr)
            return 1
        print(
            f"ok: centroid head mtp.centroids.weight shape={centroids_shape} "
            f"mtp.token_ordering.weight shape={ordering_shape} (I32) n_centroids={ce1}"
        )
    elif use_ordered_embeddings is False and centroids_shape is not None:
        print("warn: use_ordered_embeddings=false but mtp.centroids.weight present", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

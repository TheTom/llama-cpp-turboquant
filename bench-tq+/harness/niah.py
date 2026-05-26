#!/usr/bin/env python3
"""
NIAH (Needle-In-A-Haystack) retrieval test for TQ+ codec evaluation.

Plants a synthetic factoid at a known depth inside a long natural-text
haystack, asks the model to recall it, scores by string-match. Mirrors
SHARD's NIAH eval to give us a directly-comparable number against their
paper.

For each (codec, depth_pct, ctx) cell, runs llama-cli once with the
constructed prompt and parses the generation for the needle.

Usage:
    python niah.py --model M.gguf \
        --codecs "f16/f16" "q8_0/turbo3" "q8_0/turbo2" \
        --contexts 2048 4096 8192 \
        --depths 0.1 0.3 0.5 0.7 0.9 \
        --out results/niah.json
"""

import argparse
import json
import random
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CLI = REPO / "build/bin/llama-cli"


# A fixed, distinctive needle with a 5-digit numeric secret. Numeric
# answer is easier to score than free-form text.
NEEDLE_TEMPLATE = (
    "Important note: the secret code for project '{project}' is {code}. "
    "Please remember this code. End of important note. "
)
QUERY = "What is the secret code for project '{project}'? Reply with only the 5-digit number."

# Common factoids used as haystack filler. The point is to be neutral
# text the model has likely seen before — wikitext-style is fine.
HAYSTACK_INSTRUCT = (
    "Read the following passage carefully. Some of it is filler and "
    "some of it contains a single important note. After reading, you "
    "will be asked a question about the note.\n\n"
)


@dataclass
class NiahResult:
    label: str
    k_type: str
    v_type: str
    ctx: int
    depth_pct: float
    needle_pos_tokens: int
    project: str
    code: str
    response: str
    correct: bool
    elapsed_s: float
    log_path: str


def load_haystack_tokens(corpus_path: Path) -> str:
    """Load corpus text. We'll truncate by token count at run time."""
    return corpus_path.read_text(encoding="utf-8", errors="replace")


def build_prompt(haystack_text: str, target_ctx: int, depth_pct: float,
                 project: str, code: str) -> str:
    """
    Build a prompt of approximately target_ctx tokens, with the needle
    inserted at depth_pct fraction of the haystack.

    We approximate tokens as words × 0.75 (rough ratio for English
    against most BPE tokenizers). For NIAH purposes the exact token
    count doesn't matter — what matters is that the haystack fills the
    context window so the needle ends up at the intended position.
    """
    needle = NEEDLE_TEMPLATE.format(project=project, code=code)

    # Reserve ~200 tokens for instruction + query + generation budget.
    needle_words = len(needle.split())
    target_words = int(target_ctx * 0.70) - 200 - needle_words
    target_words = max(target_words, needle_words * 4)

    words = haystack_text.split()
    if len(words) < target_words:
        # Cycle through if not enough text
        n_repeats = (target_words // len(words)) + 1
        words = words * n_repeats
    haystack = " ".join(words[:target_words])

    # Split haystack at depth_pct and inject the needle
    split_idx = int(target_words * depth_pct)
    before = " ".join(words[:split_idx])
    after = " ".join(words[split_idx:target_words])
    body = before + "\n\n" + needle + "\n\n" + after

    return (
        HAYSTACK_INSTRUCT + body + "\n\n"
        + QUERY.format(project=project) + "\nAnswer: "
    )


def run_one(
    cli_bin: Path,
    model: Path,
    prompt: str,
    k_type: str,
    v_type: str,
    ctx: int,
    threads: int,
    log_path: Path,
    extra: list[str],
    n_predict: int = 32,
) -> tuple[str, float, int]:
    """Run llama-cli, return (response, elapsed_s, return_code)."""
    cmd = [
        str(cli_bin),
        "-m", str(model),
        "-c", str(ctx),
        "-n", str(n_predict),
        "--cache-type-k", k_type,
        "--cache-type-v", v_type,
        "-ngl", "999",
        "-t", str(threads),
        "-fa", "on",
        "--no-display-prompt",
        "--temp", "0",          # greedy
        "-p", prompt,
        *extra,
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=30 * 60)
    elapsed = time.time() - t0
    log_path.write_text(
        f"$ {' '.join(shlex.quote(c) for c in cmd[:11])} ... <prompt: {len(prompt)} chars>\n"
        f"# rc = {proc.returncode}, elapsed = {elapsed:.1f}s\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr (tail) ---\n{proc.stderr[-2000:]}\n",
    )
    return proc.stdout.strip(), elapsed, proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--corpus", type=Path, default=REPO / "wikitext-2-raw/wiki.test.raw")
    ap.add_argument("--codecs", nargs="+", required=True)
    ap.add_argument("--contexts", nargs="+", type=int, required=True)
    ap.add_argument("--depths", nargs="+", type=float,
                    default=[0.1, 0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=4242,
                    help="determines which needle/code is planted")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--log-dir", type=Path, default=REPO / "bench-tq+/logs")
    ap.add_argument("--cli-bin", type=Path, default=DEFAULT_CLI)
    ap.add_argument("--extra-args", default="--fit off")
    args = ap.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    extra = shlex.split(args.extra_args)

    rng = random.Random(args.seed)
    haystack = load_haystack_tokens(args.corpus)

    project_names = ["Tomato", "Orchid", "Falcon", "Mercury", "Atlas", "Crater"]

    results = []
    for codec_spec in args.codecs:
        k_type, v_type = codec_spec.split("/")
        for ctx in args.contexts:
            for depth in args.depths:
                project = rng.choice(project_names)
                code = f"{rng.randint(10000, 99999):05d}"
                prompt = build_prompt(haystack, ctx, depth, project, code)
                approx_tokens = int(len(prompt.split()) * 1.4)
                needle_pos = int(approx_tokens * depth)

                label = f"k={k_type}_v={v_type}_c{ctx}_d{int(depth*100):02d}"
                log_path = args.log_dir / f"niah_{label}.log"
                print(f"[run] {label}  needle≈{needle_pos}/{approx_tokens} project={project} code={code}")
                sys.stdout.flush()

                response, elapsed, rc = run_one(
                    args.cli_bin, args.model, prompt,
                    k_type, v_type, ctx, args.threads,
                    log_path, extra, n_predict=24,
                )
                # Score: did the generation contain the code?
                # Strip whitespace, look for the 5-digit number
                m = re.search(r"\b(\d{5})\b", response)
                guessed = m.group(1) if m else ""
                correct = (guessed == code)

                results.append(asdict(NiahResult(
                    label=label, k_type=k_type, v_type=v_type,
                    ctx=ctx, depth_pct=depth,
                    needle_pos_tokens=needle_pos,
                    project=project, code=code,
                    response=response[:200],
                    correct=correct,
                    elapsed_s=elapsed,
                    log_path=str(log_path),
                )))
                args.out.write_text(json.dumps(results, indent=2))
                print(f"  → {'✓' if correct else '✗'} got '{guessed}' (want {code}), {elapsed:.0f}s")
                sys.stdout.flush()

    # Final summary by (codec, ctx)
    print("\nNIAH accuracy (over depths):")
    codecs = sorted(set((r["k_type"], r["v_type"]) for r in results))
    ctxs = sorted(set(r["ctx"] for r in results))
    print(f"  {'codec':<18} " + " ".join(f"{c:>10}" for c in ctxs))
    for k, v in codecs:
        row = []
        for ctx in ctxs:
            rs = [r for r in results if r["k_type"] == k and r["v_type"] == v and r["ctx"] == ctx]
            if rs:
                acc = sum(r["correct"] for r in rs) / len(rs)
                row.append(f"{acc*100:.0f}% ({sum(r['correct'] for r in rs)}/{len(rs)})")
            else:
                row.append("—")
        print(f"  {k+'/'+v:<18} " + " ".join(f"{c:>10}" for c in row))


if __name__ == "__main__":
    main()

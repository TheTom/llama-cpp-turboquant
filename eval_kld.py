#!/usr/bin/env python3
"""KLD/same-top evaluation: compare f16 vs oscar2 KV cache outputs.

Usage:
    python3 eval_kld.py --model /path/to/model.gguf --prompt "prompt" -n 50
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import numpy as np
from pathlib import Path

def run_model(llama_cli, model, cache_type, prompt, n_tokens, build_dir):
    """Run llama-cli and extract logits."""
    bin_dir = os.path.join(build_dir, 'bin')
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = bin_dir
    env['CUDA_VISIBLE_DEVICES'] = '0'

    cmd = [
        llama_cli,
        '-m', model,
        '-ngl', '99',
        '-fa', 'on',
        '-c', '8192',
        '--cache-type-k', cache_type,
        '--cache-type-v', cache_type,
        '--temp', '0',
        '-p', prompt,
        '-n', str(n_tokens),
        '--single-turn',
        '--verbose-prompt',
        '--logits-all',  # dump all logits
    ]

    print(f"  Running: {' '.join(cmd[-15:])}", file=sys.stderr)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
    
    # Parse logits from stderr (--logits-all dumps them there)
    logits = []
    tokens = []
    in_logits = False
    
    for line in result.stderr.split('\n'):
        if 'logits' in line.lower() and '[' in line:
            try:
                data = json.loads(line.strip())
                if isinstance(data, list):
                    logits.append(np.array(data, dtype=np.float32))
            except (json.JSONDecodeError, ValueError):
                pass
        if line.startswith('[') and ']' in line and len(line) < 200:
            # Token output from main generation
            tokens.append(line.strip())

    # Also try to get output text
    output_text = result.stdout

    return np.array(logits) if logits else None, output_text, result.stderr


def compute_kld(p_logits, q_logits):
    """KL(P || Q) averaged over all positions.
    p = softmax(f16_logits), q = softmax(oscar2_logits)
    """
    # Compute softmax
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
    
    p = softmax(p_logits)
    q = softmax(q_logits)
    
    # Clip to avoid log(0)
    eps = 1e-10
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / np.sum(p, axis=-1, keepdims=True)
    q = q / np.sum(q, axis=-1, keepdims=True)
    
    # KL(P || Q) per position: sum(P_i * log(P_i / Q_i))
    kld_per_pos = np.sum(p * np.log(p / q), axis=-1)
    
    return float(np.mean(kld_per_pos)), kld_per_pos


def compute_same_top(p_logits, q_logits):
    """Fraction of positions where argmax matches."""
    p_top = np.argmax(p_logits, axis=-1)
    q_top = np.argmax(q_logits, axis=-1)
    same = np.mean(p_top == q_top)
    return float(same), p_top, q_top


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--prompt', default="The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain is")
    parser.add_argument('-n', type=int, default=50)
    parser.add_argument('--build-dir', default='/mnt/storage/Projects/turboquant/build-fixed-native')
    parser.add_argument('--llama-cli', default=None)
    args = parser.parse_args()
    
    cli = args.llama_cli or os.path.join(args.build_dir, 'bin/llama-cli')
    
    print("=== oscar2 KLD/same-top Evaluation ===", file=sys.stderr)
    print(f"Model: {args.model}", file=sys.stderr)
    print(f"Prompt: {args.prompt}", file=sys.stderr)
    print(f"N tokens: {args.n}", file=sys.stderr)
    print(f"Build: {args.build_dir}", file=sys.stderr)
    print(file=sys.stderr)
    
    # Run f16
    print("Step 1: Running f16 baseline...", file=sys.stderr)
    f16_logits, f16_text, f16_stderr = run_model(
        cli, args.model, 'f16', args.prompt, args.n, args.build_dir)
    if f16_logits is None:
        print("ERROR: No logits from f16 run", file=sys.stderr)
        print(f16_stderr[:500], file=sys.stderr)
        sys.exit(1)
    print(f"  Got {len(f16_logits)} logit vectors from f16", file=sys.stderr)
    print(f"  Output: {f16_text.strip()[:100]}", file=sys.stderr)
    
    # Run oscar2
    print("Step 2: Running oscar2...", file=sys.stderr)
    o2_logits, o2_text, o2_stderr = run_model(
        cli, args.model, 'oscar2', args.prompt, args.n, args.build_dir)
    if o2_logits is None:
        print("ERROR: No logits from oscar2 run", file=sys.stderr)
        print(o2_stderr[:500], file=sys.stderr)
        sys.exit(1)
    print(f"  Got {len(o2_logits)} logit vectors from oscar2", file=sys.stderr)
    print(f"  Output: {o2_text.strip()[:100]}", file=sys.stderr)
    
    # Align lengths
    min_len = min(len(f16_logits), len(o2_logits))
    f16_logits = f16_logits[:min_len]
    o2_logits = o2_logits[:min_len]
    print(f"  Aligned to {min_len} positions", file=sys.stderr)
    
    # Compute metrics
    print("Step 3: Computing KLD and same-top...", file=sys.stderr)
    kld, kld_per_pos = compute_kld(f16_logits, o2_logits)
    same_top, p_top, q_top = compute_same_top(f16_logits, o2_logits)
    
    print(file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print("RESULTS", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  KLD (avg):          {kld:.6f}", file=sys.stderr)
    print(f"  KLD (per position): {[f'{v:.4f}' for v in kld_per_pos[:10]]}...", file=sys.stderr)
    print(f"  same-top@1:         {same_top*100:.1f}%", file=sys.stderr)
    print(f"  N positions:        {min_len}", file=sys.stderr)
    
    # Output JSON for machine parsing
    result = {
        "model": args.model,
        "prompt": args.prompt,
        "n_tokens": args.n,
        "n_positions": min_len,
        "kld_mean": round(kld, 6),
        "same_top_pct": round(same_top * 100, 1),
        "f16_output": f16_text.strip()[:100],
        "oscar2_output": o2_text.strip()[:100],
    }
    print(json.dumps(result))


if __name__ == '__main__':
    main()

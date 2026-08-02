#!/usr/bin/env python3
"""
Local logit-KLD harness for OSCAR2 vs FP16 reference.
Strategy: reuse llama.cpp server endpoints if available, otherwise
fallback to a Python-side GGUF runtime and compute KL on the
final-step logprob distribution over the common token vocabulary.

This deliberately mirrors the reference doc's reporting shape.
"""
import argparse
import json
import os
import sys
import math
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="Base or rotated GGUF path")
    p.add_argument("--rotated", required=True, help="Rotated GGUF path for OSCAR2")
    p.add_argument("--corpus", required=False, help="Dataset or prompt file, one prompt per line")
    p.add_argument("--prompt", default="The capital of France is Paris. The capital of Germany is Berlin.")
    p.add_argument("--n-samples", type=int, default=16)
    return p.parse_args()

def main():
    args = parse_args()
    print(f"TODO: Local logit-KLD harness scaffold for {args.base} vs {args.rotated}")
    # TODO: Implement endpoint-binding to llama-server, or local gguf-runtime logprob extraction.

if __name__ == "__main__":
    main()

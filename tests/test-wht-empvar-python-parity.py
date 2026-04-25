#!/usr/bin/env python3

import json
import math
import subprocess
import sys
from pathlib import Path


S1 = [
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1,
]
S2 = [
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1,
]


def normalize(x):
    norm = math.sqrt(sum(v * v for v in x))
    if norm <= 1e-8:
        return list(x)
    return [v / norm for v in x]


def fwht(x):
    x = list(x)
    n = len(x)
    h = 1
    while h < n:
        for i in range(0, n, 2 * h):
            for j in range(i, i + h):
                a = x[j]
                b = x[j + h]
                x[j] = a + b
                x[j + h] = a - b
        h *= 2
    scale = 1.0 / math.sqrt(n)
    return [v * scale for v in x]


def turbo_rotate_forward(x):
    y = [x[i] * S1[i] for i in range(len(x))]
    y = fwht(y)
    return [y[i] * S2[i] for i in range(len(y))]


def turbo_rotate_inverse(x):
    y = [x[i] * S2[i] for i in range(len(x))]
    y = fwht(y)
    return [y[i] * S1[i] for i in range(len(y))]


def quantize_nearest(x, codebook):
    return min(codebook, key=lambda c: abs(x - c))


def reconstruct_dataset(samples, codebooks, renormalize_rotated):
    out = []
    for sample in samples:
        norm = max(math.sqrt(sum(v * v for v in sample)), 1e-8)
        rotated = turbo_rotate_forward(normalize(sample))
        rotated = [quantize_nearest(rotated[i], codebooks[i]) for i in range(len(rotated))]
        if renormalize_rotated:
            qr_norm = max(math.sqrt(sum(v * v for v in rotated)), 1e-8)
            rotated = [v / qr_norm for v in rotated]
        recon = turbo_rotate_inverse(rotated)
        out.append([v * norm for v in recon])
    return out


def empirical_variances(samples):
    dim = len(samples[0])
    sums = [0.0] * dim
    for sample in samples:
        rotated = turbo_rotate_forward(normalize(sample))
        for i, v in enumerate(rotated):
            sums[i] += v * v
    return [v / len(samples) for v in sums]


def empirical_variances_chunked(samples, runtime_dim):
    target_dim = len(samples[0])
    n_chunks = max(1, target_dim // runtime_dim)
    out = []
    for chunk_idx in range(n_chunks):
        chunk_samples = [sample[chunk_idx * runtime_dim:(chunk_idx + 1) * runtime_dim] for sample in samples]
        out.extend(empirical_variances(chunk_samples))
    return out


def build_empirical_codebooks(variances):
    books = []
    for var in variances:
        sigma = math.sqrt(max(var, 1e-6))
        books.append([-1.51 * sigma, -0.453 * sigma, 0.453 * sigma, 1.51 * sigma])
    return books


def build_fixed_codebooks(dim):
    sigma = 1.0 / math.sqrt(dim)
    codebook = [-1.51 * sigma, -0.453 * sigma, 0.453 * sigma, 1.51 * sigma]
    return [list(codebook) for _ in range(dim)]


def require_close(name, got, ref, tol=1e-6):
    if len(got) != len(ref):
        raise AssertionError(f"{name}: length mismatch {len(got)} != {len(ref)}")
    max_err = max(abs(a - b) for a, b in zip(got, ref))
    if max_err > tol:
        raise AssertionError(f"{name}: max error {max_err} > {tol}")


def main():
    root = Path(__file__).resolve().parents[1]
    exe = root / "build" / "bin" / "test-wht-empvar-dump"
    proc = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
    got = json.loads(proc.stdout)

    x = [((i % 7) - 3) / (i + 3) for i in range(64)]
    calib = [
        x,
        normalize(turbo_rotate_inverse(x)),
        [((i % 5) - 2) * 0.125 for i in range(64)],
        [((i % 11) - 5) / 13.0 for i in range(64)],
    ]
    eval_samples = [calib[0], calib[2]]

    normalized = normalize(x)
    rotated = turbo_rotate_forward(normalized)
    variances = empirical_variances(calib)
    chunked_calib = []
    for _ in range(3):
        chunked_calib.append([0.0] * 256)
    for i in range(256):
        chunked_calib[0][i] = math.sin(0.07 * i)
        chunked_calib[1][i] = math.cos(0.05 * i) + (0.1 if i >= 128 else 0.0)
        chunked_calib[2][i] = ((i % 9) - 4) / 5.0
    chunked_variances = empirical_variances_chunked(chunked_calib, 128)
    emp_books = build_empirical_codebooks(variances)
    fixed_books = build_fixed_codebooks(64)
    empirical_recon_first = reconstruct_dataset(eval_samples, emp_books, False)[0]
    fixed_recon_first = reconstruct_dataset(eval_samples, fixed_books, True)[0]

    require_close("normalized", got["normalized"], normalized)
    require_close("rotated", got["rotated"], rotated)
    require_close("variances", got["variances"], variances)
    require_close("chunked_variances", got["chunked_variances"], chunked_variances)
    require_close("empirical_recon_first", got["empirical_recon_first"], empirical_recon_first, tol=2e-6)
    require_close("fixed_recon_first", got["fixed_recon_first"], fixed_recon_first, tol=2e-6)
    print("test-wht-empvar-python-parity: ok")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"test-wht-empvar-python-parity: {exc}", file=sys.stderr)
        sys.exit(1)

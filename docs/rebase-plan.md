# TurboQuant `rebase` Branch Plan

**Base branch:** `feature/turboquant-kv-cache`
**Target upstream:** `upstream/master` -> `/mnt/storage/llama.cpp`
**New branch:** `rebase`
**Divergence at merge-base:** `1fd6dfe9f3d4b69cce101d832339fbda2d14b056`
**Commit delta:** `321` TurboQuant-only / `334` upstream-only (as of this snapshot)

---

## Objective

Produce a clean, linear `rebase` branch that re-applies TurboQuant feature work on top of current `llama.cpp master`, minimizing custom patch drift and preserving the existing `oscar`/upstream debug context.

---

## Step 1 — Create `rebase` from feature branch

```bash
cd /mnt/storage/Projects/turboquant
git checkout -b rebase feature/turboquant-kv-cache
git branch -u upstream/master rebase
```

Confirm upstream remote points to `/mnt/storage/llama.cpp`.

---

## Step 2 — Rebase onto upstream/master

```bash
git rebase upstream/master
```

Expect non-trivial conflicts given near-equal divergence. Resolve in order of highest churn first.

---

## Step 3 — Primary conflict zones

High-probability/modified-in-TurboQuant paths to resolve during rebase:

| Area | Files / dirs |
|---|---|
| CUDA attention kernels | `ggml/src/ggml-cuda/fattn*.cu/cuh` |
| KV cache + quantization | `src/llama-kv-cache.cpp`, `ggml/ggml.c` |
| GGML type dispatch | `ggml/include/ggml.h`, backend type enums |
| Build system | `CMakeLists.txt`, `ggml/src/CMakeLists.txt`, CUDA arch flags |
| Python binding / convert | `convert.py`, conversion helpers |
| CLI flags | `common`, `main`, sampling/context flags |
| Testing / bench harness | `examples`, `tests` parity changes |

Conflict resolution policy:
- Prefer upstream file shape where interfaces changed.
- Preserve TurboQuant custom types/dispatch branches where they add capability.
- Do not merge oscar-local debug instrumentation into `rebase`.

---

## Step 4 — Post-rebase build requirements

Rebuild with existing toolchain before declaring success:
```bash
cd /mnt/storage/Projects/turboquant
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NVFP4=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_MMA=ON
cmake --build build -j
```

Blockers to report back immediately:
- unresolved symbol errors in FA kernels
- failed NVFP4 path compile
- RTX 5090 / CUDA 13.3 arch mismatch after upstream cmake changes

---

## Step 5 — Functional validation

Target validation items, in order:
1. Model conversion sanity check for at least one supported TurboQuant model.
2. Single-GPU prompt decode/run with `--n-gpu-layers 99` on a small model.
3. KV-cache rewrite path correctness smoke test.
4. If available, compare output logits vs last known good `oscar` baseline on a fixed prompt and temp=0.

All failures are to be reported with exact command, binary path, model path, and last 80 lines of stderr.

---

## Step 6 — Commit message discipline

Final cleanup:
```bash
git commit --allow-empty -m "rebase: merge turboquant onto llama.cpp master"
git tag -a turboquant-rebase-$(date +%Y%m%d) -m "Rebase snapshot"
```

---

## Out of scope for first pass

- Cherry-picking oscar rotation / FA decode patches unless they are already in `feature/turboquant-kv-cache`.
- PR creation until the branch builds and runs a smoke test.

---

## Completion criteria

- `rebase` builds locally on RTX 5090 toolchain.
- One end-to-end convert + infer smoke test passes.
- Remaining conflict notes or build blockers documented for escalation.

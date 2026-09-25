# Instructions for llama.cpp (TurboQuant fork)

## Project Overview

This repo is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) (upstream) that adds the TurboQuant feature set on top of an upstream base. Upstream work is normally cherry-picked where it adds benefit (new model shapes, bug fixes, features that interact with the turbo paths); the tree is NOT required to contain all of upstream master. History has one wholesale sync, `a34005435` (upstream `15586e2d7`, 2026-08-10, PR #287), so do not claim bulk syncs never happened, and do not assume a commit that exists upstream also exists here (check git log).

### What TurboQuant adds

TurboQuant compresses the KV cache far beyond the standard `q8_0` by applying a fixed 128x128 orthonormal Walsh-Hadamard rotation (`GGML_OP_TURBO_WHT`) to cache vectors before quantization, which Gaussianizes the distribution, and inverse-rotating after dequantization. Head dims that are not multiples of 128 are zero-padded. MLA models (DeepSeek) have no separate V cache, so V rotation/padding is skipped for them, and K/V cache types must be identical.

The eight fork-only GGML types (registered in `ggml/include/ggml.h`):

| Type   | Enum                       | Purpose                             | Size         | Backends            |
|--------|----------------------------|-------------------------------------|--------------|---------------------|
| turbo2 | `GGML_TYPE_TURBO2_0` (43)  | KV cache only                       | 2.125 bits   | CPU, CUDA/HIP, Metal, Vulkan, SYCL |
| turbo3 | `GGML_TYPE_TURBO3_0` (44)  | KV cache only                       | 3.125 bits   | CPU, CUDA/HIP, Metal, Vulkan, SYCL |
| turbo4 | `GGML_TYPE_TURBO4_0` (47)  | KV cache only                       | 4.125 bits   | CPU, CUDA/HIP, Metal, Vulkan, SYCL |
| TQ3_1S | `GGML_TYPE_TQ3_1S` (45)    | model weights, WHT-rotated Lloyd-Max| 4.00 bits, block 32 | CPU, CUDA/HIP, Metal, Vulkan |
| TQ4_1S | `GGML_TYPE_TQ4_1S` (46)    | model weights, WHT-rotated Lloyd-Max| 5.00 bits, block 32 | CPU, CUDA/HIP, Metal, Vulkan |
| Q8_CR  | `GGML_TYPE_Q8_CR` (48)     | model weights, ConvRot              | 8.5 bits, group 256 | CPU, CUDA/HIP, Metal |
| Q5_CR  | `GGML_TYPE_Q5_CR` (49)     | model weights, ConvRot              | 5.5 bits, group 256 | CPU, CUDA/HIP, Metal |
| Q6_CR  | `GGML_TYPE_Q6_CR` (50)     | model weights, ConvRot              | 6.5625 bits, group 256 | CPU, CUDA/HIP, Metal |

The turbo blocks hold 128 values each (34, 50, 66 bytes); the `static_assert`s in `ggml/src/ggml-common.h` are the authority, not the struct comments. ggml-sycl has no TQ weight kernels, and its generic dequantize path leaves TQ weights in the WHT-rotated domain, so TQ models are unsupported there. Vulkan and SYCL reject the `*_CR` types in `supports_op`.

Turbo cache types are runtime-only, never stored in GGUF. TQ3_1S/TQ4_1S and Q8_CR/Q5_CR/Q6_CR are `llama-quantize` targets.

### Key files

- `ggml/src/ggml-turbo-quant.c` - the codec (keep byte-identical to fork tip)
- `ggml/include/ggml.h` - type enum 43-50, `GGML_OP_TURBO_WHT`
- `src/llama-kv-cache.cpp` - cache wiring, `get_k_idx`, layer-adaptive precision
- `src/llama-graph.cpp` - inverse-WHT post-processing (FA and non-FA paths)
- `ggml/src/ggml-cuda/mmvq-tq.cu` - native TQ dp4a kernels (only reached with `GGML_TQ_NATIVE=1`; by default `TQ4_1S` is converted to `q8_0` at load in `ggml-cuda.cu`)
- `ggml/src/ggml-vulkan/` - turbo FA, SET_ROWS, dequant shaders
- `ggml/src/ggml-metal/ggml-metal.metal` - TurboFlash kernels
- `docs/KV-cache-quantization.md` - usage doc (read before touching cache types; its "Environment knobs" section is currently shorter than the table above and needs a refresh)

### Usage

```bash
llama-cli -m model.gguf -c 8192 -ngl 99 --cache-type-k q8_0 --cache-type-v turbo3
```

Any combination of `f32`/`f16`/`bf16`/`q8_0`/`q4_0`/`q4_1`/`iq4_nl`/`q5_0`/`q5_1`/`turbo2`/`turbo3`/`turbo4` for K and V is supported. Turbo cache types require flash attention; it is auto-enabled with a warning before upstream's quantized-V check runs, so a turbo V cache never trips that error (it applies to the other quantized V types). On Metal, flash attention asserts unless the mixed K/V pair is turbo+turbo, turbo+q8_0 or q8_0+turbo. The same flags work in `llama-server`, `llama-bench`, `llama-perplexity`.

### Environment knobs

This table is the source of truth for variable names, defaults and semantics. `docs/KV-cache-quantization.md` carries a user-facing copy of it; if the two ever disagree, the code wins and both need fixing in the same PR. Do not document a knob only in a PR description.

| Variable                    | Default | Effect |
|-----------------------------|---------|--------|
| `TURBO_LAYER_ADAPTIVE`      | auto    | Layer-adaptive KV precision, see below |
| `TURBO_AUTO_ASYMMETRIC`     | `1`     | Auto-select asymmetric K/V types for large-GQA models (`0` disables) |
| `TURBO_SPARSE_V`            | `1`     | Sparse-V dequant skip in flash attention, Metal and `turbo3` V only (`0` disables) |
| `GGML_TQ_NATIVE`            | unset   | CUDA: `1` keeps `TQ4_1S` weights native and uses the fused TQ kernels instead of converting them to `q8_0` at load; decode about 30% faster and weight VRAM about 1.7x smaller, prefill about 2x slower |
| `GGML_CUDA_FUSE_CHAIN`      | unset   | CUDA: `0` disables elementwise chain fusion, so ADD/MUL/DIV/SCALE/CLAMP/SQR and the SILU/SIGMOID/SOFTPLUS/GELU/RELU/NEG unaries run as separate kernels |
| `GGML_CUDA_Q8CACHE`         | unset   | CUDA: `0` disables the per-graph shared `q8_1` activation cache in `mmvq`, so a repeated activation is quantized again per use |
| `LLAMA_ATTN_ROT_K_OVERRIDE` / `LLAMA_ATTN_ROT_V_OVERRIDE` | off | Optional upstream attention rotation for K / V (TurboQuant manages its own rotation); needs a quantized type and a head dim that is a multiple of 64 |
| `LLAMA_ATTN_ROT_K_NROT`     | `64`    | Rotation tile size for the optional upstream K path (`0` = upstream's largest power-of-two divisor of the head dim) |
| `LLAMA_ATTN_ROT_DISABLE`    | `0`     | Hard lock-out: force rotation off on both sides (`1` disables) |

`TURBO_LAYER_ADAPTIVE` is not simply `0`. With the variable unset, Boundary V (mode 7) turns on automatically when the V cache type is `turbo2` and the model has 8 or more layers: the first two and last two V layers become `q8_0` and every other V layer becomes `turbo2`. `TURBO_LAYER_ADAPTIVE=0` opts out. Setting the variable explicitly selects the mode, and modes 5, 6 and 7 assign the V type per layer regardless of `-ctv`: mode 7 on a `-ctv turbo3` or `-ctv turbo4` run downgrades the non-edge V layers to `turbo2`. The full mode legend is in `src/llama-kv-cache.cpp` (`kv_adaptive_mode`).

### Test gates (all must pass before touching quant/backend code)

- `test-turbo-quant` - turbo3 basis MSE=0/Cosine=1.0, turbo4 Cosine=0.9956
- `test-quantize-fns` - includes TQ3_1S/TQ4_1S and rotated-domain buffer sizing
- `test-backend-ops` - full sweep on the backend under test (23k+ cases on the RTX 5090 CUDA box; the same cases on the MI210 AMD box when CUDA/HIP kernels changed); CPU is the numeric reference and is skipped unless `-b CPU`; rejects 0/0 as FAIL
- `llama-bench` with `-ctk/-ctv turboN`; type parser accepts `tq3_1s`/`tq4_1s`

### Validation by change type

Use the narrowest set of gates that covers the change, with CPU as the numeric reference:

| Change                                                      | Required checks |
|-------------------------------------------------------------|-----------------|
| Codec `ggml/src/ggml-turbo-quant.c`                         | `test-turbo-quant` and `test-quantize-fns`, the two suites that execute the CPU codec directly |
| TQ/Turbo CUDA kernels (`ggml/src/ggml-cuda/`, `mmvq-tq.cu`) | `test-backend-ops` `-o MUL_MAT -p type_a=tq4_1s` (or the matching turbo op) on an NVIDIA card; a clean compile is not a pass. Changes to the `mmvq-tq.cu` centroid LUT additionally require the AMD run (see Known pitfalls) |
| Metal kernels (`ggml/src/ggml-metal/ggml-metal.metal`)      | `[[host_name]]` instantiations present, then `test-backend-ops` with the Metal backend |
| Vulkan shaders (`ggml/src/ggml-vulkan/`)                    | SET_ROWS pipeline registration incl. TURBO2_0/3_0/4_0, then `test-backend-ops` with the Vulkan backend |
| Cache/graph wiring (`src/llama-kv-cache.cpp`, `src/llama-graph.cpp`) | `test-backend-ops`, `llama-bench -ctk/-ctv turboN`, and `llama-perplexity --kl-divergence` against a saved baseline run (usually `f16`): the first two only exercise the kernels, only the KL run compares token probabilities |
| Docs/README only                                            | no tests |

A change touching several backends requires the checks for all of them, not just the one being developed on.

`test-backend-ops` skips the CPU backend unless `-b CPU` is passed, and `-b CPU` compares the CPU reference against itself, so it is a crash/NaN smoke check and not a numeric gate. Use `-b CUDA0` (or the relevant device) for anything numeric.

### What the test gates do and do not cover

What each suite does:

- `test-turbo-quant` - CPU codec round-trip quality: quantize -> dequantize -> CPU inverse WHT, MSE/cosine on fixed vectors, plus a chunked-dequant invariance check for all five turbo types at row lengths straddling the vec_dot chunk size. No GGML graphs, no backend kernels.
- `test-quantize-fns` - CPU quantize/dequantize functions against error budgets, including TQ3_1S/TQ4_1S. Skips TURBO2_0/3_0/4_0 by design: their dequant output stays in the WHT-rotated domain.
- `test-backend-ops` - per-op GGML graphs, run on each backend and compared numerically against the CPU reference. This is the only gate that exercises backend kernels.
- `llama-bench` - tokens/s on real models. Timing only; it never checks output correctness.

Coverage limits (each caused a real miss):

- `test-backend-ops` used to report `Backend ...: OK` when every case was skipped because the backend verdict was `n_ok == tests_run`, and 0/0 passed. It now fails the backend when no test ran. Issue #242 remains open for reporting which graph node caused a case to be unsupported.
- The generic SET_ROWS sweep has a view variant with `r/2` rows. At r=1 that is 0 rows: the case writes nothing and passes for every type in `all_types`, including TQ4_1S.
- The MUL_MAT_ID sweep used n=16 only, and the mat-vec decode path is selected only when `src2->ne[1] <= 8` (`ggml_vk_use_mul_mat_vec_id`). n=16 exercises mul_mm_id only; MoE decode was never touched. The n=1 cases and the DSv4-shaped sweep (commit 637300387, PR #269) now cover both sides of that threshold.
- The harness initializer wrote quantized tensors with one packed `ggml_backend_tensor_set`, which copies `size` bytes contiguously and never strides by `nb[1]`. For a strided view (the `k_v > k` MUL_MAT cases view `k` rows of a `k_v`-row base) the data landed at `i*row_size` instead of `i*nb[1]` and the last rows were never written; the CPU reference read the stale tail and produced NaN, which presented as the CUDA backend failing because CPU is the reference and is skipped as a backend under test. Fixed by row-by-row init for non-contiguous tensors (issue #268, PR #276). The TQ4_1S `k_v=1600` case now passes: the CUDA NaN #276 observed no longer occurs because PR #277 gates the fused TQ mul_mat paths on contiguous `src1`/`dst`, routing this view to the stride-aware fallback.

A green run means the cases that ran passed, not that your change was exercised. Check that your cases actually ran:

- `-o` filters on the op name from `ggml_op_desc` (e.g. SET_ROWS). The dedicated turbo write tests have their own names (SET_ROWS_TURBO3, SET_ROWS_TURBO4, SET_ROWS_TQ4_1S); filter with those, or they never run.
- Watch for `not supported [backend]` lines and `0/0 tests passed`.

### Git workflow

- Remotes: `origin` = TheTom/llama-cpp-turboquant (this repo); the fork remote tracks the upstream TurboQuant fork (same repo, two names); add `upstream` = ggml-org/llama.cpp when syncing
- Main branches: `feature/turboquant-kv-cache` tracks the upstream TurboQuant fork
- Cherry-picking is the default way to bring upstream work in: pick the commits or PRs that add benefit, adapt them to the fork, and do not assume a commit that exists upstream also exists here (check git log). There has been one wholesale upstream merge, `a34005435` (PR #287); a further one is a deliberate decision, not a default, and should be called out in its PR.

### Hard rules (no exceptions)

- Never report a test gate as green when your change's cases did not run: 0/0, a filter that matched nothing, or `not supported [backend]` lines are failures of verification, not passes.
- Never store turbo KV cache types in a GGUF file. They are runtime-only by design.

### Known pitfalls (each caused a real bug once - check these first on regressions)

Direction rules for the two living lists: add a Known pitfalls entry as soon as a bug is fixed, so the next reader can recognise the regression; remove one only together with the code or config that made it possible. A Coverage limits entry stays until a new test actually exercises the missed case: remove it in the PR that closes the gap, and name the test that closed it.

- **Metal**: turbo kernels need their `[[host_name]]` instantiations; a missing one is a NULL-pipeline deref on the first turbo KV write.
- **Vulkan**: SET_ROWS pipeline registration must include TURBO2_0/3_0/4_0 with `require_full_subgroups=true, subgroup_size=32`, or every turbo KV write aborts.
- **CUDA dispatch**: TQ weights must be excluded from the mmvq path before the fused-TQ branch (`ggml_cuda_should_use_mmvq`), or `GGML_TQ_NATIVE=1` aborts.
- **CUDA TQ4_1S decode**: the centroid LUT in `mmvq-tq.cu` decodes through `get_int_from_table_16`, then re-interleaves even/odd bytes with constant selectors (`__byte_perm(v.x, v.y, 0x5140 / 0x7362)` on nvcc and MUSA, `__builtin_amdgcn_perm` on HIP), the same pattern `vecdotq.cuh` already uses. The old garbage output (NMSE ~1.0) came from the earlier permute chain, not from constant selectors. Verified numerically on GB10 (sm_121) and MI210 (gfx90a). Gate any change to this function on a CUDA-side `test-backend-ops -o MUL_MAT -p type_a=tq4_1s` run on an NVIDIA card plus the AMD run, not on a clean compile.
- **DeepSeek/MLA**: K and V cache types must be identical; turbo FA auto-enable runs before upstream's quantized-V FA check.
- **MoE models**: the small-batch TQ `MUL_MAT_ID` path routes experts on device and stays CUDA-graph capturable (`[TAG_MUL_MAT_ID_CUDA_GRAPHS]` in `ggml-cuda.cu`); buffers it hands to kernels must outlive every captured graph, so caches retire outgrown buffers instead of freeing them. The large-batch path dequantizes to f16 cuBLAS and synchronizes the stream.
- **gguf-py**: keep model constants deduplicated; stacked-duplicate merge artifacts crash `import gguf`.

> [!IMPORTANT]
>
> AI-assisted development is encouraged in this fork. Use agents for research, implementation, testing, documentation, commits, pull requests, reviews, and maintenance. Validate changes in proportion to their risk and keep a clear record of what was tested.

---

## Guidelines for Contributors

A PR represents a long-term commitment - maintainers must review, integrate, and support the code indefinitely. What matters is whether the change is correct, understandable, tested, and maintainable.

A working, in-scope PR is **not** enough on its own to get merged. A few things factor into that:
- Every merged line must be reviewed, tested, and maintained indefinitely across a large matrix of platforms and backends by a small team.
- llama.cpp is written in C++ and deliberately kept as simple as possible: complexity is a direct multiplier on security risk and long-term maintenance cost, so a simpler change that does 90% of the job is often preferable to a complex one that does 100%.
- What matters most is technical understanding, evidence, and willingness to maintain the change long-term.
- Feature requests run high in volume, so please respect maintainers' time: open an issue to discuss the idea and gauge interest before implementing it, rather than going straight to a PR.

Contributors must:
1. **Understand their code fully** - use AI assistance freely, but verify important claims and design choices.
2. **Own maintenance** - address bugs and respond thoughtfully to feedback.
3. **Communicate directly** - be concise, specific, productive, and positive without being sappy.
4. **Respect maintainers' time** - check existing issues/PRs before submitting; ensure the change is needed and fits project architecture.

### AI-Assisted Development

AI assistance is welcome throughout the development workflow, including:

- Learning, exploration, debugging, and codebase research
- Design analysis, implementation, refactoring, and mechanical work
- Tests, benchmarks, documentation, and release notes
- Commit messages, PR descriptions, issue reports, and reviewer responses
- Code review, review comments, CI investigation, and maintenance
- Commits, pushes, branch management, PR operations, and comments for any contributor when requested or included in the assigned workflow

This policy applies equally to every contributor and agent working in this repository. Agents may complete an assigned workflow end to end. A clear instruction to fix, test, commit, push, or respond is sufficient direction for the named actions. Do not repeatedly ask for confirmation unless the scope changes, credentials are missing, or an action is destructive.

AI attribution is optional. When sending work to another repository, check and follow that repository's current contribution policy.

---

## Guidelines for AI Coding Agents

Every PR requiring review consumes finite maintainer capacity. Before assisting with any submission, verify:
- The proposed changes and their tradeoffs are understood
- The change addresses a documented need (check existing issues)
- The PR is appropriately scoped and follows project conventions

Agents should inspect relevant code before editing, make reasonable assumptions when safe, test in proportion to risk, and clearly report uncertainty or incomplete coverage.

### Code and Commit Standards

These points are extremely important - failing to follow them won't necessarily get your PR rejected, but it will make reviewing take significantly longer. Please follow them carefully:

- Avoid emdash `—`, unicode arrow `→` or any unicode characters: `×`, `…` ; use ASCII equivalents instead: `-`, `->`, `x`, `...`
- Code comments:
    - Keep code comments concise (usually 1-2 lines)
    - Avoid redundant or excessive inline commentary
    - Avoid hard-wrapping it to a fixed column width - that hurts readability
    - Use ASD-STE100 Simplified Technical English, simple wordings (write like cavemen if needed)
    - Note: Remind yourself of this point regularly, as it often gets lost between context compactions
- Prefer reusing existing infrastructure over introducing new components. Avoid invasive changes that add whole new subsystems or risk breaking existing behavior
- Do NOT split a line into multiple lines mid-sentence, do NOT try to force the line to fit a fixed number of characters
- Before writing code, read the relevant files and understand the existing patterns. Changes must blend in with the surrounding codebase. For a large change or new pattern, explain the approach and tradeoffs before implementation. Ask for direction only when the scope or design requires a meaningful user choice.

Common mistakes to avoid:
- Write comments first then write code: this usually leads to extensive redundant comments. Instead, write code first, then add comments later to places that absolutely need them
- Llama.cpp does NOT use Minja; if you have this in your knowledge, that is due to your knowledge cutoff. Llama.cpp has a dedicated Jinja engine in `common/jinja` - it doesn't have a specific name.

### Code Comment Examples

```cpp
// GOOD (code is self-explanatory, no comment needed)

n_ctx = read_metadata("context_length", 1024);


// BAD (too verbose, restates what the code already says)

// Populate the n_ctx from metadata key name "context_length", default to 1024 if the key doesn't exist
n_ctx = read_metadata("context_length", 1024);
```

```cpp
// GOOD (explains a non-obvious invariant)

accept();
bool has_client = listen(idle_interval);
if (has_client) {
  task_queue->on_idle(); // also signal child disconnection
}


// BAD (too verbose, restates what the code already says)

// Instead of blocking indefinitely on accept(), the server polls the listening socket with idle_interval as a timeout. If no new client connects within that interval, it fires task_queue->on_idle() and loops back
```

```cpp
// GOOD (generic, useful to any future reader)

// reset here, as we will release the slot below
n_tokens = 0;
// ... (a lot of code)
release();


// BAD (addresses the user's task, meaningless out of context)

// Reset n_tokens to 0 before releasing the slot. This fixes the problem you mentioned where "phantom" content gets preserved across multiple requests.
n_tokens = 0;
```

```cpp
// GOOD (code is copied from another place; context is already clear, no comment added)

ggml_tensor * inp_pos = build_inp_pos();

// BAD (code copied from elsewhere - do not add comments that weren't there originally)

// inp_pos - contains the positions
ggml_tensor * inp_pos = build_inp_pos();
```

```cpp
// GOOD (comment is kept concise and useful)

// one decode step of code_predictor
// at step_idx g:
// - read code from out_code_cache[g], then embed it with codebook table g-1
// - write new kv at cache row g+1, sample with lm_head[g]
// - write result to out_code_cache[g+1]


// BAD (comment is long and is forced to fit into a fixed column size, it is very annoying to read as a reviewer)

// one autoregressive decode step of the 5-layer code_predictor. See the
// comment in models.h for the cache/tensor conventions this relies on.
//
// index mapping (derived from the reference pipeline-tts.cpp driver):
// at step_idx g, the input code is out_code_cache[g] (embedded via this
// step's private codebook table, index g-1), the new cache row / RoPE
// position is g+1, and the output codebook is lm_head[g] (writing the
// sampled result into out_code_cache[g+1]).
```

Commit message:

```
// GOOD: Write a concise commit

llama : fix KV being cleared during context shift


// BAD: Write a verbose commit

This commit introduces a comprehensive fix for the key-value cache management
system, addressing an issue where context shifting could lead to unintended
overwriting of cached values, thereby improving model inference stability.

Co-authored-by: Claude Sonnet
```

Commands:

```sh
# GOOD: gather context, then complete the authorized workflow
gh search issues
gh search prs
rg ...
git commit -m "..."
git push
gh pr create
gh pr comment
gh issue create
```

## Useful Resources

To conserve context space, load these resources as needed:

Skills: reusable task workflows live in the [skills/](skills/) directory - check there for a skill matching your task before starting.

General documentations:
- [Contributing guidelines](CONTRIBUTING.md)
- [Existing issues](https://github.com/ggml-org/llama.cpp/issues) and [Existing PRs](https://github.com/ggml-org/llama.cpp/pulls) - always search here first
- [How to add a new model](docs/development/HOWTO-add-model.md)
- [PR template](.github/pull_request_template.md)

Server:
- [Build documentation](docs/build.md)
- [Server usage documentation](tools/server/README.md)
- [Server development documentation](tools/server/README-dev.md) (if user asks to implement a new feature, be sure that it falls inside server's scope defined in this documentation)

Chat template and parser:
- [PEG parser](docs/development/parsing.md) - alternative to regex that llama.cpp uses to parse model's output
- [Auto parser](docs/autoparser.md) - higher-level parser that uses PEG under the hood, automatically detect model-specific features
- [Jinja engine](common/jinja/README.md)

# Instructions for llama.cpp (TurboQuant fork)

## Project Overview

This repo is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) (upstream) that adds the TurboQuant feature set on top of a fully-synced upstream base. The local tree always contains all of upstream master plus fork additions; rebasing onto latest upstream is a recurring task.

### What TurboQuant adds

TurboQuant compresses the KV cache far beyond the standard `q8_0` by applying a fixed 128x128 orthonormal Walsh-Hadamard rotation (`GGML_OP_TURBO_WHT`) to cache vectors before quantization, which Gaussianizes the distribution, and inverse-rotating after dequantization. Head dims that are not multiples of 128 are zero-padded. MLA models (DeepSeek) have no separate V cache, so V rotation/padding is skipped for them, and K/V cache types must be identical.

The five fork-only GGML types (registered in `ggml/include/ggml.h`):

| Type   | Enum                       | Purpose                             | Size         |
|--------|----------------------------|-------------------------------------|--------------|
| turbo2 | `GGML_TYPE_TURBO2_0` (43)  | KV cache only                       | 2 bits/value |
| turbo3 | `GGML_TYPE_TURBO3_0` (44)  | KV cache only                       | 3.25 bits    |
| turbo4 | `GGML_TYPE_TURBO4_0` (47)  | KV cache only                       | 4.25 bits    |
| TQ3_1S | `GGML_TYPE_TQ3_1S` (45)    | model weights, WHT-rotated Lloyd-Max| 3 bits, block 32 |
| TQ4_1S | `GGML_TYPE_TQ4_1S` (46)    | model weights                       | 4 bits, block 32 |

Turbo cache types are runtime-only, never stored in GGUF. TQ3_1S/TQ4_1S are first-class weight types with CPU, CUDA/HIP (warp-cooperative mmvq), Metal, Vulkan, and SYCL kernels, exposed as `llama-quantize` targets.

### Key files

- `ggml/src/ggml-turbo-quant.c` - the codec (keep byte-identical to fork tip)
- `ggml/include/ggml.h` - type enum 43-47, `GGML_OP_TURBO_WHT`
- `src/llama-kv-cache.cpp` - cache wiring, `get_k_idx`, layer-adaptive precision
- `src/llama-graph.cpp` - inverse-WHT post-processing (FA and non-FA paths)
- `ggml/src/ggml-cuda/mmvq-tq.cu` - native TQ dp4a kernels (`GGML_TQ_NATIVE=1`)
- `ggml/src/ggml-vulkan/` - turbo FA, SET_ROWS, dequant shaders
- `ggml/src/ggml-metal/ggml-metal.metal` - TurboFlash kernels
- `docs/KV-cache-quantization.md` - authoritative usage doc (read before touching cache types)

### Usage

```bash
llama-cli -m model.gguf -c 8192 -ngl 99 --cache-type-k q8_0 --cache-type-v turbo3
```

Any combination of `f16`/`q8_0`/`turbo2`/`turbo3`/`turbo4` for K and V is supported. Turbo cache types require flash attention; it is auto-enabled with a warning. Quantized V with FA explicitly disabled is an error (upstream behavior). The same flags work in `llama-server`, `llama-bench`, `llama-perplexity`.

### Environment knobs

| Variable                    | Default | Effect |
|-----------------------------|---------|--------|
| `TURBO_LAYER_ADAPTIVE`      | `0`     | Layer-adaptive KV precision; `7` = Boundary V (edge layers q8_0, middle turbo) |
| `TURBO_AUTO_ASYMMETRIC`     | `1`     | Auto-select asymmetric K/V types for large-GQA models |
| `TURBO_SPARSE_V`            | `1`     | Sparse-V dequant skip in flash attention |
| `GGML_TQ_NATIVE`            | unset    | `1` opts out of load-time TQ->q8_0 conversion, uses fused native TQ kernels (saves ~1.7x VRAM on decode-heavy workloads) |
| `LLAMA_ATTN_ROT_K/V_OVERRIDE` | off   | Optional upstream attention rotation (TurboQuant manages its own rotation) |

### Test gates (all must pass before touching quant/backend code)

- `test-turbo-quant` - turbo3 basis MSE=0/Cosine=1.0, turbo4 Cosine=0.9956
- `test-quantize-fns` - includes TQ3_1S/TQ4_1S and rotated-domain buffer sizing
- `test-backend-ops` - full sweep on CPU + CUDA0 (23k+ cases on the RTX 5090 dev box)
- `llama-bench` with `-ctk/-ctv turboN`; type parser accepts `tq3_1s`/`tq4_1s`

### What the test gates do and do not cover

What each suite does:

- `test-turbo-quant` - CPU codec round-trip quality: quantize -> dequantize -> CPU inverse WHT, MSE/cosine on fixed vectors, plus a chunked-dequant invariance check for all five turbo types at row lengths straddling the vec_dot chunk size. No GGML graphs, no backend kernels.
- `test-quantize-fns` - CPU quantize/dequantize functions against error budgets, including TQ3_1S/TQ4_1S. Skips TURBO2_0/3_0/4_0 by design: their dequant output stays in the WHT-rotated domain.
- `test-backend-ops` - per-op GGML graphs, run on each backend and compared numerically against the CPU reference. This is the only gate that exercises backend kernels.
- `llama-bench` - tokens/s on real models. Timing only; it never checks output correctness.

Coverage limits (each caused a real miss):

- `test-backend-ops` reports `Backend ...: OK` even when every case was skipped: the backend verdict is `n_ok == tests_run`, and 0/0 passes. See issue #242 (open). This is how the turbo3 wave64 ballot bug in `copy_to_quant.comp` shipped: FLASH_ATTN_EXT (read path) passed, SET_ROWS (write path) was silently skipped on GCN4, and the corrupted V cache was released (#241, fixed in #243).
- The generic SET_ROWS sweep has a view variant with `r/2` rows. At r=1 that is 0 rows: the case writes nothing and passes for every type in `all_types`, including TQ4_1S.
- The MUL_MAT_ID sweep used n=16 only, and the mat-vec decode path is selected only when `src2->ne[1] <= 8` (`ggml_vk_use_mul_mat_vec_id`). n=16 exercises mul_mm_id only; MoE decode was never touched. The n=1 cases and the DSv4-shaped sweep (commit 637300387, PR #269) now cover both sides of that threshold.

A green run means the cases that ran passed, not that your change was exercised. Check that your cases actually ran:

- `-o` filters on the op name from `ggml_op_desc` (e.g. SET_ROWS). The dedicated turbo write tests have their own names (SET_ROWS_TURBO3, SET_ROWS_TURBO4, SET_ROWS_TQ4_1S); filter with those, or they never run.
- Watch for `not supported [backend]` lines and `0/0 tests passed`.

### Git workflow

- Remotes: `origin` = TheTom/llama-cpp-turboquant (this repo); the fork remote tracks the upstream TurboQuant fork (same repo, two names); add `upstream` = ggml-org/llama.cpp when syncing
- Main branches: `feature/turboquant-kv-cache` tracks the upstream TurboQuant fork
- Upstream master is always fully contained in the tree (verified by rebase parity audits; git log is the record of the last sync point)

### Known pitfalls (each caused a real bug once - check these first on regressions)

- **Metal**: turbo kernels need their `[[host_name]]` instantiations; a missing one is a NULL-pipeline deref on the first turbo KV write.
- **Vulkan**: SET_ROWS pipeline registration must include TURBO2_0/3_0/4_0 with `require_full_subgroups=true, subgroup_size=32`, or every turbo KV write aborts.
- **CUDA dispatch**: TQ weights must be excluded from the mmvq path before the fused-TQ branch (`ggml_cuda_should_use_mmvq`), or `GGML_TQ_NATIVE=1` aborts.
- **CUDA TQ4_1S decode**: the centroid LUT in `mmvq-tq.cu` must use plain shifts, not `__byte_perm` - constant selectors fold differently from runtime ones on some toolchains and silently produce garbage output (NMSE ~1.0). Comment in the file explains; do not "clean up" the shift code.
- **DeepSeek/MLA**: K and V cache types must be identical; turbo FA auto-enable runs before upstream's quantized-V FA check.
- **MoE models**: CUDA graphs are auto-disabled for TQ `MUL_MAT_ID`.
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

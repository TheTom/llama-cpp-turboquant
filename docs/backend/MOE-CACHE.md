# MoE expert cache

The MoE expert cache accelerates decode for Mixture-of-Experts (MoE) models when routed expert weights remain in host (CPU) memory. A cache hit runs the selected expert matvec on the GPU while the CPU computes the miss rows through the normal `MUL_MAT_ID` kernel. Exact gate/up/SwiGLU subgraphs can fuse rows resident in both weight tensors while half-resident and missing rows stay on the stock CPU path. The cache belongs to one backend scheduler and persists until that scheduler is destroyed.

This is an opportunistic path. Unsupported nodes, unavailable cache capacity, contention, and cache failures fall back to CPU execution. The same provider interface is implemented by the CUDA, Metal, and Vulkan backends (HIP builds share the CUDA implementation); the detailed controls documented here are read by each backend when it creates its cache session.

## What is it for

MoE models can be far too large to fit entirely in VRAM, so `llama.cpp` keeps the routed expert weights in host memory and reads them across the PCIe link on every decode step. The expert cache keeps the *working set* of recently-used experts resident in VRAM, so decode no longer pays the host-memory round trip for the experts the model is actually routing to right now. Over a decode-heavy workload, most tokens route to a small recurring set of experts, so a modest cache can serve most matvec rows from the GPU.

The cache is for **models whose routed experts do not fit in VRAM**, or for configurations that deliberately keep experts in host memory. The typical beneficiary is a large MoE model loaded across a few GPUs where the dense weights fit but the experts spill to CPU:

| Model | Cache off | Cache on | Change |
| --- | ---: | ---: | ---: |
| OLMoE-1B-7B Q4_K_M | 200.52 t/s | 254.12 t/s | +26.7% |
| ERNIE-4.5-21B-A3B Q4_K_M | 79.66 t/s | 127.40 t/s | +59.9% |
| Qwen3-30B-A3B Q4_K_XL | 71.93 t/s | 73.43 t/s | +2.1% |
| Qwen3.6-35B-A3B Q4_K_XL | 77.60 t/s | 84.04 t/s | +8.3% |
| DeepSeek-V2-Lite Q4_K_M | 65.04 t/s | 81.61 t/s | +25.5% |
| Llama-4 Scout Q4_K_XL | 25.97 t/s | 46.90 t/s | +80.6% |
| MiniMax M2.7 IQ2_XXS | 37.76 t/s | 47.81 t/s | +26.6% |

This matrix is evidence for the tested hardware, not a guarantee that every eligible GPU and model will improve.

### When it does not help

- **A model that fits completely in VRAM is not the target.** When every expert is already GPU-resident, there is nothing to cache: `auto` resolves to `off`, the cache stays dormant, and decode runs on the stock path. This is by design and costs nothing measurable (a dense control measured `182.05 t/s` in `auto` versus `182.68 t/s` off; a fully resident Qwen3.6-35B measured `144.80 t/s` off versus `144.68 t/s` auto).
- **Small models with tiny expert weights** can be below the minimum expert-size floor (see `GGML_CUDA_MOE_CACHE_MIN_EXPERT_KB`) and remain dormant because the transfer overhead would exceed the compute saved.
- **Prompt processing** is not cached. Prompt nodes above the token limit bypass the cache entirely; the benefit is decode, not prompt ingestion.
- **Dense (non-MoE) models** never engage the cache.

## Quick start

`auto` is the default, so a plain `llama-cli`/`llama-server`/`llama-bench` run already probes for cacheable CPU-resident experts. When the model fits in VRAM, nothing happens (see above); when experts spill, the cache engages with the automatic budget.

To get cache-aware placement (the recommended setup for a model that needs it), combine fit with the cache:

```sh
./build/bin/llama-server -m /path/to/model.gguf \
    --fit on --moe-cache auto \
    -c 8192 -ngl 99 -fa on
```

To force canonical CPU experts and a cache session regardless of placement:

```sh
./build/bin/llama-server -m /path/to/model.gguf --moe-cache on -ngl 99
```

To cap the per-device budget:

```sh
./build/bin/llama-server -m /path/to/model.gguf --moe-cache 4096 -ngl 99
```

To verify the cache actually engaged, look for `[moe-cache] enabled` in the log. If the cache was disabled, the eligibility probe prints the reason at normal verbosity: `MoE cache disabled (...)` lines name the blocking gate (device capability, expert size, budget, or the absence of cacheable host tensors), and `[moe-cache] session creation failed: ...` lines name a backend-side failure.

## Configuration

Use `--moe-cache MODE` with programs that use the common argument parser:

| Mode | Cache budget | Weight repacking | Device requirements |
| --- | --- | --- | --- |
| `auto` | Free VRAM minus the reserve | Preserved unless cache-aware fit selects canonical CPU experts | At least one eligible selected CUDA device that clears the 1 GiB automatic slab floor, compute capability 8.0 or newer |
| `on` | Free VRAM minus the reserve | Disabled | At least one eligible selected CUDA device, compute capability 7.0 or newer |
| `N` | At most `N` MiB per device, after the reserve | Disabled | Same as `on` |
| `soft` | Fit first tries spare VRAM with stock placement, evicting experts only as needed | Preserved unless cache-aware fit selects canonical CPU experts | Same as `on` |
| `off` or `0` | No cache session | Preserved unless changed separately | None |

`auto` is the default. When the complete model fits, or dense weights do not fit without routed experts, normal placement and repacking are preserved. When routed experts must spill and a cache is feasible, cache-aware fit keeps canonical expert weights in host memory, puts all dense layers on the main GPU when possible, and preserves the remaining VRAM for cache pools. Use `on` or a positive budget to force canonical CPU weights when automatic placement is not wanted. `soft` is like `on` with a gentler fit: it first tries to keep stock placement and fit the cache into spare VRAM, evicting experts only as needed. Compute capability floors are evaluated by CUDA and HIP devices; the Metal and Vulkan providers always report Ampere-class capability (800).

A single device is eligible for `auto` when free VRAM minus the reserve clears the 1 GiB automatic slab floor. Below that floor `auto` stays dormant so small pools do not thrash; shrink `GGML_CUDA_MOE_CACHE_RESERVE_MB` (default 3072 MiB) to give the floor room on a constrained card.

A single device is eligible for `auto` when free VRAM minus the reserve clears the 1 GiB automatic slab floor. Below that floor `auto` stays dormant so small pools do not thrash; shrink `GGML_CUDA_MOE_CACHE_RESERVE_MB` (default 3072 MiB) to give the floor room on a constrained card.

The cache only sees experts assigned to host memory. Cache-aware fit uses a no-allocation model load to inventory exact expert shapes, aliases, model memory, context memory, and compute memory before choosing between stock and cache placement. Explicit `--cpu-moe`, `--n-cpu-moe`, tensor overrides, GPU layers, or tensor splits remain authoritative and can prevent fit from changing placement.

The cache considers only backends selected for the scheduler. It does not discover or use an unselected device.

## Eligibility and allocation

With default settings, a node must satisfy all of these conditions:

- The operation is the regular CPU `MUL_MAT_ID` path with F32 activations, optionally in an exact supported gate/up/SwiGLU subgraph.
- The weight tensor name contains `_exps` or `_chexps`.
- The weight type is supported by the backend quantized matvec kernel.
- One expert meets the selected devices' effective size floor. The default is 512 KiB when every selected device is compute capability 8.0 or newer and 1 MiB otherwise. An explicit threshold remains authoritative.
- The graph node contains no more than the configured maximum token batch and no more than 64 routed rows. The default maximum is eight tokens in every active mode.
- The selected device can hold a pool of at least 64 experts of that shape. Entries are aggregated across same-shape tensors, so an individual tensor may contain fewer than 64 experts.
- In `auto`, each selected device has at least 1 GiB available for aggregate cache slabs after dispatch scratch. Forced modes retain the 64-slot floor for explicit capacity experiments.

Each physical device budget is coordinated across every target, draft, MTP, and server scheduler in the process. It is claimed once, at first eligible use. Shape inventory counts the exact number of experts in every observed tensor; mixed expert counts do not reserve nonexistent slots. Device reservations are tracked per participating session, so releasing a high-reserve target immediately restores the correct capacity for the remaining sessions. The automatic budget is:

```
min(configured budget, free VRAM - 3072 MiB reserve)
```

The configured-budget term is omitted for `auto` and `on`. A fixed `N` is a cap for cache slabs and device-side dispatch scratch together, not a guaranteed slab allocation. Pool allocation may be smaller after reserving scratch or retrying an allocation, and no pool is created when fewer than 64 slots fit.

Tensor shapes are collected before pools are allocated. Allocation waits for at least as many repeat visits as tensors observed so early graph discovery does not give all capacity to the first tensor shape. Capacity is divided among discovered shapes. A layer is assigned once to a selected device and keeps that assignment while the device remains usable. If that device cannot host a different tensor shape from the layer, the tensor receives its own stable assignment. Initial assignments are deterministic and weighted by usable slab capacity after existing pools and dispatch scratch are accounted for.

When every routed row is resident, the cache keeps a bounded amount of work on the CPU while the GPU computes the other rows. The automatic work budget is 8 MiB of expert weights per token, then capped by token count, routing width, and one quarter of the node. Four-GPU sweeps found the automatic result at or within 0.6% of the best tested fixed row count on Qwen3-30B, DeepSeek V4 target-only and DSpark, and GLM-5.2 MTP. Nodes with an actual miss are unchanged.

Common applications report an explicit or cache-aware-fit-selected mode at normal verbosity. The eligibility probe and session creation print the reason whenever the cache is disabled or cannot engage: `MoE cache disabled (...)` lines name the blocking gate, and `[moe-cache] session creation failed: ...` lines name a backend-side failure. Detailed backend messages use trace verbosity, so pass `-lv 4` for per-pool and statistics detail. The `[moe-cache] enabled` message is printed only after the first pool is allocated. If it is absent from a log at `-lv 4`, the cache did not become active, and the earlier reason lines explain which gate blocked it.

## Demand fill and eviction

The cache never transfers a missing expert synchronously for the current matvec. The miss stays on the CPU, and a bounded background request may populate the expert for a later use.

By default:

- A pool that can hold every discovered entry admits an expert after its first miss. A capacity-constrained pool admits after the second miss.
- At most 8 fills are enqueued by one node.
- A device queue is limited to 128 jobs and 512 MiB.
- Each device has one low-priority fill stream.
- Independent device workers run concurrently when at least two compute capability 8.0 or newer devices are selected. Fills remain serialized on single-device or older-device configurations.
- Full pools use heat-aware eviction. The victim is the least-recently-used expert whose resident hit count is at or below `GGML_CUDA_MOE_CACHE_HOT_USES` (default 4); only when every eligible slot is hot does the policy fall back to the LRU head. A frequently-used expert therefore stays resident instead of being evicted just because a few tokens skipped it.
- Replacement admission is throttled independently: after a successful fill, an expert needs `GGML_CUDA_MOE_CACHE_THROTTLE` (default 8) fresh misses before it can replace another full-pool entry.

The default maximum is eight tokens. Larger prompt-processing nodes bypass the cache, while single-token decode and current speculative verification batches remain eligible in `auto`, `on`, and fixed-budget modes.

## DeepSeek V4 Flash and MXFP4

The DeepSeek V4 Flash GGUF uses the conventional `ffn_gate_exps`, `ffn_up_exps`, and `ffn_down_exps` tensors. These names contain `_exps`, and CPU-resident canonical weights reach the regular `MUL_MAT_ID` path, so the MXFP4 experts are compatible with the cache. For this model, the gate and up projections have `(n_in, n_out) = (4096, 2048)` and the down projection has `(n_in, n_out) = (2048, 4096)`. Each projection occupies 4,456,448 bytes per expert.

Automatic mode accepts up to eight tokens per node. This covers the current DSpark target-verification batch, whose tested node shape is six tokens selecting six experts each. Prompt nodes above eight tokens bypass the cache and do not drive admission. The model-free cache test compares the 6 x 6 hit path with the stock CPU result.

The server understands the target/draft dependency and is the supported DSpark entry point. With automatic fit, draft-device selection, and cache policy, a four-GPU launch is:

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/bin/llama-server \
    -m /path/to/DeepSeek-V4-Flash-0731.gguf \
    -md /path/to/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf \
    --spec-type draft-dspark --spec-draft-n-max 5 \
    --fit on --moe-cache auto \
    -t 24 -tb 48 -fa on -c 8192 -np 1 \
    -ctk q8_0 -ctv q8_0
```

When no draft device is explicit, the server places the DSpark sidecar on the eligible GPU with the most available memory, excluding the target main device when possible. The draft device is moved to the front of its own device list and receives its one-layer split while the remaining target backends stay available. The server measures this draft placement against both the packed and main-device no-allocation target placements, then reserves the per-device maximum before fitting. An explicit `--spec-draft-device` remains authoritative for draft weights, but target backends are appended when needed because DFlash and DSpark share the target embedding and output tensors. `--spec-draft-device none` keeps the sidecar weights on CPU while retaining those target backends for the shared operations. Sleep/wake reload starts from the original placement request and repeats measurement and fitting instead of accumulating old reservations.

On the measured 4x RTX 3090 system, the current target-only fused path reached `28.24 +/- 0.05` t/s. Disabling fusion in the same source reduced it to `27.64 +/- 0.02` t/s, a 2.18% isolated gain; an earlier cache-off control was `23.20 +/- 0.05` t/s. A final DSpark rerun produced six exact settled responses from 69.54 to 70.25 t/s, averaging 70.06 t/s, with 213 of 215 draft tokens accepted in every request. The controlled fusion comparison averaged 69.39 t/s with multi-token fusion and 65.86 t/s on the preceding single-token-fusion branch, a 5.37% isolated gain. The earlier cache-off control was 51.63 t/s. Auto-fit projected 53,636 MiB of target cache capacity after reserving the sidecar, and the loaded target allocated 57,671 MiB of pools. Explicit GPU, CPU-only draft, and sleep/wake reload paths also produced correct output. A low-acceptance prose request remained coherent at about 27.7 t/s, illustrating that speculative gain depends on the workload rather than the cache hit rate alone. Retain the 3072 MiB reserve or remeasure when increasing context, slots, or other CUDA allocations.

### Measured DeepSeek V4 validation

The following canonical-weight scaling sweep used the Q8_K_XL target, CPU-resident experts, all dense weights on the first RTX 3090, 256 untimed generation warmup tokens, and otherwise matched cache-off and cache-on arms:

| CUDA devices | Cache off | Cache on | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 19.05 t/s | 22.54 t/s | +18.3% |
| 2 | 18.95 t/s | 26.21 t/s | +38.3% |
| 3 | 18.86 t/s | 27.47 t/s | +45.6% |
| 4 | 18.59 t/s | 27.51 t/s | +48.0% |

Putting all dense weights on the first GPU remained faster than an equal four-device tensor split. A later fixed-budget sweep with that topology reached 23.91, 25.53, 26.45, 26.92, and 27.01 t/s at 4, 8, 12, 16, and 20 GiB per device respectively. The small change above 16 GiB shows that automatic capacity is already near the useful ceiling on this model. Parallel fills were flat after warmup but improved a cold multi-GPU run by about 1.4%, so capable multi-GPU systems use them automatically.

Quality checks used the actual target GGUF, not only synthetic tensors. Perplexity over four 512-token chunks was `2.7996 +/- 0.19833` with the cache off and `2.7987 +/- 0.19824` with it on. On the final accounting code, a 54,044-token retrieval prompt returned all three exact codes placed near the beginning, middle, and end. A same-session follow-up evaluated only 18 new prompt tokens, returned the correct middle code, and remained coherent. These results show statistical quality equivalence on the tested workload, not bit-identical execution.

A target-only workload-shift check used eight unrelated prompts covering algorithms, C++ code, algebra, science, history, SQL, networking, and key rotation. The first cycle generated 1,943 tokens at a weighted 26.425 t/s; the identical prompt sequence generated 1,938 tokens at 26.408 t/s on the second cycle, a 0.06% difference. The first cold request ran at 24.99 t/s, while the remaining first-cycle requests ranged from 25.86 to 27.11 t/s. All responses were coherent, and both rounding-divergent algebra responses independently solved and verified `x = 14`. Teardown reported 85.3% to 86.5% residency-probe hits and zero cache failures. This shows quick convergence under one mixed serial workload, not a universal working-set guarantee.

Measure this model with three separate `llama-bench` arms and otherwise identical placement, context, batch, thread, and warmup settings:

1. `--moe-cache off --repack on`: optimized CPU-expert baseline.
2. `--moe-cache off --repack off`: canonical CPU-expert baseline.
3. `--moe-cache on --repack off`, or a fixed cache budget with `--repack off`: cache arm.

Other programs that use the common argument parser spell the repacking switches as `--repack` and `--no-repack` rather than `--repack on` and `--repack off`.

Comparing arm 1 with arm 3 measures the end-to-end configuration benefit; comparing arm 2 with arm 3 isolates cache execution from repacking. Run `llama-bench` with `-v`; a cache result is valid only when the trace log contains `[moe-cache] enabled`, the expected MXFP4 pools, and nonzero hits in periodic or teardown statistics. Before reporting a speedup, also compare cache off and on for perplexity or logits, coherent generation, and retrieval or prompt-following at increasing context lengths. Token streams need not be identical because CPU and CUDA rounding can change near ties.

## GLM-5.2 validation and production profile

The GLM-5.2 UD-IQ2_M tests used four RTX 3090 GPUs and the same 48-core host. The routed expert set is about 209 GiB, so this model needs a long warmup before its cache reaches steady state. Extending the same fused path from single-token nodes to MTP verification batches improved six exact-output requests from 27.38 to 29.56 t/s, or 7.96%, with statistically similar draft acceptance. A later default-policy validation produced twelve exact `1..200` responses; after the cold request, eleven samples averaged `28.251 +/- 0.527` t/s with 3272 of 3487 draft tokens accepted. An earlier cache-off configuration produced the same exact sequence at 14.46 t/s. That wider comparison is an end-to-end placement result: the off arm preserved stock placement while auto selected canonical CPU experts and retained VRAM for cache pools.

The model contains one embedded next-token prediction layer. The best measured settings remain `--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.5`. Increasing the draft maximum to five reduced throughput because acceptance fell while draft work increased.

A 58,759-token archival prompt returned all three exact retrieval markers. Its 219-token answer ran at 13.38 t/s with 159 of 173 draft tokens accepted. A same-session follow-up reused 58,769 prompt tokens, correctly classified the early, middle, and late records, and produced 294 coherent tokens at 12.84 t/s with 210 of 230 draft tokens accepted. Perplexity over four 512-token chunks was `2.2090 +/- 0.11977` with the cache off and `2.1960 +/- 0.11996` with it on.

Embedded MTP creates another context, scheduler, and cache session. Sessions retain independent cache state, but their claims are coordinated per physical GPU so target and draft contexts do not each budget the same free memory. Auto-fit measures the MTP layer-split and canonical CPU-expert main-device layouts and adds the target scheduler's compute allocation to the sidecar reservation. Using the per-device maximum prevents the target fit from consuming memory that the MTP context needs. This fixed a reproducible 2340 MiB allocation failure on the first GPU without relying on a heuristic reserve.

On three GPUs at context 65,536, target-only auto-fit settled near 16.8 t/s and the older manual target-only topology settled near 17.7 t/s. Auto-fit with MTP produced the exact expected sequence in all six requests and averaged about 24.5 t/s, approximately 38% faster than the best target-only profile. A final 8192-context three-GPU check produced six exact `1..200` responses; five settled samples averaged `25.472 +/- 0.396` t/s with 1489 of 1592 draft tokens accepted. The planner retained about 32.0 GiB of projected cache capacity at 65,536 context. Physical GPU 3 remained invisible to CUDA and stayed at 1 MiB in both runs.

A 56,045-token retrieval prompt then validated this production layout at long context. After an initial client cancellation, the replacement request reused 8,195 prefix tokens, processed the remaining 47,850 tokens at 40.74 t/s, and returned the exact middle-document key `739184`. The request completed without truncation, OOM, cache trim, or dispatch error. Shutting down the server returned all four physical GPUs to 1 MiB, including the GPU that had remained hidden throughout the run.

The validated three-GPU production profile uses cache-aware fit and embedded MTP:

```sh
CUDA_VISIBLE_DEVICES=0,1,2 \
GGML_CUDA_MOE_CACHE_RESERVE_MB=3072 \
./build/bin/llama-server \
    -m /path/to/GLM-5.2-UD-IQ2_M-00001-of-00006.gguf \
    --fit on --moe-cache auto \
    -c 65536 -np 1 -b 4096 -ub 512 \
    -t 44 -tb 48 -fa on --load-mode mmap --cache-ram 0 \
    --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.5
```

At context 65,536, a 2048-token ubatch required about 3952 MiB for the prompt graph and failed to fit; 512 was stable. Letting the planner choose placement was substantially faster than forcing all dense work onto the first GPU in this MTP configuration. The adaptive overlap, fill, admission, and matvec defaults also outperformed the previous manual overrides, so they remain automatic in the tested configuration.

`--load-mode none` was also tested against mmap with the exact 65k server profile. No-mmap took 4:31 to become ready instead of 3:03, left only about 28 GiB host memory available, increased swap use, and averaged 14.88 t/s over two steady samples versus 14.96 t/s for mmap. Mmap therefore remains the tested default for this profile.

## Broader regression matrix

These matched canonical-weight decode checks force routed experts to CPU memory. Small models used one RTX 3090; models that required or benefited from more capacity used all four. They exercise different expert counts, quantizations, and architectures:

| Model | Cache off | Cache on | Change |
| --- | ---: | ---: | ---: |
| OLMoE-1B-7B Q4_K_M | 200.52 t/s | 254.12 t/s | +26.7% |
| ERNIE-4.5-21B-A3B Q4_K_M | 79.66 t/s | 127.40 t/s | +59.9% |
| Qwen3-30B-A3B Q4_K_XL | 71.93 t/s | 73.43 t/s | +2.1% |
| Qwen3.6-35B-A3B Q4_K_XL | 77.60 t/s | 84.04 t/s | +8.3% |
| DeepSeek-V2-Lite Q4_K_M | 65.04 t/s | 81.61 t/s | +25.5% |
| Llama-4 Scout Q4_K_XL | 25.97 t/s | 46.90 t/s | +80.6% |
| MiniMax M2.7 IQ2_XXS | 37.76 t/s | 47.81 t/s | +26.6% |

The updated hardware-aware threshold made the Qwen3.6 experts eligible on four RTX 3090 devices. Its row uses 512 warmup and 512 measured generation tokens with canonical CPU experts in both arms. A fixed 4096 MiB cache reached 83.89 t/s in the same run. Llama-4 has only 16 experts per tensor, but 48 layers contribute enough entries to its shared-shape pool. Aggregating those entries instead of applying the 64-slot floor to each tensor restored cache eligibility. With 512 warmup and 512 measured generation tokens, five same-binary samples reached `25.966 +/- 0.006` t/s off, `33.541 +/- 0.320` t/s with a fixed 4096 MiB cache, and `46.901 +/- 0.235` t/s with the automatic budget. The literal default `--moe-cache auto` and default repacking reached `46.803 +/- 0.230` t/s over three samples. This exceeds the older branch's 45.03 t/s result without restoring redirect, backfill, or persistent state.

The real OLMoE model also exercised the available quant families from Q2_K through Q8_0. The matched off versus fixed 4096 MiB results were 251.54 versus 263.34 t/s for Q2_K, 215.12 versus 257.65 t/s for Q3_K_M, 200.12 versus 265.97 t/s for Q4_0, 173.49 versus 256.55 t/s for Q5_K_M, 157.75 versus 236.78 t/s for Q6_K, and 130.60 versus 224.03 t/s for Q8_0. F16 remained dormant because that cache matvec type is unsupported and preserved parity at 74.43 versus 74.78 t/s. A fully resident Qwen3.6-35B control preserved repacking and remained dormant at 144.80 t/s off versus 144.68 t/s auto.

The Ampere admission floor is 512 KiB rather than the earlier permissive 64 KiB. This is below the smallest profitable local experts: OLMoE Q2_K uses 672-880 KiB experts and reached `296.37 +/- 0.08` t/s at a 512 KiB floor versus `246.98 +/- 0.87` t/s when a 1 MiB floor made the cache dormant. Qwen3.6 uses 576-704 KiB experts and reached `92.88 +/- 0.76` versus `81.41 +/- 0.10` t/s in the same boundary test. No local measurement supports admitting experts below 512 KiB automatically. `GGML_CUDA_MOE_CACHE_MIN_EXPERT_KB` remains available for explicit experiments.

Automatic policy also requires at least 1 GiB of aggregate slab capacity per device. On OLMoE Q2_K, 160, 256, and 512 MiB caps regressed to 179.66, 203.12, and 235.29 t/s because small pools thrashed. A 640 MiB cap reached parity at 245.25 t/s, 768 MiB reached 254.21 t/s, and 1 GiB reached 268.89 t/s. This device-level floor does not replace the 64-slot shape floor: Llama-4's 22.5 MiB experts require more than 1 GiB to hold 64 slots, and 70 slots per device still improved decode from 25.97 to 31.21 t/s. Fixed modes remain available for deliberate smaller-capacity sweeps.

Prompt processing remained on the stock path. In a reverse-order Qwen3.6 2048-token prompt test, auto reached 589.07 t/s and off reached 589.38 t/s, a 0.05% difference. A dense Qwen3.5-4B control reached 182.05 t/s in auto versus 182.68 t/s off. These small differences are within run variation and show that dormant or bypassed operation does not impose a material throughput cost on the tested system.

## Concurrency, fallback, and lifetime

The scheduler binds its cache session only while it computes a graph. Within one session, per-device scratch buffers and dispatch are serialized. If another cached node in that session already owns a device, the contending node takes the regular CPU path. Separate schedulers own separate sessions, pools, streams, and dispatch locks even when they select the same physical device.

A two-slot DeepSeek server test generated unrelated Saturn and photosynthesis answers concurrently. Both responses were complete, coherent, and isolated, with no deadlock, row leakage, or cache failure. This complements the model-free concurrent-node and nested-session tests.

Valid slots are pinned for the lifetime of a node. Miss rows remain in the normal CPU work set. Hit rows are removed from that set only after the complete GPU dispatch has been accepted.

If dispatch fails, every planned hit row is restored to the CPU work set before worker threads begin. If collection fails, every skipped row is recomputed with the stock CPU helper. A fatal backend cache error disables and trims the affected device, after which nodes continue on CPU.

Output can differ slightly from CPU output because the hit path uses activation quantization and matvec arithmetic on the backend. Do not expect bit-identical logits or token streams. In particular, a small rounding difference can change a near-tie greedy token.

Public writes to a cached host weight buffer invalidate the affected byte range before the write begins. Invalidation cancels overlapping queued fills, waits for overlapping active reads and transfers, and removes overlapping slots and demand records. The same process runs before a host allocation is released or stops being a weight buffer. Callers must still obey the normal backend synchronization rules when mutating a weight used by concurrent graph execution. Scheduler teardown stops admission, cancels queued work, waits for active graph scopes and nodes, joins fill workers, and then frees device storage.

The normal backend allocator may trim an active expert cache as a last attempt to satisfy an allocation. Trimming frees all cache storage on that device and leaves it disabled for the rest of the session. Device scratch growth also retries once after releasing the superseded scratch allocation when replacement overlap causes an out-of-memory result. A session becomes permanently dormant when no nonzero device budget remains, or when `auto` has no device that satisfies its slab floor; any remaining cache devices are then trimmed as well.

## Diagnostics and developer controls

At normal verbosity, common applications report modes selected explicitly or by cache-aware fit, and the eligibility probe prints the reason whenever `auto` resolves to `off` (`MoE cache disabled (...)`). Pass `-lv 4` to see the context's requested and resolved mode and all backend diagnostics. If model shape, placement, or a fixed budget cannot satisfy automatic policy, `auto` resolves to `off` before graph execution. Backend messages are per scheduler session. With MTP or another draft context, the active target can print pools and `[moe-cache] enabled` before the secondary scheduler prints `session dormant`; that later message does not disable or trim the target session. An active backend prints its full configuration only on first eligible use, so a transient scheduler session cannot make an `off` arm look enabled. Each pool log reports the physical device, weight type, expert size, slot count, discovered entry count, complete or partial coverage, and allocated bytes. A node above the configured token limit or fixed 64-row execution bound prints one bypass warning per session and continues on the stock CPU path. Session teardown reports hits, misses, queue activity, fills, evictions, CPU-overlap rows, activation deduplication, and fallback counters for devices that processed cache nodes. `hits` and its denominator count tensor-expert residency probes. On a fused gate/up candidate, a half-resident pair therefore records one hit and one miss even though that row stays on the CPU. `fusion=A/B` separately reports the rows actually dispatched through the fused path over the candidate rows in successfully dispatched fused nodes. `pairs=both/up-only/gate-only/neither` classifies every valid gate/up candidate at the residency probe, including nodes with no dispatchable pair. Set `GGML_CUDA_MOE_CACHE_STATS=N` to print the same counters every `N` collection calls.

The following environment variables are implementation controls, not a stable command-line interface. They are read when a scheduler creates its cache session.

| Variable | Default | Meaning |
| --- | ---: | --- |
| `GGML_CUDA_MOE_CACHE_RESERVE_MB` | `3072` | VRAM left outside the cache on each device |
| `GGML_CUDA_MOE_CACHE_MIN_EXPERT_KB` | hardware dependent | Minimum bytes per expert, in KiB; `512` when all selected devices are compute capability 8.0 or newer and `1024` otherwise |
| `GGML_CUDA_MOE_CACHE_MAX_BATCH` | `8` | Maximum tokens in an eligible node |
| `GGML_CUDA_MOE_CACHE_INSERTS` | `8` | Maximum admissions per node |
| `GGML_CUDA_MOE_CACHE_ADMIT_AFTER` | adaptive | Override the initial miss count; by default it is `1` for complete pools and `2` for capacity-constrained pools |
| `GGML_CUDA_MOE_CACHE_THROTTLE` | `8` | Fresh misses required before replacing a full-pool entry |
| `GGML_CUDA_MOE_CACHE_HOT_USES` | `4` | Resident hit count at or below which a full-pool expert may be evicted; hotter entries are kept unless every slot is hot |
| `GGML_CUDA_MOE_CACHE_QUEUE` | `128` | Maximum queued jobs per device |
| `GGML_CUDA_MOE_CACHE_QUEUE_MB` | `512` | Maximum queued source bytes per device |
| `GGML_CUDA_MOE_CACHE_STATS` | `0` | Collection-call interval for periodic statistics, or `0` for teardown only |
| `GGML_CUDA_MOE_CACHE_NDEV` | all | Maximum selected devices used by a session |
| `GGML_CUDA_MOE_CACHE_SERIAL_FILL` | hardware dependent | Serialize fills across devices; defaults to `0` with at least two compute capability 8.0 or newer devices and `1` otherwise |
| `GGML_CUDA_MOE_CACHE_DEDICATED_MMV` | `0` | Force the cache-specific activation-map matvec; compatible routing uses the existing modulo-index MMV path by default |
| `GGML_CUDA_MOE_CACHE_OVERLAP_CPU_ROWS` | automatic | Rows retained on CPU only when every row is resident; an explicit value from `0` through `8` overrides the size-aware policy |
| `GGML_CUDA_MOE_CACHE_MIN_CC` | mode dependent | Override the minimum compute capability encoded as `major * 100 + minor * 10` |

Directly setting `GGML_CUDA_MOE_CACHE_MODE`, `GGML_CUDA_MOE_CACHE`, or `GGML_CUDA_MOE_CACHE_BUDGET_MB` controls the provider only when a program leaves the mode unspecified. Common applications do this for their implicit `auto` default, so provider settings affect fit and runtime consistently. An explicit `--moe-cache` or `LLAMA_ARG_MOE_CACHE` value takes precedence and also controls the model loader's repacking choice. `llama-bench` applies its selected cache arm before every model instance and overwrites the three raw backend variables; use `--moe-cache` or `LLAMA_ARG_MOE_CACHE` to select its benchmark mode.

Keep `GGML_OP_OFFLOAD_MIN_BATCH` above the decode batch size. Setting it to `1` can make the scheduler offload the complete `MUL_MAT_ID` operation to the GPU before the CPU path can split cache hits from misses.

`GGML_CUDA_MOE_CACHE_FAIL` is for fallback testing only. It accepts `dispatch`, `collect`, `insert`, or `slab`; comma-separated stages and `all` are also accepted. A CUDA build can exercise the synthetic success and failure paths with:

```sh
CUDA_VISIBLE_DEVICES=0 ./build/bin/test-moe-cache
```

## Benchmarking

Cache measurements need decode warmup. Pool creation waits for graph-shape discovery, expert admission needs repeated demand, and queued fills finish asynchronously. A one-token warmup usually measures a cold cache rather than steady state.

`llama-bench` separates devices within one tensor split with `/`, for example `-ts 1/0/0/0`; commas separate multiple benchmark values. Programs using the common parser, including `llama-server`, use commas within one split, for example `-ts 1,0,0,0`.

For an end-to-end comparison of automatic policy, let fit choose each arm and vary only `off` versus `auto`:

```sh
CUDA_VISIBLE_DEVICES=0,1,2 ./build/bin/llama-bench \
    -m /path/to/model.gguf -p 0 -n 300 --n-gen-warmup 256 -r 5 \
    -ngl 99 -fitt 1024 -t CPU_THREAD_COUNT -fa on \
    --moe-cache off,auto -v -o json
```

Replace the uppercase placeholder with an integer appropriate for the host. This measures the complete automatic choice, which may use different expert placement in the two arms. Check the logs to confirm that the `auto` arm selected cache placement and allocated pools. Record GPU models, PCIe topology, CPU model, memory channels and speed, model quantization, exact placement, build revision, and all environment overrides.

For an isolated comparison with the same canonical CPU weights in both arms, use a fixed cache budget and explicitly disable repacking:

```sh
CUDA_VISIBLE_DEVICES=0,1,2 ./build/bin/llama-bench \
    -m /path/to/model.gguf -p 0 -n 300 --n-gen-warmup 256 -r 5 \
    -ngl 99 -ncmoe MODEL_MOE_LAYER_COUNT -t CPU_THREAD_COUNT -fa on \
    -sm layer -ts 1/0/0 --moe-cache off,4096 --repack off -v -o json
```

Adjust `4096` to the intended per-device MiB cap. A layer split with trailing zero shares keeps dense work on the first GPU while still creating every selected backend for cache use. `-sm none` creates only the main backend and is not a valid multi-GPU cache-capacity comparison. `llama-bench` records the effective repack setting and rejects repacking with cache `on` or a fixed budget.

Use a long enough timed generation to amortize graph discovery and inspect both throughput variation and the final cache counters. Repeat the process after a cold process start when cold-start behavior matters. Do not infer a gain from hit rate alone: activation transfers, result transfers, fill traffic, GPU speed, PCIe speed, and CPU memory bandwidth all affect the result.

An `off` versus `on` comparison without `--repack off` includes the intended repacking-policy change. Treat results that use different repacking or expert placement as an end-to-end configuration comparison, not an isolated measurement of the cache.

## Current limitations

- The CUDA, HIP, Metal, and Vulkan backends register a provider (HIP builds share the CUDA implementation). MUSA builds compile the CUDA source to stubs and do not provide the cache. Other backends do not register a provider.
- CPU-resident expert `MUL_MAT_ID` only. Exact gate/up/SwiGLU graphs, including DeepSeek's per-input clamps, can fuse for up to the configured token batch and 64 flattened routed rows. Other GLU graphs use the stock path. There is no GPU-resident output handoff.
- Demand fill only. There is no predictive prefetch or separate prompt-time population path.
- Heat-aware eviction keeps frequently-used experts resident, but there is no strict per-layer top-K slot reservation yet; that is the planned next increment.
- No hot-set file or persistence across scheduler sessions or process restarts.
- Direct writes through a raw host pointer bypass invalidation. Mutate cached weight buffers through the backend tensor and buffer APIs.
- No runtime performance bail-out. An eligible but unprofitable cache remains active unless it fails, is trimmed, or is disabled by configuration.
- Every scheduler owns independent residency state. Process-wide physical-device coordination prevents those sessions from claiming the same free VRAM, but it does not share expert slots between them.
- Unloading a dynamic backend while a scheduler or cache session created by it is still alive is unsupported. Destroy those objects before unloading the backend module.
- Generic operation offload can bypass the cache if `GGML_OP_OFFLOAD_MIN_BATCH` is set low enough to offload decode nodes.
- Performance depends strongly on model shape, quantization, CPU memory bandwidth, PCIe link, CUDA device, spare VRAM, and routing locality.

#pragma once

#include "common.cuh"

// Device-side expert counting-sort for the large-batch MoE (MUL_MAT_ID) path.
// Replaces the host loop that copies ids to the host, sorts token-slots by expert
// on the CPU (GPU idle), and uploads the result. Produces, entirely on-device:
//   ids_to_sorted   [ne_get_rows]  gather index into src1 for each sorted slot
//   ids_from_sorted [ne_get_rows]  sorted position for each original (token,used) slot
//   tokens_per_expert_dev [ne02]   count per expert (caller D2Hs this for GEMM sizing)
// scratch_offsets / scratch_fill are caller-provided device scratch of ne02 int32 each.
//
// ids layout matches the host reader: expert = *(int32_t*)(ids + i12*ids_nb1 + iex*ids_nb0).
void ggml_cuda_moe_build_sorted_ids(
    const void * ids, int ne12, int n_expert_used, int ne02, int ne11,
    size_t ids_nb0, size_t ids_nb1,
    int32_t * ids_to_sorted, int32_t * ids_from_sorted,
    int32_t * tokens_per_expert_dev,
    int32_t * scratch_offsets, int32_t * scratch_fill,
    cudaStream_t stream);

#pragma once

// ChronoQuant runtime configuration.
//
// The implementation stores compressed K/V in backend tensors:
//   anchor: FP16/F32 keyframes
//   delta:  INT4 deltas packed eight values per int32
//   scale:  FP32 per-token, per-head scales
//
// The backend kernels own encode/decode. This header intentionally avoids
// CPU side shadow buffers so accounting and execution match the fused path.

#include <cstdint>

struct chrono_quant_config {
    uint32_t stride_k = 32;    // keyframe stride for keys (every stride-th token is I-frame)
    uint32_t stride_v = 8;     // keyframe stride for values
    uint32_t delta_bits = 4;   // bits for delta quantization (4 = INT4)
    bool     enabled = false;  // master switch
};

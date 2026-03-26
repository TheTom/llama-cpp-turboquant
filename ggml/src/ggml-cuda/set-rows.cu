#include "set-rows.cuh"
#include "cpy-utils.cuh"

typedef void (*set_rows_kernel_t)(const char * src, char * dst);

// Generic quantized set_rows kernel template
template <typename idx_t, typename block_type, int qk, void (*quantize_func)(const float *, block_type *)>
static __global__ void k_set_rows_quant(const float * __restrict__ src0,
                                        const idx_t * __restrict__ src1,
                                        block_type * __restrict__ dst,
                                        const int64_t ne_total,
                                        const int64_t ne10,
                                        const int64_t ne11,
                                        const int64_t ne12,
                                        const int64_t ne13,
                                        const int64_t s01,
                                        const int64_t s02,
                                        const int64_t s03,
                                        const int64_t s10,
                                        const int64_t s11,
                                        const int64_t s12,
                                        const int64_t s1,
                                        const int64_t s2,
                                        const int64_t s3,
                                        const uint3   ne00,
                                        const uint3   ne01,
                                        const uint3   ne02,
                                        const uint3   ne11_fd,
                                        const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    const int64_t i_base = i * qk;
    uint32_t      tmp    = (uint32_t) i_base;
    uint2         div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_type * dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_type);

    const float * src_block = src0_row + i00;
    block_type * dst_block = dst_row_ptr + i00 / qk;

    quantize_func(src_block, dst_block);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

// Template dispatch function for quantized set_rows
template<typename idx_t, typename block_type, int qk, void (*quantize_func)(const float*, block_type*)>
static void set_rows_cuda_quant(
        const float * src0_d, const idx_t * src1_d, block_type * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % qk == 0);
    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / qk;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows_quant<idx_t, block_type, qk, quantize_func><<<grid_size, block_size, 0, stream>>>(
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01, s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd,
            ne01_fd, ne02_fd, ne11_fd, ne12_fd);
    }
}

template <typename src_t, typename idx_t, typename dst_t>
static __global__ void k_set_rows(const src_t * __restrict__ src0,
                                  const idx_t * __restrict__ src1,
                                  dst_t * __restrict__ dst,
                                  const int64_t ne_total,
                                  const int64_t ne10,
                                  const int64_t ne11,
                                  const int64_t ne12,
                                  const int64_t ne13,
                                  const int64_t s01,
                                  const int64_t s02,
                                  const int64_t s03,
                                  const int64_t s10,
                                  const int64_t s11,
                                  const int64_t s12,
                                  const int64_t s1,
                                  const int64_t s2,
                                  const int64_t s3,
                                  const uint3   ne00,
                                  const uint3   ne01,
                                  const uint3   ne02,
                                  const uint3   ne11_fd,
                                  const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    uint32_t tmp = (uint32_t) i;
    uint2    div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const src_t * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    dst_t * dst_row_ptr    = dst + dst_row*s1 + i02*s2 + i03*s3;

    dst_row_ptr[i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

template<typename src_t, typename idx_t, typename dst_t>
static void set_rows_cuda(
        const src_t * src0_d, const idx_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    const int64_t ne_total = ne00 * ne01 * ne02 * ne03;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);


    const int64_t s01 = nb01/sizeof(src_t);
    const int64_t s02 = nb02/sizeof(src_t);
    const int64_t s03 = nb03/sizeof(src_t);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1/sizeof(dst_t);
    const int64_t s2  = nb2/sizeof(dst_t);
    const int64_t s3  = nb3/sizeof(dst_t);

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows<<<grid_size, block_size, 0, stream>>>(src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01,
                                                         s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd, ne01_fd, ne02_fd,
                                                         ne11_fd, ne12_fd);
    }
}

// ---- TurboQuant SET_ROWS: 128-element group processing with WHT rotation ----

// WHT sign arrays (must match CPU ops.cpp and Metal ggml-metal.metal, seed=42)
static __device__ const float turbo_sr_wht_s1[128] = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1
};
static __device__ const float turbo_sr_wht_s2[128] = {
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1
};
// QJL sign arrays (seed=1042)
static __device__ const float turbo_sr_qjl_s1[128] = {
    1,-1,-1,-1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,1,1,-1,1,1,-1,1,-1,-1,1,
    1,1,1,1,-1,-1,1,1,-1,1,-1,-1,1,-1,1,1,1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,1,
    -1,-1,-1,1,1,1,1,1,1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,1,1,-1,1,1,-1,1,1,1,1,1,1,
    1,-1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,1,1,-1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1
};
static __device__ const float turbo_sr_qjl_s2[128] = {
    1,1,-1,1,1,-1,1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,1,-1,1,1,1,-1,1,-1,-1,-1,-1,1,1,
    -1,-1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,-1,-1,1,1,1,1,1,1,1,-1,1,1,
    -1,-1,1,-1,1,1,-1,1,-1,-1,1,1,1,-1,1,-1,1,1,1,1,1,1,-1,1,-1,1,-1,1,-1,1,1,-1,
    1,-1,-1,1,1,-1,1,1,-1,1,1,1,-1,1,1,1,-1,-1,1,-1,1,-1,-1,1,-1,1,-1,1,1,1,1,-1
};

// 3-bit centroid midpoints
static __device__ const float turbo_mid_3bit[7] = {
    -0.154259f, -0.091775f, -0.043589f, 0.0f, 0.043589f, 0.091775f, 0.154259f
};
// 3-bit centroids
static __device__ const float turbo_centroids_3bit[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

// In-place WHT rotation: apply s1, WHT butterfly, normalize, apply s2 (single-thread fallback)
static __device__ void turbo_rotate_forward_cuda(float * x, const float * s1, const float * s2) {
    for (int i = 0; i < 128; i++) x[i] *= s1[i];
    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
    const float inv_sqrt_128 = 0.08838834764831845f;
    for (int i = 0; i < 128; i++) x[i] *= inv_sqrt_128 * s2[i];
}

// Warp-cooperative WHT: each thread holds 4 of the 128 elements.
// Applies signs1, 7-stage butterfly, normalize, signs2.
// lane = threadIdx.x % 32, thread holds elements [lane*4 .. lane*4+3] as x0..x3.
static __device__ void turbo_warp_wht(
        float & x0, float & x1, float & x2, float & x3,
        int lane, const float * s1, const float * s2) {

    x0 *= s1[lane*4 + 0]; x1 *= s1[lane*4 + 1];
    x2 *= s1[lane*4 + 2]; x3 *= s1[lane*4 + 3];

    // h=1: within-thread butterfly on (0,1) and (2,3)
    { float a=x0, b=x1; x0=a+b; x1=a-b; a=x2; b=x3; x2=a+b; x3=a-b; }
    // h=2: within-thread butterfly on (0,2) and (1,3)
    { float a=x0, b=x2; x0=a+b; x2=a-b; a=x1; b=x3; x1=a+b; x3=a-b; }
    // h=4..64: cross-thread butterfly via warp shuffle
    for (int h = 4; h < 128; h *= 2) {
        int partner = lane ^ (h / 4);
        float p0 = __shfl_sync(0xFFFFFFFF, x0, partner);
        float p1 = __shfl_sync(0xFFFFFFFF, x1, partner);
        float p2 = __shfl_sync(0xFFFFFFFF, x2, partner);
        float p3 = __shfl_sync(0xFFFFFFFF, x3, partner);
        if ((lane & (h / 4)) == 0) {
            x0 += p0; x1 += p1; x2 += p2; x3 += p3;
        } else {
            x0 = p0 - x0; x1 = p1 - x1; x2 = p2 - x2; x3 = p3 - x3;
        }
    }

    const float inv_sqrt_128 = 0.08838834764831845f;
    x0 *= inv_sqrt_128 * s2[lane*4 + 0]; x1 *= inv_sqrt_128 * s2[lane*4 + 1];
    x2 *= inv_sqrt_128 * s2[lane*4 + 2]; x3 *= inv_sqrt_128 * s2[lane*4 + 3];
}

static __device__ uint8_t turbo_nearest_3bit(float val) {
    if      (val < turbo_mid_3bit[0]) return 0;
    else if (val < turbo_mid_3bit[1]) return 1;
    else if (val < turbo_mid_3bit[2]) return 2;
    else if (val < turbo_mid_3bit[3]) return 3;
    else if (val < turbo_mid_3bit[4]) return 4;
    else if (val < turbo_mid_3bit[5]) return 5;
    else if (val < turbo_mid_3bit[6]) return 6;
    else                              return 7;
}

// turbo3 SET_ROWS kernel: 1 warp (32 threads) per 128-element group
// Each thread handles 4 elements. WHT butterfly uses warp shuffles.
template <typename idx_t>
static __global__ void k_set_rows_turbo3(
        const float   * __restrict__ src0,
        const idx_t   * __restrict__ src1,
        block_turbo3_0 * __restrict__ dst,
        const int64_t ne_groups,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s10, const int64_t s11, const int64_t s12,
        const int64_t s1,  const int64_t s2,  const int64_t s3,
        const int64_t ne00_grp, const int64_t ne01_val, const int64_t ne02_val) {

    const int lane = threadIdx.x % WARP_SIZE; // 0..31
    const int64_t g = (int64_t)(blockIdx.x * (blockDim.x / WARP_SIZE)) + (threadIdx.x / WARP_SIZE);
    if (g >= ne_groups) return;

    int64_t tmp = g;
    const int64_t i00_grp = tmp % ne00_grp; tmp /= ne00_grp;
    const int64_t i01 = tmp % ne01_val;     tmp /= ne01_val;
    const int64_t i02 = tmp % ne02_val;
    const int64_t i03 = tmp / ne02_val;

    const int64_t dst_row = *(src1 + (i01)*s10 + (i02 % ne11)*s11 + (i03 % ne12)*s12);

    const float * grp_src = src0 + i01*s01 + i02*s02 + i03*s03 + i00_grp * 128;
    block_turbo3_0 * dst_row_ptr = (block_turbo3_0 *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3);

    // Each thread loads 4 elements (lane*4 .. lane*4+3)
    float x0 = grp_src[lane*4 + 0];
    float x1 = grp_src[lane*4 + 1];
    float x2 = grp_src[lane*4 + 2];
    float x3 = grp_src[lane*4 + 3];

    // Warp-reduce norm_sq
    float local_sq = x0*x0 + x1*x1 + x2*x2 + x3*x3;
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        local_sq += __shfl_xor_sync(0xFFFFFFFF, local_sq, offset);
    }
    float grp_norm = sqrtf(local_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
    x0 *= inv_norm; x1 *= inv_norm; x2 *= inv_norm; x3 *= inv_norm;

    // WHT forward rotation (warp-cooperative)
    turbo_warp_wht(x0, x1, x2, x3, lane, turbo_sr_wht_s1, turbo_sr_wht_s2);

    // Quantize: each thread writes its 4 elements into the appropriate block
    // lane/8 = block index (0..3), lane%8 = which qs byte within the block
    const int blk_idx = lane / 8;
    const int qs_idx  = lane % 8; // this thread owns qs[qs_idx]
    block_turbo3_0 & blk = dst_row_ptr[i00_grp * 4 + blk_idx];

    // Quantize 4 elements → pack into one qs byte (2 bits each) + sign bits
    float vals[4] = {x0, x1, x2, x3};
    uint8_t qs_byte = 0;
    // Each thread contributes sign bits for its 4 elements within its 32-element block
    // Position within block: (lane%8)*4 + l, so bit position = (lane%8)*4 + l (0..31)
    uint32_t local_signs = 0;
    for (int l = 0; l < 4; l++) {
        uint8_t idx = turbo_nearest_3bit(vals[l]);
        qs_byte |= (idx & 0x3) << (l * 2);
        if (idx & 0x4) local_signs |= (1u << ((qs_idx) * 4 + l));
    }

    // Collect sign bits across the 8 threads within this block (same blk_idx)
    // Use warp shuffle to OR together all 8 threads' contributions
    for (int off = 4; off >= 1; off /= 2) {
        local_signs |= __shfl_xor_sync(0xFFFFFFFF, local_signs, off);
    }

    // Write norm (once per block)
    if (qs_idx == 0) {
        blk.norm = __float2half(grp_norm);
    }
    // Each thread writes its unique qs byte
    blk.qs[qs_idx] = qs_byte;
    // First thread per block writes the signs bytes
    if (qs_idx == 0) {
        blk.signs[0] = (uint8_t)(local_signs);
        blk.signs[1] = (uint8_t)(local_signs >> 8);
        blk.signs[2] = (uint8_t)(local_signs >> 16);
        blk.signs[3] = (uint8_t)(local_signs >> 24);
    }
}

template<typename idx_t>
static void set_rows_cuda_turbo3(
        const float * src0_d, const idx_t * src1_d, block_turbo3_0 * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % 128 == 0);
    const int64_t ne00_grp = ne00 / 128;
    const int64_t ne_groups = ne00_grp * ne01 * ne02 * ne03;

    // 8 warps per block = 256 threads, each warp handles 1 group
    const int warps_per_block = 8;
    const int block_size = warps_per_block * WARP_SIZE;
    const int n_blocks = (ne_groups + warps_per_block - 1) / warps_per_block;

    if (ne_groups > 0) {
        k_set_rows_turbo3<idx_t><<<n_blocks, block_size, 0, stream>>>(
            src0_d, src1_d, dst_d, ne_groups,
            ne10, ne11, ne12, ne13,
            nb01/sizeof(float), nb02/sizeof(float), nb03/sizeof(float),
            nb10/sizeof(idx_t), nb11/sizeof(idx_t), nb12/sizeof(idx_t),
            nb1, nb2, nb3,
            ne00_grp, ne01, ne02);
    }
}

// turbo4 SET_ROWS kernel: 1 warp (32 threads) per 128-element block
template <typename idx_t>
static __global__ void k_set_rows_turbo4(
        const float   * __restrict__ src0,
        const idx_t   * __restrict__ src1,
        block_turbo4_0 * __restrict__ dst,
        const int64_t ne_blocks,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s10, const int64_t s11, const int64_t s12,
        const int64_t s1,  const int64_t s2,  const int64_t s3,
        const int64_t ne00_blk, const int64_t ne01_val, const int64_t ne02_val) {

    const int lane = threadIdx.x % WARP_SIZE;
    const int64_t b_idx = (int64_t)(blockIdx.x * (blockDim.x / WARP_SIZE)) + (threadIdx.x / WARP_SIZE);
    if (b_idx >= ne_blocks) return;

    int64_t tmp = b_idx;
    const int64_t i00_blk = tmp % ne00_blk; tmp /= ne00_blk;
    const int64_t i01 = tmp % ne01_val;     tmp /= ne01_val;
    const int64_t i02 = tmp % ne02_val;
    const int64_t i03 = tmp / ne02_val;

    const int64_t dst_row = *(src1 + (i01)*s10 + (i02 % ne11)*s11 + (i03 % ne12)*s12);

    const float * blk_src = src0 + i01*s01 + i02*s02 + i03*s03 + i00_blk * 128;
    block_turbo4_0 * dst_row_ptr = (block_turbo4_0 *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3);
    block_turbo4_0 & blk = dst_row_ptr[i00_blk];

    // Each thread loads 4 elements
    float x0 = blk_src[lane*4 + 0];
    float x1 = blk_src[lane*4 + 1];
    float x2 = blk_src[lane*4 + 2];
    float x3 = blk_src[lane*4 + 3];

    // Warp-reduce norm
    float local_sq = x0*x0 + x1*x1 + x2*x2 + x3*x3;
    for (int off = WARP_SIZE/2; off > 0; off /= 2) {
        local_sq += __shfl_xor_sync(0xFFFFFFFF, local_sq, off);
    }
    float norm = sqrtf(local_sq);
    float inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;
    x0 *= inv_norm; x1 *= inv_norm; x2 *= inv_norm; x3 *= inv_norm;

    // Save normalized values for residual computation
    float n0 = x0, n1 = x1, n2 = x2, n3 = x3;

    // Step 2: WHT rotate
    turbo_warp_wht(x0, x1, x2, x3, lane, turbo_sr_wht_s1, turbo_sr_wht_s2);

    // Step 3: 3-bit quantize + compute recon for residual
    float vals[4] = {x0, x1, x2, x3};
    uint8_t indices[4];
    float recon[4];
    for (int l = 0; l < 4; l++) {
        indices[l] = turbo_nearest_3bit(vals[l]);
        recon[l] = turbo_centroids_3bit[indices[l]];
    }

    // Step 4: residual = normalized - recon
    x0 = n0 - recon[0]; x1 = n1 - recon[1]; x2 = n2 - recon[2]; x3 = n3 - recon[3];

    // Warp-reduce residual norm
    float rnorm_sq = x0*x0 + x1*x1 + x2*x2 + x3*x3;
    for (int off = WARP_SIZE/2; off > 0; off /= 2) {
        rnorm_sq += __shfl_xor_sync(0xFFFFFFFF, rnorm_sq, off);
    }

    // Step 5: QJL WHT on residual
    turbo_warp_wht(x0, x1, x2, x3, lane, turbo_sr_qjl_s1, turbo_sr_qjl_s2);

    // Collect QJL sign bits via ballot (4 ballots, one per sub-element)
    uint32_t sign_ballot0 = __ballot_sync(0xFFFFFFFF, x0 >= 0.0f);
    uint32_t sign_ballot1 = __ballot_sync(0xFFFFFFFF, x1 >= 0.0f);
    uint32_t sign_ballot2 = __ballot_sync(0xFFFFFFFF, x2 >= 0.0f);
    uint32_t sign_ballot3 = __ballot_sync(0xFFFFFFFF, x3 >= 0.0f);

    // Lane 0 serializes the writes (3-bit packing is inherently serial across bits)
    if (lane == 0) {
        blk.norm  = __float2half(norm);
        blk.rnorm = __float2half(sqrtf(rnorm_sq));

        // Zero output arrays
        for (int j = 0; j < QK_TURBO4 * 3 / 8; j++) blk.qs[j] = 0;

        // Pack 3-bit indices (collected from all lanes via shfl)
        for (int t = 0; t < 32; t++) {
            uint8_t idx0 = __shfl_sync(0xFFFFFFFF, indices[0], t);
            uint8_t idx1 = __shfl_sync(0xFFFFFFFF, indices[1], t);
            uint8_t idx2 = __shfl_sync(0xFFFFFFFF, indices[2], t);
            uint8_t idx3 = __shfl_sync(0xFFFFFFFF, indices[3], t);
            uint8_t idxs[4] = {idx0, idx1, idx2, idx3};
            for (int l = 0; l < 4; l++) {
                int j = t*4 + l;
                int bit_offset = j * 3;
                int byte_idx = bit_offset / 8;
                int bit_pos  = bit_offset % 8;
                blk.qs[byte_idx] |= (uint8_t)((idxs[l] & 0x7) << bit_pos);
                if (bit_pos > 5 && byte_idx + 1 < QK_TURBO4 * 3 / 8) {
                    blk.qs[byte_idx + 1] |= (uint8_t)((idxs[l] & 0x7) >> (8 - bit_pos));
                }
            }
        }

        // Pack QJL signs: bit j corresponds to element j
        // ballot0 has bit t set if lane t's element 0 (= element t*4) is >= 0
        // So element t*4+l is from ballot_l, bit t
        for (int j = 0; j < QK_TURBO4 / 8; j++) blk.signs[j] = 0;
        for (int t = 0; t < 32; t++) {
            uint8_t s0 = (sign_ballot0 >> t) & 1;
            uint8_t s1 = (sign_ballot1 >> t) & 1;
            uint8_t s2 = (sign_ballot2 >> t) & 1;
            uint8_t s3 = (sign_ballot3 >> t) & 1;
            int base = t * 4;
            if (s0) blk.signs[(base+0) / 8] |= (1 << ((base+0) % 8));
            if (s1) blk.signs[(base+1) / 8] |= (1 << ((base+1) % 8));
            if (s2) blk.signs[(base+2) / 8] |= (1 << ((base+2) % 8));
            if (s3) blk.signs[(base+3) / 8] |= (1 << ((base+3) % 8));
        }
    } else {
        // Other lanes participate in the shfl
        for (int t = 0; t < 32; t++) {
            __shfl_sync(0xFFFFFFFF, indices[0], t);
            __shfl_sync(0xFFFFFFFF, indices[1], t);
            __shfl_sync(0xFFFFFFFF, indices[2], t);
            __shfl_sync(0xFFFFFFFF, indices[3], t);
        }
    }
}

template<typename idx_t>
static void set_rows_cuda_turbo4(
        const float * src0_d, const idx_t * src1_d, block_turbo4_0 * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % QK_TURBO4 == 0);
    const int64_t ne00_blk = ne00 / QK_TURBO4;
    const int64_t ne_blocks = ne00_blk * ne01 * ne02 * ne03;

    const int warps_per_block = 8;
    const int block_size = warps_per_block * WARP_SIZE;
    const int n_blocks = (ne_blocks + warps_per_block - 1) / warps_per_block;

    if (ne_blocks > 0) {
        k_set_rows_turbo4<idx_t><<<n_blocks, block_size, 0, stream>>>(
            src0_d, src1_d, dst_d, ne_blocks,
            ne10, ne11, ne12, ne13,
            nb01/sizeof(float), nb02/sizeof(float), nb03/sizeof(float),
            nb10/sizeof(idx_t), nb11/sizeof(idx_t), nb12/sizeof(idx_t),
            nb1, nb2, nb3,
            ne00_blk, ne01, ne02);
    }
}

template<typename src_t, typename idx_t>
static void set_rows_cuda(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const src_t * src0_d = (const src_t *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F32) {
        set_rows_cuda(
            src0_d, src1_d, (float*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_BF16) {
        set_rows_cuda(
            src0_d, src1_d, (nv_bfloat16*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_0) {
        set_rows_cuda_quant<idx_t, block_q4_0, QK4_0, quantize_f32_q4_0_block>(
            src0_d, src1_d, (block_q4_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_1) {
        set_rows_cuda_quant<idx_t, block_q4_1, QK4_1, quantize_f32_q4_1_block>(
            src0_d, src1_d, (block_q4_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_0) {
        set_rows_cuda_quant<idx_t, block_q5_0, QK5_0, quantize_f32_q5_0_block>(
            src0_d, src1_d, (block_q5_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_1) {
        set_rows_cuda_quant<idx_t, block_q5_1, QK5_1, quantize_f32_q5_1_block>(
            src0_d, src1_d, (block_q5_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q8_0) {
        set_rows_cuda_quant<idx_t, block_q8_0, QK8_0, quantize_f32_q8_0_block>(
            src0_d, src1_d, (block_q8_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_IQ4_NL) {
        set_rows_cuda_quant<idx_t, block_iq4_nl, QK4_NL, quantize_f32_iq4_nl_block>(
            src0_d, src1_d, (block_iq4_nl*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_TURBO3_0) {
        set_rows_cuda_turbo3<idx_t>(
            src0_d, src1_d, (block_turbo3_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_TURBO4_0) {
        set_rows_cuda_turbo4<idx_t>(
            src0_d, src1_d, (block_turbo4_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}


void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);

    if (src1->type == GGML_TYPE_I64) {
        set_rows_cuda<float, int64_t>(ctx, src0, src1, dst);
    } else {
        set_rows_cuda<float, int32_t>(ctx, src0, src1, dst);
    }
}

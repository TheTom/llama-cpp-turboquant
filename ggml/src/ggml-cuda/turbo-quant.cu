/*
 * TurboQuant CUDA kernels — dequantize row + set_rows (KV cache write)
 * Port of Metal implementation from ggml-metal.metal
 */

#include "turbo-quant.cuh"
#include "dequantize.cuh"
#include "convert.cuh"

#include <cstdint>

/* ===== Constant memory initialization ===== */

void turbo_quant_init_cuda(cudaStream_t stream) {
    GGML_UNUSED(stream);
}

/* ===== turbo3 dequantize block kernel ===== */
/* Each block of 32 threads processes 8 turbo3 blocks (256 elements).
 * Each thread dequantizes 8 elements (from 1 turbo3 block per iteration). */

template<typename dst_t>
static __global__ void dequantize_block_turbo3_0(
        const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {
    const int64_t i = blockIdx.x;

    /* 32 threads, each handles 8 blocks worth across iterations */
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid / 8;   /* 0..3: which group of 8 blocks */
    const int64_t ir  = tid % 8;   /* 0..7: which block within group */
    const int64_t ib  = 8 * i + ir;

    if (ib >= nb32) {
        return;
    }

    dst_t * y = yy + 256 * i + 32 * ir + 4 * il;

    const block_turbo3_0 * x = (const block_turbo3_0 *) vx + ib;
    const float norm = __half2float(x->norm);

    /* Dequantize 4 elements at offset il*4 within this 32-element block */
    const int off = il * 4;
    const uint8_t qb = x->qs[il];  /* qs has 4 per byte, il selects the byte */
    const uint8_t sb = x->signs[il / 2];
    const int sshift = (il & 1) * 4;

    for (int k = 0; k < 4; k++) {
        const uint8_t low2 = (qb >> (k * 2)) & 0x3;
        const uint8_t hi1  = (sb >> (sshift + k)) & 0x1;
        const int idx = low2 | (hi1 << 2);
        y[k] = ggml_cuda_cast<dst_t>(turbo_centroid_3bit(idx) * norm);
    }
}

/* ===== turbo4 dequantize block kernel ===== */
/* One CUDA block per turbo4 block (128 elements). 32 threads, 4 elements each. */

template<typename dst_t>
static __global__ void dequantize_block_turbo4_0(
        const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t k) {
    const int64_t i = blockIdx.x;  /* turbo4 block index */
    const int tid = threadIdx.x;   /* 0..31 */

    if (i * QK_TURBO4 >= k) {
        return;
    }

    const block_turbo4_0 * x = (const block_turbo4_0 *) vx;
    const block_turbo4_0 & xb = x[i];

    const float norm  = __half2float(xb.norm);
    const float rnorm = __half2float(xb.rnorm);
    const float qjl_scale = TURBO_QJL_CONST / 128.0f * rnorm;

    /* Each thread handles 4 consecutive elements */
    const int base = tid * 4;
    dst_t * out = y + i * QK_TURBO4 + base;

    for (int jj = 0; jj < 4; jj++) {
        const int j = base + jj;

        /* Unpack 3-bit index */
        const int bit_offset = j * 3;
        const int byte_idx = bit_offset / 8;
        const int bit_pos = bit_offset % 8;
        uint16_t raw = (uint16_t)xb.qs[byte_idx];
        if (bit_pos > 5 && byte_idx + 1 < QK_TURBO4 * 3 / 8) {
            raw |= (uint16_t)xb.qs[byte_idx + 1] << 8;
        }
        const int idx = (raw >> bit_pos) & 0x7;
        const float centroid_val = turbo_centroid_3bit(idx);

        /* QJL reconstruction */
        const int sign_bit = (xb.signs[j / 8] >> (j % 8)) & 1;
        const float qjl_val = sign_bit ? qjl_scale : -qjl_scale;

        out[jj] = ggml_cuda_cast<dst_t>((centroid_val + qjl_val) * norm);
    }
}

/* ===== Row dequantize launchers ===== */

template<typename dst_t>
void dequantize_row_turbo3_0_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb32 = k / QK_TURBO3;
    const int nb = (k + 255) / 256;
    dequantize_block_turbo3_0<<<nb, 32, 0, stream>>>(vx, y, nb32);
}

template<typename dst_t>
void dequantize_row_turbo4_0_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    GGML_ASSERT(k % QK_TURBO4 == 0);
    const int nb = k / QK_TURBO4;
    dequantize_block_turbo4_0<<<nb, 32, 0, stream>>>(vx, y, k);
}

/* Explicit template instantiations */
template void dequantize_row_turbo3_0_cuda<float>(const void * vx, float * y, const int64_t k, cudaStream_t stream);
template void dequantize_row_turbo3_0_cuda<half>(const void * vx, half * y, const int64_t k, cudaStream_t stream);
template void dequantize_row_turbo3_0_cuda<nv_bfloat16>(const void * vx, nv_bfloat16 * y, const int64_t k, cudaStream_t stream);

template void dequantize_row_turbo4_0_cuda<float>(const void * vx, float * y, const int64_t k, cudaStream_t stream);
template void dequantize_row_turbo4_0_cuda<half>(const void * vx, half * y, const int64_t k, cudaStream_t stream);
template void dequantize_row_turbo4_0_cuda<nv_bfloat16>(const void * vx, nv_bfloat16 * y, const int64_t k, cudaStream_t stream);

/* ===== Custom set_rows kernel for turbo3 ===== */
/* Each thread processes one 128-element rotation group (4 blocks of 32).
 * WHT rotation in registers, then quantize + norm correction. */

template <typename idx_t>
static __global__ void __launch_bounds__(64)
k_set_rows_turbo3(const float * __restrict__ src0,
                   const idx_t * __restrict__ src1,
                   block_turbo3_0 * __restrict__ dst,
                   const int64_t ne_total_groups,
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

    if (i >= ne_total_groups) {
        return;
    }

    /* Decompose linear index into (i00_grp, i01, i02, i03) */
    /* i00_grp is the rotation group index within a row, units of 128 elements (= 4 blocks) */
    uint32_t tmp = (uint32_t)(i * QK_TURBO3_GROUP);  /* convert to element index for decomposition */
    uint2 div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;  /* element offset in row (multiple of 128) */
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t)i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t)i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_turbo3_0 * dst_row_ptr = (block_turbo3_0 *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3);

    const float * grp_src = src0_row + i00;
    block_turbo3_0 * grp_dst = dst_row_ptr + (i00 / QK_TURBO3);

    /* Call the device quantize function */
    turbo3_quantize_group(grp_src, grp_dst);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

/* ===== Custom set_rows kernel for turbo4 ===== */

template <typename idx_t>
static __global__ void __launch_bounds__(64)
k_set_rows_turbo4(const float * __restrict__ src0,
                   const idx_t * __restrict__ src1,
                   block_turbo4_0 * __restrict__ dst,
                   const int64_t ne_total_blocks,
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

    if (i >= ne_total_blocks) {
        return;
    }

    uint32_t tmp = (uint32_t)(i * QK_TURBO4);
    uint2 div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t)i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t)i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_turbo4_0 * dst_row_ptr = (block_turbo4_0 *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3);

    const float * blk_src = src0_row + i00;
    block_turbo4_0 * blk_dst = dst_row_ptr + (i00 / QK_TURBO4);

    turbo4_quantize_block(blk_src, blk_dst);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

/* ===== set_rows launcher for turbo3 ===== */

template<typename idx_t>
void set_rows_turbo3_cuda(
        const float * src0_d, const idx_t * src1_d, block_turbo3_0 * dst_d,
        int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
        int64_t ne10, int64_t ne11, int64_t ne12, int64_t ne13,
        size_t nb01, size_t nb02, size_t nb03,
        size_t nb10, size_t nb11, size_t nb12,
        size_t nb1, size_t nb2, size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % QK_TURBO3_GROUP == 0);
    const int64_t ne_total_groups = (ne00 * ne01 * ne02 * ne03) / QK_TURBO3_GROUP;
    const int num_blocks = (ne_total_groups + 63) / 64;

    const int64_t s01 = nb01 / sizeof(float);
    const int64_t s02 = nb02 / sizeof(float);
    const int64_t s03 = nb03 / sizeof(float);
    const int64_t s10 = nb10 / sizeof(idx_t);
    const int64_t s11 = nb11 / sizeof(idx_t);
    const int64_t s12 = nb12 / sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_total_groups > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t)ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t)ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t)ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t)ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t)ne12);

        k_set_rows_turbo3<<<num_blocks, 64, 0, stream>>>(
            src0_d, src1_d, dst_d, ne_total_groups,
            ne10, ne11, ne12, ne13,
            s01, s02, s03, s10, s11, s12, s1, s2, s3,
            ne00_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
    }
}

/* ===== set_rows launcher for turbo4 ===== */

template<typename idx_t>
void set_rows_turbo4_cuda(
        const float * src0_d, const idx_t * src1_d, block_turbo4_0 * dst_d,
        int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
        int64_t ne10, int64_t ne11, int64_t ne12, int64_t ne13,
        size_t nb01, size_t nb02, size_t nb03,
        size_t nb10, size_t nb11, size_t nb12,
        size_t nb1, size_t nb2, size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % QK_TURBO4 == 0);
    const int64_t ne_total_blocks = (ne00 * ne01 * ne02 * ne03) / QK_TURBO4;
    const int num_blocks = (ne_total_blocks + 63) / 64;

    const int64_t s01 = nb01 / sizeof(float);
    const int64_t s02 = nb02 / sizeof(float);
    const int64_t s03 = nb03 / sizeof(float);
    const int64_t s10 = nb10 / sizeof(idx_t);
    const int64_t s11 = nb11 / sizeof(idx_t);
    const int64_t s12 = nb12 / sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_total_blocks > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t)ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t)ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t)ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t)ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t)ne12);

        k_set_rows_turbo4<<<num_blocks, 64, 0, stream>>>(
            src0_d, src1_d, dst_d, ne_total_blocks,
            ne10, ne11, ne12, ne13,
            s01, s02, s03, s10, s11, s12, s1, s2, s3,
            ne00_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
    }
}

/* Explicit template instantiations for set_rows launchers */
template void set_rows_turbo3_cuda<int64_t>(
    const float *, const int64_t *, block_turbo3_0 *,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, cudaStream_t);
template void set_rows_turbo3_cuda<int32_t>(
    const float *, const int32_t *, block_turbo3_0 *,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, cudaStream_t);
template void set_rows_turbo4_cuda<int64_t>(
    const float *, const int64_t *, block_turbo4_0 *,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, cudaStream_t);
template void set_rows_turbo4_cuda<int32_t>(
    const float *, const int32_t *, block_turbo4_0 *,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, cudaStream_t);

/* ===== TURBO_WHT kernel — graph-level Walsh-Hadamard rotation ===== */
/* Applies forward (direction=0) or inverse (direction=1) WHT rotation to fp32 data.
 * Each thread processes one 128-element group.
 * This is used at graph level to pre-rotate queries and inverse-rotate attention output,
 * so that the dequantize path can be a simple centroid lookup without rotation. */

static __global__ void k_turbo_wht(
        const float * __restrict__ src,
        float       * __restrict__ dst,
        const int64_t n_groups,
        const int     direction) {
    const int64_t group_idx = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    if (group_idx >= n_groups) return;

    const float * in  = src + group_idx * 128;
    float       * out = dst + group_idx * 128;

    float x[128];

    if (direction == 0) {
        /* Forward: signs1 * FWHT * signs2 */
        for (int i = 0; i < 128; i++) x[i] = in[i] * (float)turbo_wht_signs1_d[i];
        turbo_fwht_128(x);
        for (int i = 0; i < 128; i++) out[i] = x[i] * (float)turbo_wht_signs2_d[i];
    } else {
        /* Inverse: signs2 * FWHT * signs1 (FWHT is self-inverse) */
        for (int i = 0; i < 128; i++) x[i] = in[i] * (float)turbo_wht_signs2_d[i];
        turbo_fwht_128(x);
        for (int i = 0; i < 128; i++) out[i] = x[i] * (float)turbo_wht_signs1_d[i];
    }
}

void ggml_cuda_op_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;

    int direction;
    memcpy(&direction, dst->op_params, sizeof(int));


    const int64_t n_elements = ggml_nelements(src0);
    GGML_ASSERT(n_elements % 128 == 0);
    const int64_t n_groups = n_elements / 128;

    cudaStream_t stream = ctx.stream();
    const int block_size = 256;
    const int grid_size = (n_groups + block_size - 1) / block_size;

    k_turbo_wht<<<grid_size, block_size, 0, stream>>>(src0_d, dst_d, n_groups, direction);
}

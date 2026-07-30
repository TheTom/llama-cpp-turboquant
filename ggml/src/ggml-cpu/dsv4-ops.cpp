//
// ggml_compute_forward_dsv4_hc_comb
//

static void ggml_dsv4_hc_comb_norm_cols(float * comb, float eps) {
    constexpr int64_t hc = 4;

    for (int64_t idst = 0; idst < hc; ++idst) {
        float sum = eps;
        for (int64_t isrc = 0; isrc < hc; ++isrc) {
            sum += comb[idst + hc*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int64_t isrc = 0; isrc < hc; ++isrc) {
            comb[idst + hc*isrc] *= inv_sum;
        }
    }
}

static void ggml_dsv4_hc_comb_norm_rows(float * comb, float eps) {
    constexpr int64_t hc = 4;

    for (int64_t isrc = 0; isrc < hc; ++isrc) {
        float sum = eps;
        for (int64_t idst = 0; idst < hc; ++idst) {
            sum += comb[idst + hc*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int64_t idst = 0; idst < hc; ++idst) {
            comb[idst + hc*isrc] *= inv_sum;
        }
    }
}

static void ggml_compute_forward_dsv4_hc_comb_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc = 4;
    constexpr int64_t comb_offset = 2*hc;
    constexpr int64_t hc_mix_dim = (2 + hc)*hc;

    const int64_t n_tokens = mixes->ne[1];

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0] == hc);
    GGML_ASSERT(dst->ne[1] == hc);
    GGML_ASSERT(dst->ne[2] == n_tokens);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(base->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, base,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    const float eps = ggml_get_op_params_f32(dst, 0);
    const int32_t n_iter = ggml_get_op_params_i32(dst, 1);
    GGML_ASSERT(n_iter > 0);

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t dr  = (n_tokens + nth - 1) / nth;
    const int64_t it0 = dr * ith;
    const int64_t it1 = MIN(it0 + dr, n_tokens);

    const float scale_comb = *(const float *) ((const char *) scale->data + 2*nbs0);

    for (int64_t it = it0; it < it1; ++it) {
        float comb[hc*hc];

        for (int64_t isrc = 0; isrc < hc; ++isrc) {
            float max = -INFINITY;
            for (int64_t idst = 0; idst < hc; ++idst) {
                const int64_t idx = idst + hc*isrc;
                const float xv = *(const float *) ((const char *) mixes->data + (comb_offset + idx)*nbm0 + it*nbm1);
                const float bv = *(const float *) ((const char *) base->data  + (comb_offset + idx)*nbb0);
                const float v = xv * scale_comb + bv;
                comb[idx] = v;
                max = MAX(max, v);
            }

            float sum = 0.0f;
            for (int64_t idst = 0; idst < hc; ++idst) {
                const int64_t idx = idst + hc*isrc;
                const float v = expf(comb[idx] - max);
                comb[idx] = v;
                sum += v;
            }

            const float inv_sum = 1.0f / sum;
            for (int64_t idst = 0; idst < hc; ++idst) {
                const int64_t idx = idst + hc*isrc;
                comb[idx] = comb[idx] * inv_sum + eps;
            }
        }

        ggml_dsv4_hc_comb_norm_cols(comb, eps);
        for (int32_t i = 1; i < n_iter; ++i) {
            ggml_dsv4_hc_comb_norm_rows(comb, eps);
            ggml_dsv4_hc_comb_norm_cols(comb, eps);
        }

        for (int64_t isrc = 0; isrc < hc; ++isrc) {
            for (int64_t idst = 0; idst < hc; ++idst) {
                const int64_t idx = idst + hc*isrc;
                *(float *) ((char *) dst->data + idst*nbd0 + isrc*nbd1 + it*nbd2) = comb[idx];
            }
        }
    }
}

void ggml_compute_forward_dsv4_hc_comb(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_comb_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_dsv4_hc_pre

static void ggml_compute_forward_dsv4_hc_pre_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t n_embd   = x->ne[0];
    const int64_t hc       = x->ne[1];
    const int64_t n_tokens = x->ne[2];

    GGML_ASSERT(dst->ne[0] == n_embd);
    GGML_ASSERT(dst->ne[1] == n_tokens);
    GGML_ASSERT(weights->ne[0] == hc);
    GGML_ASSERT(weights->ne[1] == n_tokens);

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t nr  = n_embd * n_tokens;
    const int64_t dr  = (nr + nth - 1) / nth;
    const int64_t ir0 = dr * ith;
    const int64_t ir1 = MIN(ir0 + dr, nr);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i0 = ir % n_embd;
        const int64_t it = ir / n_embd;

        float sum = 0.0f;
        for (int64_t ih = 0; ih < hc; ++ih) {
            const float xv = *(const float *) ((const char *) x->data       + i0*nbx0 + ih*nbx1 + it*nbx2);
            const float wv = *(const float *) ((const char *) weights->data + ih*nbw0 + it*nbw1);
            sum += xv * wv;
        }

        *(float *) ((char *) dst->data + i0*nbd0 + it*nbd1) = sum;
    }
}

void ggml_compute_forward_dsv4_hc_pre(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_pre_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_dsv4_hc_post

static void ggml_compute_forward_dsv4_hc_post_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * residual = dst->src[1];
    const ggml_tensor * post     = dst->src[2];
    const ggml_tensor * comb     = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(comb->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t hc       = residual->ne[1];

    GGML_ASSERT(dst->ne[0] == n_embd);
    GGML_ASSERT(dst->ne[1] == hc);
    GGML_ASSERT(dst->ne[2] == n_tokens);
    GGML_ASSERT(residual->ne[0] == n_embd);
    GGML_ASSERT(residual->ne[2] == n_tokens);
    GGML_ASSERT(post->ne[0] == hc);
    GGML_ASSERT(post->ne[1] == n_tokens);
    GGML_ASSERT(comb->ne[0] == hc);
    GGML_ASSERT(comb->ne[1] == hc);
    GGML_ASSERT(comb->ne[2] == n_tokens);

    GGML_TENSOR_LOCALS(size_t, nbx, x,        nb);
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbp, post,     nb);
    GGML_TENSOR_LOCALS(size_t, nbc, comb,     nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t nr  = n_embd * hc * n_tokens;
    const int64_t dr  = (nr + nth - 1) / nth;
    const int64_t ir0 = dr * ith;
    const int64_t ir1 = MIN(ir0 + dr, nr);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i0     = ir % n_embd;
        const int64_t idst   = (ir / n_embd) % hc;
        const int64_t it     = ir / (n_embd * hc);

        const float xv = *(const float *) ((const char *) x->data    + i0*nbx0 + it*nbx1);
        const float pv = *(const float *) ((const char *) post->data + idst*nbp0 + it*nbp1);

        float sum = xv * pv;
        for (int64_t isrc = 0; isrc < hc; ++isrc) {
            const float rv = *(const float *) ((const char *) residual->data + i0*nbr0 + isrc*nbr1 + it*nbr2);
            const float cv = *(const float *) ((const char *) comb->data     + idst*nbc0 + isrc*nbc1 + it*nbc2);
            sum += rv * cv;
        }

        *(float *) ((char *) dst->data + i0*nbd0 + idst*nbd1 + it*nbd2) = sum;
    }
}

void ggml_compute_forward_dsv4_hc_post(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_post_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_lightning_indexer

void ggml_compute_forward_lightning_indexer(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * w = dst->src[2]; // weights
    const ggml_tensor * m = dst->src[3]; // mask

    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(   q->type == GGML_TYPE_F32);
    GGML_ASSERT(   w->type == GGML_TYPE_F32);
    GGML_ASSERT(   m->type == GGML_TYPE_F16);

    GGML_TENSOR_LOCALS(int64_t, neq,  q, ne)
    GGML_TENSOR_LOCALS(size_t,  nbq,  q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek,  k, ne)
    GGML_TENSOR_LOCALS(size_t,  nbk,  k, nb)
    GGML_TENSOR_LOCALS(int64_t, new,  w, ne)
    GGML_TENSOR_LOCALS(size_t,  nbw,  w, nb)
    GGML_TENSOR_LOCALS(int64_t, nem,  m, ne)
    GGML_TENSOR_LOCALS(size_t,  nbm,  m, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb)

    GGML_ASSERT( nb0 == ggml_type_size(dst->type));
    GGML_ASSERT(nbq0 == ggml_type_size(  q->type));
    GGML_ASSERT(nbk0 == ggml_type_size(  k->type));
    GGML_ASSERT(nbw0 == ggml_type_size(  w->type));
    GGML_ASSERT(nbm0 == ggml_type_size(  m->type));

    const int n_embd    = q->ne[0];
    const int n_head    = q->ne[1];
    const int n_tokens  = q->ne[2];
    const int n_stream  = q->ne[3];
    const int n_kv      = k->ne[2];

    ggml_to_float_t const k_to_float = ggml_get_type_traits(k->type)->to_float;
    GGML_ASSERT((k->type == GGML_TYPE_F32 || k_to_float) && "lightning indexer: unsupported K-type");

    const int nr  = n_kv;
    const int ith = params->ith;
    const int nth = params->nth;

    // (temporary) buffer for K converted to float
    float * k_row_f32 = (float *) params->wdata + ith*(1*n_embd + CACHE_LINE_SIZE_F32);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int s = 0; s < n_stream; ++s) {
        for (int t = 0; t < n_tokens; ++t) {
            const float       *   w_row =       (float *) ((char *)   w->data + t*nbw1 +        s*nbw3);
            const ggml_fp16_t *   m_row = (ggml_fp16_t *) ((char *)   m->data + t*nbm1 + (s%nem3)*nbm3);
            float             * dst_row =       (float *) ((char *) dst->data + t*nb1  +        s*nb3 );
            for (int ik = ir0; ik < ir1; ++ik) {
                char * k_row = (char *) k->data + ik*nbk2 + s*nbk3;
                if (k_to_float) {
                    k_to_float(k_row, k_row_f32, n_embd);
                } else {
                    k_row_f32 = (float *) k_row;
                }
                float score = 0.0f;
                for (int h = 0; h < n_head; ++h) {
                    // dot product of q and k for head h
                    float qk = 0.0f;
                    const float * q_row = (float *) ((char *) q->data + h*nbq1 + t*nbq2 + s*nbq3);
                    ggml_vec_dot_f32(n_embd, &qk, 0, q_row, 0, k_row_f32, 0, 1);
                    // ReLU and weights (prescaled)
                    score += MAX(qk, 0.0f) * w_row[h];
                }
                // apply mask
                dst_row[ik] = score + GGML_CPU_FP16_TO_FP32(m_row[ik]);
            }
        }
    }
}

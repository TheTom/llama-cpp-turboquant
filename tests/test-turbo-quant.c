#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>

extern void quantize_row_turbo3_0_ref(const float * x, void * y, long long k);
extern void dequantize_row_turbo3_0(const void * x, float * y, long long k);
extern void quantize_row_turbo4_0_ref(const float * x, void * y, long long k);
extern void dequantize_row_turbo4_0(const void * x, float * y, long long k);
extern void turbo_cpu_fwht_inverse(float * x, int group_size);

static bool check_metrics(const char * name, float mse, float cosv, float ni, float no, float max_mse, float min_cos) {
    const float cosine = ni > 0.0f && no > 0.0f ? cosv / sqrtf(ni) / sqrtf(no) : 0.0f;

    printf("  MSE=%.8f Cosine=%.6f InNorm=%.6f OutNorm=%.6f\n\n",
            (double) mse, (double) cosine, (double) sqrtf(ni), (double) sqrtf(no));

    if (mse > max_mse || cosine < min_cos) {
        fprintf(stderr, "%s failed: MSE %.8f > %.8f or cosine %.6f < %.6f\n",
                name, (double) mse, (double) max_mse, (double) cosine, (double) min_cos);
        return false;
    }

    return true;
}

int main(void) {
    const int d = 128;
    char buf[256];
    float input[128], output[128];
    float mse, cosv, ni, no;
    bool ok = true;

    printf("=== TurboQuant C Round-Trip Test ===\n\n");

    /* Test 1: basis vector
     *
     * dequantize_row_turbo3_0 leaves output in the WHT-rotated domain (Q is
     * also rotated by the graph, so <Q_rot, K_rot> yields correct attention
     * scores without an explicit inverse). To verify the round-trip, apply
     * the inverse WHT before comparing against the original input. */
    memset(input, 0, sizeof(input));
    input[0] = 1.0f;
    quantize_row_turbo3_0_ref(input, buf, d);
    dequantize_row_turbo3_0(buf, output, d);
    turbo_cpu_fwht_inverse(output, d);
    printf("Test 1 (turbo3): e0 = [1, 0, ...]\n");
    printf("  In:  [%.6f, %.6f, %.6f, %.6f]\n",
            (double) input[0], (double) input[1], (double) input[2], (double) input[3]);
    printf("  Out: [%.6f, %.6f, %.6f, %.6f]\n",
            (double) output[0], (double) output[1], (double) output[2], (double) output[3]);
    mse = cosv = ni = no = 0;
    for (int i = 0; i < d; i++) { mse += (input[i]-output[i])*(input[i]-output[i]); cosv += input[i]*output[i]; ni += input[i]*input[i]; no += output[i]*output[i]; }
    ok &= check_metrics("turbo3 basis", mse/d, cosv, ni, no, 0.0001f, 0.995f);

    /* Test 2: large-norm vector */
    for (int i = 0; i < d; i++) input[i] = sinf(i*0.1f+0.5f) * 10.0f;
    quantize_row_turbo3_0_ref(input, buf, d);
    dequantize_row_turbo3_0(buf, output, d);
    turbo_cpu_fwht_inverse(output, d);
    printf("Test 2 (turbo3): sin*10\n");
    printf("  In:  [%.4f, %.4f, %.4f, %.4f]\n",
            (double) input[0], (double) input[1], (double) input[2], (double) input[3]);
    printf("  Out: [%.4f, %.4f, %.4f, %.4f]\n",
            (double) output[0], (double) output[1], (double) output[2], (double) output[3]);
    mse = cosv = ni = no = 0;
    for (int i = 0; i < d; i++) { mse += (input[i]-output[i])*(input[i]-output[i]); cosv += input[i]*output[i]; ni += input[i]*input[i]; no += output[i]*output[i]; }
    ok &= check_metrics("turbo3 sin", mse/d, cosv, ni, no, 1.50f, 0.98f);

    /* Test 3: turbo4
     *
     * Same convention as turbo3: dequant leaves output in the rotated domain
     * (see comment in dequantize_row_turbo4_0 @ ggml-turbo-quant.c). Apply
     * the inverse WHT before comparing. */
    for (int i = 0; i < d; i++) input[i] = cosf(i*0.2f) * 5.0f;
    quantize_row_turbo4_0_ref(input, buf, d);
    dequantize_row_turbo4_0(buf, output, d);
    turbo_cpu_fwht_inverse(output, d);
    printf("Test 3 (turbo4): cos*5\n");
    printf("  In:  [%.4f, %.4f, %.4f, %.4f]\n",
            (double) input[0], (double) input[1], (double) input[2], (double) input[3]);
    printf("  Out: [%.4f, %.4f, %.4f, %.4f]\n",
            (double) output[0], (double) output[1], (double) output[2], (double) output[3]);
    mse = cosv = ni = no = 0;
    for (int i = 0; i < d; i++) { mse += (input[i]-output[i])*(input[i]-output[i]); cosv += input[i]*output[i]; ni += input[i]*input[i]; no += output[i]*output[i]; }
    ok &= check_metrics("turbo4 cos", mse/d, cosv, ni, no, 0.35f, 0.98f);

    printf("=== Done ===\n");
    return ok ? 0 : 1;
}

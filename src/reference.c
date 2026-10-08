#include <stddef.h>
#include <stdint.h>

/* Sums in int64 so that no narrower accumulator in the kernel can agree with it by accident. */
void matvec_int8_reference(const int8_t *A, const int8_t *x, const float *scale, float *y, int M, int K)
{
    for (int m = 0; m < M; m++) {
        int64_t sum = 0;
        for (int k = 0; k < K; k++)
            sum += (int64_t)A[(size_t)m * K + k] * (int64_t)x[k];
        y[m] = scale[m] * (float)sum;
    }
}

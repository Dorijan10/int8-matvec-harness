#include <stddef.h>
#include <stdint.h>

/* The test compares y bit-exactly, so keep the exact int32 sum and a single float multiply. */
void matvec_int8(const int8_t *A, const int8_t *x, const float *scale, float *y, int M, int K)
{
    for (int m = 0; m < M; m++) {
        const int8_t *row = A + (size_t)m * K;
        int32_t sum = 0;
        for (int k = 0; k < K; k++)
            sum += (int32_t)row[k] * (int32_t)x[k];
        y[m] = scale[m] * (float)sum;
    }
}

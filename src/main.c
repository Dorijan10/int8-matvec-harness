#define _POSIX_C_SOURCE 199309L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void matvec_int8(const int8_t *A, const int8_t *x, const float *scale, float *y, int M, int K);
void matvec_int8_reference(const int8_t *A, const int8_t *x, const float *scale, float *y, int M, int K);

enum fill { RANDOM, EXTREME, MIN };

struct shape {
    int M, K;
    enum fill fill;
};

static const struct shape shapes[] = {
    {1, 1, RANDOM},
    {5, 1, MIN},           /* K = 1 leaves no full step for any unrolled or vector loop */
    {3, 7, RANDOM},
    {17, 12, RANDOM},      /* leaves a remainder for loops 8 or more wide */
    {9, 24, RANDOM},       /* leaves a remainder for loops 16 or more wide */
    {33, 48, RANDOM},      /* leaves a remainder for loops 64 wide */
    {65, 127, EXTREME},
    {7, 4096, MIN},        /* pairs of -128 * -128 overflow int16, and sums of 2^26 still fit int32 */
    {31, 4099, EXTREME},
    {4096, 4096, RANDOM},  /* the benchmark shape */
    {4095, 4097, RANDOM},
};

enum { GUARD = 16 };

static uint64_t lcg_state;

static uint32_t lcg_next(void)
{
    lcg_state = lcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)(lcg_state >> 32);
}

static void *xmalloc(size_t n)
{
    void *p = malloc(n);
    if (!p) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }
    return p;
}

static void fill_int8(int8_t *v, size_t n, enum fill fill)
{
    for (size_t i = 0; i < n; i++) {
        uint32_t r = lcg_next();
        if (fill == MIN)
            v[i] = -128;
        else if (fill == EXTREME)
            v[i] = (r >> 31) ? 127 : -128;
        else
            v[i] = (int8_t)((int)(r >> 24) - 128);
    }
}

static void fill_scale(float *v, size_t n)
{
    for (size_t i = 0; i < n; i++)
        v[i] = (float)((int32_t)(lcg_next() >> 16) - 32768) / 4194304.0f;
}

static uint32_t bits(float f)
{
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    return u;
}

static int check_reference(void)
{
    /* Worked by hand, so a mistake shared by kernel and reference cannot pass. */
    const int8_t A[6] = {1, 2, 3, -4, 5, -6};
    const int8_t x[3] = {7, -8, 9};
    const float scale[2] = {0.5f, 2.0f};
    float y[2];
    matvec_int8_reference(A, x, scale, y, 2, 3);
    int ok = y[0] == 9.0f && y[1] == -244.0f;
    printf("%s reference against hand-worked values\n", ok ? "ok  " : "FAIL");
    return ok;
}

static int check_shape(const struct shape *s, uint64_t seed)
{
    size_t M = (size_t)s->M, K = (size_t)s->K;
    int8_t *A = xmalloc(M * K);
    int8_t *x = xmalloc(K);
    float *scale = xmalloc(M * sizeof *scale);
    float *got = xmalloc((M + GUARD) * sizeof *got);
    float *want = xmalloc(M * sizeof *want);

    lcg_state = seed;
    fill_int8(A, M * K, s->fill);
    fill_int8(x, K, s->fill);
    fill_scale(scale, M);
    /* All-ones bytes are a NaN no correct kernel writes, so a skipped row or stray write shows. */
    memset(got, 0xFF, (M + GUARD) * sizeof *got);
    matvec_int8(A, x, scale, got, s->M, s->K);
    matvec_int8_reference(A, x, scale, want, s->M, s->K);

    int ok = 1;
    for (size_t m = 0; m < M && ok; m++) {
        if (bits(got[m]) != bits(want[m])) {
            printf("FAIL M=%d K=%d row %zu: got %.9g (0x%08x), want %.9g (0x%08x)\n", s->M, s->K, m,
                   got[m], bits(got[m]), want[m], bits(want[m]));
            ok = 0;
        }
    }
    for (size_t g = 0; g < GUARD && ok; g++) {
        if (bits(got[M + g]) != 0xFFFFFFFFu) {
            printf("FAIL M=%d K=%d: wrote past the end of y\n", s->M, s->K);
            ok = 0;
        }
    }
    if (ok)
        printf("ok   M=%d K=%d\n", s->M, s->K);

    free(A);
    free(x);
    free(scale);
    free(got);
    free(want);
    return ok;
}

static int run_tests(void)
{
    int failed = !check_reference();
    size_t count = sizeof shapes / sizeof shapes[0];
    for (size_t i = 0; i < count; i++)
        failed += !check_shape(&shapes[i], 1000 + i);
    printf("%s\n", failed ? "FAILED" : "PASSED");
    return failed ? 1 : 0;
}

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static int bench(void)
{
    const int M = 4096, K = 4096;
    int8_t *A = xmalloc((size_t)M * K);
    int8_t *x = xmalloc(K);
    float *scale = xmalloc(M * sizeof *scale);
    float *y = xmalloc(M * sizeof *y);

    lcg_state = 42;
    fill_int8(A, (size_t)M * K, RANDOM);
    fill_scale(scale, M);

    /* Warm caches and CPU frequency so the timed repetitions are steady. */
    double start = now();
    while (now() - start < 1.0) {
        fill_int8(x, K, RANDOM);
        matvec_int8(A, x, scale, y, M, K);
    }

    /* x changes every repetition so a kernel cannot reuse an earlier result; only the kernel is timed. */
    double seconds = 0.0, checksum = 0.0;
    long reps = 0;
    while (seconds < 8.0) {
        fill_int8(x, K, RANDOM);
        double t0 = now();
        matvec_int8(A, x, scale, y, M, K);
        seconds += now() - t0;
        checksum += y[reps % M];
        reps++;
    }

    double gops = 2.0 * M * K * (double)reps / seconds / 1e9;
    FILE *out = fopen("artemis_results.json", "w");
    if (!out) {
        perror("artemis_results.json");
        return 1;
    }
    fprintf(out, "{\"gops\": %.4f}\n", gops);
    fclose(out);
    printf("M=%d K=%d reps=%ld seconds=%.3f gops=%.4f checksum=%.6g\n", M, K, reps, seconds, gops, checksum);

    free(A);
    free(x);
    free(scale);
    free(y);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc == 2 && strcmp(argv[1], "test") == 0)
        return run_tests();
    if (argc == 2 && strcmp(argv[1], "bench") == 0)
        return bench();
    fprintf(stderr, "usage: %s test|bench\n", argv[0]);
    return 2;
}

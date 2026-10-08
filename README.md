# int8 matrix-vector harness

A benchmark subject for code optimisation: a naive int8 quantised matrix-vector
product.

## What is measured

`src/matvec.c` computes

    y[m] = scale[m] * sum_k A[m][k] * x[k]

where `A` is an int8 `M x K` matrix in row-major order, `x` is an int8 vector,
the sum is accumulated in int32, `scale` is one float per row, and `y` is float.
The benchmark shape is `M = K = 4096`. All inputs come from a seeded linear
congruential generator, so every run sees identical data.

`src/matvec.c` is the only file an optimiser should need to edit. Its function
signature must stay the same.

| File | Purpose |
| --- | --- |
| `src/matvec.c` | The naive scalar kernel under test. |
| `src/reference.c` | A deliberately simple implementation that sums in int64, used only by the test. |
| `src/main.c` | The test and bench drivers. |
| `Makefile` | `compile`, `test`, `bench`, `clean`. |

## How to run

    make compile   # gcc -O2 -std=c11 -Wall -Wextra
    make test      # exits non-zero on any mismatch
    make bench     # writes artemis_results.json
    make clean

The harness needs only gcc, make and a POSIX shell. To run it as the benchmark
does, in the `example-cpp` image:

    docker run --rm --user "$(id -u):$(id -g)" -v "$PWD":/work -w /work \
      --cpus 2 --memory 8g example-cpp \
      bash -lc 'make clean compile && make test && make bench && cat artemis_results.json'

## Correctness

`make test` first checks the reference against values worked by hand, then
compares the kernel with the reference bit for bit on 11 shapes: `M` and `K`
from 1 to 4099, including `K = 1`, sizes that leave a remainder for loops 4, 8,
16 or 64 wide, and inputs of -128 and 127 whose sums overflow int16 but not
int32. It also fails if the kernel skips a row or writes past the end of `y`.

To match bit for bit, an optimised kernel must compute the exact int32 sum and
then `scale[m] * (float)sum`, one float multiply per row.

## Metric

`make bench` runs the kernel for 1 s to warm up, then repeats it until 8 s of
kernel time have passed, and writes

    {"gops": 2 * M * K * reps / seconds / 1e9}

to `artemis_results.json` in the repository root. Higher is better. `seconds`
counts only the kernel calls. `x` is regenerated before each repetition, so a
kernel cannot reuse an earlier result; `A` and `scale` stay fixed.

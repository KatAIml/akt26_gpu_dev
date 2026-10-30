// =============================================================================
// 5-extras / gemm_opt.cu  —  going further than 4-gemm (not part of the graded code)
//
// Batched GEMM  C[b] = A[b] * B[b],  n x n column-major, n <= 32, FP32.
// Three kernels are compared with cuBLAS (cublasSgemmStridedBatched) for every n = 1..32:
//
//   v0  baseline     : the 4-gemm kernel — one block per matrix, one thread per element of C,
//                      A[b] and B[b] staged in shared memory.
//   v1  multi-matrix : several small matrices per block (blockDim.z = matrices per block), so
//                      that small n still fill whole warps (n = 7 -> 49 threads = 1.5 warps,
//                      15 idle lanes per block with v0).
//   v2  register tile: each thread computes a 2 x 2 tile of C, keeping 2 values of A and 2 of B
//                      in registers per k step -> 1 shared-memory load per FMA instead of 2.
//
// Every kernel is checked against a CPU reference (same forward error as 4-gemm, tol 1e-6).
// Output: a table on screen and a CSV file (default gemm_sweep.csv).
//
// usage: ./gemm_opt [batch_size = 1000] [csv = gemm_sweep.csv]
// =============================================================================
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define NMAX 32

#define CHECK(call)                                                              \
    do {                                                                         \
        cudaError_t e_ = (call);                                                 \
        if (e_ != cudaSuccess) {                                                 \
            std::fprintf(stderr, "CUDA error '%s' at %s:%d\n",                   \
                         cudaGetErrorString(e_), __FILE__, __LINE__);            \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

// -----------------------------------------------------------------------------
// v0: baseline (identical algorithm to 4-gemm/gemm.cu)
__global__ void gemm_v0(int n, int batch, const float* A, const float* B, float* C)
{
    __shared__ float sA[NMAX][NMAX];
    __shared__ float sB[NMAX][NMAX];

    const int b = blockIdx.x, row = threadIdx.x, col = threadIdx.y;
    if (b >= batch) return;                       // uniform for the whole block

    const size_t base = (size_t)b * n * n;
    sA[col][row] = A[base + (size_t)col * n + row];
    sB[col][row] = B[base + (size_t)col * n + row];
    __syncthreads();

    float sum = 0.0f;
    for (int k = 0; k < n; ++k) sum += sA[k][row] * sB[col][k];
    C[base + (size_t)col * n + row] = sum;
}

// -----------------------------------------------------------------------------
// v1: several matrices per block. block = (n, n, mpb), dynamic shared memory = mpb * 2 * n*n floats.
// Threads are numbered x fastest, then y, then z, so for small n one warp spans several matrices.
__global__ void gemm_v1_multi(int n, int batch, const float* A, const float* B, float* C)
{
    extern __shared__ float smem[];

    const int row = threadIdx.x, col = threadIdx.y, z = threadIdx.z;
    const int nn  = n * n;
    const int b   = blockIdx.x * blockDim.z + z;  // matrix handled by this z-slice
    const bool active = (b < batch);              // the last block may be partially filled

    float* sA = smem + (size_t)z * 2 * nn;
    float* sB = sA + nn;
    const size_t base = (size_t)b * nn;

    if (active) {
        sA[col * n + row] = A[base + (size_t)col * n + row];
        sB[col * n + row] = B[base + (size_t)col * n + row];
    }
    __syncthreads();                              // reached by every thread (no early return)

    if (active) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) sum += sA[k * n + row] * sB[col * n + k];
        C[base + (size_t)col * n + row] = sum;
    }
}

// -----------------------------------------------------------------------------
// v2: 2 x 2 register tile per thread. block = (ceil(n/2), ceil(n/2)), one matrix per block.
// Thread (tx, ty) computes rows {tx, tx + RT} and columns {ty, ty + CT} of C.
__global__ void gemm_v2_regtile(int n, int batch, const float* A, const float* B, float* C)
{
    __shared__ float sA[NMAX * NMAX];
    __shared__ float sB[NMAX * NMAX];

    const int b = blockIdx.x;
    if (b >= batch) return;                       // uniform for the whole block

    const int RT = blockDim.x, CT = blockDim.y;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int nthreads = RT * CT, tid = ty * RT + tx;
    const int nn = n * n;
    const size_t base = (size_t)b * nn;

    // cooperative, coalesced load of A[b] and B[b] (fewer threads than elements)
    for (int idx = tid; idx < nn; idx += nthreads) {
        sA[idx] = A[base + idx];
        sB[idx] = B[base + idx];
    }
    __syncthreads();

    const int r0 = tx, r1 = tx + RT;              // the two rows of this thread
    const int c0 = ty, c1 = ty + CT;              // the two columns of this thread
    const bool vr1 = (r1 < n), vc1 = (c1 < n);

    float acc00 = 0.f, acc01 = 0.f, acc10 = 0.f, acc11 = 0.f;
    for (int k = 0; k < n; ++k) {
        const float a0 = sA[k * n + r0];
        const float a1 = vr1 ? sA[k * n + r1] : 0.f;
        const float b0 = sB[c0 * n + k];
        const float b1 = vc1 ? sB[c1 * n + k] : 0.f;
        acc00 += a0 * b0;  acc01 += a0 * b1;      // 4 FMAs for 4 shared loads
        acc10 += a1 * b0;  acc11 += a1 * b1;
    }

    if (r0 < n && c0 < n) C[base + (size_t)c0 * n + r0] = acc00;
    if (r0 < n && vc1)    C[base + (size_t)c1 * n + r0] = acc01;
    if (vr1 && c0 < n)    C[base + (size_t)c0 * n + r1] = acc10;
    if (vr1 && vc1)       C[base + (size_t)c1 * n + r1] = acc11;
}

// -----------------------------------------------------------------------------
void batched_gemm_cpu(int n, int batch, const float* A, const float* B, float* C)
{
    const size_t ms = (size_t)n * n;
    for (int b = 0; b < batch; ++b) {
        const size_t base = (size_t)b * ms;
        for (int col = 0; col < n; ++col)
            for (int row = 0; row < n; ++row) {
                float sum = 0.0f;
                for (int k = 0; k < n; ++k)
                    sum += A[base + (size_t)k * n + row] * B[base + (size_t)col * n + k];
                C[base + (size_t)col * n + row] = sum;
            }
    }
}

// same metric as 4-gemm: max over the batch of ||C - Cref||_F / (n ||Cref||_F)
double max_batch_forward_error(int n, int batch, const float* R, const float* ref)
{
    const size_t ms = (size_t)n * n;
    double worst = 0.0;
    for (int b = 0; b < batch; ++b) {
        double d2 = 0.0, r2 = 0.0;
        for (size_t i = 0; i < ms; ++i) {
            const double d = (double)R[b * ms + i] - (double)ref[b * ms + i];
            d2 += d * d;
            r2 += (double)ref[b * ms + i] * ref[b * ms + i];
        }
        const double e = (r2 > 0.0) ? std::sqrt(d2) / (n * std::sqrt(r2)) : (d2 == 0.0 ? 0.0 : INFINITY);
        worst = std::max(worst, e);
    }
    return worst;
}

// -----------------------------------------------------------------------------
int main(int argc, char** argv)
{
    const int batch   = (argc > 1) ? std::atoi(argv[1]) : 1000;
    const char* csv   = (argc > 2) ? argv[2] : "gemm_sweep.csv";
    const int repeats = 100;
    const double tol  = 1e-6;
    if (batch <= 0) { std::fprintf(stderr, "usage: %s [batch_size > 0] [csv]\n", argv[0]); return EXIT_FAILURE; }

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s (compute capability %d.%d) | batch = %d | repeats = %d\n\n",
                prop.name, prop.major, prop.minor, batch, repeats);

    FILE* f = std::fopen(csv, "w");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", csv); return EXIT_FAILURE; }
    std::fprintf(f, "gpu,n,batch,impl,time_us,gflops,fwd_error,pass\n");

    std::printf("%3s | %22s | %22s | %22s | %22s | %s\n", "n",
                "v0 baseline us (GF/s)", "v1 multi us (GF/s)", "v2 regtile us (GF/s)", "cuBLAS us (GF/s)", "best");
    std::printf("----+------------------------+------------------------+------------------------+------------------------+------------\n");

    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));

    bool all_pass = true;
    for (int n = 1; n <= NMAX; ++n) {
        const size_t total = (size_t)n * n * batch, bytes = total * sizeof(float);
        std::vector<float> hA(total), hB(total), hRef(total), hC(total);
        for (size_t i = 0; i < total; ++i) {
            hA[i] = std::sin(0.013f * (float)(i + 1));
            hB[i] = std::cos(0.017f * (float)(i + 1));
        }
        batched_gemm_cpu(n, batch, hA.data(), hB.data(), hRef.data());

        float *dA, *dB, *dC;
        CHECK(cudaMalloc(&dA, bytes)); CHECK(cudaMalloc(&dB, bytes)); CHECK(cudaMalloc(&dC, bytes));
        CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

        // launch configurations
        // v1: matrices per block (~256 threads), capped at 64 because blockDim.z <= 64 in CUDA
        const int mpb = std::min(64, std::max(1, 256 / (n * n)));
        const dim3 blk0(n, n, 1),   grd0(batch, 1, 1);
        const dim3 blk1(n, n, mpb), grd1((batch + mpb - 1) / mpb, 1, 1);
        const size_t shm1 = (size_t)mpb * 2 * n * n * sizeof(float);
        const int h2 = (n + 1) / 2;
        const dim3 blk2(h2, h2, 1), grd2(batch, 1, 1);
        const float one = 1.0f, zero = 0.0f;
        const long long stride = (long long)n * n;

        const char* names[4] = {"v0_baseline", "v1_multi", "v2_regtile", "cublas"};
        double us[4], err[4];
        for (int impl = 0; impl < 4; ++impl) {
            auto run = [&]() {
                switch (impl) {
                    case 0: gemm_v0<<<grd0, blk0>>>(n, batch, dA, dB, dC); break;
                    case 1: gemm_v1_multi<<<grd1, blk1, shm1>>>(n, batch, dA, dB, dC); break;
                    case 2: gemm_v2_regtile<<<grd2, blk2>>>(n, batch, dA, dB, dC); break;
                    default: cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one,
                                                       dA, n, stride, dB, n, stride, &zero, dC, n, stride, batch);
                }
            };
            CHECK(cudaMemset(dC, 0, bytes));
            run();                                                         // warm-up + result for the check
            CHECK(cudaGetLastError());
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));
            err[impl] = max_batch_forward_error(n, batch, hC.data(), hRef.data());

            CHECK(cudaEventRecord(t0));
            for (int r = 0; r < repeats; ++r) run();
            CHECK(cudaEventRecord(t1));
            CHECK(cudaEventSynchronize(t1));
            float ms = 0.f;
            CHECK(cudaEventElapsedTime(&ms, t0, t1));
            us[impl] = ms * 1000.0 / repeats;
        }

        const double flops = 2.0 * n * n * n * batch;
        int best = 0;
        for (int i = 1; i < 4; ++i) if (us[i] < us[best]) best = i;
        std::printf("%3d |", n);
        for (int i = 0; i < 4; ++i) {
            const bool ok = err[i] <= tol;
            all_pass = all_pass && ok;
            std::printf(" %9.2f (%8.1f)%s |", us[i], flops / (us[i] * 1e3), ok ? "   " : " !!");
            std::fprintf(f, "%s,%d,%d,%s,%.4f,%.4f,%.6e,%d\n", prop.name, n, batch, names[i], us[i],
                         flops / (us[i] * 1e3), err[i], ok ? 1 : 0);
        }
        std::printf(" %s\n", names[best]);

        CHECK(cudaFree(dA)); CHECK(cudaFree(dB)); CHECK(cudaFree(dC));
    }

    std::fclose(f);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cublasDestroy(handle);
    std::printf("\n%s (every kernel checked against the CPU reference, tol = %.0e) | CSV: %s\n",
                all_pass ? "ALL TESTS PASS" : "SOME TESTS FAIL (!!)", tol, csv);
    return all_pass ? 0 : 1;
}

// =============================================================================
// 5-extras / team_compare.cu  —  our two independent 4-gemm solutions, side by side
//
// Two team members (Khalil and Achraf) solved 4-gemm independently and reached the same
// design: one block per matrix, one thread per element of C, A and B staged in shared memory.
// The only difference is the SHARED-MEMORY LAYOUT:
//   khalil_2d     : float sA[32][32], element (row, col) at sA[col][row]   -> column stride 32
//   achraf_packed : float sA[32*32],  element (row, col) at sA[col*n + row] -> column stride n
// With stride 32, two different columns of B that are read by the same warp (n < 32) map to
// the SAME shared-memory bank -> bank conflict. With stride n they usually do not.
// This program measures both (and cuBLAS) for every n = 1..32, checks every result against
// the CPU, and writes a CSV.
//
// usage: ./team_compare [batch = 1000] [csv = team_compare.csv]
// =============================================================================
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call)                                                              \
    do {                                                                         \
        cudaError_t e_ = (call);                                                 \
        if (e_ != cudaSuccess) {                                                 \
            std::fprintf(stderr, "CUDA error '%s' at %s:%d\n",                   \
                         cudaGetErrorString(e_), __FILE__, __LINE__);            \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

// Khalil's version (first 4-gemm solution in this repo): 2D shared arrays, column stride 32
__global__ void gemm_khalil_2d(int n, int batch, const float* A, const float* B, float* C)
{
    __shared__ float sA[32][32];
    __shared__ float sB[32][32];
    const int b = blockIdx.x, row = threadIdx.x, col = threadIdx.y;
    if (b >= batch) return;
    const size_t base = (size_t)b * n * n;
    sA[col][row] = A[base + (size_t)col * n + row];
    sB[col][row] = B[base + (size_t)col * n + row];
    __syncthreads();
    float sum = 0.0f;
    for (int k = 0; k < n; ++k) sum += sA[k][row] * sB[col][k];
    C[base + (size_t)col * n + row] = sum;
}

// Achraf's version (now the graded 4-gemm kernel): packed shared arrays, column stride n
__global__ void gemm_achraf_packed(int n, int batch, const float* A, const float* B, float* C)
{
    __shared__ float sA[32 * 32];
    __shared__ float sB[32 * 32];
    const int b = blockIdx.x;
    if (b >= batch) return;
    const int row = threadIdx.x, col = threadIdx.y;
    const size_t base = (size_t)b * n * n;
    sA[col * n + row] = A[base + (size_t)col * n + row];
    sB[col * n + row] = B[base + (size_t)col * n + row];
    __syncthreads();
    float sum = 0.0f;
    for (int k = 0; k < n; ++k) sum += sA[k * n + row] * sB[col * n + k];
    C[base + (size_t)col * n + row] = sum;
}

void batched_gemm_cpu(int n, int batch, const float* A, const float* B, float* C)
{
    const size_t ms = (size_t)n * n;
    for (int b = 0; b < batch; ++b) {
        const size_t base = (size_t)b * ms;
        for (int col = 0; col < n; ++col)
            for (int row = 0; row < n; ++row) {
                float sum = 0.0f;
                for (int k = 0; k < n; ++k) sum += A[base + (size_t)k * n + row] * B[base + (size_t)col * n + k];
                C[base + (size_t)col * n + row] = sum;
            }
    }
}

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
        worst = std::max(worst, (r2 > 0.0) ? std::sqrt(d2) / (n * std::sqrt(r2)) : (d2 == 0.0 ? 0.0 : INFINITY));
    }
    return worst;
}

int main(int argc, char** argv)
{
    const int batch = (argc > 1) ? std::atoi(argv[1]) : 1000;
    const char* csv = (argc > 2) ? argv[2] : "team_compare.csv";
    const int repeats = 200;
    const double tol = 1e-6;
    if (batch <= 0) { std::fprintf(stderr, "usage: %s [batch > 0] [csv]\n", argv[0]); return EXIT_FAILURE; }

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s | batch = %d | repeats = %d\n\n", prop.name, batch, repeats);

    FILE* f = std::fopen(csv, "w");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", csv); return EXIT_FAILURE; }
    std::fprintf(f, "gpu,n,batch,impl,time_us,gflops,fwd_error,pass\n");
    std::printf("%3s | %22s | %22s | %22s | %s\n", "n", "khalil_2d us (GF/s)", "achraf_packed us (GF/s)", "cuBLAS us (GF/s)", "packed vs 2d");
    std::printf("----+------------------------+------------------------+------------------------+-------------\n");

    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));
    bool all_pass = true;

    for (int n = 1; n <= 32; ++n) {
        const size_t total = (size_t)n * n * batch, bytes = total * sizeof(float);
        std::vector<float> hA(total), hB(total), hRef(total), hC(total);
        for (size_t i = 0; i < total; ++i) { hA[i] = std::sin(0.013f * (float)(i + 1)); hB[i] = std::cos(0.017f * (float)(i + 1)); }
        batched_gemm_cpu(n, batch, hA.data(), hB.data(), hRef.data());

        float *dA, *dB, *dC;
        CHECK(cudaMalloc(&dA, bytes)); CHECK(cudaMalloc(&dB, bytes)); CHECK(cudaMalloc(&dC, bytes));
        CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

        const dim3 blk(n, n, 1), grd(batch, 1, 1);
        const float one = 1.0f, zero = 0.0f;
        const long long stride = (long long)n * n;
        const char* names[3] = {"khalil_2d", "achraf_packed", "cublas"};
        double us[3], err[3];
        for (int impl = 0; impl < 3; ++impl) {
            auto run = [&]() {
                if (impl == 0)      gemm_khalil_2d<<<grd, blk>>>(n, batch, dA, dB, dC);
                else if (impl == 1) gemm_achraf_packed<<<grd, blk>>>(n, batch, dA, dB, dC);
                else cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one,
                                               dA, n, stride, dB, n, stride, &zero, dC, n, stride, batch);
            };
            CHECK(cudaMemset(dC, 0, bytes));
            run();
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
        std::printf("%3d |", n);
        for (int i = 0; i < 3; ++i) {
            const bool ok = err[i] <= tol;
            all_pass = all_pass && ok;
            std::printf(" %9.2f (%8.1f)%s |", us[i], flops / (us[i] * 1e3), ok ? "   " : " !!");
            std::fprintf(f, "%s,%d,%d,%s,%.4f,%.4f,%.6e,%d\n", prop.name, n, batch, names[i], us[i], flops / (us[i] * 1e3), err[i], ok ? 1 : 0);
        }
        std::printf(" %6.2fx\n", us[0] / us[1]);
        CHECK(cudaFree(dA)); CHECK(cudaFree(dB)); CHECK(cudaFree(dC));
    }
    std::fclose(f);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cublasDestroy(handle);
    std::printf("\n%s | 'packed vs 2d' > 1 means the packed layout is faster | CSV: %s\n",
                all_pass ? "ALL TESTS PASS" : "SOME TESTS FAIL (!!)", csv);
    return all_pass ? 0 : 1;
}

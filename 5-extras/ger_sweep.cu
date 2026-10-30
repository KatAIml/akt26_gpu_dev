// =============================================================================
// 5-extras / ger_sweep.cu  —  launch-configuration study for the rank-1 update (not graded)
//
// A <- A + alpha * x * y^T, m x n column-major, FP32. Same kernel as 3-ger/ger.cu.
// For several matrix sizes and thread-block shapes we measure time, GFLOP/s and the
// effective memory bandwidth (GER is memory-bound: A is read and written once,
// 8 bytes per element for 2 flops). cuBLAS SGER is measured as the reference.
//
// The interesting comparison: block.x along the ROWS (consecutive threads -> consecutive
// addresses, coalesced) versus block.x = 1 (consecutive threads -> addresses m*4 bytes
// apart, uncoalesced).
//
// usage: ./ger_sweep [csv = ger_sweep.csv]
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

__global__ void ger_kernel(int m, int n, float alpha, const float* x, const float* y, float* A)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < m && j < n) A[(size_t)j * m + i] += alpha * x[i] * y[j];
}

void ger_cpu(int m, int n, float alpha, const float* x, const float* y, float* A)
{
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) A[(size_t)j * m + i] += alpha * x[i] * y[j];
}

double error_norm(size_t size, const float* r, const float* ref)
{
    double e = 0.0, nrm = 0.0;
    for (size_t i = 0; i < size; ++i) {
        const double d = (double)r[i] - (double)ref[i];
        e += d * d;
        nrm += (double)ref[i] * ref[i];
    }
    return std::sqrt(e) / std::sqrt(nrm);
}

int main(int argc, char** argv)
{
    const char* csv = (argc > 1) ? argv[1] : "ger_sweep.csv";
    const int repeats = 100;
    const float alpha = 1.5f;
    const double tol = 1e-6;

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s (compute capability %d.%d) | repeats = %d\n", prop.name, prop.major, prop.minor, repeats);

    const int sizes[][2] = {{4000, 2000}, {1000, 777}, {8192, 8192}, {4001, 3}};
    const int shapes[][2] = {{32, 1}, {32, 4}, {32, 8}, {32, 16}, {32, 32}, {64, 4}, {128, 2}, {256, 1},
                             {16, 16}, {8, 32}, {1, 256}};
    const int nshapes = sizeof(shapes) / sizeof(shapes[0]);

    FILE* f = std::fopen(csv, "w");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", csv); return EXIT_FAILURE; }
    std::fprintf(f, "gpu,m,n,block_x,block_y,impl,time_us,gflops,bandwidth_GBs,error,pass\n");

    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));
    bool all_pass = true;

    for (const auto& s : sizes) {
        const int m = s[0], n = s[1];
        const size_t ms = (size_t)m * n, bytes = ms * sizeof(float);
        std::vector<float> hx(m), hy(n), hA0(ms), hRef, hOut(ms);
        for (int i = 0; i < m; ++i) hx[i] = std::sin(0.1f * (i + 1));
        for (int j = 0; j < n; ++j) hy[j] = std::cos(0.2f * (j + 1));
        for (size_t i = 0; i < ms; ++i) hA0[i] = 0.1f * std::sin(0.1f * (float)(i + 1));
        hRef = hA0;
        ger_cpu(m, n, alpha, hx.data(), hy.data(), hRef.data());

        float *dx, *dy, *dA;
        CHECK(cudaMalloc(&dx, m * sizeof(float))); CHECK(cudaMalloc(&dy, n * sizeof(float))); CHECK(cudaMalloc(&dA, bytes));
        CHECK(cudaMemcpy(dx, hx.data(), m * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dy, hy.data(), n * sizeof(float), cudaMemcpyHostToDevice));

        const double flops = 2.0 * m * n;
        const double traffic = 2.0 * bytes + (m + n) * sizeof(float);   // read + write A, read x and y

        std::printf("\nMatrix %d x %d\n%-14s %12s %10s %12s %12s\n", m, n, "block (x,y)", "time (us)", "GFLOP/s", "GB/s", "error");
        for (int c = 0; c <= nshapes; ++c) {
            const bool is_cublas = (c == nshapes);
            dim3 block, grid;
            if (!is_cublas) {
                block = dim3(shapes[c][0], shapes[c][1], 1);
                grid  = dim3((m + block.x - 1) / block.x, (n + block.y - 1) / block.y, 1);
                if (grid.y > 65535) continue;                             // grid.y limit
            }
            auto run = [&]() {
                if (is_cublas) cublasSger(handle, m, n, &alpha, dx, 1, dy, 1, dA, m);
                else ger_kernel<<<grid, block>>>(m, n, alpha, dx, dy, dA);
            };
            // correctness: one update from the initial matrix
            CHECK(cudaMemcpy(dA, hA0.data(), bytes, cudaMemcpyHostToDevice));
            run();
            CHECK(cudaGetLastError());
            CHECK(cudaMemcpy(hOut.data(), dA, bytes, cudaMemcpyDeviceToHost));
            const double err = error_norm(ms, hOut.data(), hRef.data());
            const bool ok = err <= tol;
            all_pass = all_pass && ok;

            // timing (warm-up done above), starting again from the initial matrix
            CHECK(cudaMemcpy(dA, hA0.data(), bytes, cudaMemcpyHostToDevice));
            CHECK(cudaEventRecord(t0));
            for (int r = 0; r < repeats; ++r) run();
            CHECK(cudaEventRecord(t1));
            CHECK(cudaEventSynchronize(t1));
            float msec = 0.f;
            CHECK(cudaEventElapsedTime(&msec, t0, t1));
            const double us = msec * 1000.0 / repeats;

            char label[32];
            if (is_cublas) std::snprintf(label, sizeof(label), "cuBLAS SGER");
            else std::snprintf(label, sizeof(label), "(%d, %d)", shapes[c][0], shapes[c][1]);
            std::printf("%-14s %12.2f %10.1f %12.1f %12.3e%s\n", label, us, flops / (us * 1e3), traffic / (us * 1e3), err, ok ? "" : "  FAIL");
            std::fprintf(f, "%s,%d,%d,%d,%d,%s,%.4f,%.4f,%.4f,%.6e,%d\n", prop.name, m, n,
                         is_cublas ? 0 : shapes[c][0], is_cublas ? 0 : shapes[c][1],
                         is_cublas ? "cublas" : "custom", us, flops / (us * 1e3), traffic / (us * 1e3), err, ok ? 1 : 0);
        }
        CHECK(cudaFree(dx)); CHECK(cudaFree(dy)); CHECK(cudaFree(dA));
    }

    std::fclose(f);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cublasDestroy(handle);
    std::printf("\n%s | CSV: %s\n", all_pass ? "ALL TESTS PASS" : "SOME TESTS FAIL", csv);
    return all_pass ? 0 : 1;
}

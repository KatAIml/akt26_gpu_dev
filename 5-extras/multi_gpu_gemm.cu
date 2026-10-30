// =============================================================================
// 5-extras / multi_gpu_gemm.cu  —  batched GEMM split across several GPUs (not graded)
//
// C[b] = A[b] * B[b], n x n column-major, FP32. The products are independent, so the batch is
// split into contiguous chunks, one per GPU (GPU g gets matrices [first_g, first_g + count_g)).
// Each GPU uses the fastest kernel from gemm_opt.cu for that n (v1 for n <= 11, v2 otherwise).
//
// Strong scaling (fixed total batch) for 1 .. #GPUs, measured in two ways:
//   (A) kernels only : A and B already on the GPUs, we time the computation alone
//   (B) end-to-end   : copy A and B in (pinned host memory, async copies), compute, copy C out
// The gap between (A) and (B) shows how much the PCIe transfers limit multi-GPU scaling.
// Every result is checked against a CPU reference (same forward error as 4-gemm, tol 1e-6).
//
// usage: ./multi_gpu_gemm [batch = 50000] [csv = multi_gpu.csv]
// =============================================================================
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
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

// v1: several matrices per block (same as gemm_opt.cu)
__global__ void gemm_v1_multi(int n, int batch, const float* A, const float* B, float* C)
{
    extern __shared__ float smem[];
    const int row = threadIdx.x, col = threadIdx.y, z = threadIdx.z;
    const int nn = n * n;
    const int b = blockIdx.x * blockDim.z + z;
    const bool active = (b < batch);
    float* sA = smem + (size_t)z * 2 * nn;
    float* sB = sA + nn;
    const size_t base = (size_t)b * nn;
    if (active) {
        sA[col * n + row] = A[base + (size_t)col * n + row];
        sB[col * n + row] = B[base + (size_t)col * n + row];
    }
    __syncthreads();
    if (active) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) sum += sA[k * n + row] * sB[col * n + k];
        C[base + (size_t)col * n + row] = sum;
    }
}

// v2: 2 x 2 register tile per thread (same as gemm_opt.cu)
__global__ void gemm_v2_regtile(int n, int batch, const float* A, const float* B, float* C)
{
    __shared__ float sA[NMAX * NMAX];
    __shared__ float sB[NMAX * NMAX];
    const int b = blockIdx.x;
    if (b >= batch) return;
    const int RT = blockDim.x, CT = blockDim.y, tx = threadIdx.x, ty = threadIdx.y;
    const int nthreads = RT * CT, tid = ty * RT + tx, nn = n * n;
    const size_t base = (size_t)b * nn;
    for (int idx = tid; idx < nn; idx += nthreads) { sA[idx] = A[base + idx]; sB[idx] = B[base + idx]; }
    __syncthreads();
    const int r0 = tx, r1 = tx + RT, c0 = ty, c1 = ty + CT;
    const bool vr1 = (r1 < n), vc1 = (c1 < n);
    float acc00 = 0.f, acc01 = 0.f, acc10 = 0.f, acc11 = 0.f;
    for (int k = 0; k < n; ++k) {
        const float a0 = sA[k * n + r0], a1 = vr1 ? sA[k * n + r1] : 0.f;
        const float b0 = sB[c0 * n + k], b1 = vc1 ? sB[c1 * n + k] : 0.f;
        acc00 += a0 * b0; acc01 += a0 * b1; acc10 += a1 * b0; acc11 += a1 * b1;
    }
    if (r0 < n && c0 < n) C[base + (size_t)c0 * n + r0] = acc00;
    if (r0 < n && vc1)    C[base + (size_t)c1 * n + r0] = acc01;
    if (vr1 && c0 < n)    C[base + (size_t)c0 * n + r1] = acc10;
    if (vr1 && vc1)       C[base + (size_t)c1 * n + r1] = acc11;
}

// launch the best kernel for this n on the current device / stream
void launch_gemm(int n, int count, const float* A, const float* B, float* C, cudaStream_t s)
{
    if (count <= 0) return;
    if (n <= 11) {
        const int mpb = std::min(64, std::max(1, 256 / (n * n)));
        const dim3 blk(n, n, mpb), grd((count + mpb - 1) / mpb, 1, 1);
        gemm_v1_multi<<<grd, blk, (size_t)mpb * 2 * n * n * sizeof(float), s>>>(n, count, A, B, C);
    } else {
        const int h = (n + 1) / 2;
        gemm_v2_regtile<<<dim3(count, 1, 1), dim3(h, h, 1), 0, s>>>(n, count, A, B, C);
    }
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

struct Part {                 // one GPU's share of the batch
    int dev = 0, first = 0, count = 0;
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaStream_t s = nullptr;
};

static double now_us()
{
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv)
{
    const int batch = (argc > 1) ? std::atoi(argv[1]) : 50000;
    const char* csv = (argc > 2) ? argv[2] : "multi_gpu.csv";
    const int reps_kernel = 50, reps_e2e = 10;
    const double tol = 1e-6;
    if (batch <= 0) { std::fprintf(stderr, "usage: %s [batch > 0] [csv]\n", argv[0]); return EXIT_FAILURE; }

    int ndev = 0;
    CHECK(cudaGetDeviceCount(&ndev));
    std::printf("GPUs found: %d\n", ndev);
    for (int d = 0; d < ndev; ++d) {
        cudaDeviceProp p;
        CHECK(cudaGetDeviceProperties(&p, d));
        std::printf("  GPU %d: %s (compute capability %d.%d, %.1f GiB)\n", d, p.name, p.major, p.minor, p.totalGlobalMem / 1073741824.0);
    }
    cudaDeviceProp p0;
    CHECK(cudaGetDeviceProperties(&p0, 0));

    FILE* f = std::fopen(csv, "w");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", csv); return EXIT_FAILURE; }
    std::fprintf(f, "gpu,n,batch,ngpu,mode,time_us,gflops,speedup,efficiency,fwd_error,pass\n");

    bool all_pass = true;
    const int ns[] = {8, 16, 32};
    for (int n : ns) {
        const size_t ms = (size_t)n * n, total = ms * batch, bytes = total * sizeof(float);
        const double flops = 2.0 * n * n * n * batch;

        // pinned host memory: required for truly asynchronous copies that overlap across GPUs
        float *hA, *hB, *hC;
        CHECK(cudaMallocHost(&hA, bytes)); CHECK(cudaMallocHost(&hB, bytes)); CHECK(cudaMallocHost(&hC, bytes));
        std::vector<float> ref(total);
        for (size_t i = 0; i < total; ++i) { hA[i] = std::sin(0.013f * (float)(i + 1)); hB[i] = std::cos(0.017f * (float)(i + 1)); }
        batched_gemm_cpu(n, batch, hA, hB, ref.data());

        std::printf("\nn = %d, batch = %d (%.1f MB per operand, %.2f GFLOP per batch)\n", n, batch, bytes / 1e6, flops / 1e9);
        std::printf("%5s | %24s %8s %6s | %24s %8s %6s | %s\n", "GPUs", "(A) kernels only us (GF/s)", "speedup", "eff.",
                    "(B) end-to-end us (GF/s)", "speedup", "eff.", "error");

        double t1_kernel = 0, t1_e2e = 0;
        for (int ng = 1; ng <= ndev; ++ng) {
            // split the batch into ng contiguous chunks
            std::vector<Part> parts(ng);
            for (int g = 0; g < ng; ++g) {
                parts[g].dev = g;
                parts[g].first = (int)((long long)batch * g / ng);
                parts[g].count = (int)((long long)batch * (g + 1) / ng) - parts[g].first;
                const size_t b_g = (size_t)parts[g].count * ms * sizeof(float);
                CHECK(cudaSetDevice(g));
                CHECK(cudaStreamCreate(&parts[g].s));
                CHECK(cudaMalloc(&parts[g].dA, b_g)); CHECK(cudaMalloc(&parts[g].dB, b_g)); CHECK(cudaMalloc(&parts[g].dC, b_g));
            }
            auto off = [&](const Part& pt) { return (size_t)pt.first * ms; };
            auto bytes_of = [&](const Part& pt) { return (size_t)pt.count * ms * sizeof(float); };
            auto sync_all = [&]() { for (auto& pt : parts) { CHECK(cudaSetDevice(pt.dev)); CHECK(cudaStreamSynchronize(pt.s)); } };

            // ---------- (B) end-to-end: H2D A,B -> kernel -> D2H C, all GPUs working concurrently ----------
            auto e2e_once = [&]() {
                for (auto& pt : parts) {
                    CHECK(cudaSetDevice(pt.dev));
                    CHECK(cudaMemcpyAsync(pt.dA, hA + off(pt), bytes_of(pt), cudaMemcpyHostToDevice, pt.s));
                    CHECK(cudaMemcpyAsync(pt.dB, hB + off(pt), bytes_of(pt), cudaMemcpyHostToDevice, pt.s));
                    launch_gemm(n, pt.count, pt.dA, pt.dB, pt.dC, pt.s);
                    CHECK(cudaMemcpyAsync(hC + off(pt), pt.dC, bytes_of(pt), cudaMemcpyDeviceToHost, pt.s));
                }
                sync_all();
            };
            std::fill(hC, hC + total, 0.0f);
            e2e_once();                                        // warm-up, and it leaves A and B on the GPUs
            CHECK(cudaGetLastError());
            const double err = max_batch_forward_error(n, batch, hC, ref.data());
            const bool ok = err <= tol;
            all_pass = all_pass && ok;

            double t0 = now_us();
            for (int r = 0; r < reps_e2e; ++r) e2e_once();
            const double t_e2e = (now_us() - t0) / reps_e2e;

            // ---------- (A) kernels only: data already resident on every GPU ----------
            for (auto& pt : parts) { CHECK(cudaSetDevice(pt.dev)); launch_gemm(n, pt.count, pt.dA, pt.dB, pt.dC, pt.s); }
            sync_all();
            t0 = now_us();
            for (int r = 0; r < reps_kernel; ++r)
                for (auto& pt : parts) { CHECK(cudaSetDevice(pt.dev)); launch_gemm(n, pt.count, pt.dA, pt.dB, pt.dC, pt.s); }
            sync_all();
            const double t_kernel = (now_us() - t0) / reps_kernel;

            if (ng == 1) { t1_kernel = t_kernel; t1_e2e = t_e2e; }
            const double sk = t1_kernel / t_kernel, se = t1_e2e / t_e2e;
            std::printf("%5d | %12.1f (%9.1f) %8.2f %6.2f | %12.1f (%9.1f) %8.2f %6.2f | %.2e%s\n", ng,
                        t_kernel, flops / (t_kernel * 1e3), sk, sk / ng, t_e2e, flops / (t_e2e * 1e3), se, se / ng, err, ok ? "" : "  FAIL");
            std::fprintf(f, "%s,%d,%d,%d,kernels_only,%.3f,%.3f,%.4f,%.4f,%.6e,%d\n", p0.name, n, batch, ng, t_kernel, flops / (t_kernel * 1e3), sk, sk / ng, err, ok);
            std::fprintf(f, "%s,%d,%d,%d,end_to_end,%.3f,%.3f,%.4f,%.4f,%.6e,%d\n", p0.name, n, batch, ng, t_e2e, flops / (t_e2e * 1e3), se, se / ng, err, ok);

            // end-to-end traffic: A, B in and C out = 3 operands over PCIe
            if (ng == ndev)
                std::printf("      end-to-end moves %.0f MB over PCIe per batch: %.1f GB/s aggregate with %d GPU(s); the kernel is %.0f%% of the 1-GPU end-to-end time\n",
                            3.0 * bytes / 1e6, 3.0 * bytes / (t_e2e * 1e3), ng, 100.0 * t1_kernel / t1_e2e);

            for (auto& pt : parts) {
                CHECK(cudaSetDevice(pt.dev));
                CHECK(cudaFree(pt.dA)); CHECK(cudaFree(pt.dB)); CHECK(cudaFree(pt.dC));
                CHECK(cudaStreamDestroy(pt.s));
            }
        }
        CHECK(cudaFreeHost(hA)); CHECK(cudaFreeHost(hB)); CHECK(cudaFreeHost(hC));
    }
    std::fclose(f);
    std::printf("\n%s (every configuration checked against the CPU reference, tol = %.0e) | CSV: %s\n",
                all_pass ? "ALL TESTS PASS" : "SOME TESTS FAIL", tol, csv);
    return all_pass ? 0 : 1;
}

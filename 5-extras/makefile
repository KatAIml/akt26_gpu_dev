# 5-extras: optimized batched GEMM study, GER launch-configuration sweep, multi-GPU batched GEMM (not graded)
CUDADIR ?= /usr/local/cuda

# gpu architecture: Ampere by default, override e.g. `make sm=75` on a Tesla T4 (Colab) or `make sm=60` on a P100
sm=80
GENCODE_SM   = -gencode arch=compute_$(sm),code=sm_$(sm)
GENCODE_COMP = -gencode arch=compute_$(sm),code=compute_$(sm)

LIB=-L${CUDADIR}/lib64 -lcublas

all:
	nvcc -O3 ${GENCODE_SM} ${GENCODE_COMP} gemm_opt.cu -o gemm_opt ${LIB}
	nvcc -O3 ${GENCODE_SM} ${GENCODE_COMP} ger_sweep.cu -o ger_sweep ${LIB}
	nvcc -O3 ${GENCODE_SM} ${GENCODE_COMP} multi_gpu_gemm.cu -o multi_gpu_gemm

clean:
	rm -rf gemm_opt ger_sweep multi_gpu_gemm

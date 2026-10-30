#!/usr/bin/env bash
# 5-extras: multi-GPU batched GEMM study (e.g. Kaggle "GPU T4 x2")
#   usage (from the repository root):  bash 5-extras/run_multi_gpu.sh <sm>     e.g. 75 for Tesla T4
set -u
SM=${1:-80}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
OUT="$ROOT/5-extras/results_${GPU}_x${NGPU}"
mkdir -p "$OUT"
export PATH=/usr/local/cuda/bin:$PATH
echo "GPUs: ${NGPU} x ${GPU} | sm=$SM | results -> $OUT"
nvidia-smi topo -m > "$OUT/topology.txt" 2>&1 || true              # how the GPUs are connected (PCIe / NVLink)

( cd "$ROOT/5-extras" && make -s sm=$SM ) || exit 1
( cd "$ROOT/5-extras" && ./multi_gpu_gemm 50000 "$OUT/multi_gpu.csv" ) 2>&1 | tee "$OUT/multi_gpu.txt"
python3 "$ROOT/5-extras/plot_multi_gpu.py" "$OUT"
echo "done: $OUT"

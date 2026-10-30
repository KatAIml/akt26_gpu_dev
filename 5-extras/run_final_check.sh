#!/usr/bin/env bash
# Final check of the merged team repository on one GPU:
#   graded programs (default sizes + edge cases), compute-sanitizer, and the team comparison.
#   usage (from the repository root):  bash 5-extras/run_final_check.sh <sm>     e.g. 75 for a Tesla T4
set -u
SM=${1:-80}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
OUT="$ROOT/5-extras/results_${GPU}/final_check"
mkdir -p "$OUT"
export PATH=/usr/local/cuda/bin:$PATH
SAN=$(command -v compute-sanitizer || echo /usr/local/cuda/bin/compute-sanitizer)
echo "GPU: $GPU | sm=$SM | results -> $OUT"

( cd "$ROOT/3-ger"  && make -s sm=$SM ) || exit 1
( cd "$ROOT/4-gemm" && make -s sm=$SM ) || exit 1
( cd "$ROOT/5-extras" && nvcc -O3 -gencode arch=compute_$SM,code=sm_$SM team_compare.cu -o team_compare -lcublas ) || exit 1

{
  for args in "" "1000 777" "4001 3" "1 1" "33 65"; do (cd "$ROOT/3-ger" && ./ger $args); echo; done
  for args in "" "32 1000" "1 1000" "7 5000" "31 20000" "17 1"; do (cd "$ROOT/4-gemm" && ./gemm $args); echo; done
} 2>&1 | tee "$OUT/graded_runs.txt"

{
  echo "== memcheck ger 300 200 2"; (cd "$ROOT/3-ger"  && "$SAN" --tool memcheck --leak-check full ./ger 300 200 2 | tail -3)
  echo "== memcheck gemm 7 50";     (cd "$ROOT/4-gemm" && "$SAN" --tool memcheck --leak-check full ./gemm 7 50   | tail -3)
  echo "== racecheck gemm 16 20";   (cd "$ROOT/4-gemm" && "$SAN" --tool racecheck ./gemm 16 20                   | tail -3)
  echo "== synccheck gemm 32 10";   (cd "$ROOT/4-gemm" && "$SAN" --tool synccheck ./gemm 32 10                   | tail -3)
} 2>&1 | tee "$OUT/sanitizer.txt"

( cd "$ROOT/5-extras" && ./team_compare 1000 "$OUT/team_compare.csv" ) 2>&1 | tee "$OUT/team_compare.txt"
python3 "$ROOT/5-extras/plot_team_compare.py" "$OUT"
echo "done: $OUT"

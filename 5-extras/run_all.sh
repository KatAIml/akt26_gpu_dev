#!/usr/bin/env bash
# 5-extras: build and run everything on one GPU, save the outputs in results_<gpu>/
#   usage (from the repository root):  bash 5-extras/run_all.sh <sm>      e.g. 75 for a Tesla T4 (Colab), 60 for a P100
set -u
SM=${1:-80}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
OUT="$ROOT/5-extras/results_${GPU}"
mkdir -p "$OUT"
SAN=$(command -v compute-sanitizer || echo /usr/local/cuda/bin/compute-sanitizer)
echo "GPU: $GPU | sm=$SM | results -> $OUT"

# 1 · build the graded programs and the extras (the makefiles are not modified, sm is overridden)
( cd "$ROOT/3-ger"    && make -s sm=$SM ) || exit 1
( cd "$ROOT/4-gemm"   && make -s sm=$SM ) || exit 1
( cd "$ROOT/5-extras" && make -s sm=$SM ) || exit 1

# 2 · graded programs: default sizes + edge cases
{
  for args in "" "1000 777" "4001 3" "1 1" "33 65"; do (cd "$ROOT/3-ger" && ./ger $args); echo; done
  for args in "" "32 1000" "1 1000" "7 5000" "31 20000" "17 1"; do (cd "$ROOT/4-gemm" && ./gemm $args); echo; done
} 2>&1 | tee "$OUT/graded_runs.txt"

# 3 · compute-sanitizer: memory errors (memcheck), shared-memory races (racecheck), barrier misuse (synccheck)
{
  echo "== memcheck ger 300 200 2";   (cd "$ROOT/3-ger"  && "$SAN" --tool memcheck  --leak-check full ./ger 300 200 2   | tail -3)
  echo "== memcheck gemm 7 50";       (cd "$ROOT/4-gemm" && "$SAN" --tool memcheck  --leak-check full ./gemm 7 50     | tail -3)
  echo "== racecheck gemm 16 20";     (cd "$ROOT/4-gemm" && "$SAN" --tool racecheck ./gemm 16 20                     | tail -3)
  echo "== synccheck gemm 32 10";     (cd "$ROOT/4-gemm" && "$SAN" --tool synccheck ./gemm 32 10                     | tail -3)
  echo "== racecheck gemm_opt 30";    (cd "$ROOT/5-extras" && "$SAN" --tool racecheck ./gemm_opt 30 "$OUT/sanitizer_tmp.csv" | tail -3)
} 2>&1 | tee "$OUT/sanitizer.txt"
rm -f "$OUT/sanitizer_tmp.csv"

# 4 · studies
( cd "$ROOT/5-extras" && ./gemm_opt 1000 "$OUT/gemm_sweep.csv" ) 2>&1 | tee "$OUT/gemm_sweep.txt"
( cd "$ROOT/5-extras" && ./ger_sweep "$OUT/ger_sweep.csv" )      2>&1 | tee "$OUT/ger_sweep.txt"

# 5 · plots
python3 "$ROOT/5-extras/plot_results.py" "$OUT"
echo "done: $OUT"

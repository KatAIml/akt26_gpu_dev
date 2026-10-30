"""Plots for 5-extras. Usage: python plot_results.py results_dir   (reads gemm_sweep.csv and ger_sweep.csv)"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

d = sys.argv[1] if len(sys.argv) > 1 else "results"
gemm = pd.read_csv(os.path.join(d, "gemm_sweep.csv"))
ger = pd.read_csv(os.path.join(d, "ger_sweep.csv"))
gpu = gemm["gpu"].iloc[0]
labels = {"v0_baseline": "v0 baseline (4-gemm)", "v1_multi": "v1 several matrices / block",
          "v2_regtile": "v2 2x2 register tile", "cublas": "cuBLAS strided batched"}

# 1 · GEMM GFLOP/s vs n
fig, ax = plt.subplots(figsize=(9, 4.8))
for impl, g in gemm.groupby("impl"):
    ax.plot(g["n"], g["gflops"], marker="o", ms=3, lw=2 if impl == "cublas" else 1.5,
            ls="--" if impl == "cublas" else "-", label=labels.get(impl, impl))
ax.set_xlabel("matrix size n"); ax.set_ylabel("GFLOP/s"); ax.grid(alpha=.3); ax.legend()
ax.set_title(f"Batched GEMM, batch = {gemm['batch'].iloc[0]}, FP32 · {gpu}")
fig.tight_layout(); fig.savefig(os.path.join(d, "gemm_gflops.png"), dpi=130)

# 2 · GEMM speed-up of each kernel over cuBLAS
piv = gemm.pivot(index="n", columns="impl", values="time_us")
fig, ax = plt.subplots(figsize=(9, 4.2))
for impl in ["v0_baseline", "v1_multi", "v2_regtile"]:
    ax.plot(piv.index, piv["cublas"] / piv[impl], marker="o", ms=3, label=labels[impl])
ax.axhline(1, color="k", lw=1, ls="--")
ax.set_xlabel("matrix size n"); ax.set_ylabel("speed-up over cuBLAS (×)"); ax.grid(alpha=.3); ax.legend()
ax.set_title(f"Speed-up over cuBLAS (> 1 = faster than cuBLAS) · {gpu}")
fig.tight_layout(); fig.savefig(os.path.join(d, "gemm_speedup.png"), dpi=130)

# 3 · GER bandwidth per block shape
big = ger[(ger["m"] == 4000) & (ger["n"] == 2000)].copy()
big["label"] = big.apply(lambda r: "cuBLAS" if r["impl"] == "cublas" else f"({int(r['block_x'])},{int(r['block_y'])})", axis=1)
fig, ax = plt.subplots(figsize=(9, 4.2))
colors = ["#d62728" if l == "cuBLAS" else ("#7f7f7f" if l.startswith("(1,") else "#1f77b4") for l in big["label"]]
ax.bar(big["label"], big["bandwidth_GBs"], color=colors)
ax.set_ylabel("effective bandwidth (GB/s)"); ax.set_xlabel("thread-block shape (x = rows, y = columns)")
ax.set_title(f"GER 4000 x 2000: block shape vs memory bandwidth · {gpu}")
ax.tick_params(axis="x", rotation=30); ax.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(d, "ger_blocks.png"), dpi=130)
print("plots written to", d)

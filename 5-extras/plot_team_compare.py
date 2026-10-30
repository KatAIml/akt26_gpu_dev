"""Plot for the team comparison. Usage: python plot_team_compare.py results_dir   (reads team_compare.csv)"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

d = sys.argv[1] if len(sys.argv) > 1 else "results"
df = pd.read_csv(os.path.join(d, "team_compare.csv"))
p = df.pivot(index="n", columns="impl", values="time_us")
gpu = df["gpu"].iloc[0]

fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
labels = {"khalil_2d": "Khalil · 2D shared (stride 32)", "achraf_packed": "Achraf · packed shared (stride n)", "cublas": "cuBLAS"}
for impl, g in df.groupby("impl"):
    ax[0].plot(g["n"], g["gflops"], marker="o", ms=3, ls="--" if impl == "cublas" else "-", label=labels[impl])
ax[0].set_xlabel("matrix size n"); ax[0].set_ylabel("GFLOP/s"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[0].set_title(f"Our two 4-gemm solutions · {gpu}")
ratio = p["khalil_2d"] / p["achraf_packed"]
ax[1].bar(ratio.index, ratio.values, color=["#2ca02c" if v > 1 else "#d62728" for v in ratio.values])
ax[1].axhline(1, color="k", lw=1)
ax[1].set_xlabel("matrix size n"); ax[1].set_ylabel("time(2D) / time(packed)")
ax[1].set_title("> 1: packed layout faster (green) · < 1: 2D faster (red)"); ax[1].grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(d, "team_compare.png"), dpi=130)
print("plot written to", os.path.join(d, "team_compare.png"))

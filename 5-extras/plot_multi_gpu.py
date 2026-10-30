"""Plot for the multi-GPU study. Usage: python plot_multi_gpu.py results_dir   (reads multi_gpu.csv)"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

d = sys.argv[1] if len(sys.argv) > 1 else "results"
df = pd.read_csv(os.path.join(d, "multi_gpu.csv"))
gpu, batch, ngmax = df["gpu"].iloc[0], df["batch"].iloc[0], df["ngpu"].max()

fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
modes = {"kernels_only": ("(A) kernels only", "#1f77b4"), "end_to_end": ("(B) end-to-end incl. PCIe copies", "#d62728")}
ns = sorted(df["n"].unique())
w = 0.38
for i, (mode, (label, color)) in enumerate(modes.items()):
    sub = df[(df["mode"] == mode) & (df["ngpu"] == ngmax)].set_index("n").loc[ns]
    xs = [k + (i - 0.5) * w for k in range(len(ns))]
    ax[0].bar(xs, sub["speedup"], w, label=label, color=color)
    for x, v in zip(xs, sub["speedup"]):
        ax[0].text(x, v + 0.03, f"{v:.2f}×", ha="center", fontsize=9)
ax[0].axhline(ngmax, color="k", ls="--", lw=1, label=f"ideal ({ngmax}×)")
ax[0].set_xticks(range(len(ns))); ax[0].set_xticklabels([f"n = {n}" for n in ns])
ax[0].set_ylabel(f"speed-up with {ngmax} GPUs vs 1"); ax[0].legend(fontsize=8)
ax[0].set_title(f"Strong scaling, batch = {batch}")

for mode, (label, color) in modes.items():
    for n, ls in zip(ns, ["-", "--", ":"]):
        sub = df[(df["mode"] == mode) & (df["n"] == n)].sort_values("ngpu")
        ax[1].plot(sub["ngpu"], sub["gflops"], marker="o", color=color, ls=ls, label=f"{label.split(' ')[0]} n = {n}")
ax[1].set_xlabel("number of GPUs"); ax[1].set_ylabel("GFLOP/s"); ax[1].set_xticks(range(1, ngmax + 1))
ax[1].set_yscale("log"); ax[1].legend(fontsize=7, ncol=2); ax[1].grid(alpha=.3)
ax[1].set_title(f"Throughput · {gpu} x{ngmax}")
fig.tight_layout(); fig.savefig(os.path.join(d, "multi_gpu.png"), dpi=130)
print("plot written to", os.path.join(d, "multi_gpu.png"))

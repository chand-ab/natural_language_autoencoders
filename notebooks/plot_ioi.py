"""Replot a finished IOI shot sweep from its CSV.

Standalone on purpose: importing ioi_intervention would load Gemma-3-27B,
launch SGLang, and take three GPUs just to draw a line chart. Panel format
matches executor.py's plot_sweep (CONDITIONS table kept in sync by hand).

    python plot_ioi.py 20260806_050948
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT_DIR = Path("/mnt/ssd-1/soar-nla/abhinav")

# Mirror of executor.py:1106 -- (column, marker, legend label).
CONDITIONS = [
    ("acc_baseline", "o", "Baseline (no NLA)"),
    ("acc_nla", "^", "NLA round-trip"),
    ("acc_para_medium", "s", "Paraphrase round-trip (medium)"),
    ("acc_para_heavy", "d", "Paraphrase round-trip (heavy)"),
]

if len(sys.argv) != 2:
    sys.exit(f"usage: {sys.argv[0]} <run_id>   e.g. 20260806_050948")
run_id = sys.argv[1]

csv_path = OUT_DIR / f"fv_shot_sweep_ioi_{run_id}.csv"
if not csv_path.exists():
    sys.exit(f"no CSV for run_id {run_id!r} at {csv_path}")
df = pd.read_csv(csv_path)

fig, ax = plt.subplots(figsize=(5, 4))
for col, marker, label in CONDITIONS:
    if col not in df.columns:
        print(f"[warn] missing column {col!r}, skipping that line")
        continue
    ax.plot(df["n_shots"], df[col], marker=marker, label=label)
ax.set_xlabel("Number of shots")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1)
ax.set_title("ioi")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()

png_path = OUT_DIR / f"fv_shot_sweep_ioi_{run_id}.png"
fig.savefig(png_path, dpi=150)
print(f"[ok] {csv_path} -> {png_path}")

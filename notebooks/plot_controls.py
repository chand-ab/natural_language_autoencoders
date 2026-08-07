"""Plot the IOI 0-shot control results from their CSV.

Standalone like plot_ioi.py -- importing ioi_controls would load Gemma-3-27B
and take three GPUs just to draw two charts.

    python plot_controls.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT_DIR = Path("/mnt/ssd-1/soar-nla/abhinav")

# Spelled out on both figures so they stand alone outside this conversation.
CAPTION = ("AR(explanation) = reconstructed activation;   "
           "$h_{orig}$ = original activation")
csv_path = OUT_DIR / "ioi_controls_0shot.csv"
df = pd.read_csv(csv_path)

named = df[df["condition"] != "alpha"]
acc = named.groupby("condition")["correct"].mean()
alpha = df[df["condition"] == "alpha"].groupby("alpha")["correct"].mean()
n = df["sample_idx"].nunique()

# --- 1. accuracy by condition -------------------------------------------
order = ["baseline", "nla", "mismatch", "random"]
# Wrapped onto two lines so the long names don't collide at this figure width.
LABELS = {
    "baseline": "Baseline\n(no NLA)",
    "nla": "NLA\nround-trip",
    "mismatch": "mismatch AR(explanation)\nround-trip",
    "random": "random vector\nof norm $\\|h_{orig}\\|$",
}
fig, ax = plt.subplots(figsize=(6.5, 4))
bars = ax.bar([LABELS[c] for c in order], [acc[c] for c in order], color="tab:blue")
for b, c in zip(bars, order):
    ax.text(b.get_x() + b.get_width() / 2, acc[c] + 0.02, f"{acc[c]:.2f}",
            ha="center", fontsize=9)
ax.tick_params(axis="x", labelsize=9)
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1)
ax.set_title(f"IOI 0-shot, patch conditions (n={n})")
ax.grid(alpha=0.3, axis="y")
fig.text(0.5, 0.015, CAPTION, ha="center", fontsize=8, color="dimgray")
fig.tight_layout(rect=(0, 0.05, 1, 1))
p1 = OUT_DIR / "ioi_controls_conditions.png"
fig.savefig(p1, dpi=150)
print(f"[ok] {p1}")

# --- 2. magnitude sweep --------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(alpha.index, alpha.values, marker="o", label="Patch = direction of AR(expl)")
ax.axhline(acc["baseline"], ls="--", color="gray", label="Baseline (no NLA)")
ax.axhline(acc["random"], ls=":", color="gray",
           label="random vector of norm $\\|h_{orig}\\|$")
ax.set_xlabel(r"$\alpha$  (patch norm as multiple of $\|h_{orig}\|$)")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1)
ax.set_title(f"Magnitude sweep, direction held (n={n})")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.text(0.5, 0.015, CAPTION, ha="center", fontsize=8, color="dimgray")
fig.tight_layout(rect=(0, 0.05, 1, 1))
p2 = OUT_DIR / "ioi_controls_alpha.png"
fig.savefig(p2, dpi=150)
print(f"[ok] {p2}")

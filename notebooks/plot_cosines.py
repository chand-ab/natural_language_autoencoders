"""Cosine-similarity distributions from the dumped control vectors.

Offline: reads the .npz, needs no GPU and no models.

    python plot_cosines.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("/mnt/ssd-1/soar-nla/abhinav")
d = np.load(OUT_DIR / "ioi_controls_0shot_vectors.npz")
h_orig, h_hat = d["h_orig"], d["h_hat"]
n = len(h_orig)


def unit(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def offdiag(a):
    C = unit(a) @ unit(a).T
    return C[~np.eye(len(a), dtype=bool)]


cross_hat = offdiag(h_hat)
cross_orig = offdiag(h_orig)
own = (unit(h_hat) * unit(h_orig)).sum(-1)

fig, ax = plt.subplots(figsize=(6.5, 4))
bins = np.linspace(0.94, 1.0, 60)
# density=True: the cross sets have 2450 pairs each, the own set only 50 --
# raw counts would flatten the own-pair curve into the axis.
ax.hist(cross_orig, bins=bins, alpha=0.55, density=True,
        label=f"$h_{{orig,i}}$ vs $h_{{orig,j}}$  (mean {cross_orig.mean():.4f})")
ax.hist(cross_hat, bins=bins, alpha=0.55, density=True,
        label=f"AR$_i$ vs AR$_j$  (mean {cross_hat.mean():.4f})")
ax.hist(own, bins=bins, alpha=0.75, density=True,
        label=f"AR$_i$ vs its own $h_{{orig,i}}$  (mean {own.mean():.4f})")
ax.set_xlabel("Cosine similarity")
ax.set_ylabel("Density")
ax.set_title(f"Reconstructions cluster tighter than their targets (n={n})")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.text(0.5, 0.015,
         "AR$_i$ = AR(explanation$_i$) = reconstructed activation;   "
         "$h_{orig}$ = original activation",
         ha="center", fontsize=8, color="dimgray")
fig.tight_layout(rect=(0, 0.05, 1, 1))
p = OUT_DIR / "ioi_controls_cosines.png"
fig.savefig(p, dpi=150)
print(f"[ok] {p}")
print(f"  cross AR   mean={cross_hat.mean():.4f} min={cross_hat.min():.4f}")
print(f"  cross orig mean={cross_orig.mean():.4f} min={cross_orig.min():.4f}")
print(f"  own pair   mean={own.mean():.4f} min={own.min():.4f}")

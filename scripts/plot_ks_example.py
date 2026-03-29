"""
Illustrative figure for the KS statistic.

Produces two side-by-side panels showing empirical CDFs:
  Left  — real brain and a synthetic network that closely matches (low KS)
  Right — same real brain and a synthetic network that looks very different (high KS)

The real brain distribution is identical in both panels.
Data is made up purely for illustration.
Run from the project root:
    python scripts/plot_ks_example.py

Output:
    images/GenerativeModels/ks_example.png
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

def ecdf(data):
    """Return (x, y) arrays for the empirical CDF as a step function."""
    x = np.sort(data)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y

# ── Shared real brain distribution ──
real = np.random.normal(loc=18, scale=4, size=2000).clip(2, 40)

# ── Low KS: synthetic that closely matches real ──
synth_low  = np.random.normal(loc=17, scale=4.5, size=2000).clip(2, 40)

# ── High KS: synthetic that is very different from real ──
synth_high = np.random.normal(loc=28, scale=3, size=2000).clip(2, 40)

ks_low,  _ = stats.ks_2samp(real, synth_low)
ks_high, _ = stats.ks_2samp(real, synth_high)

BLUE  = "#2E86AB"   # real brain
CORAL = "#E84855"   # synthetic network

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

for ax, synth, ks_val, title in zip(
    axes,
    [synth_low,  synth_high],
    [ks_low,     ks_high],
    ["Low KS",   "High KS"],
):
    xr, yr = ecdf(real)
    xs, ys = ecdf(synth)

    ax.step(xr, yr, where="post", color=BLUE,  lw=2, label="Real brain")
    ax.step(xs, ys, where="post", color=CORAL, lw=2, label="Synthetic network")

    ax.set_xlabel("Degree", fontsize=13)
    ax.set_ylabel("Cumulative probability", fontsize=13)
    ax.set_title(f"{title}   (KS = {ks_val:.2f})", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()

out = "./images/GenerativeModels/ks_example.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
plt.show()

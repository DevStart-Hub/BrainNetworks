"""
Plots from the coarse sweep.

1. Group-level energy landscape (energy_landscape.png)
2. Individual optima overlaid on the coarse landscape (individual_landscape_coarse.png)

Run from the project root:
    python scripts/plot_coarse.py

Requires:
    resources/GenerativeModels/experiments.pkl

Outputs:
    images/GenerativeModels/energy_landscape.png
    images/GenerativeModels/individual_landscape_coarse.png
"""

import pickle
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# ============================================================
# 1. Load coarse experiments
# ============================================================
with open("./resources/GenerativeModels/experiments.pkl", "rb") as f:
    experiments = pickle.load(f)

print(f"Loaded {len(experiments)} coarse experiments")

# ============================================================
# 2. Build landscape
# ============================================================
mean_energies = [float(exp['energy_tensor'].mean()) for exp in experiments]

rows = [
    {"eta": exp['eta'], "gamma": exp['gamma'], "energy": e}
    for exp, e in zip(experiments, mean_energies)
]
df        = pd.DataFrame(rows)
landscape = df.pivot(index="eta", columns="gamma", values="energy")

best_idx    = int(np.argmin(mean_energies))
best_eta    = experiments[best_idx]['eta']
best_gamma  = experiments[best_idx]['gamma']
best_energy = mean_energies[best_idx]
print(f"Best energy: {best_energy:.3f}  eta={best_eta:.2f}  gamma={best_gamma:.2f}")

# ============================================================
# 3. Group-level energy landscape
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(
    landscape.values,
    origin="lower", aspect="auto", cmap="viridis_r",
    extent=[
        landscape.columns.min(), landscape.columns.max(),
        landscape.index.min(),   landscape.index.max(),
    ],
)
ax.scatter(best_gamma, best_eta, color="red", s=120, zorder=5, label="Best Fit")
plt.colorbar(im, ax=ax, label="Energy (max KS)")
ax.set_xlabel("gamma (homophily)", fontsize=12)
ax.set_ylabel("eta (distance penalty)", fontsize=12)
ax.set_title("Energy landscape", fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("./images/GenerativeModels/energy_landscape.png", dpi=150, bbox_inches="tight")
print("Saved energy_landscape.png")
plt.show()

# ============================================================
# 4. Per-brain coarse optima
# ============================================================
per_brain_energy      = torch.stack(
    [exp['energy_tensor'].mean(dim=0) for exp in experiments]
)  # [n_configs, n_brains]
best_config_per_brain = per_brain_energy.argmin(dim=0)  # [n_brains]

ind_eta_c   = [experiments[int(idx)]['eta']   for idx in best_config_per_brain]
ind_gamma_c = [experiments[int(idx)]['gamma'] for idx in best_config_per_brain]

counts = Counter(zip(ind_eta_c, ind_gamma_c))
print(f"Unique coarse optima: {len(counts)} out of {len(best_config_per_brain)} participants")

# ============================================================
# 5. Plot individual optima on coarse landscape
# X markers in red; count labels inline to the right of X; no group dot
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(
    landscape.values,
    origin="lower", aspect="auto", cmap="viridis_r",
    extent=[
        landscape.columns.min(), landscape.columns.max(),
        landscape.index.min(),   landscape.index.max(),
    ],
)
for (eta_val, gamma_val), n in counts.items():
    ax.scatter(gamma_val, eta_val, marker="x", s=140, color="red",
               linewidths=2.5, zorder=5)
    if n > 1:
        ax.text(gamma_val + 0.02, eta_val, str(n),
                color="red", fontsize=9, fontweight="bold",
                va="center", zorder=6)
plt.colorbar(im, ax=ax, label="Energy (max KS)")
ax.set_xlabel("gamma (homophily)", fontsize=12)
ax.set_ylabel("eta (distance penalty)", fontsize=12)
ax.set_title("Individual optima on coarse grid", fontsize=13)
plt.tight_layout()
plt.savefig("./images/GenerativeModels/individual_landscape_coarse.png",
            dpi=150, bbox_inches="tight")
print("Saved individual_landscape_coarse.png")
plt.show()

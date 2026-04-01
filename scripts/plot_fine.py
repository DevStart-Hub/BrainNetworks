"""
Plots from the fine sweep.

1. Individual optima on fine landscape (individual_landscape_fine.png)
2. GNM parameters vs early life stress (individual_correlations.png)

Run from the project root:
    python scripts/plot_fine.py

Requires:
    resources/GenerativeModels/individual_experiments.pkl
    resources/BrainNetworks/DATA/stress_scores.npy

Outputs:
    images/GenerativeModels/individual_landscape_fine.png
    images/GenerativeModels/individual_correlations.png
"""

import pickle
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy import stats

# ============================================================
# 1. Load fine experiments and stress scores
# ============================================================
with open("./resources/GenerativeModels/individual_experiments.pkl", "rb") as f:
    experiments = pickle.load(f)

stress_scores = np.load("./resources/BrainNetworks/DATA/stress_scores.npy")
num_brains    = len(stress_scores)

print(f"Loaded {len(experiments)} fine experiments, {num_brains} participants")

# ============================================================
# 2. Per-brain fine optima
# ============================================================
per_brain_energy = torch.stack(
    [exp['energy_tensor'].mean(dim=0) for exp in experiments]
)  # [n_configs, n_brains]
best_config_fine = per_brain_energy.argmin(dim=0)  # [n_brains]

ind_eta_f   = [experiments[int(idx)]['eta']   for idx in best_config_fine]
ind_gamma_f = [experiments[int(idx)]['gamma'] for idx in best_config_fine]

counts_fine = Counter(zip(ind_eta_f, ind_gamma_f))
print(f"Unique fine optima: {len(counts_fine)} out of {num_brains}")

# ============================================================
# 3. Build fine landscape
# ============================================================
mean_energies_fine = [float(exp['energy_tensor'].mean()) for exp in experiments]
rows_fine = [
    {"eta": exp['eta'], "gamma": exp['gamma'], "energy": e}
    for exp, e in zip(experiments, mean_energies_fine)
]
df_fine        = pd.DataFrame(rows_fine)
landscape_fine = df_fine.pivot(index="eta", columns="gamma", values="energy")

# ============================================================
# 4. Plot individual optima on fine landscape
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(
    landscape_fine.values,
    origin="lower", aspect="auto", cmap="viridis_r",
    extent=[
        landscape_fine.columns.min(), landscape_fine.columns.max(),
        landscape_fine.index.min(),   landscape_fine.index.max(),
    ],
)
for (eta_val, gamma_val), n in counts_fine.items():
    ax.scatter(gamma_val, eta_val, marker="x", s=140, color="red",
               linewidths=2.5, zorder=5)
    if n > 1:
        ax.text(gamma_val + 0.006, eta_val, str(n),
                color="red", fontsize=9, fontweight="bold",
                va="center", zorder=6)
plt.colorbar(im, ax=ax, label="Energy (max KS)")
ax.set_xlabel("gamma (homophily)", fontsize=12)
ax.set_ylabel("eta (distance penalty)", fontsize=12)
ax.set_title("Individual optima on fine grid", fontsize=13)
plt.tight_layout()
plt.savefig("./images/GenerativeModels/individual_landscape_fine.png",
            dpi=150, bbox_inches="tight")
print("Saved individual_landscape_fine.png")
plt.show()

# ============================================================
# 5. Build individual dataframe
# ============================================================
rows_ind = []
for brain_idx, idx in enumerate(best_config_fine):
    best_exp = experiments[int(idx)]
    rows_ind.append({
        "brain":       brain_idx,
        "eta":         best_exp['eta'],
        "gamma":       best_exp['gamma'],
        "best_energy": float(best_exp['energy_tensor'].mean(dim=0)[brain_idx]),
        "stress":      float(stress_scores[brain_idx]),
    })
df_ind = pd.DataFrame(rows_ind)
print(df_ind.to_string(index=False))

# ============================================================
# 6. Correlate with early life stress
# ============================================================
r_eta,   p_eta   = stats.pearsonr(df_ind["stress"], df_ind["eta"])
r_gamma, p_gamma = stats.pearsonr(df_ind["stress"], df_ind["gamma"])

print(f"\neta   vs stress: r = {r_eta:+.2f},  p = {p_eta:.3f}")
print(f"gamma vs stress: r = {r_gamma:+.2f}, p = {p_gamma:.3f}")

COLORS = ["#C0392B", "#2980B9"]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, param, r_val, p_val, ylabel, color in zip(
    axes,
    ["eta",                  "gamma"],
    [r_eta,                  r_gamma],
    [p_eta,                  p_gamma],
    ["η (distance penalty)", "γ (homophily)"],
    COLORS,
):
    sns.regplot(
        x=df_ind["stress"], y=df_ind[param],
        ax=ax,
        scatter_kws={"color": color, "edgecolors": "white",
                     "linewidths": 0.6, "s": 60, "zorder": 3},
        line_kws={"color": color, "linewidth": 2},
        ci=95,
        color=color,
    )
    p_str = f"p = {p_val:.3f}" if p_val >= 0.001 else "p < 0.001"
    ax.set_title(f"r = {r_val:+.2f},  {p_str}", fontsize=12)
    ax.set_xlabel("Early life stress score", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("GNM Parameters vs. Early Life Stress", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("./images/GenerativeModels/individual_correlations.png",
            dpi=150, bbox_inches="tight")
print("Saved individual_correlations.png")
plt.show()

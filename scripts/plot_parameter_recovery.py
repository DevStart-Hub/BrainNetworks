"""
Parameter recovery check (internal diagnostic, not part of any tutorial).

Compares the ground-truth eta and gamma used to generate each of the 20
synthetic brains (generation_params.npy) against the individual parameters
recovered from the fine-grained GNM sweep (individual_experiments.pkl).

Run from the project root:
    python scripts/plot_parameter_recovery.py

Requires:
    resources/BrainNetworks/DATA/generation_params.npy
    resources/GenerativeModels/individual_experiments.pkl

Outputs:
    images/GenerativeModels/parameter_recovery.png
"""

import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================
# 1. Load ground truth parameters
# generation_params.npy has shape [n_brains, 2]: columns are [eta, gamma]
# ============================================================
gen_params  = np.load("./resources/BrainNetworks/DATA/generation_params.npy")
true_eta    = gen_params[:, 0]
true_gamma  = gen_params[:, 1]
num_brains  = len(true_eta)

print(f"Ground truth eta:   {true_eta.min():.2f} to {true_eta.max():.2f}")
print(f"Ground truth gamma: {true_gamma.min():.2f} to {true_gamma.max():.2f}")

# ============================================================
# 2. Load recovered parameters from fine sweep
# ============================================================
with open("./resources/GenerativeModels/individual_experiments.pkl", "rb") as f:
    experiments = pickle.load(f)

per_brain_energy = torch.stack(
    [exp['energy_tensor'].mean(dim=0) for exp in experiments]
)  # [n_configs, n_brains]
best_config_fine = per_brain_energy.argmin(dim=0)

rec_eta   = np.array([experiments[int(idx)]['eta']   for idx in best_config_fine])
rec_gamma = np.array([experiments[int(idx)]['gamma'] for idx in best_config_fine])

print(f"Recovered eta:   {rec_eta.min():.2f} to {rec_eta.max():.2f}")
print(f"Recovered gamma: {rec_gamma.min():.2f} to {rec_gamma.max():.2f}")

# ============================================================
# 3. Correlate ground truth vs recovered
# ============================================================
r_eta,   p_eta   = stats.pearsonr(true_eta,   rec_eta)
r_gamma, p_gamma = stats.pearsonr(true_gamma, rec_gamma)

print(f"\nParameter recovery:")
print(f"  eta:   r = {r_eta:+.3f},  p = {p_eta:.3f}")
print(f"  gamma: r = {r_gamma:+.3f}, p = {p_gamma:.3f}")

# ============================================================
# 4. Plot
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, true_vals, rec_vals, r_val, p_val, label, color in zip(
    axes,
    [true_eta,   true_gamma],
    [rec_eta,    rec_gamma],
    [r_eta,      r_gamma],
    [p_eta,      p_gamma],
    ["η (distance penalty)", "γ (homophily)"],
    ["#C0392B", "#2980B9"],
):
    ax.scatter(true_vals, rec_vals, color=color, edgecolors="white",
               linewidths=0.6, s=70, zorder=3)

    # Identity line over the full range
    all_vals = np.concatenate([true_vals, rec_vals])
    lims = [all_vals.min() - 0.05 * all_vals.ptp(),
            all_vals.max() + 0.05 * all_vals.ptp()]
    ax.plot(lims, lims, "k--", linewidth=1, zorder=2, label="Identity")

    p_str = f"p = {p_val:.3f}" if p_val >= 0.001 else "p < 0.001"
    ax.set_title(f"r = {r_val:+.2f},  {p_str}", fontsize=12)
    ax.set_xlabel(f"True {label}", fontsize=12)
    ax.set_ylabel(f"Recovered {label}", fontsize=12)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Parameter Recovery: Ground Truth vs Recovered", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("./images/GenerativeModels/parameter_recovery.png",
            dpi=150, bbox_inches="tight")
print("Saved parameter_recovery.png")
plt.show()

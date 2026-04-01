"""
All correlations with early life stress, in three sections:

  1. Ground-truth GNM parameters (eta, gamma used to generate each brain)
  2. Binary topology metrics (degree, betweenness, clustering, edge length)
  3. Recovered GNM parameters from the fine-grained sweep

Stress scores are created and saved here.

Run from the project root:
    python scripts/correlations.py

Requires:
    resources/BrainNetworks/DATA/brain_networks_20_preprocessed.npy
    resources/BrainNetworks/DATA/distance_matrix.npy
    resources/BrainNetworks/DATA/generation_params.npy
    resources/GenerativeModels/individual_experiments.pkl

Outputs:
    resources/BrainNetworks/DATA/stress_scores.npy
    images/GenerativeModels/correlations_ground_truth.png
    images/BrainNetworks/topology_descriptive.png
    images/BrainNetworks/topology_stress_correlations.png
    images/GenerativeModels/individual_correlations.png
"""

import pickle
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy import stats
from scipy.stats import gaussian_kde

# ============================================================
# 1. Generate and save stress scores
# Higher brain index = weaker constraints (less negative eta, lower gamma)
# = more stochastic developmental process = higher stress
# ============================================================
gen_params  = np.load("./resources/BrainNetworks/DATA/generation_params.npy")
true_eta    = gen_params[:, 0]
true_gamma  = gen_params[:, 1]
num_brains  = len(true_eta)

np.random.seed(11)
base_stress   = np.linspace(3, 17, num_brains)
noise         = np.random.normal(0, 4, num_brains)
stress        = np.clip(base_stress + noise, 0, 20)
stress        = np.round(stress, 1)

#np.save("./resources/BrainNetworks/DATA/stress_scores.npy", stress)
#print(f"Stress scores: {stress}")

# ============================================================
# 2. Load preprocessed brains and binarize
# ============================================================
brains_w = np.load("./resources/BrainNetworks/DATA/brain_networks_20_preprocessed.npy")
brains_b = (brains_w > 0).astype(float)   # binarized
D        = np.load("./resources/BrainNetworks/DATA/distance_matrix.npy")

# ============================================================
# 3. Section 1 — Ground-truth GNM parameters vs stress
# ============================================================
print("\nGround-truth GNM parameters vs stress:")
for label, vals in [("eta", true_eta), ("gamma", true_gamma)]:
    r, p = stats.pearsonr(stress, vals)
    print(f"  {label}: r = {r:+.3f},  p = {p:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, vals, ylabel, color in zip(
    axes,
    [true_eta,               true_gamma],
    ["η true (distance penalty)", "γ true (homophily)"],
    ["#C0392B", "#2980B9"],
):
    sns.regplot(x=stress, y=vals, ax=ax,
                scatter_kws={"color": color, "edgecolors": "white",
                             "linewidths": 0.6, "s": 60, "zorder": 3},
                line_kws={"color": color, "linewidth": 2},
                ci=95, color=color)
    r, p = stats.pearsonr(stress, vals)
    p_str = f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"
    ax.set_title(f"r = {r:+.2f},  {p_str}", fontsize=12)
    ax.set_xlabel("Early life stress score", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Ground-Truth GNM Parameters vs. Early Life Stress",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("./images/GenerativeModels/correlations_ground_truth.png",
            dpi=150, bbox_inches="tight")
print("Saved correlations_ground_truth.png")
plt.show()

# ============================================================
# 4. Section 2 — Binary topology metrics vs stress
# ============================================================
results = []
for i in range(num_brains):
    G = nx.from_numpy_array(brains_b[i])
    edges = np.triu(brains_b[i], k=1) > 0
    results.append({
        "degree":      np.mean([d for _, d in G.degree()]),
        "betweenness": np.mean(list(nx.betweenness_centrality(G, normalized=True).values())),
        "clustering":  np.mean(list(nx.clustering(G).values())),
        "edge_length": D[edges].mean(),
    })
df_topo = pd.DataFrame(results)

metrics       = ["degree", "betweenness", "clustering", "edge_length"]
metric_labels = ["Degree", "Betweenness\nCentrality", "Clustering\nCoefficient", "Edge Length (mm)"]
colors        = ["#C0392B", "#2980B9", "#27AE60", "#8E44AD"]

# Descriptive plot
fig, axes = plt.subplots(1, 4, figsize=(13, 4))
rng = np.random.default_rng(seed=0)
for ax, metric, label, color in zip(axes, metrics, metric_labels, colors):
    values   = df_topo[metric].values
    kde      = gaussian_kde(values, bw_method=0.4)
    y_range  = np.linspace(values.min() - 0.05 * values.ptp(),
                           values.max() + 0.05 * values.ptp(), 200)
    kde_vals = kde(y_range)
    kde_vals = kde_vals / kde_vals.max() * 0.4
    ax.fill_betweenx(y_range, 0, kde_vals, color=color, alpha=0.6)
    ax.plot(kde_vals, y_range, color=color, linewidth=1.5)
    jitter = rng.uniform(-0.25, -0.05, len(values))
    ax.scatter(jitter, values, color=color, edgecolors="white",
               linewidths=0.6, s=55, zorder=3, alpha=0.9)
    ax.axhline(values.mean(), color="black", linewidth=1.2,
               linestyle="--", xmin=0, xmax=0.5, zorder=2)
    ax.set_xlim(-0.4, 0.55)
    ax.set_ylabel(label, fontsize=11)
    ax.set_xticks([])
    ax.spines[["top", "right", "bottom"]].set_visible(False)

fig.suptitle("Topological Metrics Across 20 Brains", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("./images/BrainNetworks/topology_descriptive.png", dpi=200, bbox_inches="tight")
print("Saved topology_descriptive.png")
plt.show()

# Correlation plot
print("\nBinary topology metrics vs stress:")
for metric in metrics:
    r, p = stats.pearsonr(df_topo[metric], stress)
    print(f"  {metric:>15s}: r = {r:+.3f},  p = {p:.3f}")

fig, axes = plt.subplots(1, 4, figsize=(13, 4))
for ax, metric, label, color in zip(axes, metrics, metric_labels, colors):
    sns.regplot(x=stress, y=df_topo[metric], ax=ax,
                scatter_kws={"color": color, "edgecolors": "white",
                             "linewidths": 0.6, "s": 60, "zorder": 3},
                line_kws={"color": color, "linewidth": 2},
                ci=95, color=color)
    r, p = stats.pearsonr(df_topo[metric], stress)
    p_str = f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"
    ax.set_title(f"r = {r:+.2f}, {p_str}", fontsize=10)
    ax.set_xlabel("Early Life Stress", fontsize=11)
    ax.set_ylabel(label, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Brain Topology vs. Early Life Stress", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("./images/BrainNetworks/topology_stress_correlations.png",
            dpi=200, bbox_inches="tight")
print("Saved topology_stress_correlations.png")
plt.show()

# ============================================================
# 5. Section 3 — Recovered GNM parameters vs stress
# ============================================================
with open("./resources/GenerativeModels/individual_experiments.pkl", "rb") as f:
    experiments = pickle.load(f)

per_brain_energy = torch.stack(
    [exp['energy_tensor'].mean(dim=0) for exp in experiments]
)
best_config_fine = per_brain_energy.argmin(dim=0)

rows_ind = []
for brain_idx, idx in enumerate(best_config_fine):
    best_exp = experiments[int(idx)]
    rows_ind.append({
        "eta":    best_exp['eta'],
        "gamma":  best_exp['gamma'],
        "stress": float(stress[brain_idx]),
    })
df_ind = pd.DataFrame(rows_ind)

r_eta,   p_eta   = stats.pearsonr(df_ind["stress"], df_ind["eta"])
r_gamma, p_gamma = stats.pearsonr(df_ind["stress"], df_ind["gamma"])

print("\nRecovered GNM parameters vs stress:")
print(f"  eta:   r = {r_eta:+.3f},  p = {p_eta:.3f}")
print(f"  gamma: r = {r_gamma:+.3f}, p = {p_gamma:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, param, r_val, p_val, ylabel, color in zip(
    axes,
    ["eta",                  "gamma"],
    [r_eta,                  r_gamma],
    [p_eta,                  p_gamma],
    ["η (distance penalty)", "γ (homophily)"],
    ["#C0392B", "#2980B9"],
):
    sns.regplot(x=df_ind["stress"], y=df_ind[param], ax=ax,
                scatter_kws={"color": color, "edgecolors": "white",
                             "linewidths": 0.6, "s": 60, "zorder": 3},
                line_kws={"color": color, "linewidth": 2},
                ci=95, color=color)
    p_str = f"p = {p_val:.3f}" if p_val >= 0.001 else "p < 0.001"
    ax.set_title(f"r = {r_val:+.2f},  {p_str}", fontsize=12)
    ax.set_xlabel("Early life stress score", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("./images/GenerativeModels/individual_correlations.png",
            dpi=150, bbox_inches="tight")
print("Saved individual_correlations.png")
plt.show()

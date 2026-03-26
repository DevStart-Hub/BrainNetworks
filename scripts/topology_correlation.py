import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde

# Load the preprocessed brains
brains = np.load("./resources/BrainNetworks/DATA/brain_networks_20_preprocessed.npy")

# Load node coordinates (100 x 3: x, y, z in MNI space)
coords = np.load("./resources/BrainNetworks/DATA/coordinates.npy")

# Load the precomputed pairwise Euclidean distance matrix (100 x 100)
D = np.load("./resources/BrainNetworks/DATA/distance_matrix.npy")
num_brains = brains.shape[0]

results = []

for i in range(num_brains):
    G = nx.from_numpy_array(brains[i])

    mean_deg   = np.mean([d for _, d in G.degree()])
    mean_bc    = np.mean(list(nx.betweenness_centrality(G, normalized=True).values()))
    mean_clust = np.mean(list(nx.clustering(G).values()))
    edges      = np.triu(brains[i], k=1) > 0
    mean_len   = D[edges].mean()

    results.append({
        "brain":       i + 1,
        "degree":      mean_deg,
        "betweenness": mean_bc,
        "clustering":  mean_clust,
        "edge_length": mean_len,
    })

df = pd.DataFrame(results)

# ============================================================
# Descriptive plot: half violins + data points, one per metric
# ============================================================
metrics      = ["degree", "betweenness", "clustering", "edge_length"]
metric_labels = ["Degree", "Betweenness\nCentrality", "Clustering\nCoefficient", "Edge Length (mm)"]
colors       = ["#C0392B", "#2980B9", "#27AE60", "#8E44AD"]

fig, axes = plt.subplots(1, 4, figsize=(13, 4))

rng = np.random.default_rng(seed=0)

for ax, metric, label, color in zip(axes, metrics, metric_labels, colors):
    values = df[metric].values

    # --- half violin (right side only) ---
    kde = gaussian_kde(values, bw_method=0.4)
    y_range = np.linspace(values.min() - 0.05 * values.ptp(),
                          values.max() + 0.05 * values.ptp(), 200)
    kde_vals = kde(y_range)
    kde_vals = kde_vals / kde_vals.max() * 0.4   # scale max width to 0.4 data units

    ax.fill_betweenx(y_range, 0, kde_vals, color=color, alpha=0.6)
    ax.plot(kde_vals, y_range, color=color, linewidth=1.5)

    # --- jittered data points (left of the violin) ---
    jitter = rng.uniform(-0.25, -0.05, len(values))
    ax.scatter(jitter, values, color=color, edgecolors="white",
               linewidths=0.6, s=55, zorder=3, alpha=0.9)

    ax.axhline(values.mean(), color="black", linewidth=1.2,
               linestyle="--", xmin=0, xmax=0.5, zorder=2)

    ax.set_xlim(-0.4, 0.55)
    ax.set_xlabel("")
    ax.set_ylabel(label, fontsize=11)
    ax.set_xticks([])
    ax.spines[["top", "right", "bottom"]].set_visible(False)

fig.suptitle("Topological Metrics Across 20 Brains", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("./images/BrainNetworks/topology_descriptive.png", dpi=200, bbox_inches="tight")
plt.show()

# ============================================================
# Correlations with early life stress
# ============================================================
stress = np.load("./resources/BrainNetworks/DATA/stress_scores.npy")

print("Correlation with early life stress:\n")
for metric in metrics:
    r, p = stats.pearsonr(df[metric], stress)
    print(f"  {metric:>15s}: r = {r:+.3f},  p = {p:.3f}")

# ============================================================
# Regression plots: each metric vs stress
# ============================================================
fig, axes = plt.subplots(1, 4, figsize=(13, 4))

for ax, metric, label, color in zip(axes, metrics, metric_labels, colors):
    sns.regplot(
        x=stress, y=df[metric],
        ax=ax,
        scatter_kws={"color": color, "edgecolors": "white",
                     "linewidths": 0.6, "s": 60, "zorder": 3},
        line_kws={"color": color, "linewidth": 2},
        ci=95,
        color=color,
    )

    r, p = stats.pearsonr(df[metric], stress)
    p_str = f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"
    ax.set_title(f"r = {r:+.2f}, {p_str}", fontsize=10)
    ax.set_xlabel("Early Life Stress", fontsize=11)
    ax.set_ylabel(label, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Brain Topology vs. Early Life Stress", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("./images/BrainNetworks/topology_stress_correlations.png", dpi=200, bbox_inches="tight")
plt.show()

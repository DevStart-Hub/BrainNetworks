"""
Generate 3 example brain networks illustrating the role of eta and gamma,
and save them as a single figure: images/GenerativeModels/3_example_brains.png

The three networks show:
  Left   — eta very negative (-4.0), gamma = 0.0  →  strongly local, distance dominates
  Centre — eta negative (-2.0),      gamma = 0.4  →  balanced (realistic brains live here)
  Right  — eta near zero  (-0.3),    gamma = 0.8  →  weak distance penalty, homophily dominates
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial.distance import pdist, squareform

from gnm import BinaryGenerativeParameters, GenerativeNetworkModel
from gnm.generative_rules import MatchingIndex

# ============================================================
# 1. Load coordinates and build distance matrix
# ============================================================
coords = np.load('./resources/BrainNetworks/DATA/coordinates.npy')
num_nodes = coords.shape[0]
print(f"Loaded {num_nodes} node coordinates.")

dist_np = squareform(pdist(coords, metric='euclidean'))
distance_matrix = torch.tensor(dist_np, dtype=torch.float32)

# ============================================================
# 2. Define the three parameter sets
# ============================================================
parameter_sets = [
    {"eta": -4.0, "gamma": 0.0, "label": "η = −4,  γ = 0"},
    {"eta": -2.0, "gamma": 0.4, "label": "η = −2.2,  γ = 0.2"},
    {"eta": -0.3, "gamma": 0.8, "label": "η = −0.1,  γ = 0.8"},
]

# Target: ~8% density — enough edges to see structure without saturating the matrix
num_connections = int(num_nodes * (num_nodes - 1) / 2 * 0.08)
print(f"Target connections per network: {num_connections}")

# ============================================================
# 3. Generate each network
# ============================================================
networks = []

for ps in parameter_sets:
    print(f"\nGenerating: {ps['label']}")

    binary_params = BinaryGenerativeParameters(
        eta=ps["eta"],
        gamma=ps["gamma"],
        lambdah=0.0,
        distance_relationship_type='powerlaw',
        preferential_relationship_type='powerlaw',
        heterochronicity_relationship_type='powerlaw',
        generative_rule=MatchingIndex(),
        num_iterations=num_connections,
    )

    model = GenerativeNetworkModel(
        binary_parameters=binary_params,
        num_simulations=1,
        distance_matrix=distance_matrix,
    )

    model.run_model()
    adj = model.adjacency_matrix.squeeze(0).cpu().numpy()
    networks.append(adj)
    print(f"  -> {int((adj > 0).sum() / 2)} edges")

# ============================================================
# 4. Plot
# ============================================================
TEAL   = "#5e0505"
DARK   = "#2b2b2b"

fig = plt.figure(figsize=(13, 5.5), facecolor="white")

# Outer gridspec: 3 columns for the three networks
gs = gridspec.GridSpec(
    1, 3,
    figure=fig,
    wspace=0.08,
    left=0.04, right=0.96,
    top=0.82, bottom=0.05,
)

for col, (ps, adj) in enumerate(zip(parameter_sets, networks)):
    ax = fig.add_subplot(gs[col])

    # Show binary matrix (0/1) — sort nodes by hemisphere (left half vs right half)
    # so the block structure is easier to read
    sorted_idx = np.argsort(coords[:, 0])   # sort by x-coordinate (L → R)
    adj_sorted = adj[np.ix_(sorted_idx, sorted_idx)]

    ax.imshow(
        adj_sorted,
        cmap="Reds",
        interpolation="nearest",
        aspect="equal",
        vmin=0, vmax=1,
    )

    # Parameter label above the matrix
    ax.set_title(ps["label"], fontsize=13, fontweight="bold", color=DARK, pad=8)
    ax.set_xticks([])
    ax.set_yticks([])

    # Thin coloured border to visually separate the panels
    for spine in ax.spines.values():
        spine.set_edgecolor(TEAL)
        spine.set_linewidth(1.5)

# Overall title
fig.suptitle(
    "Three GNM networks",
    fontsize=14, fontweight="bold", color=DARK, y=0.95,
)

# ============================================================
# 5. Save
# ============================================================
out_path = "./images/GenerativeModels/3_example_brains.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"\nSaved figure to {out_path}")
plt.show()

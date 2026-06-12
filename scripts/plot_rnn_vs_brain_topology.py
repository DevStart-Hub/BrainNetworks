"""
Compare the trained (basic) RNN's W_rec to the 20 brains in WEIGHTED topology
space: weighted modularity (segregation) vs weighted global efficiency (integration).

Both are put on the same footing: positive, symmetric, no diagonal, thresholded to
the brains' density, and normalised to [0, 1] (so weighted efficiency is comparable).

Saves images/RecurrentNetworks/rnn_vs_brain_topology.png and prints the numbers.
"""

import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Weighted topology metrics
# ------------------------------------------------------------------
def weighted_modularity(W):
    G = nx.from_numpy_array(W)
    comm = nx.community.louvain_communities(G, weight="weight", seed=42)
    return nx.community.modularity(G, comm, weight="weight")

def weighted_efficiency(W):
    # distance = 1 / strength; efficiency = mean over pairs of 1 / shortest weighted path
    G = nx.from_numpy_array(W)
    for _, _, d in G.edges(data=True):
        w = d["weight"]
        d["dist"] = 1.0 / w if w > 0 else np.inf
    n = G.number_of_nodes()
    tot = 0.0
    for src, lengths in nx.all_pairs_dijkstra_path_length(G, weight="dist"):
        for tgt, dl in lengths.items():
            if src != tgt and dl > 0 and np.isfinite(dl):
                tot += 1.0 / dl
    return tot / (n * (n - 1))

def prep(W, density=None):
    """positive, symmetric, no diagonal, (optionally) thresholded, normalised to [0,1]."""
    W = np.abs(np.asarray(W, dtype=float))
    W = (W + W.T) / 2.0
    np.fill_diagonal(W, 0.0)
    if density is not None:
        n = W.shape[0]
        triu = W[np.triu_indices(n, 1)]
        k = int(len(triu) * density)
        thr = np.sort(triu)[::-1][k]
        W = np.where(W >= thr, W, 0.0)
    mx = W.max()
    if mx > 0:
        W = W / mx
    return W

# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
brains = np.load("./resources/BrainNetworks/DATA/brain_networks_20_preprocessed.npy")
W_rec = torch.load("./resources/RecurrentNetworks/trained_rnn.pt")["W_rec.weight"].numpy()

# brains are already thresholded to a common density; measure it
b0 = brains[0]
dens = np.count_nonzero(np.triu(b0, 1)) / (100 * 99 / 2)
print(f"brain density = {dens:.3f}")

brain_mod, brain_eff = [], []
for i in range(brains.shape[0]):
    Wb = prep(brains[i])               # already at common density
    brain_mod.append(weighted_modularity(Wb))
    brain_eff.append(weighted_efficiency(Wb))

Wr = prep(W_rec, density=dens)         # trim RNN to the brains' density
rnn_mod = weighted_modularity(Wr)
rnn_eff = weighted_efficiency(Wr)

print(f"BRAINS  modularity {np.mean(brain_mod):.3f}±{np.std(brain_mod):.3f}   "
      f"efficiency {np.mean(brain_eff):.3f}±{np.std(brain_eff):.3f}")
print(f"RNN     modularity {rnn_mod:.3f}              efficiency {rnn_eff:.3f}")

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(brain_mod, brain_eff, s=70, color="#C0392B", alpha=0.8,
           edgecolor="white", linewidth=0.5, label="Brains (n=20)")
ax.scatter([rnn_mod], [rnn_eff], s=320, color="#1E3A8A", marker="*",
           edgecolor="white", linewidth=1.0, zorder=5, label="Basic RNN")
ax.set_xlabel("Weighted modularity  (segregation)", fontsize=12)
ax.set_ylabel("Weighted efficiency  (integration)", fontsize=12)
ax.set_title("Topology: the basic RNN vs real brains", fontsize=13, fontweight="bold")
ax.legend(fontsize=11, loc="best")
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("./images/RecurrentNetworks/rnn_vs_brain_topology.png", dpi=200, bbox_inches="tight")
print("Saved rnn_vs_brain_topology.png")

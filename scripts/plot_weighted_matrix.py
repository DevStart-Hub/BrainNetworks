"""
Make the weighted demo connectivity matrix (for Introduction + Installation):
  - saves resources/BrainNetworks/DATA/weighted_network.npy  (a single weighted brain)
  - saves images/BrainNetworks/weighted_matrix_heatmap.png
"""

import numpy as np
import matplotlib.pyplot as plt

brains = np.load('./resources/BrainNetworks/DATA/brain_networks_20.npy')
adj = brains[0]   # one weighted brain (streamline counts)

np.save('./resources/BrainNetworks/DATA/weighted_network.npy', adj)
print("Saved weighted_network.npy", adj.shape, "max", adj.max())

fig, ax = plt.subplots(figsize=(8, 7))
vmax = np.percentile(adj[adj > 0], 99)
im = ax.imshow(adj, cmap='Reds', interpolation='nearest', aspect='equal', vmax=vmax)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Connection strength (streamline count)', fontsize=12)

ax.set_xlabel('Brain Region', fontsize=13)
ax.set_ylabel('Brain Region', fontsize=13)
ax.set_title('Connectivity Matrix', fontsize=15, fontweight='bold')

plt.tight_layout()
plt.savefig('./images/BrainNetworks/weighted_matrix_heatmap.png', dpi=200, bbox_inches='tight')
print("Saved weighted_matrix_heatmap.png")

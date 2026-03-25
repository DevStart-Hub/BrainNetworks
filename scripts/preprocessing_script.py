"""
Preprocessing script: density check, thresholding, and connectivity check.
Mirrors the code from Preprocessing.qmd for testing.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from netneurotools.networks import threshold_network

# ============================================================
# 1. Load the brain networks
# ============================================================
brains = np.load('./resources/BrainNetworks/DATA/brain_networks_20.npy')
num_brains = brains.shape[0]
num_nodes = brains.shape[1]
print(f"Loaded {num_brains} brains with {num_nodes} nodes each.")

# ============================================================
# 2. Compute density for each brain
# ============================================================
densities = []

for i in range(num_brains):
    G = nx.from_numpy_array(brains[i])
    d = nx.density(G)
    densities.append(d)
    print(f"Brain {i+1:2d}: density = {d:.4f}")

print(f"\nRange: {min(densities):.4f} to {max(densities):.4f}")

# ============================================================
# 3. Plot density histogram
# ============================================================
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(densities, bins=10, color='firebrick', edgecolor='black', alpha=0.8)
ax.set_xlabel('Density', fontsize=13)
ax.set_ylabel('Number of brains', fontsize=13)
ax.set_title('Density Distribution Across 20 Brains', fontsize=14, fontweight='bold')
ax.axvline(min(densities), color='black', linestyle='--', linewidth=1.5, label=f'Min = {min(densities):.4f}')
ax.legend()
plt.tight_layout()
plt.savefig('./images/BrainNetworks/density_histogram.png', dpi=200, bbox_inches='tight')
print("Saved density histogram to images/BrainNetworks/density_histogram.png")
plt.show()

# ============================================================
# 4. Threshold all brains to the minimum density
# ============================================================
target_density = min(densities)
target_retain = target_density * 100  # threshold_network expects a percentage (0-100)
print(f"\nThresholding all brains to density = {target_density:.4f} (retain = {target_retain:.2f}%)")

brains_thresholded = np.zeros_like(brains)

for i in range(num_brains):
    brains_thresholded[i] = threshold_network(brains[i], retain=target_retain)
    
    G = nx.from_numpy_array(brains_thresholded[i])
    new_density = nx.density(G)
    print(f"Brain {i+1:2d}: {densities[i]:.4f} → {new_density:.4f}")

# ============================================================
# 5. Check connectivity
# ============================================================
print("\n--- Connectivity Check ---")
disconnected = []

for i in range(num_brains):
    G = nx.from_numpy_array(brains_thresholded[i])
    connected = nx.is_connected(G)
    num_components = nx.number_connected_components(G)
    
    status = "✓ Connected" if connected else f"✗ DISCONNECTED ({num_components} components)"
    print(f"Brain {i+1:2d}: {status}")
    
    if not connected:
        disconnected.append(i)

if len(disconnected) == 0:
    print("\n🎉 All networks are fully connected!")
else:
    print(f"\n⚠️  {len(disconnected)} network(s) are disconnected: {[d+1 for d in disconnected]}")

# ============================================================
# 6. Save preprocessed networks
# ============================================================
np.save('./resources/BrainNetworks/DATA/brain_networks_20_preprocessed.npy', brains_thresholded)
print(f"\nSaved preprocessed networks with shape {brains_thresholded.shape}")
print(f"All networks now have density = {target_density:.4f}")

"""
Generate a binary brain network using the GNM toolbox.

This script:
  1. Loads node coordinates from data/coordinates.npy
  2. Computes a Euclidean distance matrix from those coordinates
  3. Generates a binary network using the GenerativeNetworkModel
  4. Saves the resulting adjacency matrix and distance matrix to data/
  5. Plots the connectivity matrix as a heatmap and saves it to images/
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

from gnm import BinaryGenerativeParameters, GenerativeNetworkModel
from gnm.generative_rules import MatchingIndex

# ============================================================
# 1. Load coordinates
# ============================================================
coords = np.load('./resources/BrainNetworks/DATA/coordinates.npy')
num_nodes = coords.shape[0]
print(f"Loaded {num_nodes} node coordinates with shape {coords.shape}")

# ============================================================
# 2. Compute Euclidean distance matrix
# ============================================================
dist_np = squareform(pdist(coords, metric='euclidean'))
distance_matrix = torch.tensor(dist_np, dtype=torch.float32)
print(f"Distance matrix shape: {distance_matrix.shape}")

# Save the distance matrix for future use
np.save('./resources/BrainNetworks/DATA/distance_matrix.npy', dist_np)
print("Saved distance matrix to data/distance_matrix.npy")

# ============================================================
# 3. Set up and run the Generative Network Model
# ============================================================
# Choose a target number of connections (edges)
# A typical brain network has ~10-15% density
num_possible_edges = num_nodes * (num_nodes - 1) // 2
target_density = 0.20  # 20% density
num_connections = int(num_possible_edges * target_density)
print(f"Target: {num_connections} connections ({target_density*100:.0f}% density)")

# Define binary generative parameters
binary_params = BinaryGenerativeParameters(
    eta=-2.2,           # Distance penalty (negative = prefer short-range)
    gamma=0.25,          # Topological preference (positive = prefer similar nodes)
    lambdah=0.0,        # No heterochronicity
    distance_relationship_type='powerlaw',
    preferential_relationship_type='powerlaw',
    heterochronicity_relationship_type='powerlaw',
    generative_rule=MatchingIndex(),
    num_iterations=num_connections,
)

# Create and run the model
model = GenerativeNetworkModel(
    binary_parameters=binary_params,
    num_simulations=1,
    distance_matrix=distance_matrix,
)

print("Running generative model...")
model.run_model()
print("Done!")

# Extract the generated adjacency matrix
adj_matrix = model.adjacency_matrix.squeeze(0).cpu().numpy()
print(f"Generated network: {num_nodes} nodes, {int(adj_matrix.sum()/2)} edges")

# Save the generated network
np.save('./resources/BrainNetworks/DATA/generated_binary_network.npy', adj_matrix)
print("Saved generated network to data/generated_binary_network.npy")

# ============================================================
# 4. Plot the connectivity matrix as a heatmap
# ============================================================
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(adj_matrix, cmap='Reds', interpolation='nearest', aspect='equal')
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Connection', fontsize=12)

ax.set_xlabel('Brain Region', fontsize=13)
ax.set_ylabel('Brain Region', fontsize=13)
ax.set_title('Binary Connectivity Matrix', fontsize=15, fontweight='bold')

plt.tight_layout()
plt.savefig('./images/binary_matrix_heatmap.png', dpi=200, bbox_inches='tight')
print("Saved heatmap to images/binary_matrix_heatmap.png")
plt.show()

# ============================================================
# 5. Plot nodes and edges on brain silhouette using nilearn
# ============================================================
from nilearn import plotting

num_possible_edges = num_nodes * (num_nodes - 1) // 2
target_density = 0.07  # 20% density
num_connections = int(num_possible_edges * target_density)
print(f"Target: {num_connections} connections ({target_density*100:.0f}% density)")

# Define binary generative parameters
binary_params = BinaryGenerativeParameters(
    eta=-2.2,           # Distance penalty (negative = prefer short-range)
    gamma=0.25,          # Topological preference (positive = prefer similar nodes)
    lambdah=0.0,        # No heterochronicity
    distance_relationship_type='powerlaw',
    preferential_relationship_type='powerlaw',
    heterochronicity_relationship_type='powerlaw',
    generative_rule=MatchingIndex(),
    num_iterations=num_connections,
)

# Create and run the model
model = GenerativeNetworkModel(
    binary_parameters=binary_params,
    num_simulations=1,
    distance_matrix=distance_matrix,
)

print("Running generative model...")
model.run_model()
print("Done!")

# Extract the generated adjacency matrix
adj_matrix = model.adjacency_matrix.squeeze(0).cpu().numpy()
print(f"Generated network: {num_nodes} nodes, {int(adj_matrix.sum()/2)} edges")


# Load coordinates (x, y, z for each node)
coords = np.load('./resources/BrainNetworks/DATA/coordinates.npy')


# Plot the connectome on a glass brain (sagittal view)
fig, ax = plt.subplots(figsize=(8, 6))

display = plotting.plot_connectome(
    adjacency_matrix=adj_matrix,
    node_coords=coords,
    node_size=50,
    node_color='firebrick',
    edge_cmap='Reds',
    edge_vmin=0,
    edge_vmax=1,
    display_mode='x',
    axes=ax,
    title='Sagittal',
    alpha=0.4,
)

plt.tight_layout()
plt.savefig('./images/brain_connectome_generated.png', dpi=200, bbox_inches='tight')
print("Saved brain connectome to images/brain_connectome_generated.png")
plt.show()

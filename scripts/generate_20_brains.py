"""
Generate 20 brain networks with varying eta and gamma parameters.

Each brain is generated using a different combination of eta and gamma,
simulating individual differences in network wiring. The resulting
networks are saved as a single array with shape [20, 100, 100].

We also generate a fake "early life stress" score for each brain,
where higher stress corresponds to weaker constraints (less negative eta,
lower gamma) — consistent with the Adaptive Stochasticity Hypothesis.
"""

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform

from gnm import BinaryGenerativeParameters, GenerativeNetworkModel
from gnm.generative_rules import MatchingIndex

# ============================================================
# 1. Load coordinates and compute distance matrix
# ============================================================
coords = np.load('./resources/BrainNetworks/DATA/coordinates.npy')
num_nodes = coords.shape[0]
print(f"Loaded {num_nodes} node coordinates.")

dist_np = squareform(pdist(coords, metric='euclidean'))
distance_matrix = torch.tensor(dist_np, dtype=torch.float32)

# ============================================================
# 2. Define parameter ranges for 20 brains
# ============================================================
num_brains = 20

# Eta: from -3.0 (strong distance penalty) to -1.8 (weak distance penalty)
eta_values = np.linspace(-2.2, -1.0, num_brains)

# Gamma: from 0.3 (strong topological preference) to 0.02 (weak preference)
gamma_values = np.linspace(0.3, 0.1, num_brains)

# Target density: varies between 18% and 22% across brains
num_possible_edges = num_nodes * (num_nodes - 1) // 2
np.random.seed(123)
density_values = np.random.uniform(0.18, 0.22, num_brains)
connections_per_brain = [int(num_possible_edges * d) for d in density_values]

print(f"Generating {num_brains} brains")
print(f"Density range: {density_values.min():.3f} to {density_values.max():.3f}")
print(f"Eta range:     {eta_values[0]:.2f} to {eta_values[-1]:.2f}")
print(f"Gamma range:   {gamma_values[0]:.2f} to {gamma_values[-1]:.2f}")

# ============================================================
# 3. Generate each brain network
# ============================================================
all_brains = np.zeros((num_brains, num_nodes, num_nodes))

for i in range(num_brains):
    eta = float(eta_values[i])
    gamma = float(gamma_values[i])
    num_connections = connections_per_brain[i]
    
    print(f"\nBrain {i+1}/{num_brains}: eta={eta:.3f}, gamma={gamma:.3f}, density={density_values[i]:.3f}")
    
    connected = False
    attempt = 0
    
    while not connected:
        attempt += 1
        
        binary_params = BinaryGenerativeParameters(
            eta=eta,
            gamma=gamma,
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
        
        # Check connectivity
        import networkx as nx
        G = nx.from_numpy_array(adj)
        connected = nx.is_connected(G)
        
        if not connected:
            n_components = nx.number_connected_components(G)
            print(f"  Attempt {attempt}: disconnected ({n_components} components), regenerating...")
        
        if attempt > 50:
            print(f"  WARNING: Could not generate connected network after 50 attempts, using last result.")
            break
    
    print(f"  ✓ Connected after {attempt} attempt(s)")
    
    # ----- Convert binary to weighted using distances -----
    # Biologically, shorter distances between brain regions tend to have
    # MORE white matter streamlines (stronger connections). We simulate this
    # using an inverse-distance relationship with some noise.
    
    rng = np.random.default_rng(seed=i)
    max_dist = dist_np[dist_np > 0].max()
    
    # Normalise distances to [0, 1] range
    dist_norm = dist_np / max_dist
    
    # Inverse distance: shorter distance -> higher streamline count
    # Scale to roughly 5-500 streamlines
    raw_weights = 500 * (1 - dist_norm) ** 2
    
    # Add some noise (±20%) to make it more realistic
    noise_factor = rng.uniform(0.8, 1.2, size=raw_weights.shape)
    weights = raw_weights * noise_factor
    
    # Apply only where connections exist, clip to minimum 5 streamlines
    weighted_adj = adj * weights
    weighted_adj = np.clip(weighted_adj, 0, None)
    weighted_adj[weighted_adj > 0] = np.maximum(weighted_adj[weighted_adj > 0], 5)
    
    # Make sure it stays symmetric
    weighted_adj = (weighted_adj + weighted_adj.T) / 2
    
    all_brains[i] = weighted_adj
    print(f"  -> {int((adj > 0).sum()/2)} edges, weight range: {weighted_adj[weighted_adj > 0].min():.0f}-{weighted_adj[weighted_adj > 0].max():.0f} streamlines")

# ============================================================
# 4. Generate stress scores
# ============================================================
# Higher index = weaker constraints = more stochastic = higher stress
# We create a simple linear mapping with some noise
np.random.seed(1)
base_stress = np.linspace(4, 16, num_brains)  # Smooth gradient from low to high
noise = np.random.normal(0, 6, num_brains)   # Add some noise
stress_scores = np.clip(base_stress + noise, 0, 20)  # Clip to 0-20 range
stress_scores = np.round(stress_scores, 1)

print(f"\nStress scores: {stress_scores}")

# ============================================================
# 5. Save everything
# ============================================================
np.save('./resources/BrainNetworks/DATA/brain_networks_20.npy', all_brains)
np.save('./resources/BrainNetworks/DATA/stress_scores.npy', stress_scores)

# Also save the parameter values for reference
params = np.column_stack([eta_values, gamma_values])
np.save('./resources/BrainNetworks/DATA/generation_params.npy', params)

print(f"\nSaved:")
print(f"  brain_networks_20.npy  shape={all_brains.shape}")
print(f"  stress_scores.npy      shape={stress_scores.shape}")
print(f"  generation_params.npy  shape={params.shape}")
print("Done!")

# ============================================================
# 6. Plot all 20 brains as small matrix subplots
# ============================================================
import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 5, figsize=(10, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(all_brains[i], cmap='Reds', interpolation='nearest', aspect='equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'{i+1}', fontsize=8, pad=2)

#fig.text(0.5, 0.01, 'Brain Region', ha='center', fontsize=12)
#fig.text(0.02, 0.5, 'Brain Region', va='center', rotation='vertical', fontsize=12)
fig.suptitle('20 Brain Networks', fontsize=14, fontweight='bold', y=1.01)

plt.tight_layout()
plt.savefig('./images/BrainNetworks/all_20_brains_4x5.png', dpi=200, bbox_inches='tight')
print("Saved 4x5 grid to images/BrainNetworks/all_20_brains_4x5.png")
plt.show()

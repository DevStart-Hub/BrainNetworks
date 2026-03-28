"""
Group-level GNM parameter sweep.

Reproduces the code from the Model Fitting tutorial (ModelFitting.qmd).
Run from the project root:
    python scripts/sweep.py

Outputs:
    images/GenerativeModels/energy_landscape.png
"""

import time
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from gnm import fitting, generative_rules, evaluation

# ============================================================
# 1. Load the data
# ============================================================
brains = np.load("./resources/BrainNetworks/DATA/brain_networks_20_preprocessed.npy")
num_brains, num_nodes, _ = brains.shape

binary_brains = (brains > 0).astype(float)
binary_brains_tensor = torch.tensor(binary_brains, dtype=torch.float32)

print(f"Loaded {num_brains} brains, each {num_nodes} x {num_nodes}")

dist_np = np.load("./resources/BrainNetworks/DATA/distance_matrix.npy")
distance_matrix = torch.tensor(dist_np, dtype=torch.float32)

num_connections = int(binary_brains_tensor[0].sum().item() / 2)
print(f"Target connections per network: {num_connections}")

# ============================================================
# 2. Define the sweep
# ============================================================
n_eta        = 10
n_gamma      = 10
n_simulations = 10

eta_values   = torch.linspace(-5,   0,   n_eta)
gamma_values = torch.linspace(-0.5, 0.5, n_gamma)

binary_sweep_parameters = fitting.BinarySweepParameters(
    eta    = eta_values,
    gamma  = gamma_values,
    distance_relationship_type         = ["powerlaw"],
    preferential_relationship_type     = ["powerlaw"],
    generative_rule  = [generative_rules.MatchingIndex()],
    num_iterations   = [num_connections],
)

sweep_config = fitting.SweepConfig(
    binary_sweep_parameters = binary_sweep_parameters,
    num_simulations         = n_simulations,
    distance_matrix         = [distance_matrix],
)

# ============================================================
# 3. Define the energy function
# ============================================================
criteria = [
    evaluation.DegreeKS(),
    evaluation.BetweennessKS(),
    evaluation.ClusteringKS(),
    evaluation.EdgeLengthKS(distance_matrix),
]

energy = evaluation.MaxCriteria(criteria)
energy_key = str(energy)

# ============================================================
# 4. Run the sweep
# ============================================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on: {DEVICE}")

start = time.perf_counter()

experiments = fitting.perform_sweep(
    sweep_config         = sweep_config,
    binary_evaluations   = [energy],
    real_binary_matrices = binary_brains_tensor,
    save_model           = False,
    save_run_history     = False,
)

elapsed = time.perf_counter() - start
n_configs = n_eta * n_gamma
print(f"Sweep complete in {elapsed:.1f} s  ({elapsed / (n_configs * n_simulations):.3f} s per simulation)")

# ============================================================
# 5. Find the best-fitting parameters
# ============================================================
# experiments has one entry per parameter config (n_eta * n_gamma configs).
# evaluation_results.binary_evaluations[key] has shape [num_simulations, num_brains].
# We average over both dims to get a single energy per config, then pick the minimum.

mean_energies = [
    float(exp.evaluation_results.binary_evaluations[energy_key].mean())
    for exp in experiments
]

best_idx    = int(np.argmin(mean_energies))
best_exp    = experiments[best_idx]
best_energy = mean_energies[best_idx]
best_eta    = float(best_exp.run_config.binary_parameters.eta)
best_gamma  = float(best_exp.run_config.binary_parameters.gamma)

print(f"Best energy : {best_energy:.3f}")
print(f"Best eta    : {best_eta:.2f}")
print(f"Best gamma  : {best_gamma:.2f}")

# ============================================================
# 6. Plot the energy landscape
# ============================================================
# Build one row per config — guaranteed unique (eta, gamma) pairs.
rows = []
for exp, e_val in zip(experiments, mean_energies):
    rows.append({
        "eta":    float(exp.run_config.binary_parameters.eta),
        "gamma":  float(exp.run_config.binary_parameters.gamma),
        "energy": e_val,
    })

df_sweep = pd.DataFrame(rows)
landscape = df_sweep.pivot(index="eta", columns="gamma", values="energy")

fig, ax = plt.subplots(figsize=(7, 5))

im = ax.imshow(
    landscape.values,
    origin="lower",
    aspect="auto",
    cmap="viridis_r",
    extent=[
        landscape.columns.min(), landscape.columns.max(),
        landscape.index.min(),   landscape.index.max(),
    ],
)

ax.scatter(best_gamma, best_eta, color="red", s=120, zorder=5,
           label=f"Best Fit")

plt.colorbar(im, ax=ax, label="Energy (max KS)")
ax.set_xlabel("gamma (homophily)", fontsize=12)
ax.set_ylabel("eta (distance penalty)", fontsize=12)
ax.set_title("Energy landscape", fontsize=13)
ax.legend(fontsize=10)

plt.tight_layout()

out_path = "./images/GenerativeModels/energy_landscape.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved energy landscape to {out_path}")
plt.show()

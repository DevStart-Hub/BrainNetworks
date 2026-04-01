"""
Individual-level GNM parameter sweep (fine grid).

Runs a 20x20 sweep over a narrower eta/gamma range chosen by visual
inspection of the coarse landscape. Saves results as plain dicts.

Run from the project root:
    python scripts/sweep_fine.py

Outputs:
    resources/GenerativeModels/individual_experiments.pkl
"""

import pickle
import time
import numpy as np
import torch

from gnm import fitting, generative_rules, evaluation

# ============================================================
# 1. Load the data
# ============================================================
brains = np.load("./resources/BrainNetworks/DATA/brain_networks_20_preprocessed.npy")
num_brains, num_nodes, _ = brains.shape
binary_brains = (brains > 0).astype(float)
binary_brains_tensor = torch.tensor(binary_brains, dtype=torch.float32)

dist_np = np.load("./resources/BrainNetworks/DATA/distance_matrix.npy")
distance_matrix = torch.tensor(dist_np, dtype=torch.float32)

num_connections = int(binary_brains_tensor[0].sum().item() / 2)
print(f"Loaded {num_brains} brains, {num_connections} connections each")

# ============================================================
# 2. Define the energy function
# ============================================================
criteria = [
    evaluation.DegreeKS(),
    evaluation.BetweennessKS(),
    evaluation.ClusteringKS(),
    evaluation.EdgeLengthKS(distance_matrix),
]
energy     = evaluation.MaxCriteria(criteria)
energy_key = str(energy)

# ============================================================
# 3. Define the fine sweep
# Range chosen by visual inspection of the coarse landscape.
# ============================================================
n_eta   = 30
n_gamma = 30
n_sims  = 30

eta_values   = torch.linspace(-3.0, -0.5, n_eta)
gamma_values = torch.linspace( 0.05,  0.4, n_gamma)

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
    num_simulations         = n_sims,
    distance_matrix         = [distance_matrix],
)

# ============================================================
# 4. Run the sweep
# ============================================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on: {DEVICE}")
print(f"Fine sweep: {n_eta} eta x {n_gamma} gamma x {n_sims} simulations")

start = time.perf_counter()

raw = fitting.perform_sweep(
    sweep_config         = sweep_config,
    binary_evaluations   = [energy],
    real_binary_matrices = binary_brains_tensor,
    save_model           = False,
    save_run_history     = False,
)

elapsed = time.perf_counter() - start
print(f"Sweep complete in {elapsed:.1f} s")

# ============================================================
# 5. Save as plain dicts (same format as coarse experiments)
# energy_tensor shape: [num_simulations, num_brains]
# ============================================================
experiments = [
    {
        "eta":           float(exp.run_config.binary_parameters.eta),
        "gamma":         float(exp.run_config.binary_parameters.gamma),
        "energy_tensor": exp.evaluation_results.binary_evaluations[energy_key],
    }
    for exp in raw
]

pkl_path = "./resources/GenerativeModels/individual_experiments.pkl"
with open(pkl_path, "wb") as f:
    pickle.dump(experiments, f)
print(f"Saved {len(experiments)} experiments to {pkl_path}")

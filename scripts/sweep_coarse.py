"""
Group-level GNM parameter sweep (coarse grid).

Runs a 10x10 sweep over eta and gamma, evaluates against 20 real brains,
and saves the results as a list of plain dicts to experiments.pkl.

Run from the project root:
    python scripts/sweep_coarse.py

Outputs:
    resources/GenerativeModels/experiments.pkl
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
# 2. Define the sweep
# ============================================================
n_eta        = 10
n_gamma      = 10
n_simulations = 10

eta_values   = torch.linspace(-5, 0, n_eta)
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
energy     = evaluation.MaxCriteria(criteria)
energy_key = str(energy)

# ============================================================
# 4. Run the sweep
# ============================================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on: {DEVICE}")

start = time.perf_counter()

raw = fitting.perform_sweep(
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
# 5. Save as plain dicts
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

pkl_path = "./resources/GenerativeModels/experiments.pkl"
with open(pkl_path, "wb") as f:
    pickle.dump(experiments, f)
print(f"Saved {len(experiments)} experiments to {pkl_path}")

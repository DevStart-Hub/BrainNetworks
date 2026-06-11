"""
Figures for the "Looking inside the RNN" tutorial.

  1. all_weights.png   - the three learned weight matrices (W_in, W_rec, W_out)
  2. wrec_vs_brain.png  - W_rec side by side with a real brain connectivity matrix
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Load the trained weights
# ------------------------------------------------------------------
state = torch.load('./resources/RecurrentNetworks/trained_rnn.pt')
W_in = state['W_in.weight'].numpy()    # (100, 3)
W_rec = state['W_rec.weight'].numpy()  # (100, 100)
W_out = state['W_out.weight'].numpy()  # (1, 100)
print("shapes:", W_in.shape, W_rec.shape, W_out.shape)

# a real (weighted) brain connectivity matrix from Section 1 (100 nodes)
brain = np.load('./resources/BrainNetworks/DATA/brain_networks_20.npy')[0]


def sym(m):
    return float(np.abs(m).max())


# ------------------------------------------------------------------
# Figure 1: all three weight matrices
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
panels = [
    (W_in,  "Input weights  $W_{in}$\n(100 units x 3 inputs)"),
    (W_rec, "Recurrent weights  $W_{rec}$\n(100 x 100)"),
    (W_out, "Output weights  $W_{out}$\n(1 output x 100 units)"),
]
for ax, (M, title) in zip(axes, panels):
    m = sym(M)
    im = ax.imshow(M, cmap='RdBu_r', vmin=-m, vmax=m, aspect='auto', interpolation='nearest')
    ax.set_title(title, fontsize=12, fontweight='bold')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig('./images/RecurrentNetworks/all_weights.png', dpi=200, bbox_inches='tight')
print("Saved all_weights.png")

# ------------------------------------------------------------------
# Figure 2: W_rec vs a real brain connectivity matrix
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

vmax_brain = np.percentile(brain[brain > 0], 99)
im0 = axes[0].imshow(brain, cmap='Reds', interpolation='nearest', aspect='equal', vmax=vmax_brain)
axes[0].set_title("A brain connectivity matrix\n(from Section 1)", fontsize=13, fontweight='bold')
axes[0].set_xlabel("brain region", fontsize=11)
axes[0].set_ylabel("brain region", fontsize=11)
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

m = sym(W_rec)
im1 = axes[1].imshow(W_rec, cmap='RdBu_r', vmin=-m, vmax=m, interpolation='nearest', aspect='equal')
axes[1].set_title("Our RNN's recurrent weights\n$W_{rec}$", fontsize=13, fontweight='bold')
axes[1].set_xlabel("unit", fontsize=11)
axes[1].set_ylabel("unit", fontsize=11)
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('./images/RecurrentNetworks/wrec_vs_brain.png', dpi=200, bbox_inches='tight')
print("Saved wrec_vs_brain.png")

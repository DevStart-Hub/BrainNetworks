"""
Normalised W_rec vs brain comparison for the "Looking inside the RNN" tutorial.
Takes |W_rec|, normalises both matrices to [0, 1], and plots them on a shared scale.
Saves images/RecurrentNetworks/wrec_vs_brain_normalized.png
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

brain = np.load('./resources/BrainNetworks/DATA/brain_networks_20.npy')[0]
W_rec = torch.load('./resources/RecurrentNetworks/trained_rnn.pt')['W_rec.weight'].numpy()

# match the tutorial's preprocessing
W_rec = np.abs(W_rec)            # strength, ignoring excitatory/inhibitory sign
brain = brain / brain.max()      # normalise each to [0, 1]
W_rec = W_rec / W_rec.max()

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
panels = [(brain, "A brain connectivity matrix\n(strength, normalised)", "brain region"),
          (W_rec, "Our RNN's $W_{rec}$\n(magnitude, normalised)", "unit")]
for ax, (M, title, lab) in zip(axes, panels):
    im = ax.imshow(M, cmap='Reds', vmin=0, vmax=1, interpolation='nearest', aspect='equal')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel(lab, fontsize=11)
    ax.set_ylabel(lab, fontsize=11)

fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04, label='normalised connection strength')
plt.savefig('./images/RecurrentNetworks/wrec_vs_brain_normalized.png', dpi=200, bbox_inches='tight')
print("Saved wrec_vs_brain_normalized.png")

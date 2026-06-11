"""
Final preprocessed W_rec vs brain comparison for "Looking inside the RNN".
Pipeline (matching the tutorial): |W_rec| -> normalise -> symmetrise -> drop diagonal.
Saves images/RecurrentNetworks/wrec_vs_brain_symmetric.png
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

brain = np.load('./resources/BrainNetworks/DATA/brain_networks_20.npy')[0].astype(float)
W_rec = torch.load('./resources/RecurrentNetworks/trained_rnn.pt')['W_rec.weight'].numpy().astype(float)

# match the tutorial's preprocessing, in order
W_rec = np.abs(W_rec)
brain = brain / brain.max()
W_rec = W_rec / W_rec.max()
W_rec = (W_rec + W_rec.T) / 2          # symmetrise: drop direction info
np.fill_diagonal(W_rec, 0)             # drop self-connections, like the brain
np.fill_diagonal(brain, 0)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
panels = [(brain, "A brain connectivity matrix\n(preprocessed)", "brain region"),
          (W_rec, "Our RNN's $W_{rec}$\n(preprocessed)", "unit")]
for ax, (M, title, lab) in zip(axes, panels):
    im = ax.imshow(M, cmap='Reds', vmin=0, vmax=1, interpolation='nearest', aspect='equal')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel(lab, fontsize=11)
    ax.set_ylabel(lab, fontsize=11)

fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04, label='normalised connection strength')
plt.savefig('./images/RecurrentNetworks/wrec_vs_brain_symmetric.png', dpi=200, bbox_inches='tight')
print("Saved wrec_vs_brain_symmetric.png")

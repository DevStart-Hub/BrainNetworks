"""
Plot all 20 preprocessed brain networks as a 4x5 grid.

Run from the project root:
    python scripts/plot_20_brains.py

Requires:
    resources/BrainNetworks/DATA/brain_networks_20_preprocessed.npy

Outputs:
    images/BrainNetworks/all_20_brains_4x5.png
"""

import numpy as np
import matplotlib.pyplot as plt

brains = np.load("./resources/BrainNetworks/DATA/brain_networks_20_preprocessed.npy")

fig, axes = plt.subplots(4, 5, figsize=(10, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(brains[i], cmap="Reds", interpolation="nearest", aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{i+1}", fontsize=8, pad=2)

fig.suptitle("20 Brain Networks", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("./images/BrainNetworks/all_20_brains_4x5.png", dpi=200, bbox_inches="tight")
print("Saved all_20_brains_4x5.png")
plt.show()

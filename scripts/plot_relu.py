"""Plot the ReLU activation function for the Building-the-RNN tutorial."""

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 400)
y = np.maximum(0, x)

fig, ax = plt.subplots(figsize=(7, 4.2))

# the two regimes, shaded
ax.axvspan(-3, 0, color='#F2F2F2', zorder=0)
ax.axvspan(0, 3, color='#E3F4E1', zorder=0)

ax.plot(x, y, color='#C0392B', linewidth=3)

# axes through the origin
ax.axhline(0, color='#999999', linewidth=0.8)
ax.axvline(0, color='#999999', linewidth=0.8)

# annotations
ax.text(-1.5, 1.6, "input negative\n→ unit stays silent\n(response = 0)",
        ha='center', va='center', fontsize=10, color='#555555')
ax.text(1.6, 0.7, "input positive\n→ unit responds,\nmore strongly with\nmore input",
        ha='center', va='center', fontsize=10, color='#1E6B34')

ax.set_xlabel("total input arriving at the unit", fontsize=12)
ax.set_ylabel("unit's response", fontsize=12)
ax.set_title("The ReLU activation function", fontsize=13, fontweight='bold')
ax.set_xlim(-3, 3)
ax.set_ylim(-0.4, 3)
ax.set_xticks(range(-3, 4))

plt.tight_layout()
out = './images/RecurrentNetworks/relu.png'
plt.savefig(out, dpi=200, bbox_inches='tight')
print("Saved", out)

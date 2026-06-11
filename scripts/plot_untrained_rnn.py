"""
Build the LeakyRNN from the Building-the-RNN tutorial, run it UNTRAINED on the
Go/No-Go task, report its accuracy, and plot its output vs the target on an
example Go and No-Go trial.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

np.random.seed(0)
torch.manual_seed(0)

# ------------------------------------------------------------------
# Task (from the Task Design tutorial)
# ------------------------------------------------------------------
def generate_batch(batch_size=32, dt=100):
    fixation, stimulus, delay, decision = 500, 500, 500, 500
    n_fix, n_stim, n_del, n_dec = (fixation // dt, stimulus // dt,
                                   delay // dt, decision // dt)
    T = n_fix + n_stim + n_del + n_dec
    stim_on, stim_off = n_fix, n_fix + n_stim
    dec_on = n_fix + n_stim + n_del

    x = np.zeros((T, batch_size, 3))
    y = np.zeros((T, batch_size, 1))
    labels = np.random.randint(0, 2, batch_size)
    go, nogo = labels == 1, labels == 0
    x[:dec_on, :, 0] = 1
    x[stim_on:stim_off, go, 1] = 1
    x[stim_on:stim_off, nogo, 2] = 1
    y[dec_on:, go, 0] = 1
    return x, y, labels


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
class LeakyRNN(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, alpha=0.2):
        super().__init__()
        self.alpha = alpha
        self.n_hidden = n_hidden
        self.W_in = nn.Linear(n_input, n_hidden)
        self.W_rec = nn.Linear(n_hidden, n_hidden)
        self.W_out = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        T, batch, _ = x.shape
        h = torch.zeros(batch, self.n_hidden)
        outputs = []
        for t in range(T):
            pre = self.W_in(x[t]) + self.W_rec(h)
            h = (1 - self.alpha) * h + self.alpha * torch.relu(pre)
            outputs.append(self.W_out(h))
        return torch.stack(outputs)


# ------------------------------------------------------------------
# Run untrained
# ------------------------------------------------------------------
dt = 100
dec_on = 15
model = LeakyRNN(n_input=3, n_hidden=100, n_output=1)

x, y, labels = generate_batch(batch_size=500)
x_t = torch.tensor(x, dtype=torch.float32)
out = model(x_t).detach().numpy()          # (T, batch, 1)

resp = out[dec_on:, :, 0].mean(0)          # mean output in the decision period
pred = (resp > 0.5).astype(int)
acc = (pred == labels).mean()
print(f"Untrained accuracy: {acc:.2f}")

# ------------------------------------------------------------------
# Plot output vs target on one Go and one No-Go trial
# ------------------------------------------------------------------
go_idx = np.where(labels == 1)[0][0]
nogo_idx = np.where(labels == 0)[0][0]
T = out.shape[0]
t_ms = np.arange(T) * dt

bounds = [0, 500, 1000, 1500, 2000]
colors = ['#ECECEC', '#FDE9D9', '#E8F1FB', '#E3F4E1']
names = ['fixation', 'stimulus', 'delay', 'decision']

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, idx, title in zip(axes, [go_idx, nogo_idx], ["Go trial", "No-Go trial"]):
    for k in range(4):
        ax.axvspan(bounds[k], bounds[k + 1], color=colors[k], zorder=0)
        ax.text((bounds[k] + bounds[k + 1]) / 2, 1.02, names[k],
                transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=9, color='#666')
    ax.plot(t_ms, y[:, idx, 0], color='#1E8449', lw=2.5, label='target')
    ax.plot(t_ms, out[:, idx, 0], color='#C0392B', lw=2, label='network output')
    ax.axhline(0, color='#cccccc', lw=0.8)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=18)
    ax.set_xlabel('Time (ms)', fontsize=11)
    ax.set_xlim(0, 2000)
axes[0].set_ylabel('output', fontsize=11)
axes[0].legend(loc='upper left', fontsize=9)
plt.suptitle(f'Untrained network (accuracy = {acc:.0%})', fontsize=14, y=1.04)
plt.tight_layout()
out_path = './images/RecurrentNetworks/untrained_rnn.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print("Saved", out_path)

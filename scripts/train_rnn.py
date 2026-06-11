"""
Train the LeakyRNN on the Go/No-Go task for the Training tutorial.

Produces:
  - images/RecurrentNetworks/training_curves.png   (loss + accuracy over training)
  - images/RecurrentNetworks/trained_rnn.png       (trained output on Go / No-Go)
  - resources/RecurrentNetworks/trained_rnn.pt      (weights, for later tutorials)
Prints the final accuracy.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

np.random.seed(0)
torch.manual_seed(0)

# ------------------------------------------------------------------
# Task + model (as in the tutorials)
# ------------------------------------------------------------------
def generate_batch(batch_size=128, dt=100):
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
            drive = torch.relu(self.W_in(x[t]) + self.W_rec(h))
            h = (1 - self.alpha) * h + self.alpha * drive
            outputs.append(self.W_out(h))
        return torch.stack(outputs)


dec_on = 15


def accuracy(model, n=500):
    x, y, labels = generate_batch(batch_size=n)
    with torch.no_grad():
        out = model(torch.tensor(x, dtype=torch.float32)).numpy()
    resp = out[dec_on:, :, 0].mean(0)
    pred = (resp > 0.5).astype(int)
    return (pred == labels).mean()


# ------------------------------------------------------------------
# Train
# ------------------------------------------------------------------
model = LeakyRNN(n_input=3, n_hidden=100, n_output=1)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

n_steps = 2000
steps_rec, loss_rec, acc_rec = [], [], []

for step in range(n_steps):
    x, y, labels = generate_batch(batch_size=128)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    outputs = model(x)
    loss = ((outputs - y) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0 or step == n_steps - 1:
        a = accuracy(model)
        steps_rec.append(step)
        loss_rec.append(loss.item())
        acc_rec.append(a)
        print(f"step {step:4d}   loss {loss.item():.4f}   acc {a:.2f}")

final_acc = accuracy(model, n=2000)
print(f"\nFinal accuracy: {final_acc:.3f}")

# ------------------------------------------------------------------
# Save weights
# ------------------------------------------------------------------
os.makedirs('./resources/RecurrentNetworks', exist_ok=True)
torch.save(model.state_dict(), './resources/RecurrentNetworks/trained_rnn.pt')
np.save('./resources/RecurrentNetworks/trained_W_rec.npy',
        model.W_rec.weight.detach().numpy())
print("Saved weights to resources/RecurrentNetworks/")

# ------------------------------------------------------------------
# Plot 1: training curves
# ------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(steps_rec, loss_rec, color='#C0392B', lw=2)
ax[0].set_xlabel('training step', fontsize=12)
ax[0].set_ylabel('loss', fontsize=12)
ax[0].set_title('Loss goes down', fontsize=13, fontweight='bold')
ax[1].plot(steps_rec, np.array(acc_rec) * 100, color='#1E8449', lw=2)
ax[1].axhline(50, color='#999', ls='--', lw=1, label='chance')
ax[1].set_xlabel('training step', fontsize=12)
ax[1].set_ylabel('accuracy (%)', fontsize=12)
ax[1].set_ylim(40, 102)
ax[1].set_title('Accuracy goes up', fontsize=13, fontweight='bold')
ax[1].legend(fontsize=10)
plt.tight_layout()
plt.savefig('./images/RecurrentNetworks/training_curves.png', dpi=200, bbox_inches='tight')
print("Saved training_curves.png")

# ------------------------------------------------------------------
# Plot 2: trained network in action
# ------------------------------------------------------------------
x, y, labels = generate_batch(batch_size=500)
with torch.no_grad():
    out = model(torch.tensor(x, dtype=torch.float32)).numpy()
go_idx = np.where(labels == 1)[0][0]
nogo_idx = np.where(labels == 0)[0][0]
T = out.shape[0]
t_ms = np.arange(T) * 100
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
plt.suptitle(f'Trained network (accuracy = {final_acc:.0%})', fontsize=14, y=1.04)
plt.tight_layout()
plt.savefig('./images/RecurrentNetworks/trained_rnn.png', dpi=200, bbox_inches='tight')
print("Saved trained_rnn.png")

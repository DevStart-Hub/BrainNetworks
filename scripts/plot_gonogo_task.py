"""
Generate the Go/No-Go task figure for the Task Design tutorial.

Builds the task FROM SCRATCH (no neurogym) but keeps the period structure
(fixation -> stimulus -> delay -> decision) and ~500 ms timing used in the
lab's neurogym Poli/Yang tasks. Saves an example Go trial and No-Go trial.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ------------------------------------------------------------------
# The task generator (this is the code shown in the tutorial)
# ------------------------------------------------------------------
def generate_batch(batch_size=32, dt=100):
    # period durations in milliseconds
    fixation, stimulus, delay, decision = 500, 500, 500, 500

    # turn milliseconds into a number of timesteps
    n_fix  = fixation // dt
    n_stim = stimulus // dt
    n_del  = delay    // dt
    n_dec  = decision // dt
    T = n_fix + n_stim + n_del + n_dec          # total timesteps

    # where each period starts / ends (in timesteps)
    stim_on  = n_fix
    stim_off = n_fix + n_stim
    dec_on   = n_fix + n_stim + n_del

    # 3 input channels: 0 = fixation cue, 1 = Go cue, 2 = No-Go cue
    x    = np.zeros((T, batch_size, 3))
    y    = np.zeros((T, batch_size, 1))         # target response
    mask = np.zeros((T, batch_size))            # when we score the network

    # half the trials are Go (1), half are No-Go (0)
    labels = np.random.randint(0, 2, batch_size)
    go, nogo = labels == 1, labels == 0

    # fixation cue stays ON until the decision period begins
    x[:dec_on, :, 0] = 1

    # the stimulus: a Go cue on Go trials, a No-Go cue on No-Go trials
    x[stim_on:stim_off, go,   1] = 1
    x[stim_on:stim_off, nogo, 2] = 1

    # respond (1) only on Go trials, and only during the decision period
    y[dec_on:, go, 0] = 1

    # we score the network only during the decision period
    mask[dec_on:, :] = 1

    return x, y, mask, labels


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------
dt = 100
fixation = stimulus = delay = decision = 500
bounds_ms = [0, fixation, fixation + stimulus,
             fixation + stimulus + delay,
             fixation + stimulus + delay + decision]
period_names = ['fixation', 'stimulus', 'delay', 'decision']
period_colors = ['#ECECEC', '#FDE9D9', '#E8F1FB', '#E3F4E1']

# one Go trial and one No-Go trial, built explicitly
x = np.zeros((20, 2, 3)); y = np.zeros((20, 2, 1))
# trial 0 = Go, trial 1 = No-Go
x[:15, :, 0] = 1                 # fixation cue (steps 0-14)
x[5:10, 0, 1] = 1                # Go cue on trial 0
x[5:10, 1, 2] = 1                # No-Go cue on trial 1
y[15:, 0, 0] = 1                 # respond on the Go trial

T = x.shape[0]
t_ms = np.arange(T) * dt

traces = [
    ("Fixation cue", x[:, :, 0], '#555555'),
    ("Go cue",       x[:, :, 1], '#C0392B'),
    ("No-Go cue",    x[:, :, 2], '#2E86C1'),
    ("Target output", y[:, :, 0], '#1E8449'),
]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for col, (ax, title) in enumerate(zip(axes, ["Go trial", "No-Go trial"])):
    # shade periods
    for k in range(4):
        ax.axvspan(bounds_ms[k], bounds_ms[k + 1], color=period_colors[k], zorder=0)
        ax.text((bounds_ms[k] + bounds_ms[k + 1]) / 2, len(traces) - 0.35,
                period_names[k], ha='center', va='top', fontsize=9, color='#666666')
    # stacked traces (each offset vertically)
    for row, (label, data, color) in enumerate(reversed(traces)):
        base = row * 1.0
        ax.step(t_ms, base + 0.7 * data[:, col], where='post', color=color, linewidth=2)
        ax.axhline(base, color='#DDDDDD', linewidth=0.8, zorder=0)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Time (ms)', fontsize=11)
    ax.set_xlim(0, bounds_ms[-1])

axes[0].set_yticks([i * 1.0 for i in range(len(traces))])
axes[0].set_yticklabels([t[0] for t in reversed(traces)], fontsize=10)
axes[0].set_ylim(-0.3, len(traces) + 0.1)

plt.tight_layout()
out = './images/RecurrentNetworks/gonogo_task.png'
plt.savefig(out, dpi=200, bbox_inches='tight')
print("Saved", out)

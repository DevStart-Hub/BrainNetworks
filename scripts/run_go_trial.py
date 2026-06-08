"""Run the single Go-trial code from the Task Design tutorial and print x and y."""

import numpy as np

# period durations in milliseconds
fixation, stimulus, delay, decision = 500, 500, 500, 500

dt = 100   # milliseconds per timestep

# turn milliseconds into a number of timesteps
n_fix  = fixation // dt
n_stim = stimulus // dt
n_del  = delay    // dt
n_dec  = decision // dt
T = n_fix + n_stim + n_del + n_dec

# where each period starts / ends
stim_on  = n_fix
stim_off = n_fix + n_stim
dec_on   = n_fix + n_stim + n_del

# a single Go trial: T timesteps x 3 input channels
x = np.zeros((T, 3))
x[:dec_on, 0] = 1            # fixation cue
x[stim_on:stim_off, 1] = 1   # Go cue

y = np.zeros((T, 1))
y[dec_on:, 0] = 1            # respond during the decision period

print("x (inputs) — columns: [fixation, Go, No-Go]")
print(x)
print()
print("y (target output)")
print(y)

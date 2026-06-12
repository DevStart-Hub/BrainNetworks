# GNM Tutorials — Next Steps & Full Plan

## General Setup

- **Create a dedicated Conda environment** that includes:
  - `GenerativeNetworkModels` (the GNM toolbox)
  - `jupyter` (so that Quarto can execute Python chunks during render)
  - Plotting libraries (e.g., `matplotlib`, `nilearn`, `networkx`)
  - Any other dependencies needed across all tutorials
- **We will NOT use real data!** All brain networks used throughout the tutorials will be **generated using the GNM toolbox itself**.

------------------------------------------------------------------------

## Section 1: Brain Networks

### Tutorial: Introduction (DONE — `Introduction.qmd`)

- Conceptual intro to brain networks, graph theory, and connectivity matrices.
- Under the hood (without showing the code overtly), we produce **two figures**:
  1.  A **3D brain connectivity plot** using `nilearn`, displaying the GNM default network with default coordinates.
  2.  A **connectivity matrix** of the same data.
- These figures are embedded as images — the reader sees the visuals but not the code that generated them.

### Tutorial: Installation (DONE — `Installation.qmd`)

- Install Python, set up Conda environment, `pip install GenerativeNetworkModels`.
- Callout box briefly explaining what GNMs are (simple rules to create artificial networks mimicking real brains — full details later).
- Load the default built-in network using `gnm.defaults.get_binary_network()`.
- **Now we show the script!** We reproduce the same connectivity matrix from the Introduction tutorial, but this time the reader sees and runs the code themselves.

### Tutorial: Research Question

- Introduce a **general scientific question** that will drive the rest of the tutorial series.
- **Scenario:** We have **20 brains**, and we want to investigate whether brain network organisation changes depending on **early life stress**.
- Each brain has an associated **early life stress score** from a real questionnaire (we need to identify which one — e.g., the Childhood Trauma Questionnaire, CTQ, or the Adverse Childhood Experiences, ACE).
- Frame the question: *Do individuals with higher early life stress show different network properties?*
- This question will be tackled using **different approaches** across the following tutorials.

### Tutorial: Preprocessing

- Before analysing the networks, we must check the data is usable!
  - **Threshold** all networks to the **same density** (so they are comparable).
  - Check: are the networks **fully connected**? Identify and discard any that are not.
  - Any other quality-control steps.

### Tutorial: Topology

- **First approach** to the research question: look at **topology metrics** and see if they correlate with early life stress.
- Compute network metrics for each of the 20 brains (e.g., clustering coefficient, path length, modularity, degree distribution, betweenness centrality, etc.).
- Possibly run a **PCA** on the metrics to show that they naturally cluster into **integration** vs **segregation** dimensions.
- Build **regression models in R** to test whether topology metrics (or PCA components) correlate with early life stress scores.

------------------------------------------------------------------------

## Section 2: Generative Models

### Tutorial: What is a GNM?

- We got very weak correlations from topology alone — but there is more we can do!
- Introduce the idea of **Generative Network Models**: finding a **parsimonious, small set of parameters** that captures complex brain topology.
- Explain **what a GNM is** conceptually: using simple wiring rules to grow artificial networks that look like real brains.
- Introduce the **two key parameters**:
  - **eta (η):** controls the spatial/distance penalty (how much wiring cost matters).
  - **gamma (γ):** controls the topological preference (e.g., homophily — connecting to nodes with similar existing connectivity).
- Explain what each parameter does and how they interact to shape the generated network.

### Tutorial: Model Fitting

- Now that we understand GNMs, we apply them!
- Perform a **binary sweep** over the eta-gamma parameter space.
- Define the **energy function** (how we measure how close a generated network is to a real one).
- Identify the **lowest energy model** (the parameter combination that best reproduces the observed brain network).
- **Interpret the results at group level**: what do the best-fitting eta and gamma values tell us about how brains are wired?

### Tutorial: Individual Differences

- Move from group-level to **individual-level** parameter estimation.
- Perform a **second sweep** that is:
  - **More focused / fine-grained** (narrower parameter range around the group optimum).
  - Run over **multiple runs** to ensure robust individual estimates.
- Extract **individual-level eta and gamma estimates** for each of the 20 brains.
- **Correlate** these individual parameter estimates with **early life stress scores**.
- Interpret the results: what do individual differences in wiring rules tell us?
- **Speculate on links to cognition**: how might differences in eta/gamma relate to cognitive outcomes?

## Section 3: Artificial Neural Networks

> **Big idea:** We've seen that brains are networks. We've built generative models to grow brain-like networks. Now we ask: can we build *artificial* networks that *learn* — and do they end up looking like real brains?

> **Architecture note (important for planning):** We use a **Leaky Recurrent Neural Network (RNN)** throughout this section, not a feedforward network. The key reason: the recurrent weight matrix `W_rec` is already `N×N`, making it a direct connectivity matrix with no tricks needed. This mirrors the architecture used in state-of-the-art brain-like ANN research (Yang et al. 2019; Khona et al. 2023), but built here from scratch — across several short tutorials (task → model → training) — with **no extra dependencies** beyond what is already in `gnm_box` (`torch`, `numpy`, `matplotlib`).

> **Rendering note (important for authoring):** Training an RNN live at render time would make every Quarto build slow and **non-deterministic** (different weights each build → drifting figures). Follow the same approach as the brain networks in Sections 1–2: **train once in a standalone script, save the trained weights (and the loss/accuracy history) to disk, and have the tutorials load those artifacts** for plotting. Set a fixed random seed (`torch.manual_seed`, `np.random.seed`) in the training script for reproducibility. The training-loop code is still *shown* in the Training tutorial so the reader understands it; it just isn't re-executed on every render.

> **Dependencies:** No new packages required. Everything runs in the existing `gnm_box` conda environment.

------------------------------------------------------------------------

### Tutorial: What is a Recurrent Neural Network?

**Goal:** Bridge from brain networks to RNNs conceptually — show that the architecture is already familiar before any code is written.

**Conceptual walkthrough:** - Recap: brain neurons receive inputs from many other neurons, integrate those signals over time, and fire if the integrated input crosses a threshold - Real neurons don't reset instantly — they have a **membrane time constant**: past activity decays gradually - The leaky integration formula captures this:

`h(t+1) = (1 - alpha) * h(t) + alpha * phi( W_rec * h(t) + W_in * x(t) )`

- `h(t)`: the activity of all N hidden units at time t — a snapshot of the network's internal state
- `W_rec`: the N×N **recurrent weight matrix** — who influences whom (this is the connectivity matrix!)
- `W_in`: input weights (stimulus arriving from outside, like thalamic input)
- `alpha = dt/tau`: the leak rate, set by the membrane time constant tau (\~100ms in cortex)
- `phi`: activation function (ReLU) — neurons fire positively above threshold
- The network processes a stimulus **one timestep at a time** — it has memory built in via `h(t)`
- Compare to GNMs: GNMs grew a network forward in time using wiring rules; RNNs update a network's activity forward in time using learned weights. Both are defined by the structure of their N×N connectivity matrix.

**The key insight to emphasise:** - `W_rec` is an N×N matrix where entry `W_ij` is the synaptic weight from unit j to unit i - This is structurally identical to the brain connectivity matrices studied since Tutorial 1 - We are not going to look *into* a black box — the connectivity matrix IS the model, and it's the same object we already know how to analyse

**Show a diagram:** - Input layer (stimulus) → hidden layer (N recurrent units, fully connected via W_rec) → output layer (decision) - Annotate `W_rec` as the connectivity matrix; explicitly draw the parallel to brain networks from Section 1

------------------------------------------------------------------------

### Tutorial: Task Design (DONE — `TaskDesign.qmd`)

**Goal:** Introduce cognitive tasks and build a Go/No-Go generator from scratch (no external dependencies). This tutorial is *only* about the task; the network that solves it comes next.

**Stress that many tasks exist:** reference the [neurogym](https://neurogym.github.io/) library and the multi-task work of Yang et al. (2019). We build ours from scratch so every moving part is visible. Tie Go/No-Go to **impulse control / executive function**, and back to the early-life-stress theme.

**Task design (from scratch, but matching the neurogym/Poli period + timing conventions):** - Four periods, **500 ms each at dt = 100 → T = 20 timesteps**: **fixation → stimulus → delay → decision** (same structure used in the lab's neurogym tasks). - **3 input channels:** a **fixation cue** (on until the decision period), a **Go cue**, and a **No-Go cue** (exactly one of the two is on during the stimulus period). - **1 output:** target = 1 during the decision period on Go trials, 0 otherwise (including all of the No-Go trial). - `mask` marks the decision period — we score the network only there. - ⚠️ Consequence for the next tutorials: the model's **`n_input = 3`** (not 1).

``` python
import numpy as np

def generate_batch(batch_size=32, dt=100):
    # period durations in milliseconds
    fixation, stimulus, delay, decision = 500, 500, 500, 500

    # turn milliseconds into a number of timesteps
    n_fix, n_stim, n_del, n_dec = (fixation // dt, stimulus // dt,
                                   delay // dt, decision // dt)
    T = n_fix + n_stim + n_del + n_dec          # total timesteps

    stim_on  = n_fix
    stim_off = n_fix + n_stim
    dec_on   = n_fix + n_stim + n_del

    # 3 input channels: 0 = fixation cue, 1 = Go cue, 2 = No-Go cue
    x    = np.zeros((T, batch_size, 3))
    y    = np.zeros((T, batch_size, 1))         # target response
    mask = np.zeros((T, batch_size))            # when we score the network

    labels = np.random.randint(0, 2, batch_size)   # 1 = Go, 0 = No-Go
    go, nogo = labels == 1, labels == 0

    x[:dec_on, :, 0] = 1                         # fixation cue until decision
    x[stim_on:stim_off, go,   1] = 1            # Go cue on Go trials
    x[stim_on:stim_off, nogo, 2] = 1            # No-Go cue on No-Go trials
    y[dec_on:, go, 0] = 1                        # respond only on Go decision
    mask[dec_on:, :] = 1                         # score the decision period

    return x, y, mask, labels
```

**Plot to produce:** - A single example trial: the input signal `x`, the target `y`, and the shaded response window, for both a Go and a No-Go trial side by side. This makes the task structure (stimulus → delay → respond) concrete before any network exists.

**Closing hook:** We now have a task. But who solves it? In the next tutorial we build the network — and we'll see it's the same N×N connectivity matrix we've worked with since Section 1.

------------------------------------------------------------------------

### Tutorial: Building the RNN

**Goal:** Implement the Leaky RNN from scratch in PyTorch and run it — on an **untrained** network — to see that it cannot yet do the task. This sets up the need for training in the following tutorial.

**Leaky RNN implementation (\~60 lines of PyTorch):**

``` python
import torch
import torch.nn as nn

class LeakyRNN(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, alpha=0.2):
        super().__init__()
        self.alpha = alpha
        self.n_hidden = n_hidden
        self.W_rec = nn.Linear(n_hidden, n_hidden, bias=False)  # the N×N connectivity matrix
        self.W_in  = nn.Linear(n_input,  n_hidden, bias=True)
        self.W_out = nn.Linear(n_hidden, n_output, bias=True)

    def forward(self, x):
        # x shape: (T, batch, n_input)
        h = torch.zeros(x.size(1), self.n_hidden)
        outputs = []
        for t in range(x.size(0)):
            pre = self.W_rec(h) + self.W_in(x[t])
            h = (1 - self.alpha) * h + self.alpha * torch.relu(pre)
            outputs.append(self.W_out(h))
        return torch.stack(outputs), h  # (T, batch, n_output)
```

- `model.W_rec.weight` is the N×N connectivity matrix, accessible at any point during or after training
- **Hidden size: N = 100** — deliberately matching the 100-node brain networks used throughout the series
- Walk through `forward` line by line: it is just the leaky-integration formula from the previous tutorial, applied once per timestep in a loop. Connect each line back to the equation `h(t+1) = (1 - alpha) * h(t) + alpha * phi(W_rec h + W_in x)`.

**The untrained network fails (the motivating hook):** Before any training, the weights are random, so the network has no idea what to do. Show this explicitly — it makes the next tutorial's payoff land:

``` python
model = LeakyRNN(n_input=3, n_hidden=100, n_output=1)

x, y, mask, labels = generate_batch(batch_size=200)
outputs, _ = model(torch.tensor(x, dtype=torch.float32))

# Did it respond correctly on Go vs No-Go? (final-timestep output thresholded at 0.5)
pred = (outputs[-1, :, 0] > 0.5).float().numpy()
print(f"Untrained accuracy: {(pred == labels).mean():.2f}")   # ~0.50 — pure chance!
```

- The untrained network sits at \~50% — it is guessing. Emphasise: the *architecture* is in place, but the *weights are meaningless* until the network learns them.

**Closing hook:** The network has the right shape but the wrong numbers. How does it go from random weights to weights that solve the task? That is **learning** — the subject of the next tutorial.

------------------------------------------------------------------------

### Tutorial: Training the RNN

**Goal:** Introduce how a network learns (gradient descent) and implement the training loop that turns the random `W_rec` into one that solves the Go/No-Go task.

**How does the network learn? (the key new concept — give it explicit, gentle treatment)**

This is the one genuinely new idea in the whole section, and it must not be assumed. In Section 2 the GNM toolbox did all the fitting internally, so the reader has *never* seen gradient descent. Explain the mechanism in plain language **before** showing any training code:

- The network starts with **random** `W_rec`, `W_in`, `W_out`, so at first it produces nonsense on the task.
- We summarise "how wrong it is" with a single number, the **loss** (here, the squared difference between its output and the correct response, counted *only* during the response window).
- The key step: PyTorch automatically works out, for **every single weight**, which direction to nudge it to make the loss a little smaller. This is **backpropagation / gradient descent** — the "gradient" is just the slope telling each weight which way is "downhill" toward lower error.
- We take one small step in that direction (the step size is the **learning rate**) and repeat thousands of times. Slowly, the random matrix turns into a matrix that solves the task.
- **Crucially:** we never tell the network *what* `W_rec` should be — we only tell it *whether it got the task right*, and the connectivity emerges on its own. That is exactly why inspecting the learned `W_rec` in the next tutorial is interesting.
- Callback to Section 2: a GNM *grows* a network forward using fixed wiring rules; here the network *learns* its wiring by repeatedly nudging weights to reduce error. Same object (`W_rec`), two opposite routes to it (grow forward vs. train backward).

**The training loop (show this code — it is the part the reader actually runs):**

``` python
import torch.optim as optim

model = LeakyRNN(n_input=3, n_hidden=100, n_output=1)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

losses, accuracies = [], []
for step in range(2000):
    # --- get a fresh batch of trials ---
    x, y, mask, labels = generate_batch(batch_size=32)
    x    = torch.tensor(x,    dtype=torch.float32)   # numpy -> torch
    y    = torch.tensor(y,    dtype=torch.float32)
    mask = torch.tensor(mask, dtype=torch.float32)

    # --- run the network forward over all timesteps ---
    outputs, _ = model(x)

    # --- measure error ONLY in the response window (that is what `mask` does) ---
    loss = (((outputs - y) ** 2).squeeze(-1) * mask).sum() / mask.sum()

    # --- the three lines that make learning happen ---
    optimizer.zero_grad()   # clear the gradients left over from the previous step
    loss.backward()         # backprop: compute the downhill direction for every weight
    optimizer.step()        # nudge every weight one small step downhill

    # --- track progress for the plots ---
    losses.append(loss.item())
    pred = (outputs[-1, :, 0] > 0.5).float()
    accuracies.append((pred == torch.tensor(labels, dtype=torch.float32)).float().mean().item())
```

Walk through the three-line core (`zero_grad` → `backward` → `step`) explicitly — this is the heart of *all* deep learning and the reader is meeting it for the first time. Everything else in the loop is bookkeeping (make a batch, run forward, score it, record progress).

**Glossary — define each term as it appears in the code above:**

- **Timestep**: one time point; the network updates `h(t)` at each step inside `model(x)` — unlike a feedforward network, it processes a *sequence*.
- **Trial / batch**: a set of simultaneous trials; `batch_size` sets how many trials are averaged per weight update.
- **Training step / iteration**: one pass through the loop — one batch in, one weight update out (we run 2000 of them).
- **Epoch** (related term worth a sentence): in a dataset of *fixed* size, one full pass over all the data; because we *generate* fresh trials on the fly, there is no fixed dataset, so here we simply count training steps.
- **Learning rate** (`lr=1e-3`): the step size; too large → unstable training, too small → painfully slow convergence.
- **Loss** (masked MSE): how wrong the network is, counted only when it should be responding.
- **Accuracy**: fraction of trials whose final output lands on the correct side of 0.5.

**Plots to produce:** - Loss curve and accuracy curve over training steps — the network improves over time - Example trial: plot `h(t)` for a handful of units, overlaid with the input stimulus and the network's output response - This makes the "network processes time" intuition concrete before we look inside at weights

------------------------------------------------------------------------

### Tutorial: W_rec is a Connectivity Matrix

**Goal:** The "aha" moment — the recurrent weight matrix is already, structurally, a brain connectivity matrix.

**The reveal:** - Extract `model.W_rec.weight.detach().numpy()` — a 100×100 matrix - Plot it as a heatmap — it looks exactly like the connectivity matrices from Tutorial 1 - The analogy is now exact, not approximate: - Brain: entry `(i, j)` = structural connection from region j to region i - Leaky RNN: entry `(i, j)` = synaptic weight from unit j to unit i - Both tell us: *who influences whom, and how strongly* - No binarisation or Gram-matrix tricks needed for the visual — the N×N structure is already there

**Continuous vs weighted vs binary:** - Recall from Sections 1–2 that the raw brain networks are **weighted** (streamline counts), and we *thresholded then binarised* them for the GNM analysis - `W_rec` is likewise **continuous** (varying, and crucially *signed*, weight strengths) - So `W_rec` is most directly analogous to a **weighted brain network** (white matter tract strength), before binarisation - For the topological-metric comparison later, we apply the **same recipe used in the Preprocessing tutorial**: threshold `|W_rec|` to the same density as the brain networks, then binarise — putting RNN and brain on the same footing

**Visualisations:** - Heatmap of raw `W_rec` (signed: blue = negative/inhibitory weights, red = positive/excitatory weights) - Heatmap of `|W_rec|` (unsigned weight magnitude) - Binarised version (top X% absolute weights) placed side by side with a real brain connectivity matrix

**Conceptual bridge to GNMs:** - In Section 2 we asked: *what wiring rules generated the brain network?* - Here we ask: *what wiring did the network develop on its own, just by learning the task?* - Is task-driven learning alone enough to produce brain-like connectivity? Let's find out.

------------------------------------------------------------------------

### Tutorial: Making the RNN More Brain-Like — Biological Constraints

**Goal:** Add two biologically motivated constraints to the RNN training — a distance penalty (the direct analogue of GNM's η) and Dale's Law (excitatory/inhibitory sign separation). Train three models and compare their connectivity and performance.

**The three models:** 1. **Basic RNN** — standard training, no biological constraints 2. **Distance RNN** — distance penalty on `W_rec` (the λ/η analogue) 3. **Dale RNN** — distance penalty + Dale's Law (excitatory/inhibitory constraint)

#### Constraint 1: Distance Penalty (λ — the η analogue)

- Use the same **MNI brain coordinates** from the GNM default network (100 regions)

- Assign each hidden unit i to brain region i — the same distance matrix `D` used in Sections 1 and 2

- Add a distance regularisation term to the loss:

  `L_total = L_task + lambda * sum_ij( |W_rec_ij| * D_ij )`

  Large weights between spatially distant units are penalised — exactly what η does in the GNM

- λ is the direct analogue of η: higher λ → stronger distance penalty → more local, short-range connectivity

- Implementation: one extra line in the training loop

``` python
dist_penalty = lambda_dist * (torch.abs(model.W_rec.weight) * D_tensor).sum()
loss = task_loss + dist_penalty
```

#### Constraint 2: Dale's Law

- In real cortex, \~80% of neurons are **excitatory** (can only send positive signals to others) and \~20% are **inhibitory** (can only send negative signals)
- This is Dale's Principle: each neuron has a fixed sign for all its outgoing connections
- Implementation: assign each of the 100 units a sign at initialisation; enforce this sign after every gradient step

``` python
# Assign signs: 80 excitatory (+1), 20 inhibitory (-1)
signs = torch.ones(n_hidden)
signs[int(0.8 * n_hidden):] = -1

# After each gradient step, project weights back to sign-constrained space:
with torch.no_grad():
    model.W_rec.weight.data = torch.abs(model.W_rec.weight.data) * signs.unsqueeze(0)
```

- This constraint changes the structure of `W_rec` fundamentally: some columns must be entirely non-negative (excitatory units), others entirely non-positive (inhibitory units)
- It is biologically realistic, reduces model degrees of freedom, and produces a visually distinctive sign structure in the weight matrix

**What to compare across the three models:** - **Performance**: accuracy curves over training — does adding constraints hurt task performance? - **W_rec heatmaps**: visual side-by-side comparison of the three matrices - **Edge length distributions**: does the distance RNN shift toward shorter connections? - **Sign structure**: does the Dale RNN show the column-wise excitatory/inhibitory organisation visible in real cortex?

**Expected qualitative results:** - Basic RNN: high accuracy, arbitrary connectivity (long-range weights, mixed signs throughout) - Distance RNN: similar accuracy, connectivity shifted toward shorter-range connections - Dale RNN: slightly lower accuracy (more constrained), but `W_rec` structure that visually and topologically resembles real brain organisation more closely

**Conceptual message:** - Biological constraints do not necessarily hurt performance — the brain achieves high cognitive performance despite (or because of?) these constraints - The distance penalty is the exact same wiring principle that GNMs use to grow realistic brain networks; here the network learns to respect it during task training

------------------------------------------------------------------------

### Tutorial: Comparing RNNs and Brains

**Goal:** Quantitatively compare the connectivity structure of all three trained RNNs against real brain networks using the same metrics from Section 1 — closing the loop of the entire tutorial series.

**The four networks to compare:** 1. **Basic RNN** — `W_rec` binarised to match brain network density 2. **Real brain networks** — the GNM default network (or the 20 brains from Section 1)

**Topological metrics (all introduced in Tutorial: Topology):** - **Degree distribution**: are some units/regions disproportionately connected (hubs)? - **Betweenness centrality**: which units act as information bottlenecks? - **Clustering coefficient**: do connected units form dense local clusters (modularity)? - **Edge length distribution**: how long are the connections on average in MNI space?

**KS statistics — callback to the GNM energy function:** - Compute the same four KS statistics between each RNN and the real brain networks - This is the exact same energy formula used in Tutorial: Model Fitting (Section 2) - Rank the three models by total KS energy — lower energy = more brain-like connectivity

**Visualisations:** - Side-by-side heatmaps: Basic \| Distance \| Dale \| Real Brain - Violin plots of each topological metric, one group per network type - **Radar/spider plot**: all four metrics at once, one polygon per network type — immediate visual comparison of the full profile - Bar chart of KS energy scores across the three RNN models vs real brain baseline

**Narrative payoff — closing the series:** - Section 1 described brain networks with topology metrics - Section 2 grew brain-like networks using distance and homophily rules (GNMs) - Section 3 trained networks to do a cognitive task and asked whether they develop brain-like connectivity - The answer: task-driven learning alone is not enough — adding the same biological wiring constraints used in GNMs (distance penalty, Dale's Law) pushes the learned connectivity meaningfully closer to real brains - **The key message:** the brain's wiring constraints are not just metabolic necessities — they shape the kind of network that emerges from learning to perform cognitive tasks, and these constraints are the same whether you grow a network forward (GNM) or train it backward (RNN gradient descent)

------------------------------------------------------------------------

### Open Questions & Future Directions

- **Fit a GNM to the trained RNN weight matrices** — what η and γ best describe the Basic vs Dale RNN? Does the Dale RNN produce η/γ values closer to the real brain estimates from Section 2? This is a testable, potentially publishable comparison.
- **Add a homophily regulariser** (analogous to GNM's γ) — penalise or reward connections between units that share common partners, in addition to the distance penalty
- **Train on multiple tasks** — does the RNN develop more modular, brain-like connectivity when it must learn several cognitive tasks at once? (The core research question in Yang et al. 2019 and Khona et al. 2023)
- **Individual differences** — train 20 RNNs with slightly different noise levels (analogous to early life stress) and ask whether the resulting `W_rec` matrices show topology differences mirroring the brain differences from Section 1
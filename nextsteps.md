# GNM Tutorials — Next Steps & Full Plan

## General Setup

- **Create a dedicated Conda environment** that includes:
  - `GenerativeNetworkModels` (the GNM toolbox)
  - `jupyter` (so that Quarto can execute Python chunks during render)
  - Plotting libraries (e.g., `matplotlib`, `nilearn`, `networkx`)
  - Any other dependencies needed across all tutorials

- **We will NOT use real data!** All brain networks used throughout the tutorials will be **generated using the GNM toolbox itself**.

---

## Section 1: Brain Networks

### Tutorial: Introduction (DONE — `Introduction.qmd`)
- Conceptual intro to brain networks, graph theory, and connectivity matrices.
- Under the hood (without showing the code overtly), we produce **two figures**:
  1. A **3D brain connectivity plot** using `nilearn`, displaying the GNM default network with default coordinates.
  2. A **connectivity matrix** of the same data.
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

---

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

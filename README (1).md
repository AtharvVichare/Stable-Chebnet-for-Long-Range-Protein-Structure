# 🧬 Stable-ChebNet GVAE — Long-Range Protein Structure Analysis

> **Graph Variational Autoencoder with Stable Chebyshev Convolutions for Protein Anomaly Detection on the ENZYMES Dataset**

Based on:
- *Return of ChebNet: Understanding and Improving an Overlooked GNN on Long-Range Tasks* — Ali Hariri
- *Graph learning for capturing long-range dependencies in protein structures* — Ali Hariri, Pierre Vandergheynst (EPFL)

---

##  Overview

Proteins are inherently graph-structured: secondary structure elements (helices, sheets) form nodes, and edges encode amino-acid sequence proximity or spatial nearest-neighbour relations. Modelling such graphs requires capturing **long-range dependencies** — interactions between residues that are far apart in the sequence but spatially close — which standard MPNNs (GCN, GAT) fail to handle due to **over-squashing**.

This project applies **Stable-ChebNet**, a theoretically grounded spectral GNN, inside a **Graph Variational Autoencoder (GVAE)** to learn compact representations of protein graphs. Anomalous proteins (those not reflecting a valid catalysed chemical reaction for their class) produce high reconstruction and representational errors.

---

##  Architecture

```
Input Protein Graph G = (X, A)
         │
         ▼
┌─────────────────────────────────────────┐
│          Stable-ChebNet Encoder          │
│                                          │
│  X^(l+1) = X^(l) + ε · Σ_k T_k(L̃)    │
│            X^(l) (W_k − W_k^T − γI)    │
│                                          │
│  → Z_node  [N × d_latent]               │
│  → Z_g     [1 × d_latent]  (mean pool)  │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│               Decoder                    │
│  MLP(Z_node)  → X̂  (node features)     │
│  σ(Z·Zᵀ)     → Â  (adjacency)          │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│    Shared Stable-ChebNet Re-Encoder      │
│  (same weights as Encoder)               │
│  → Ẑ_node, Z_g'                         │
└─────────────────────────────────────────┘
         │
         ▼
   Loss = L1 + λ · L3
```

### Why Stable-ChebNet?

Classical ChebNet with large polynomial order K enables long-range communication (Theorem 2 in the paper) but suffers from **unstable training dynamics** — the Jacobian singular-value spectrum grows exponentially with K (Theorem 1).

Stable-ChebNet fixes this with two modifications:

| Modification | Effect |
|---|---|
| **Antisymmetric weights** `W_eff = W − Wᵀ − γI` | Jacobian eigenvalues become purely imaginary (Theorem 3) |
| **Forward-Euler discretisation** with step ε | `‖J‖₂ = 1 + O(ε²)` — no exponential growth or decay (Theorem 4) |

The result: arbitrarily large K for long-range protein interactions, with provably stable gradient dynamics.

---

##  Loss Functions

### L1 — Reconstruction Loss
```
L1 = ‖A − Â‖_F² + ‖X − X̂‖_F²
```
Measures how faithfully the model reconstructs both the graph topology (adjacency) and node features.

### L3 — Representational Loss
```
L_node  = (1/|G|) · Σ_g  (1/|V_g|) · Σ_i  ‖Z_node,i − Ẑ_node,i‖²

L_graph = (1/|G|) · Σ_g  ‖Z_g − Z_g'‖²

L3 = L_node + L_graph
```
Enforces **consistency** between the encoder's latent representation and the re-encoder's representation of the reconstructed graph. A high L3 indicates the reconstructed graph no longer preserves the structural properties of the original — the hallmark of an anomalous protein.

### Total Loss
```
L_total = L1 + λ · L3        (λ = 0.5 by default)
```

---

##  Dataset — ENZYMES

| Property | Value |
|---|---|
| Source | TU Dortmund / PyG `TUDataset` |
| Graphs | 600 protein graphs |
| Classes | 6 enzyme classes (EC top-level) |
| Nodes | Secondary structure elements (helices, sheets) |
| Node features | 18-dimensional (type + physical/chemical properties) |
| Edges | Amino-acid sequence neighbours OR 3 nearest spatial neighbours |
| Anomaly definition | Graphs not belonging to any of the 6 valid classes, or not reflecting the catalysed chemical reaction |

### Enzyme Classes

| Label | Class | Catalysis |
|---|---|---|
| 0 | Oxidoreductases | Redox reactions |
| 1 | Transferases | Group transfer |
| 2 | Hydrolases | Bond hydrolysis |
| 3 | Lyases | Bond cleavage (non-hydrolytic) |
| 4 | Isomerases | Structural rearrangement |
| 5 | Ligases | Bond formation (ATP-dependent) |

---

##  Project Structure

```
.
├── Stable_ChebNet_GVAE_ENZYMES.ipynb   ← Main notebook (run this)
├── README.md                            ← This file
└── dataset/
    └── ENZYMES/
        ├── raw/                         ← Auto-downloaded TUDataset files
        │   ├── ENZYMES_A.txt
        │   ├── ENZYMES_graph_indicator.txt
        │   ├── ENZYMES_graph_labels.txt
        │   ├── ENZYMES_node_attributes.txt
        │   └── ENZYMES_node_labels.txt
        └── processed/
            └── enzymes_data.pt          ← Cached PyG tensors
```

The dataset folder is **created automatically** on first run — no manual download needed.

---

##  Installation

### Requirements

```bash
pip install torch torchvision torchaudio
pip install torch_geometric
pip install torch_scatter torch_sparse   # PyG optional dependencies
pip install plotly kaleido
pip install scikit-learn scipy tqdm matplotlib numpy
```

### Quick start (Colab / local Jupyter)

```bash
# Clone or copy the notebook, then simply run all cells in order.
# Cell 1 handles all pip installs automatically.
jupyter notebook Stable_ChebNet_GVAE_ENZYMES.ipynb
```

---

##  Notebook Walkthrough

### Cell 1 — Install Dependencies
Auto-installs all required packages via `subprocess`.

### Cell 2 — Imports
PyTorch, PyG, Plotly, scikit-learn, tqdm.

### Cell 3 — ENZYMES PyG InMemoryDataset
Wraps `TUDataset` as a proper `InMemoryDataset` with:
- Raw files persisted to `dataset/ENZYMES/raw/`
- Processed tensors cached to `dataset/ENZYMES/processed/enzymes_data.pt`
- Prints per-class graph counts, node/edge statistics

### Cell 4 — 3D Protein Graph Visualisation
One interactive 3D Plotly graph per enzyme class. Node positions are derived by PCA on the 18-dim node feature vectors (PCA₁₋₃ as x, y, z). Node colour encodes feature type (helix vs. sheet proxy).

### Cell 5 — Dataset Statistics Dashboard
- Per-class degree distribution histograms
- Scatter plot: #nodes vs #edges, coloured by class

### Cell 6 — Dirichlet Energy & Best K Selection
Computes:
```
E(X) = tr(Xᵀ L X) / n
```
for K = 0..12 over 30 sampled protein graphs under both vanilla and Stable ChebNet propagation. Plots mean ± 1σ energy bands. Best K is chosen where the vanilla energy gradient `|∂E/∂k|` first stabilises.

### Cell 7 — StableChebNetLayer
Implements the exact update rule from Theorem 4:
```python
W_eff = W - W.T - gamma * I          # antisymmetric
X_out = X + eps * sum_k( T_k(L̃) @ X @ W_eff_k )
```
With Chebyshev recurrence `T_0=I, T_1=L̃, T_k = 2L̃T_{k-1} − T_{k-2}`.

### Cell 8 — StableChebGVAE
Full model with:
- `StableChebEncoder` (2 layers) → µ, logσ → Z_node via reparameterisation → Z_g via mean pooling
- `NodeDecoder` (3-layer MLP) → X̂
- Inner-product decoder → Â (sigmoid, threshold 0.5)
- Shared re-encoder on (X̂, Â) → Ẑ_node, Z_g'

### Cell 9 — Loss Functions
Exact implementations of L1 and L3 as specified.

### Cell 10 — Training
AdamW optimiser + ReduceLROnPlateau scheduler. Trains for 80 epochs (configurable). Saves best checkpoint by validation loss.

### Cell 11 — Training Curves
Three-panel Plotly figure: Total loss, L1, L3 for train and val.

### Cell 12 — Z_g vs Z_g' Visualisation
PCA-2D scatter of:
- Z_g (circles) — encoder graph embeddings
- Z_g' (crosses) — re-encoder graph embeddings
- Overlay comparison showing alignment/divergence

### Cell 13 — Original vs Reconstructed 3D Graphs
Side-by-side 3D Plotly for 4 test proteins:
- **Left**: Original G — true edges + original node features
- **Right**: Reconstructed Ĝ — decoded edges (threshold Â) + decoded X̂

### Cell 14 — Anomaly Score Analysis
Per-class box plots of anomaly score = `‖Z_g − Z_g'‖²`. Classes with consistently higher scores have more anomalous reconstructions.

### Cell 15 — Summary
Prints full theoretical summary, model config, and final metrics.

---

##  Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `K` | Auto (Dirichlet) | Chebyshev polynomial order |
| `eps` (ε) | 0.45 | Forward-Euler step size |
| `gamma` (γ) | 0.10 | Dissipative force for numerical stability |
| `hidden_dim` | 64 | Hidden layer dimension |
| `latent_dim` | 32 | Latent space dimension |
| `lr` | 1e-3 | AdamW learning rate |
| `epochs` | 80 | Training epochs |
| `batch_size` | 32 | Graphs per batch |
| `lambda_rep` | 0.5 | Weight for representational loss L3 |

These follow the values in `config_StableCheb.json` from the original Stable-ChebNet codebase (ε=0.45, γ=0.1).

---

##  Theoretical Background

### Over-Squashing in Protein GNNs
MPNNs aggregate information over local neighbourhoods. For a node to receive information from a node k hops away, the signal must pass through all intermediate nodes. With exponentially growing neighbourhoods, information gets "squashed" into fixed-size vectors — destroying long-range protein interaction signals that are critical for:
- Tertiary structure stabilisation
- Ligand binding site prediction
- Allosteric regulation modelling

### ChebNet vs GCN
GCN is a special case of ChebNet with K=1 and additional approximations. This strong **locality bias** makes GCN unable to capture long-range dependencies. ChebNet with K>1 directly captures K-hop structural information via the Chebyshev polynomial basis of the graph Laplacian.

### Stable-ChebNet Key Theorems

**Theorem 1** (Jacobian instability): For vanilla ChebNet, the mean squared singular values of the layer-wise Jacobian scale as `σ² · Σ_k λᵢ^{2k}` — growing exponentially with K.

**Theorem 3** (Purely imaginary eigenvalues): With antisymmetric weights and symmetric normalised Laplacian, the ODE Jacobian has `Re(λᵢ(J)) = 0` for all i.

**Theorem 4** (Non-exponential dynamics): With antisymmetric weights and step size ε, `‖J^(l)‖₂ = 1 + O(ε²)` — the Jacobian norm stays bounded across layers.

### GVAE Anomaly Detection
The core insight: a **normal** protein graph should reconstruct faithfully and the re-encoder should agree with the original encoder (`Z_g ≈ Z_g'`). An **anomalous** graph violates the structural patterns learned during training, producing:
- High `‖A − Â‖_F²` (topology mismatch)
- High `‖X − X̂‖_F²` (feature mismatch)  
- High `‖Z_g − Z_g'‖²` (representation inconsistency)

---

##  References

```bibtex
@article{hariri2024return,
  title={Return of ChebNet: Understanding and Improving an Overlooked GNN on Long-Range Tasks},
  author={Hariri, Ali},
  year={2024}
}

@article{hariri2022graph,
  title={Graph learning for capturing long-range dependencies in protein structures},
  author={Hariri, Ali and Vandergheynst, Pierre},
  institution={EPFL},
  year={2022}
}

@inproceedings{defferrard2016convolutional,
  title={Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering},
  author={Defferrard, Micha{\"e}l and Bresson, Xavier and Vandergheynst, Pierre},
  booktitle={NeurIPS},
  year={2016}
}

@article{schomburg2004brenda,
  title={BRENDA, the enzyme database: updates and major new developments},
  author={Schomburg, Ida and others},
  journal={Nucleic Acids Research},
  year={2004}
}
```

---

##  Author Notes

This implementation was developed for GSoC 2026 ML4Sci preparation, targeting the project *"Deep Graph Anomaly Detection with Contrastive Learning for New Physics Searches"*. The Stable-ChebNet GVAE framework here directly mirrors the architecture in the referenced papers, with the GVAE reconstruction + representational loss formulation designed to surface anomalous protein graphs that deviate from learned structural priors.

---

*Built with PyTorch Geometric · Plotly · ENZYMES (TU Dortmund)*

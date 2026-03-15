# GID: Graph Integrated Omics using Dual view Framework 

[![Status](https://img.shields.io/badge/Status-Pre--publication-orange.svg)]()
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-red.svg)]()

> **Note:** This repository contains the core methodology and modular codebase for an upcoming publication. Full datasets and pre-trained weights will be made publicly available upon peer-review completion.

## 📌 Overview
The **Graph Integrated Omics using Dual view (GID)** framework is a novel deep learning architecture designed for the seamless integration of heterogeneous multi-omics data. By leveraging both graph topology and global sample manifolds, GID learns highly informative representations of cancer samples and infers individualized, patient-specific regulatory networks.

![GID Architecture](https://github.com/Mohamedema/Graph-based-Omics-Integration-and-Dimensionality-reduction-GID/blob/f82e14725e5ed32a3313564de277491a24d7b5db/image.png?raw=true)
*Figure 1: The GID Architecture comprising a Graph Autoencoder (GAE) for feature topology and a parallel Sample Autoencoder (AE) for global manifold learning.*

## 🧠 Methodology & Architecture
The GID framework addresses the challenge of multi-omics integration through a dual-pathway, end-to-end optimizable approach:

1. **Graph Convolutional Network (GCN) Autoencoder:**  Learns feature topology by utilizing a weighted Protein-Protein Interaction (PPI) network (derived from STRING) as prior biological knowledge.
    Processes multi-omics node features via GCN layers equipped with Batch Normalization, PReLU activations, and residual linear projections to ensure stable gradient flow.
2. **Sample-Level Autoencoder:**  A parallel Multilayer Perceptron (MLP) autoencoder designed to capture global sample-specific signals from the concatenated multi-omics matrices.
3. **Hybrid Embedding via Attention Pooling:**
   * Node embeddings are aggregated using an expression-driven **Softmax pooling strategy**.
   * The pooled graph-topology vector is concatenated with the MLP sample embedding to form a unified, highly robust latent representation.

## 🔬 Single-Sample Network Inference (LIONESS)
A core capability of this framework is the extraction of **individualized patient networks**. Following the generation of a global continuous adjacency matrix via Cosine Similarity, we apply the **LIONESS** (Linear Interpolation to Obtain Network Estimates for Single Samples) mathematical framework. 


## 📊 Supported Data Modalities
The framework effectively integrates five distinct molecular layers:
* Transcriptomics (RNA-seq)
* Proteomics
* DNA Methylation
* CRISPR-Cas9 Gene Essentiality Screens
* Copy Number Variations (CNV - Handled via discrete cross-entropy loss)

## 📂 Repository Structure
The codebase adheres to software engineering best practices, ensuring high modularity and reproducibility:

```text
Graph-based-Omics-Integration-and-Dimensionality-reduction-GID/
│
├── outputs/               # Generate UMAPs, heatmaps, and CSV metrics
├── src/                   
│   ├── __init__.py
│   ├── dataset.py         # PyTorch CustomDataset and DataLoader preparation
│   ├── models.py          # GCN, MLP, and Dual-Autoencoder definitions
│   ├── train.py           # Training loop, loss computation, and optimization logic
│   ├── inference.py       # Memory-safe LIONESS network inference
│   └── evaluation.py      # Downstream analysis (CORUM validation, Clustering, UMAP)
│
├── main.py                # Main execution pipeline
├── environment.yml        # Conda environment dependencies
└── README.md              # Project documentation
```

🚀 Installation & Reproducibility
To ensure full reproducibility and avoid dependency conflicts (specifically with PyTorch Geometric and CUDA toolkits), we provide a Conda environment file.

1. Clone the repository:
git clone [https://github.com/Mohamedema/Graph-based-Omics-Integration-and-Dimensionality-reduction-GID.git](https://github.com/Mohamedema/Graph-based-Omics-Integration-and-Dimensionality-reduction-GID.git)
cd Graph-based-Omics-Integration-and-Dimensionality-reduction-GID

git clone [https://github.com/Mohamedema/Graph-based-Omics-Integration-and-Dimensionality-reduction-GID.git](https://github.com/Mohamedema/Graph-based-Omics-Integration-and-Dimensionality-reduction-GID.git)
cd Graph-based-Omics-Integration-and-Dimensionality-reduction-GID

2. Create and activate the Conda environment:

conda env create -f environment.yml
conda activate gid_env

3. Run the complete pipeline:

python main.py

📈 Downstream Analysis
The automated evaluation module (src/evaluation.py) performs rigorous biological and statistical validation:

Clustering Quality: Computes Calinski-Harabasz and Davies-Bouldin indices.

Latent Space Visualization: Generates UMAP projections of the hybrid embeddings.

Complex Density Analysis: Validates inferred networks against known human protein complexes (CORUM database), generating tissue-specific complex density heatmaps and calculating ROC-AUC scores.

🤝 Contact & Academic Use
This repository is currently maintained for peer-review and academic evaluation purposes. For inquiries regarding the methodology, codebase, or potential postdoctoral collaborations, please contact the author via GitHub.




import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap
from sklearn.metrics import roc_curve, auc, average_precision_score, calinski_harabasz_score, davies_bouldin_score

# =========================================================
# 1. CORUM-based Validation
# =========================================================
def run_corum_validation(continuous_adj_matrix, reconstructed_continuous_adj, corum_path="corum_humanComplexes.txt", output_dir="outputs/"):
    print("\nStarting CORUM-based Validation...")
    if not os.path.exists(corum_path):
        print(f"[WARNING] {corum_path} not found. Skipping CORUM validation.")
        return

    corum_df = pd.read_csv(corum_path, sep="\t")
    complex_col = "subunits_gene_name"

    positive_pairs = set()
    for proteins in corum_df[complex_col].dropna():
        genes = [g.strip() for g in proteins.split(";") if g.strip() != ""]
        for g1, g2 in itertools.combinations(genes, 2):
            positive_pairs.add(tuple(sorted([g1, g2])))

    protein_list = list(continuous_adj_matrix.index)
    n = len(protein_list)
    protein_to_idx = {p: i for i, p in enumerate(protein_list)}
    corum_adj = np.zeros((n, n), dtype=int)

    for g1, g2 in positive_pairs:
        if g1 in protein_to_idx and g2 in protein_to_idx:
            i, j = protein_to_idx[g1], protein_to_idx[g2]
            corum_adj[i, j] = 1
            corum_adj[j, i] = 1

    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    y_true = corum_adj[mask]
    y_scores = reconstructed_continuous_adj[mask]

    if len(np.unique(y_true)) < 2:
        print("WARNING: Only one class found. ROC cannot be computed.")
        return

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    ap = average_precision_score(y_true, y_scores)

    pd.DataFrame({"FPR": fpr, "TPR": tpr}).to_csv(f"{output_dir}all_omics_corum_model_validation.csv", index=False)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve: Reconstruction vs CORUM')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(f"{output_dir}corum_roc_curve.png")
    plt.close()
    
    print(f"CORUM ROC AUC: {roc_auc:.4f} | PR AUC: {ap:.4f}")

# =========================================================
# 2. Tissue-Specific Complex Analysis (Density)
# =========================================================
# =========================================================
# 2. Tissue-Specific Complex Analysis (Density)
# =========================================================
def run_tissue_complex_analysis(continuous_adj_matrix, reconstructed_adj, tissue_labels, corum_path="corum_humanComplexes.txt", output_dir="outputs/"):
    print("\nStarting Tissue-Specific Complex Analysis...")
    if not os.path.exists(corum_path):
        return

    # 1. Create merged_tissue_adj dynamically
    unique_tissues = np.unique(tissue_labels)
    tissue_adj_matrices = {t: [] for t in unique_tissues}
    for i, t in enumerate(tissue_labels):
        tissue_adj_matrices[t].append(reconstructed_adj[i])
        
    merged_tissue_adj = {}
    for t in unique_tissues:
        if len(tissue_adj_matrices[t]) > 0:
            avg_adj = np.mean(tissue_adj_matrices[t], axis=0)
            merged_tissue_adj[t] = (avg_adj + avg_adj.T) / 2

    # 2. Load CORUM and Map
    corum_df = pd.read_csv(corum_path, sep="\t")
    protein_to_idx = {p: i for i, p in enumerate(continuous_adj_matrix.index)}
    complex_dict = {}
    
    for idx, row in corum_df.iterrows():
        if pd.isna(row["subunits_gene_name"]): continue
        genes = [g.strip() for g in row["subunits_gene_name"].split(";") if g.strip() != ""]
        genes_in_network = [g for g in genes if g in protein_to_idx]
        if len(genes_in_network) >= 3:
            complex_name = f"{row.get('ComplexName', 'Complex')}_{idx}"
            complex_dict[complex_name] = genes_in_network

    # 3. Binarize & Score
    binary_tissue_adj = {}
    for t, adj in merged_tissue_adj.items():
        adj_proc = adj.copy()
        np.fill_diagonal(adj_proc, 0)
        threshold = np.percentile(adj_proc, 95)
        binary_tissue_adj[t] = (adj_proc >= threshold).astype(int)

    complex_scores = []
    for c_name, genes in complex_dict.items():
        indices = [protein_to_idx[g] for g in genes]
        n_nodes = len(indices)
        possible_edges = (n_nodes * (n_nodes - 1)) / 2
        if possible_edges == 0: continue
        
        for t, bin_adj in binary_tissue_adj.items():
            sub_adj = bin_adj[np.ix_(indices, indices)]
            actual_edges = sub_adj[np.triu_indices_from(sub_adj, k=1)].sum()
            complex_scores.append({"Complex": c_name, "Tissue": t, "Score": actual_edges / possible_edges})

    if not complex_scores:
        print("No complexes matched for density analysis.")
        return

    complex_df = pd.DataFrame(complex_scores)
    heatmap_df = complex_df.pivot(index="Complex", columns="Tissue", values="Score")
    
    # Z-score normalization (Your original implementation)
    heatmap_z = (heatmap_df - heatmap_df.mean(axis=1).values.reshape(-1,1)) / \
                heatmap_df.std(axis=1).values.reshape(-1,1)
    
    # CLEANING: Replace Inf/NaN with 0
    heatmap_z = heatmap_z.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    heatmap_z.to_csv(f"{output_dir}complex_tissue_density_zscores.csv")

    # =========================================================
    # 7) Extract Tissue-Specific Complexes (Z > 2)
    # =========================================================
    tissue_specific = []
    for complex_name in heatmap_z.index:
        row = heatmap_z.loc[complex_name]
        max_tissue = row.idxmax()
        max_value = row.max()
        
        if max_value > 2:
            tissue_specific.append({
                "Complex": complex_name,
                "Specific_Tissue": max_tissue,
                "Z_score": max_value
            })

    tissue_specific_df = pd.DataFrame(tissue_specific)
    if not tissue_specific_df.empty:
        tissue_specific_df = tissue_specific_df.sort_values("Z_score", ascending=False)
        tissue_specific_df.to_csv(f"{output_dir}tissue_specific_complexes_density.csv", index=False)
    
    # =========================================================
    # 8) Heatmap of All Complexes
    # =========================================================
    heatmap_z_clean = heatmap_z.loc[~(heatmap_z == 0).all(axis=1)]
    if len(heatmap_z_clean) > 1:
        plt.figure(figsize=(16, 14))
        try:
            sns.clustermap(heatmap_z_clean, cmap="coolwarm", metric="cosine", method="average", figsize=(16, 14), xticklabels=True, yticklabels=False)
            plt.savefig(f"{output_dir}complex_tissue_heatmap_density_all.png", dpi=300)
        except Exception:
            sns.clustermap(heatmap_z_clean, cmap="coolwarm", metric="euclidean", method="average", figsize=(16, 14), xticklabels=True, yticklabels=False)
            plt.savefig(f"{output_dir}complex_tissue_heatmap_density_all_euclidean.png", dpi=300)
        plt.close()

    # =========================================================
    # 9) Heatmap of Tissue-Specific Only (The Missing Plot!)
    # =========================================================
    if len(tissue_specific_df) > 1:
        specific_complex_names = tissue_specific_df["Complex"].values
        valid_names = [n for n in specific_complex_names if n in heatmap_z_clean.index]
        
        if len(valid_names) > 1:
            heatmap_specific = heatmap_z_clean.loc[valid_names]
            
            plt.figure(figsize=(16, 12))
            sns.clustermap(heatmap_specific, cmap="coolwarm", metric="cosine", method="average", figsize=(16, 12), xticklabels=True, yticklabels=True)
            plt.savefig(f"{output_dir}complex_tissue_heatmap_density_specific.png", dpi=300)
            plt.close()
            
    print("Tissue-Specific Complex Analysis Complete.")

# =========================================================
# 3. Hybrid Embeddings & Metrics
# =========================================================
def run_embedding_analysis(model, sample_ae, reshaped_data, scaled_data, edge_index, edge_weight, final_meta_data2, device, output_dir="outputs/"):
    print("\nStarting Hybrid Embeddings & Clustering Metrics...")
    model.eval()
    sample_ae.eval()
    
    labels = final_meta_data2['tissue'].values
    G, L, num_samples = 2002, 5, scaled_data.shape[1]

    with torch.no_grad():
        # GCN H
        _, H = model(reshaped_data.to(device), edge_index.to(device), edge_weight.to(device))
        # GCN H
        # Softmax Pooling
        sample_data_tensor = torch.FloatTensor(scaled_data).view(L, G, num_samples).mean(dim=0).t().to(device)
        attention_weights = F.softmax(sample_data_tensor, dim=1)
        h_pooled = torch.mm(attention_weights, H)
        
        # Sample AE Z
        sample_input_tensor = torch.FloatTensor(scaled_data.T).to(device)
        _, z_emb = sample_ae(sample_input_tensor)

        # Hybrid
        h_pooled_norm = F.normalize(h_pooled, p=2, dim=1)
        z_emb_norm = F.normalize(z_emb, p=2, dim=1)
        hybrid_emb = torch.cat([h_pooled_norm, z_emb_norm], dim=1)

    # UMAP for Hybrid
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, metric='cosine', random_state=42)
    umap_coords = reducer.fit_transform(hybrid_emb.cpu().numpy())
    
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x=umap_coords[:,0], y=umap_coords[:,1], hue=labels, palette='tab20', s=60, alpha=0.8, edgecolor='w')
    plt.title('UMAP of Hybrid Embeddings (Graph H + Sample Z)')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_dir}umap_hybrid_concat.png", dpi=300)
    plt.close()

    # Metrics
    results = []
    for name, emb in [("Sample_AE", z_emb.cpu().numpy()), ("GCN_Pooled", h_pooled.cpu().numpy()), ("Hybrid", hybrid_emb.cpu().numpy())]:
        results.append({
            "Embedding": name,
            "Calinski_Harabasz": calinski_harabasz_score(emb, labels),
            "Davies_Bouldin": davies_bouldin_score(emb, labels)
        })
        
    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(f"{output_dir}embedding_clustering_metrics.csv", index=False)
    
    metrics_melted = metrics_df.melt(id_vars="Embedding", value_vars=["Calinski_Harabasz", "Davies_Bouldin"], var_name="Metric", value_name="Score")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=metrics_melted, x="Embedding", y="Score", hue="Metric")
    plt.title("Clustering Performance per Embedding Type")
    plt.tight_layout()
    plt.savefig(f"{output_dir}embedding_barplot.png", dpi=300)
    plt.close()
    
    print("Embedding Analysis Complete.")
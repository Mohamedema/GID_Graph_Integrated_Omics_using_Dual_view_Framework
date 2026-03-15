import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap
from scipy.stats import pearsonr
from sklearn.metrics import roc_curve, auc

# ==========================================
# Visualization & Validation Utilities
# ==========================================

def plot_training_metrics(train_losses, sample_losses, ch_scores, db_scores, output_dir="outputs/"):
    """Plots training losses and clustering scores."""
    # 1. Training Loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='GCN Reconstruction Loss')
    plt.plot(sample_losses, label='Sample AE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss (Dual View)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}pic1_training_loss.png")
    plt.close()

    # 2. Clustering Metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.plot(ch_scores)
    ax1.set_title('Calinski-Harabasz Score')
    ax1.set_xlabel('Epoch')
    ax1.grid(True)
    
    ax2.plot(db_scores)
    ax2.set_title('Davies-Bouldin Score')
    ax2.set_xlabel('Epoch')
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}pic3_metrics.png")
    plt.close()

def plot_umap(all_latent_np, tissue_labels, output_dir="outputs/"):
    """Generates UMAP for the latent space."""
    reducer = umap.UMAP(random_state=42)
    all_umap = reducer.fit_transform(all_latent_np)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=all_umap[:, 0], y=all_umap[:, 1], hue=tissue_labels, palette='tab20', s=50)
    plt.title('UMAP of Sample Latent Space')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_dir}pic2_umap.png")
    plt.close()

def plot_tissue_networks_umap(reconstructed_adj, tissue_labels, output_dir="outputs/"):
    """Merges tissue networks and plots UMAP + Clustered Heatmap."""
    unique_tissues = np.unique(tissue_labels)
    num_tissues = len(unique_tissues)
    num_sam = len(tissue_labels)

    tissue_adj_matrices = {tissue: [] for tissue in unique_tissues}
    for i in range(num_sam):
        tissue = tissue_labels[i]
        tissue_adj_matrices[tissue].append(reconstructed_adj[i])

    merged_tissue_adj = {}
    for tissue in unique_tissues:
        if len(tissue_adj_matrices[tissue]) > 0:
            merged_tissue_adj[tissue] = np.mean(tissue_adj_matrices[tissue], axis=0)
            merged_tissue_adj[tissue] = (merged_tissue_adj[tissue] + merged_tissue_adj[tissue].T) / 2

    tissue_adj_array = np.array([merged_tissue_adj[tissue] for tissue in unique_tissues])
    tissue_adj_flat = tissue_adj_array.reshape(num_tissues, -1)

    # UMAP
    adj_reducer = umap.UMAP(random_state=42, metric='cosine')
    adj_umap = adj_reducer.fit_transform(tissue_adj_flat)

    plt.figure(figsize=(12, 10))
    sns.scatterplot(x=adj_umap[:, 0], y=adj_umap[:, 1], hue=unique_tissues, palette='tab20', s=100)
    plt.title('UMAP of Tissue-Specific PPI Networks')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_dir}pic10.png")
    plt.close()

    # Clustered Heatmap
    tissue_df = pd.DataFrame(tissue_adj_flat, index=unique_tissues)
    sns.clustermap(
        tissue_df, cmap="viridis", metric="cosine", method="average", 
        figsize=(16, 10), row_cluster=True, col_cluster=False, 
        xticklabels=False, yticklabels=True, cbar_kws={"label": "Mean Inferred Edge Weight"}
    )
    plt.suptitle("Clustered Heatmap of Merged Tissue-Specific Inferred Adjacencies", fontsize=16)
    plt.savefig(f"{output_dir}pic12.png")
    plt.close()

def validate_and_plot_roc(continuous_adj, reconstructed_continuous_adj, output_dir="outputs/", binary_threshold=0.7):
    """Calculates Pearson correlation, ROC AUC, and plots distributions."""
    original_values = continuous_adj.flatten()
    reconstructed_values = reconstructed_continuous_adj.flatten()

    pearson_corr, p_value = pearsonr(original_values, reconstructed_values)
    print(f"Global Pearson Correlation: {pearson_corr:.4f} (p-value: {p_value:.4g})")

    y_true = (original_values >= binary_threshold).astype(int)
    y_scores = reconstructed_values

    if len(np.unique(y_true)) >= 2:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})
        roc_df.to_csv(f"{output_dir}all_omics_string_model_validation.csv", index=False)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title(f'ROC Curve: Reconstruction vs Original (Threshold >= {binary_threshold})')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}pic16_roc_curve.png")
        plt.close()
    else:
        print("WARNING: Threshold too high/low. Only one class found. Skipping ROC.")

    # Error Histogram
    errors = np.abs(original_values - reconstructed_values)
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, color='teal', alpha=0.7)
    plt.xlabel('Absolute Reconstruction Error (|Original - Reconstructed|)')
    plt.ylabel('Frequency (Edges)')
    plt.title('Distribution of Reconstruction Errors')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}pic15_error_hist.png")
    plt.close()

    # Correlation Scatter (Sampled)
    plt.figure(figsize=(10, 8))
    sample_indices = np.random.choice(len(original_values), size=min(50000, len(original_values)), replace=False)
    plt.scatter(original_values[sample_indices], reconstructed_values[sample_indices], alpha=0.3, s=5)
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Reconstruction')
    plt.xlabel('Original Edge Weights')
    plt.ylabel('Reconstructed Edge Weights')
    plt.title(f'Reconstruction Correlation (Sampled)\nPearson r = {pearson_corr:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}pic14_correlation.png")
    plt.close()

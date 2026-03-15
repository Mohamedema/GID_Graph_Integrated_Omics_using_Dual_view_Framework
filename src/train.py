import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from torch_geometric.utils import subgraph

# ==========================================
# Training Loop Function
# ==========================================

def train_models(
    gcn_model, sample_ae, 
    train_loader, sample_loader, sample_iter,  
    sample_data_transposed, tissue_labels, 
    edge_index, edge_weight, 
    G, split_point_gcn, split_point_sae,
    num_epochs=1000, learning_rate=0.000289, weight_decay=0.000108,
    device='cpu'
):
    # 1. Setup Optimizers and Loss
    gcn_optimizer = optim.Adam(gcn_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    sample_optimizer = optim.Adam(sample_ae.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss(reduction='none')


    # Move to device (GPU/CPU)
    gcn_model = gcn_model.to(device)
    sample_ae = sample_ae.to(device)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    sample_data_transposed = sample_data_transposed.to(device)

    # 2. Tracking metrics
    train_losses = []
    sample_losses = []
    ch_scores = []
    db_scores = []

    sample_iter = iter(sample_loader)

    print(f"Starting Training on {device}...")
    
    for epoch in tqdm(range(num_epochs), desc="Training"):
        gcn_model.train()
        sample_ae.train()
        running_train_loss = 0.0
        running_sample_loss  = 0.0
        
        # Iterate over Genes (GCN)
        for features, mask, batch_indices in train_loader:
            features = features.to(device)
            mask = mask.to(device)
            batch_indices = batch_indices.to(device)

            gcn_optimizer.zero_grad()
            sample_optimizer.zero_grad()
            
            # --- A. GCN TRAIN STEP ---
            batch_edge_index, batch_edge_weight = subgraph(
                batch_indices, edge_index, edge_attr=edge_weight,
                relabel_nodes=True, num_nodes=G
            )
            
            outputs, _ = gcn_model(features, batch_edge_index, batch_edge_weight)
            valid_mask = ~mask
            
            # Continuous Loss
            mse_gcn = criterion(outputs[:, :split_point_gcn], features[:, :split_point_gcn])
            mse_gcn = mse_gcn[valid_mask[:, :split_point_gcn]].mean() if valid_mask[:, :split_point_gcn].any() else 0.0
            
            # Discrete Loss
            mask_disc = valid_mask[:, split_point_gcn:]
            if mask_disc.any():
                pred_disc = outputs[:, split_point_gcn:][mask_disc]
                true_disc = features[:, split_point_gcn:][mask_disc]
                true_cls = (true_disc + 2).long().clamp(0, 4)
                centers = torch.tensor([-2., -1., 0., 1., 2.], device=device).view(1, 5)
                logits = -1.0 * (pred_disc.unsqueeze(1) - centers) ** 2
                ce_gcn = F.cross_entropy(logits, true_cls)
            else:
                ce_gcn = torch.tensor(0., device=device)
                
            recon_loss = mse_gcn + ce_gcn

            # --- B. SAMPLE AE TRAIN STEP ---
            try:
                sample_batch, mask_batch = next(sample_iter)
            except StopIteration:
                sample_iter = iter(sample_loader)
                sample_batch, mask_batch = next(sample_iter)
                
            sample_batch = sample_batch.to(device)
            mask_batch = mask_batch.to(device)

            sample_recon, _ = sample_ae(sample_batch)
            sample_valid_mask = ~mask_batch.bool()
            
            # Continuous Loss
            mse_sae = criterion(sample_recon[:, :split_point_sae], sample_batch[:, :split_point_sae])
            mse_sae = mse_sae[sample_valid_mask[:, :split_point_sae]].mean() if sample_valid_mask[:, :split_point_sae].any() else 0.0
            
            # Discrete Loss
            mask_disc_s = sample_valid_mask[:, split_point_sae:]
            if mask_disc_s.any():
                pred_disc_s = sample_recon[:, split_point_sae:][mask_disc_s]
                true_disc_s = sample_batch[:, split_point_sae:][mask_disc_s]
                true_cls_s = (true_disc_s + 2).long().clamp(0, 4)
                centers_s = torch.tensor([-2., -1., 0., 1., 2.], device=device).view(1, 5)
                logits_s = -1.0 * (pred_disc_s.unsqueeze(1) - centers_s) ** 2
                ce_sae = F.cross_entropy(logits_s, true_cls_s)
            else:
                ce_sae = torch.tensor(0., device=device)
                
            sample_loss_val = mse_sae + ce_sae

            # --- C. BACKPROP ---
            loss = recon_loss + sample_loss_val
            loss.backward()
            gcn_optimizer.step()
            sample_optimizer.step()
            
            running_train_loss += recon_loss.item()
            running_sample_loss += sample_loss_val.item()

        # Store Metrics
        train_losses.append(running_train_loss / len(train_loader))
        sample_losses.append(running_sample_loss / len(train_loader))
        
        # Calculate Clustering Scores (On Sample Embeddings)
        sample_ae.eval()
        with torch.no_grad():
            _, full_z = sample_ae(sample_data_transposed)
            all_latent_np = full_z.cpu().numpy()
            
            if len(np.unique(tissue_labels)) > 1: # Safety check
                ch_scores.append(calinski_harabasz_score(all_latent_np, tissue_labels))
                db_scores.append(davies_bouldin_score(all_latent_np, tissue_labels))
            else:
                ch_scores.append(0)
                db_scores.append(0)

    print("Training Complete!")
    return train_losses, sample_losses, ch_scores, db_scores, all_latent_np

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import gc

# ==========================================
# 1. Helper Function: Cosine Similarity
# ==========================================
def get_similarity_matrix(H_emb):
    H_norm = F.normalize(H_emb, p=2, dim=1)
    adj = torch.mm(H_norm, H_norm.t())
    return adj

# ==========================================
# 2. Main LIONESS Inference Function (CPU Fallback for Full Model)
# ==========================================
def run_lioness_inference(model, full_features_tensor, edge_index, edge_weight, num_sam, G, L, device='cpu'):
    """
    Runs LIONESS logic using the FULL model (Encoder + Decoder) 
    forced on the CPU to prevent CUDA Out Of Memory errors.
    """
    model.eval()
    

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()

    print("Moving model and data to CPU for memory-safe Inference...")

    model = model.to('cpu')
    features_cpu = full_features_tensor.to('cpu')
    edge_index_cpu = edge_index.to('cpu')
    edge_weight_cpu = edge_weight.to('cpu')

    with torch.no_grad():
        # --- PART A: Population-Level Reconstruction ---
        print("Computing Full Network Baseline (Full Model)...")
        

        _, full_H = model(features_cpu, edge_index_cpu, edge_weight_cpu)
        adj_full_tensor = get_similarity_matrix(full_H)
        
        reconstructed_continuous_adj = adj_full_tensor.numpy()
        reconstructed_continuous_adj = 0.5 * (reconstructed_continuous_adj + reconstructed_continuous_adj.T)
        
        del full_H

        # --- PART B: Sample-Specific Inference (LIONESS) ---
        print("Starting Sample-Specific Inference (LIONESS)...")
        reconstructed_adj = np.zeros((num_sam, G, G), dtype=np.float32)
        N = num_sam
        
        for i in tqdm(range(num_sam), desc="Inferring Single Samples"):
            features_minus_i = features_cpu.clone()
            indices_to_zero = [i + (layer * num_sam) for layer in range(L)]
            features_minus_i[:, indices_to_zero] = 0.0 
            

            _, H_minus_i = model(features_minus_i, edge_index_cpu, edge_weight_cpu)
            
            adj_minus_i_tensor = get_similarity_matrix(H_minus_i)
            
            lioness_matrix = (N * adj_full_tensor) - ((N - 1) * adj_minus_i_tensor)
            reconstructed_adj[i] = lioness_matrix.numpy()
            

            del features_minus_i, H_minus_i, adj_minus_i_tensor, lioness_matrix


    model = model.to(device)
    
    return reconstructed_continuous_adj, reconstructed_adj

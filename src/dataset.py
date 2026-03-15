import torch
from torch.utils.data import Dataset

# ==========================================
# Custom Dataset Class for Omics/Graph Data
# ==========================================

class CustomDataset(Dataset):
    """
    A custom PyTorch Dataset for handling features, masks, and optional metadata.
    Useful for GCN and Autoencoder batching.
    """
    def __init__(self, features, mask, metadata=None):
        self.features = torch.FloatTensor(features)
        self.mask = torch.BoolTensor(mask)
        self.metadata = metadata if metadata is None else torch.FloatTensor(metadata)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.metadata is None:
            return self.features[idx], self.mask[idx], idx
        return self.features[idx], self.mask[idx], self.metadata[idx]

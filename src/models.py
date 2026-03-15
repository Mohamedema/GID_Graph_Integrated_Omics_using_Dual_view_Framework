import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

# ==========================================
# 1. Multi-Layer Perceptron (MLP) Blocks
# ==========================================

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout_rate=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.prelu = nn.PReLU()
        self.residual_transform = nn.Linear(input_dim, latent_dim)

    def forward(self, x):
        x_initial = self.residual_transform(x)       
        x = self.fc1(x)
        x = self.prelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x + x_initial                         

class MLPDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, dropout_rate=0.3):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.residual_transform = nn.Linear(latent_dim, output_dim)

    def forward(self, z):
        x_initial = self.residual_transform(z)       
        x = self.fc1(z)
        x = self.prelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x + x_initial

class SampleAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout_rate=0.3):
        super(SampleAutoencoder, self).__init__()
        self.encoder = MLPEncoder(input_dim, hidden_dim, latent_dim, dropout_rate)
        self.decoder = MLPDecoder(latent_dim, hidden_dim, input_dim, dropout_rate)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

# ==========================================
# 2. Graph Convolutional Network (GCN) Blocks
# ==========================================

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout_rate=0.3):
        super(Encoder, self).__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, latent_dim)
        self.bn2 = nn.BatchNorm1d(latent_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.prelu = nn.PReLU()
        self.residual_transform = nn.Linear(input_dim, latent_dim)

    def forward(self, x, edge_index, edge_weight):
        x_initial = self.residual_transform(x)
        x = self.gcn1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.dropout(x)
        x = self.gcn2(x, edge_index, edge_weight)
        x = self.bn2(x)
        return x + x_initial

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, dropout_rate=0.3):
        super(Decoder, self).__init__()
        self.gcn1 = GCNConv(latent_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.residual_transform = nn.Linear(latent_dim, output_dim)

    def forward(self, x, edge_index, edge_weight):
        x_initial = self.residual_transform(x)
        x = self.gcn1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.dropout(x)
        x = self.gcn2(x, edge_index, edge_weight)
        x = self.bn2(x)
        return x + x_initial

class WeightedGCNAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout_rate=0.3):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, dropout_rate)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, dropout_rate)

    def forward(self, x, edge_index, edge_weight):
        H = self.encoder(x, edge_index, edge_weight)
        x_recon = self.decoder(H, edge_index, edge_weight)
        return x_recon, H  

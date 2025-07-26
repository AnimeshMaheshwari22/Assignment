import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- Define required args only ---
class Args:
    use_color = False
    enc_dim = 256
    preenc_npoints = 2048
    enc_type = "vanilla"
    enc_nhead = 4
    enc_ffn_dim = 128
    enc_dropout = 0.1
    enc_activation = "relu"
    mlp_dropout = 0.3
    nqueries = 256
    pos_embed = "fourier"
    enc_nlayers = 3

args = Args()

# --- 3D Deformable Self-Attention Module ---
# This is a complete, self-contained implementation.
class MultiHeadDeformableAttention3D(nn.Module):
    """
    Implements 3D Deformable Self-Attention, inspired by Deformable DETR.
    This module samples features from a point cloud at locations predicted by the model.
    """
    def __init__(self, embed_dim, num_heads, num_sampling_points, k_neighbors=4):
        """
        Args:
            embed_dim (int): The embedding dimension of the input features.
            num_heads (int): The number of attention heads.
            num_sampling_points (int): The number of points to sample for each query per head.
            k_neighbors (int): The number of nearest neighbors for Inverse Distance Weighting (IDW) sampling.
        """
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_sampling_points = num_sampling_points
        self.k_neighbors = k_neighbors
        self.head_dim = embed_dim // num_heads

        self.sampling_offset_proj = nn.Linear(embed_dim, num_heads * num_sampling_points * 3)
        self.attention_weights_proj = nn.Linear(embed_dim, num_heads * num_sampling_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query_features, reference_points):
        """
        Args:
            query_features (Tensor): Input features. Shape: (N, L, E)
            reference_points (Tensor): 3D coordinates of the input points. Shape: (N, L, 3)
        Returns:
            Tensor: The output features after attention. Shape: (N, L, E)
        """
        N, L, E = query_features.shape
        device = query_features.device

        value = self.value_proj(query_features)
        value = value.view(N, L, self.num_heads, self.head_dim).transpose(1, 2)

        sampling_offsets = self.sampling_offset_proj(query_features)
        sampling_offsets = sampling_offsets.view(N, L, self.num_heads, self.num_sampling_points, 3).transpose(1, 2)

        attention_weights = self.attention_weights_proj(query_features)
        attention_weights = attention_weights.view(N, L, self.num_heads, self.num_sampling_points).transpose(1, 2)
        attention_weights = F.softmax(attention_weights, dim=-1)

        sampling_locations = reference_points.unsqueeze(1).unsqueeze(3) + sampling_offsets

        sampling_locations_flat = sampling_locations.reshape(N * self.num_heads, L * self.num_sampling_points, 3)
        reference_points_flat = reference_points.repeat(self.num_heads, 1, 1)

        dists = torch.cdist(sampling_locations_flat, reference_points_flat)
        topk_dists, topk_indices = torch.topk(dists, self.k_neighbors, dim=-1, largest=False)

        idw_weights = 1.0 / (topk_dists + 1e-8)
        idw_weights = idw_weights / idw_weights.sum(dim=-1, keepdim=True)

        value_flat = value.transpose(0, 1).reshape(N * self.num_heads, L, self.head_dim)
        batch_indices = torch.arange(N * self.num_heads, device=device).view(-1, 1, 1)
        neighbor_features = value_flat[batch_indices, topk_indices]

        sampled_features = (neighbor_features * idw_weights.unsqueeze(-1)).sum(dim=2)
        sampled_features = sampled_features.view(N, self.num_heads, L, self.num_sampling_points, self.head_dim)

        output = (sampled_features * attention_weights.unsqueeze(-1)).sum(dim=3)
        output = output.transpose(1, 2).contiguous().view(N, L, E)
        output = self.output_proj(output)

        return output

try:
    # This block will run if the model file is available
    from models.pointnet_enc import build_pointnetencoder

    pc_path = "D:\Assignment\scripts\pc_1.npy"  # <--- REPLACE WITH YOUR FILE PATH
    pc_np = np.load(pc_path)
    pc_np = pc_np.transpose(1, 2, 0)
    pc_np = pc_np.reshape(-1, 3)
    print("Point cloud shape is: ", pc_np.shape)
    pc_tensor = torch.tensor(pc_np, dtype=torch.float32).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pc_tensor = pc_tensor.to(device)
    
    # --- Build and run model ---
    model = build_pointnetencoder(args)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        inputs = {"point_clouds": pc_tensor}
        enc_xyz, enc_features, enc_inds = model.run_encoder(inputs["point_clouds"])

except (ImportError, FileNotFoundError):
    # This block runs as a fallback if the model file isn't found
    print("Could not load PointNet model. Using dummy data for demonstration.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc_xyz = torch.randn(1, args.preenc_npoints, 3).to(device)
    enc_features = torch.randn(args.preenc_npoints, 1, args.enc_dim).to(device)


# --- Apply Deformable Self-Attention ---
print("Applying 3D Deformable Self-Attention...")

# 1. Prepare inputs for the attention module
# The module expects (Batch, Seq_len, Dim), so we permute the features
query_features = enc_features.permute(1, 0, 2)
reference_points = enc_xyz # Already in the correct format

print(f"Permuted Feature shape (Input to Attention): {query_features.shape}")

# 2. Define attention hyperparameters
# These can be tuned based on your model's needs
D_MODEL = args.enc_dim
N_HEADS = args.enc_nhead
N_SAMPLING_POINTS = 4   # A common choice for deformable attention
K_NEIGHBORS_FOR_IDW = 4 # Neighbors for interpolation

# 3. Instantiate and run the attention layer
deformable_attn_layer = MultiHeadDeformableAttention3D(
    embed_dim=D_MODEL,
    num_heads=N_HEADS,
    num_sampling_points=N_SAMPLING_POINTS,
    k_neighbors=K_NEIGHBORS_FOR_IDW
).to(device)

with torch.no_grad():
    # Apply the attention layer to the encoder's output
    final_features = deformable_attn_layer(query_features, reference_points)


# --- Final Output ---
print("\n✅ Deformable Self-Attention applied successfully!")
print(f"Final Features shape: {final_features.shape}")

# The output shape should match the input feature shape
assert final_features.shape == query_features.shape
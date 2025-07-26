import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms

# --- Helper Function to Generate 2D Reference Points ---
def get_reference_points(H, W, device):
    y, x = torch.meshgrid(
        torch.linspace(0.5, H - 0.5, H, device=device),
        torch.linspace(0.5, W - 0.5, W, device=device),
        indexing="ij",
    )
    y = y / H
    x = x / W
    ref_points = torch.stack((x, y), -1)
    return ref_points

# --- 2D Deformable Self-Attention Module (Corrected) ---
class MultiHeadDeformableAttention2D(nn.Module):
    def __init__(self, embed_dim, num_heads, num_sampling_points):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_sampling_points = num_sampling_points
        self.head_dim = embed_dim // num_heads

        self.sampling_offset_proj = nn.Linear(embed_dim, num_heads * num_sampling_points * 2)
        self.attention_weights_proj = nn.Linear(embed_dim, num_heads * num_sampling_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, reference_points):
        N, H, W, E = query.shape
        L = H * W

        sampling_offsets = self.sampling_offset_proj(query)
        sampling_offsets = sampling_offsets.view(N, L, self.num_heads, self.num_sampling_points, 2).transpose(1, 2)

        attention_weights = self.attention_weights_proj(query)
        attention_weights = attention_weights.view(N, L, self.num_heads, self.num_sampling_points).transpose(1, 2)
        attention_weights = F.softmax(attention_weights, dim=-1)

        sampling_locations = reference_points.view(N, 1, L, 1, 2) + sampling_offsets
        
        query_for_sampling = query.permute(0, 3, 1, 2)
        
        # Reshape grid for grid_sample
        sampling_grid = (sampling_locations.reshape(N * self.num_heads, L, self.num_sampling_points, 2) * 2) - 1

        # We need to sample from the query for each head.
        # So we expand the query to match the grid's batch size (N * Heads)
        query_expanded = query_for_sampling.repeat_interleave(self.num_heads, dim=0)

        # Sample features from the expanded query
        # Output shape: (N * Heads, E, L, K)
        sampled_features_raw = F.grid_sample(
            query_expanded, sampling_grid, mode='bilinear', padding_mode='zeros', align_corners=False
        )
        
        # Reshape and permute to bring features into a usable format
        # (N * Heads, E, L, K) -> (N, Heads, L, K, E)
        sampled_features = sampled_features_raw.view(N, self.num_heads, E, L, self.num_sampling_points)
        sampled_features = sampled_features.permute(0, 1, 3, 4, 2)

        # Apply attention weights and sum over sampling points
        # (N, Heads, L, K, E) * (N, Heads, L, K, 1) -> sum -> (N, Heads, L, E)
        output = (sampled_features * attention_weights.unsqueeze(-1)).sum(dim=3)

        # Combine heads. A simple and effective way is to average them.
        output = output.mean(dim=1)  # Average across the heads dimension -> (N, L, E)
        
        # Reshape back to 2D spatial format and apply final projection
        output = output.view(N, H, W, E)
        output = self.output_proj(output)
        
        return output

# --- Main Execution ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("1. Loading Swin-B model...")
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, features_only=True)
    model = model.to(device)
    model.eval()

    data_config = timm.data.resolve_model_data_config(model)
    transforms_pipeline = timm.data.create_transform(**data_config, is_training=False)
    
    try:
        image = Image.open(r"D:\Assignment\processed_dataset\rgb_img_1.png").convert("RGB")
    except FileNotFoundError:
        print(r"ERROR: 'D:\Assignment\processed_dataset\rgb_img_1.png' not found. Please verify the file path.")
        exit()

    input_tensor = transforms_pipeline(image).unsqueeze(0).to(device)
    print(f"Input image tensor shape: {input_tensor.shape}")
    print("-" * 50)
    
    print("2. Extracting feature maps from Swin-B...")
    with torch.no_grad():
        feature_maps = model(input_tensor)
        features = feature_maps[-1]
    
    N, H, W, E = features.shape
    features_2d = features
    
    print(f"Original feature shape (N, H, W, E): {features.shape}")
    print(f"Feature shape for attention (N, H, W, E): {features_2d.shape}")
    print("-" * 50)
    
    print("3. Generating reference points...")
    ref_points = get_reference_points(H, W, device).unsqueeze(0).expand(N, -1, -1, -1)
    print(f"Reference points shape: {ref_points.shape}")
    print("-" * 50)

    print("4. Applying 2D Deformable Self-Attention...")
    deformable_attn_layer = MultiHeadDeformableAttention2D(
        embed_dim=E,
        num_heads=8,
        num_sampling_points=4
    ).to(device)
    
    with torch.no_grad():
        refined_features = deformable_attn_layer(features_2d, ref_points)
        
    print("\n✅ Pipeline complete!")
    print(f"Final refined features shape: {refined_features.shape}")

    assert refined_features.shape == features_2d.shape
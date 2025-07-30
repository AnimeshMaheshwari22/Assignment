import torch
import torch.nn as nn
import torch.nn.functional as F

def get_reference_points_2d(H, W, device):
    y, x = torch.meshgrid(
        torch.linspace(0.5, H - 0.5, H, device=device),
        torch.linspace(0.5, W - 0.5, W, device=device),
        indexing="ij",
    )
    y = y / H
    x = x / W
    ref_points = torch.stack((x, y), -1)
    return ref_points

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

        spatial_value = self.value_proj(query).view(N, H, W, self.num_heads, self.head_dim)
        spatial_value = spatial_value.permute(0, 3, 4, 1, 2)
        spatial_value = spatial_value.reshape(N * self.num_heads, self.head_dim, H, W)

        query_flat = query.view(N, L, E)
        sampling_offsets = self.sampling_offset_proj(query_flat).view(N, L, self.num_heads, self.num_sampling_points, 2)
        attention_weights = self.attention_weights_proj(query_flat).view(N, L, self.num_heads, self.num_sampling_points)
        attention_weights = F.softmax(attention_weights, dim=-1)

        ref_points_flat = reference_points.view(N, L, 2)
        sampling_locations = ref_points_flat.unsqueeze(2).unsqueeze(3) + sampling_offsets
        sampling_grid = 2 * sampling_locations - 1

        sampling_grid = sampling_grid.permute(0, 2, 1, 3, 4).reshape(N * self.num_heads, L, self.num_sampling_points, 2)
        sampled_value = F.grid_sample(spatial_value, sampling_grid, mode='bilinear', padding_mode='zeros', align_corners=False)

        attention_weights = attention_weights.permute(0, 2, 1, 3).reshape(N * self.num_heads, L, self.num_sampling_points)
        sampled_value = sampled_value.permute(0, 2, 3, 1)

        output = (sampled_value * attention_weights.unsqueeze(-1)).sum(dim=2)

        output = output.view(N, self.num_heads, L, self.head_dim).permute(0, 2, 1, 3).reshape(N, L, E)
        output = self.output_proj(output)
        output = output.view(N, H, W, E)

        return output

class MultiHeadDeformableAttention3D(nn.Module):
    def __init__(self, embed_dim, num_heads, num_sampling_points, k_neighbors=4):
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

class ImageToDepthCrossAttention(nn.Module):
    def __init__(self, fusion_dim=256, nhead=8, ffn_dim=1024):
        super().__init__()
        self.attn = nn.MultiheadAttention(fusion_dim, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(fusion_dim)

        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, fusion_dim)
        )
        self.norm2 = nn.LayerNorm(fusion_dim)

    def forward(self, img_feats, pc_feats):
        attended_img_feats, _ = self.attn(query=img_feats, key=pc_feats, value=pc_feats)
        img_feats_fused = self.norm1(img_feats + attended_img_feats)
        img_feats_out = self.ffn(img_feats_fused)
        return self.norm2(img_feats_fused + img_feats_out)

class DepthToImageCrossAttention(nn.Module):
    def __init__(self, fusion_dim=256, nhead=8, ffn_dim=1024):
        super().__init__()
        self.attn = nn.MultiheadAttention(fusion_dim, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(fusion_dim)

        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, fusion_dim)
        )
        self.norm2 = nn.LayerNorm(fusion_dim)

    def forward(self, pc_feats, img_feats):
        attended_pc_feats, _ = self.attn(query=pc_feats, key=img_feats, value=img_feats)
        pc_feats_fused = self.norm1(pc_feats + attended_pc_feats)
        pc_feats_out = self.ffn(pc_feats_fused)
        return self.norm2(pc_feats_fused + pc_feats_out)

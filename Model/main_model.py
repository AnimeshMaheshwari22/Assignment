# =================================================================================
# File: main_model.py
# Description: Main model architecture, helpers, and demonstration script.
# =================================================================================
import torch
import torch.nn as nn
import math
import numpy as np
from functools import partial
from typing import Optional
import torch.nn.functional as F

# Imports from other modules
from transformer_blocks import VisionLanguageEncoder, TransformerDecoder, TransformerDecoderLayer
from feature_extractor import get_bert_backbone, get_swin_b_backbone

# --- Helper Functions and Classes for the Main Model ---

def furthest_point_sample(xyz, npoint):
    """A simple PyTorch implementation of Furthest Point Sampling."""
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

class GenericMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, norm_fn_name=None, activation="relu", use_conv=True, dropout=0.0, hidden_use_bias=False, output_use_activation=False, output_use_norm=False, output_use_bias=True):
        super().__init__()
        hidden_dims = [] if hidden_dims is None else hidden_dims
        all_dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for i in range(len(all_dims) - 1):
            is_last_layer = i == len(all_dims) - 2
            in_dim, out_dim = all_dims[i], all_dims[i+1]
            use_bias = hidden_use_bias if not is_last_layer else output_use_bias
            layer_fn = nn.Conv1d if use_conv else nn.Linear
            layers.append(layer_fn(in_dim, out_dim, 1 if use_conv else (), bias=use_bias))
            if (not is_last_layer or output_use_norm) and norm_fn_name is not None:
                layers.append(nn.BatchNorm1d(out_dim) if norm_fn_name == "bn1d" else nn.LayerNorm(out_dim))
            if (not is_last_layer or output_use_activation) and activation is not None:
                layers.append(nn.ReLU() if activation == "relu" else nn.GELU())
            if dropout > 0 and not is_last_layer:
                layers.append(nn.Dropout(p=dropout))
        self.mlp = nn.Sequential(*layers)
    def forward(self, x): return self.mlp(x)

class PositionEmbeddingCoordsSine(nn.Module):
    """
    Corrected sinusoidal position embedding for 3D coordinates that handles
    any dimension size by ensuring sub-dimensions are even.
    """
    def __init__(self, d_pos=256, temperature=10000, normalize=True):
        super().__init__()
        self.d_pos, self.temperature, self.normalize = d_pos, temperature, normalize
        
        # Distribute d_pos into three parts for x, y, z, ensuring each part is even
        d_per_dim = d_pos // 3
        self.d_x = (d_per_dim // 2) * 2
        self.d_y = (d_per_dim // 2) * 2
        self.d_z = d_pos - self.d_x - self.d_y
        assert self.d_z % 2 == 0, f"Remaining dimension for z is not even: {self.d_z}"

        # Denominators for positional encoding
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dim_t_x = self.temperature ** (2 * torch.arange(self.d_x // 2, device=device) / self.d_x)
        self.dim_t_y = self.temperature ** (2 * torch.arange(self.d_y // 2, device=device) / self.d_y)
        self.dim_t_z = self.temperature ** (2 * torch.arange(self.d_z // 2, device=device) / self.d_z)

    def forward(self, xyz, input_range=None):
        if self.normalize:
            min_coord, max_coord = input_range
            xyz = (xyz - min_coord) / (max_coord - min_coord + 1e-6)
            
        # Calculate sinusoidal embeddings for each coordinate
        pos_x = xyz[..., 0, None] / self.dim_t_x
        pos_y = xyz[..., 1, None] / self.dim_t_y
        pos_z = xyz[..., 2, None] / self.dim_t_z
        
        # Apply sin and cos and interleave
        pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=-1).flatten(-2)
        pos_z = torch.stack((pos_z.sin(), pos_z.cos()), dim=-1).flatten(-2)
        
        # Concatenate and permute
        pos = torch.cat((pos_x, pos_y, pos_z), dim=-1)
        return pos.permute(0, 2, 1)

class BoxProcessor:
    def __init__(self, dataset_config): self.dataset_config = dataset_config
    def compute_predicted_center(self, c_off, q_xyz, pc_dims):
        c_unnorm = q_xyz + c_off
        c_norm = (c_unnorm - pc_dims[0]) / (pc_dims[1] - pc_dims[0] + 1e-6)
        return c_norm, c_unnorm
    def compute_predicted_size(self, s_norm, pc_dims):
        return s_norm * (pc_dims[1] - pc_dims[0])
    def compute_predicted_angle(self, ang_log, ang_res):
        if ang_log.shape[-1] == 1: return ang_log * 0
        ang_per_cls = 2*np.pi/self.dataset_config.num_angle_bin
        pred_cls = ang_log.argmax(dim=-1).detach()
        ang = ang_per_cls*pred_cls + ang_res.gather(2, pred_cls.unsqueeze(-1)).squeeze(-1)
        ang[ang > np.pi] -= 2*np.pi
        return ang
    def compute_objectness_and_cls_prob(self, cls_logits):
        cls_prob = F.softmax(cls_logits, dim=-1)
        return cls_prob[..., :-1], 1 - cls_prob[..., -1]
    def box_parametrization_to_corners(self, center, size, angle):
        return self.dataset_config.box_parametrization_to_corners(center, size, angle)

# --- Main Model Class ---

class VisionLanguage3DBoxModel(nn.Module):
    def __init__(self, encoder, decoder, dataset_config, encoder_dim=768, decoder_dim=256, num_queries=256, mlp_dropout=0.3):
        super().__init__()
        self.encoder, self.decoder, self.num_queries = encoder, decoder, num_queries
        self.box_processor = BoxProcessor(dataset_config)
        self.dataset_config = dataset_config # Store dataset_config
        self.encoder_to_decoder_projection = GenericMLP(encoder_dim, [encoder_dim], decoder_dim, "bn1d", use_conv=True)
        self.pos_embedding = PositionEmbeddingCoordsSine(d_pos=decoder_dim, normalize=True)
        self.query_projection = GenericMLP(decoder_dim, [decoder_dim], decoder_dim, use_conv=True, output_use_activation=True)
        self.build_mlp_heads(dataset_config, decoder_dim, mlp_dropout)

    def build_mlp_heads(self, dataset_config, decoder_dim, mlp_dropout):
        mlp_func = partial(GenericMLP, norm_fn_name="bn1d", activation="relu", use_conv=True, hidden_dims=[decoder_dim, decoder_dim], dropout=mlp_dropout, input_dim=decoder_dim)
        self.mlp_heads = nn.ModuleDict({
            "sem_cls_head": mlp_func(output_dim=dataset_config.num_semcls + 1), "center_head": mlp_func(output_dim=3),
            "size_head": mlp_func(output_dim=3), "angle_cls_head": mlp_func(output_dim=dataset_config.num_angle_bin),
            "angle_residual_head": mlp_func(output_dim=dataset_config.num_angle_bin)
        })

    def get_query_embeddings(self, enc_xyz, pc_dims):
        q_inds = furthest_point_sample(enc_xyz, self.num_queries).long()
        q_xyz = torch.gather(enc_xyz, 1, q_inds.unsqueeze(-1).repeat(1, 1, 3))
        pos_embed = self.pos_embedding(q_xyz, input_range=pc_dims)
        return q_xyz, self.query_projection(pos_embed)

    def get_box_predictions(self, q_xyz, pc_dims, box_feats):
        box_feats = box_feats.permute(0, 2, 3, 1) # L, B, C, Q
        L, B, C, Q = box_feats.shape
        box_feats = box_feats.reshape(L * B, C, Q)
        preds = {k: h(box_feats).transpose(1, 2).reshape(L, B, Q, -1) for k, h in self.mlp_heads.items()}
        preds["center_head"] = preds["center_head"].sigmoid() - 0.5
        preds["size_head"] = preds["size_head"].sigmoid()
        # Correctly access the number of angle bins from the dataset_config
        preds["angle_residual_head"] *= (np.pi / self.dataset_config.num_angle_bin)
        
        outputs = []
        for l in range(L):
            center_norm, center_unnorm = self.box_processor.compute_predicted_center(preds["center_head"][l], q_xyz, pc_dims)
            size_unnorm = self.box_processor.compute_predicted_size(preds["size_head"][l], pc_dims)
            angle = self.box_processor.compute_predicted_angle(preds["angle_cls_head"][l], preds["angle_residual_head"][l])
            corners = self.box_processor.box_parametrization_to_corners(center_unnorm, size_unnorm, angle)
            with torch.no_grad():
                sem_prob, obj_prob = self.box_processor.compute_objectness_and_cls_prob(preds["sem_cls_head"][l])
            outputs.append({"box_corners": corners, "sem_cls_prob": sem_prob, "objectness_prob": obj_prob})
        return {"outputs": outputs[-1], "aux_outputs": outputs[:-1]}

    def forward(self, img_features, txt_features, point_cloud_dims):
        enc_features = self.encoder(img_features, txt_features)
        print(f"✅ Step 1: Encoder Output Shape: {enc_features.shape}")
        B, N, _ = enc_features.shape
        H = W = int(math.sqrt(N))
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        enc_xyz = torch.stack([grid_x.flatten()/ (W-1), grid_y.flatten()/(H-1), torch.zeros_like(grid_x.flatten())], dim=-1).unsqueeze(0).repeat(B, 1, 1).to(enc_features.device)
        enc_features_proj = self.encoder_to_decoder_projection(enc_features.permute(0, 2, 1)).permute(2, 0, 1)
        q_xyz, q_embed = self.get_query_embeddings(enc_xyz, point_cloud_dims)
        enc_pos = self.pos_embedding(enc_xyz, input_range=point_cloud_dims).permute(2, 0, 1)
        q_embed = q_embed.permute(2, 0, 1)
        box_features, _ = self.decoder(torch.zeros_like(q_embed), enc_features_proj, query_pos=q_embed, pos=enc_pos)
        print(f"✅ Step 2: Decoder Output Shape (Box Features): {box_features.shape} (Layers, Queries, Batch, Dim)")
        return self.get_box_predictions(q_xyz, point_cloud_dims, box_features)

# --- Demonstration Script ---
if __name__ == '__main__':
    class DummyDatasetConfig:
        def __init__(self): self.num_semcls, self.num_angle_bin = 18, 12
        def box_parametrization_to_corners(self, center, size, angle):
            B, Q, _ = center.shape
            l, w, h = size[..., 0], size[..., 1], size[..., 2]
            
            x_corners = torch.stack([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2], dim=-1)
            y_corners = torch.stack([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2], dim=-1)
            z_corners = torch.stack([h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2], dim=-1)
            corners = torch.stack([x_corners, y_corners, z_corners], dim=-2) # B, Q, 3, 8
            
            cos_a, sin_a = torch.cos(angle), torch.sin(angle)
            # Rotation matrix for Z-axis
            row1 = torch.stack([cos_a, -sin_a, torch.zeros_like(cos_a)], dim=-1)
            row2 = torch.stack([sin_a, cos_a, torch.zeros_like(cos_a)], dim=-1)
            row3 = torch.stack([torch.zeros_like(cos_a), torch.zeros_like(cos_a), torch.ones_like(cos_a)], dim=-1)
            rot_matrix = torch.stack([row1, row2, row3], dim=-2) # B, Q, 3, 3
            
            # Apply rotation
            rotated_corners = torch.einsum('bqij,bqjk->bqik', rot_matrix, corners)
            
            # Add center offset
            return rotated_corners + center.unsqueeze(-1)

    dataset_config = DummyDatasetConfig()
    BATCH_SIZE, IMG_PATCHES, TXT_TOKENS = 1, 196, 3
    ENC_D, DEC_D = 768, 256
    ENC_L, DEC_L = 4, 6
    ENC_H, DEC_H = 12, 8
    N_QUERIES = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("--- Initializing Full 3D Box Prediction Model for a Single Example ---")
    encoder = VisionLanguageEncoder(num_layers=ENC_L, d_model=ENC_D, n_heads=ENC_H).to(device)
    decoder_layer = TransformerDecoderLayer(d_model=DEC_D, nhead=DEC_H).to(device)
    decoder = TransformerDecoder(decoder_layer, num_layers=DEC_L, return_intermediate=True).to(device)
    model = VisionLanguage3DBoxModel(encoder, decoder, dataset_config, encoder_dim=ENC_D, decoder_dim=DEC_D, num_queries=N_QUERIES).to(device)

    # --- Create Dummy Input Data ---
    image_features = torch.randn(BATCH_SIZE, IMG_PATCHES, ENC_D).to(device)
    text_features = torch.randn(BATCH_SIZE, TXT_TOKENS, ENC_D).to(device)
    pc_dims = [torch.tensor([0.,0.,0.]).to(device), torch.tensor([1.,1.,1.]).to(device)]

    print(f"--- Running Forward Pass ---")
    model.eval()
    with torch.no_grad():
        predictions = model(image_features, text_features, pc_dims)

    box_corners = predictions['outputs']['box_corners']
    print(f"✅ Step 3: Final Bounding Box Corners Shape: {box_corners.shape}")
    assert box_corners.shape == (BATCH_SIZE, N_QUERIES, 3, 8), "The output shape is incorrect!"
    print("\n✅ Pipeline execution successful.")

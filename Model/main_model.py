# =================================================================================
# File: main_model.py
# Description: Main model architecture that integrates all components.
# =================================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from functools import partial
from typing import Optional

# --- Imports from other project files ---
from feature_extractor import get_swin_b_backbone, get_bert_backbone
from transformer_blocks import VisionLanguageEncoder, TransformerDecoder, TransformerDecoderLayer
from matcher_and_criterion import HungarianMatcher3D, SetCriterion

# =================================================================================
# SECTION 1: HELPER CLASSES & FUNCTIONS FOR THE MAIN MODEL
# =================================================================================

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
    def __init__(self, d_pos=256, temperature=10000, normalize=True):
        super().__init__()
        self.d_pos, self.temperature, self.normalize = d_pos, temperature, normalize
        d_per_dim = d_pos // 3
        self.d_x = (d_per_dim // 2) * 2
        self.d_y = (d_per_dim // 2) * 2
        self.d_z = d_pos - self.d_x - self.d_y
        assert self.d_z % 2 == 0, f"Remaining dimension for z is not even: {self.d_z}"
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dim_t_x = self.temperature ** (2 * torch.arange(self.d_x // 2, device=device) / self.d_x)
        self.dim_t_y = self.temperature ** (2 * torch.arange(self.d_y // 2, device=device) / self.d_y)
        self.dim_t_z = self.temperature ** (2 * torch.arange(self.d_z // 2, device=device) / self.d_z)

    def forward(self, xyz, input_range=None):
        if self.normalize:
            min_coord, max_coord = input_range
            xyz = (xyz - min_coord) / (max_coord - min_coord + 1e-6)
        pos_x = xyz[..., 0, None] / self.dim_t_x
        pos_y = xyz[..., 1, None] / self.dim_t_y
        pos_z = xyz[..., 2, None] / self.dim_t_z
        pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=-1).flatten(-2)
        pos_z = torch.stack((pos_z.sin(), pos_z.cos()), dim=-1).flatten(-2)
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

# =================================================================================
# SECTION 2: MAIN MODEL
# =================================================================================

class VisionLanguage3DBoxModel(nn.Module):
    def __init__(self, dataset_config, 
                 encoder_dim=768, decoder_dim=256, num_queries=256, mlp_dropout=0.3,
                 enc_layers=4, enc_nhead=12, enc_ffn_dim=3072, enc_dropout=0.1,
                 dec_layers=6, dec_nhead=8, dec_ffn_dim=2048, dec_dropout=0.1,
                 cost_class=1.0, cost_center=1.0, cost_size=1.0, cost_angle=1.0,
                 loss_weight_ce=1.0, loss_weight_center=1.0, loss_weight_size=1.0, loss_weight_angle=1.0):
        super().__init__()
        
        # --- Instantiate Trainable Backbones ---
        self.rgb_backbone = get_swin_b_backbone(pretrained=True)
        self.pc_backbone = get_swin_b_backbone(pretrained=True)
        self.text_backbone, self.tokenizer = get_bert_backbone()
        
        # --- Instantiate Transformer Components ---
        self.encoder = VisionLanguageEncoder(
            num_layers=enc_layers,
            d_model=encoder_dim,
            n_heads=enc_nhead,
            d_ffn=enc_ffn_dim,
            dropout=enc_dropout
        )
        decoder_layer = TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=dec_nhead,
            dim_feedforward=dec_ffn_dim,
            dropout=dec_dropout
        )
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_layers=dec_layers,
            return_intermediate=True
        )

        self.num_queries = num_queries
        
        # --- Instantiate Matcher and Criterion ---
        matcher = HungarianMatcher3D(
            cost_class=cost_class, 
            cost_center=cost_center, 
            cost_size=cost_size, 
            cost_angle=cost_angle
        )
        weight_dict = {
            'loss_ce': loss_weight_ce, 
            'loss_center': loss_weight_center, 
            'loss_size': loss_weight_size, 
            'loss_angle': loss_weight_angle
        }
        losses = ['labels', 'boxes']
        self.criterion = SetCriterion(
            num_classes=dataset_config.num_semcls, 
            matcher=matcher, 
            weight_dict=weight_dict, 
            losses=losses
        )
        
        # --- Helper Modules ---
        self.box_processor = BoxProcessor(dataset_config)
        self.dataset_config = dataset_config
        
        # --- Projection Layers ---
        swin_feat_dim = 1024
        bert_feat_dim = 768
        self.rgb_proj = nn.Linear(swin_feat_dim, encoder_dim)
        self.pc_proj = nn.Linear(swin_feat_dim, encoder_dim)
        assert bert_feat_dim == encoder_dim, "BERT output dim must match encoder dim"

        self.encoder_to_decoder_projection = GenericMLP(encoder_dim, [encoder_dim], decoder_dim, "bn1d", use_conv=True)
        self.pos_embedding = PositionEmbeddingCoordsSine(d_pos=decoder_dim, normalize=True)
        self.query_projection = GenericMLP(decoder_dim, [decoder_dim], decoder_dim, use_conv=True, output_use_activation=True)
        
        # --- Prediction Heads ---
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
        box_feats_permuted = box_feats.permute(0, 2, 3, 1) # L, B, C, Q
        L, B, C, Q = box_feats_permuted.shape
        box_feats_reshaped = box_feats_permuted.reshape(L * B, C, Q)
        
        preds = {k: h(box_feats_reshaped).transpose(1, 2).reshape(L, B, Q, -1) for k, h in self.mlp_heads.items()}
        
        center_offset = preds["center_head"].sigmoid() - 0.5
        size_normalized = preds["size_head"].sigmoid()
        angle_residual_normalized = preds["angle_residual_head"]
        angle_residual = angle_residual_normalized * (np.pi / self.dataset_config.num_angle_bin)
        
        outputs = []
        for l in range(L):
            center_norm, center_unnorm = self.box_processor.compute_predicted_center(center_offset[l], q_xyz, pc_dims)
            size_unnorm = self.box_processor.compute_predicted_size(size_normalized[l], pc_dims)
            angle = self.box_processor.compute_predicted_angle(preds["angle_cls_head"][l], angle_residual[l])
            corners = self.box_processor.box_parametrization_to_corners(center_unnorm, size_unnorm, angle)
            with torch.no_grad():
                sem_prob, obj_prob = self.box_processor.compute_objectness_and_cls_prob(preds["sem_cls_head"][l])
            
            outputs.append({
                "sem_cls_logits": preds["sem_cls_head"][l], "center_unnormalized": center_unnorm,
                "size_unnormalized": size_unnorm, "angle_continuous": angle, "box_corners": corners,
                "sem_cls_prob": sem_prob, "objectness_prob": obj_prob
            })
        return {"outputs": outputs[-1], "aux_outputs": outputs[:-1]}

    def forward(self, inputs, targets=None):
        # 1. Tokenize text and extract features from all three backbones
        device = inputs['rgb_input'].device
        tokenized = self.tokenizer(inputs['text_prompts'], padding='longest', return_tensors='pt').to(device)
        
        rgb_feats_raw = self.rgb_backbone(inputs['rgb_input'])
        pc_feats_raw = self.pc_backbone(inputs['pc_input'])
        text_feats_raw = self.text_backbone(input_ids=tokenized['input_ids'], attention_mask=tokenized['attention_mask'])[0]

        # 2. Project visual features and fuse them
        rgb_feats = self.rgb_proj(rgb_feats_raw)
        pc_feats = self.pc_proj(pc_feats_raw)
        fused_visual_features = rgb_feats + pc_feats
        
        # 3. Run the rest of the pipeline
        enc_features = self.encoder(fused_visual_features, text_feats_raw)
        B, N, _ = enc_features.shape
        H = W = int(math.sqrt(N))
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        
        point_cloud_dims = [inputs['point_cloud_dims_min'], inputs['point_cloud_dims_max']]
        
        enc_xyz = torch.stack([grid_x.flatten()/(W-1), grid_y.flatten()/(H-1), torch.zeros_like(grid_x.flatten())], dim=-1).unsqueeze(0).repeat(B, 1, 1)
        enc_features_proj = self.encoder_to_decoder_projection(enc_features.permute(0, 2, 1)).permute(2, 0, 1)
        q_xyz, q_embed = self.get_query_embeddings(enc_xyz, point_cloud_dims)
        enc_pos = self.pos_embedding(enc_xyz, input_range=point_cloud_dims).permute(2, 0, 1)
        q_embed = q_embed.permute(2, 0, 1)
        box_features, _ = self.decoder(torch.zeros_like(q_embed), enc_features_proj, query_pos=q_embed, pos=enc_pos)
        
        predictions = self.get_box_predictions(q_xyz, point_cloud_dims, box_features)
        
        if self.training and targets is not None:
            loss_dict = self.criterion(predictions, targets)
            return predictions, loss_dict
            
        return predictions

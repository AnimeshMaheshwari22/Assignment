import torch
from torch import nn
import torch.nn.functional as F
import copy

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class FusionDecoderLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn_pc = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn_img = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

    def forward(self, queries, pc_memory, img_memory):
        q_sa = self.norm1(queries)
        sa_out, _ = self.self_attn(q_sa, q_sa, q_sa)
        queries = queries + self.dropout1(sa_out)
        q_ca_pc = self.norm2(queries)
        ca_pc_out, _ = self.cross_attn_pc(query=q_ca_pc, key=pc_memory, value=pc_memory)
        queries = queries + self.dropout2(ca_pc_out)
        q_ca_img = self.norm3(queries)
        B, H, W, C = img_memory.shape
        img_memory_flat = img_memory.view(B, H * W, C)
        ca_img_out, _ = self.cross_attn_img(query=q_ca_img, key=img_memory_flat, value=img_memory_flat)
        queries = queries + self.dropout3(ca_img_out)
        q_ffn = self.norm4(queries)
        ffn_out = self.ffn(q_ffn)
        queries = queries + self.dropout4(ffn_out)
        return queries

class FusionDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, d_model=256, num_classes=1, num_queries=16):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.d_model = d_model
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.sem_cls_head = nn.Linear(d_model, num_classes)
        self.bbox_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, 8)
        )

    def forward(self, pc_features, img_features, xyz_coords):
        B = pc_features.shape[0]
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        aux_outputs = []
        for layer in self.layers:
            queries = layer(queries, pc_features, img_features)
            sem_cls_logits = self.sem_cls_head(queries)
            bbox_params = self.bbox_head(queries)
            aux_outputs.append({
                'sem_cls_logits': sem_cls_logits, 
                'bbox_params': bbox_params,
            })
        final_output = aux_outputs.pop()
        return {'outputs': final_output, 'aux_outputs': aux_outputs}

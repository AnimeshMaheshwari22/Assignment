# =================================================================================
# File: transformer_blocks.py
# Description: Core Transformer components for the model.
# =================================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Helper to clone modules
def get_clones(module, N):
    return nn.ModuleList([module for i in range(N)])

class FFN(nn.Module):
    """A standard Feed-Forward Network (FFN) module."""
    def __init__(self, d_model=768, d_ffn=3072, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout2(self.linear2(self.dropout1(self.activation(self.linear1(x)))))

class DeformableSelfAttention(nn.Module):
    """A placeholder for the Deformable Self-Attention module."""
    def __init__(self, d_model=768, n_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

    def forward(self, query):
        attn_output, _ = self.attention(query, query, query)
        return attn_output

class VisionLanguageEncoderLayer(nn.Module):
    """A single layer of the dual-stream Vision-Language Transformer Encoder."""
    def __init__(self, d_model=768, n_heads=8, d_ffn=3072, dropout=0.1):
        super().__init__()
        self.img_self_attn = DeformableSelfAttention(d_model=d_model, n_heads=n_heads)
        self.txt_self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.cross_attn_i2t = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.cross_attn_t2i = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.ffn = FFN(d_model=d_model, d_ffn=d_ffn, dropout=dropout)
        self.norm_img1, self.norm_txt1 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.norm_img2, self.norm_txt2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.norm_img3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, img_features, txt_features, img_mask=None, txt_mask=None):
        img_sa_out = self.img_self_attn(query=img_features)
        img_features_sa = self.norm_img1(img_features + self.dropout(img_sa_out))
        txt_sa_out, _ = self.txt_self_attn(query=txt_features, key=txt_features, value=txt_features, key_padding_mask=txt_mask)
        txt_features_sa = self.norm_txt1(txt_features + self.dropout(txt_sa_out))
        i2t_out, _ = self.cross_attn_i2t(query=txt_features_sa, key=img_features_sa, value=img_features_sa, key_padding_mask=img_mask)
        txt_features_updated = self.norm_txt2(txt_features_sa + self.dropout(i2t_out))
        t2i_out, _ = self.cross_attn_t2i(query=img_features_sa, key=txt_features_updated, value=txt_features_updated, key_padding_mask=txt_mask)
        img_features_updated = self.norm_img2(img_features_sa + self.dropout(t2i_out))
        ffn_out = self.ffn(img_features_updated)
        return self.norm_img3(img_features_updated + self.dropout(ffn_out))

class VisionLanguageEncoder(nn.Module):
    """A stack of VisionLanguageEncoderLayer modules."""
    def __init__(self, num_layers=6, d_model=768, n_heads=8, d_ffn=3072, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            VisionLanguageEncoderLayer(d_model, n_heads, d_ffn, dropout) for _ in range(num_layers)
        ])
    def forward(self, img_features, txt_features, img_mask=None, txt_mask=None):
        output = img_features
        for layer in self.layers:
            output = layer(output, txt_features, img_mask, txt_mask)
        return output

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout1, self.dropout2, self.dropout3 = nn.Dropout(dropout), nn.Dropout(dropout), nn.Dropout(dropout)
        self.activation = F.relu

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, query_pos, pos, **kwargs):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2, _ = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos), key=self.with_pos_embed(memory, pos), value=memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, None

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(decoder_layer.linear2.out_features)
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, **kwargs):
        output = tgt
        intermediate = []
        for layer in self.layers:
            output, attn = layer(output, memory, **kwargs)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        if self.return_intermediate:
            return torch.stack(intermediate), attn
        return self.norm(output), attn
import torch
from torch import nn
import torch.nn.functional as F

# A standard helper function for DETR-style models to clone a module N times
def _get_clones(module, N):
    return nn.ModuleList([module for i in range(N)])

class FusionDecoderLayer(nn.Module):
    """
    A single layer of the multi-modal fusion decoder.
    It refines a set of object queries by attending to both point cloud and image features.
    """
    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        # Self-attention over the object queries
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Cross-attention for queries to attend to point cloud features
        self.cross_attn_pc = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Cross-attention for queries to attend to image features
        self.cross_attn_img = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )

        # Layer Normalization and Dropout layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

    def forward(self, queries, pc_memory, img_memory):
        """
        Args:
            queries (Tensor): The set of object queries, shape (B, Num_Queries, d_model)
            pc_memory (Tensor): The point cloud features from the encoder, shape (B, Num_Points, d_model)
            img_memory (Tensor): The image features from the encoder, shape (B, H, W, d_model)
        """
        # 1. Self-attention on object queries
        q_sa = self.norm1(queries)
        sa_out, _ = self.self_attn(q_sa, q_sa, q_sa)
        queries = queries + self.dropout1(sa_out)

        # 2. Cross-attention: Queries attend to Point Cloud features
        q_ca_pc = self.norm2(queries)
        ca_pc_out, _ = self.cross_attn_pc(query=q_ca_pc, key=pc_memory, value=pc_memory)
        queries = queries + self.dropout2(ca_pc_out)
        
        # 3. Cross-attention: Queries attend to Image features
        q_ca_img = self.norm3(queries)
        # Flatten the image features' spatial dimensions for attention
        B, H, W, C = img_memory.shape
        img_memory_flat = img_memory.view(B, H * W, C)
        ca_img_out, _ = self.cross_attn_img(query=q_ca_img, key=img_memory_flat, value=img_memory_flat)
        queries = queries + self.dropout3(ca_img_out)

        # 4. Feed-Forward Network
        q_ffn = self.norm4(queries)
        ffn_out = self.ffn(q_ffn)
        queries = queries + self.dropout4(ffn_out)

        return queries

class FusionDecoder(nn.Module):
    """
    The full decoder that stacks multiple FusionDecoderLayers and adds prediction heads.
    """
    def __init__(self, decoder_layer, num_layers, d_model=256, num_classes=10, num_queries=256):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.d_model = d_model

        # Learnable object queries, one for each potential object
        self.query_embed = nn.Embedding(num_queries, d_model)

        # Prediction heads for class and 3D bounding box
        self.class_head = nn.Linear(d_model, num_classes)
        # Bbox head predicts 8 parameters: center(x,y,z), size(w,h,l), yaw(sin,cos)
        self.bbox_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 8)
        )

    def forward(self, pc_features, img_features, xyz_coords):
        """
        Args:
            pc_features (Tensor): (B, Num_Points, d_model)
            img_features (Tensor): (B, H, W, d_model)
            xyz_coords (Tensor): (B, Num_Points, 3), used for reference points
        """
        B = pc_features.shape[0]
        # Initialize queries for the batch
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        
        # The decoder outputs predictions from each layer (auxiliary losses)
        intermediate_outputs = []
        for layer in self.layers:
            queries = layer(queries, pc_features, img_features)
            
            # Get predictions from the current layer's refined queries
            output_class = self.class_head(queries)
            output_bbox = self.bbox_head(queries)
            
            # The decoder predicts offsets relative to the point cloud's center.
            # Add the point cloud center as a reference point to the predicted box center.
            # This helps ground the predictions in the 3D scene.
            pc_center = xyz_coords.mean(dim=1, keepdim=True)
            output_bbox[:, :, :3] += pc_center
            
            intermediate_outputs.append({'pred_logits': output_class, 'pred_boxes': output_bbox})

        # Return the output from the final decoder layer
        return intermediate_outputs[-1]

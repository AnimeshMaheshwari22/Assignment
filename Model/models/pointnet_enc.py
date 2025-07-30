# pointnet_enc.py (Corrected Version)

# Copyright (c) Facebook, Inc. and its affiliates.
import math
from functools import partial

import torch
import torch.nn as nn
from third_party.pointnet2.pointnet2_modules import PointnetSAModuleVotes

from models.helpers import GenericMLP
from models.position_embedding import PositionEmbeddingCoordsSine
from models.transformer import (MaskedTransformerEncoder, TransformerEncoder,
                                TransformerEncoderLayer)

class PointNetEncoder(nn.Module):
    def __init__(
        self,
        pre_encoder,
        encoder,
        encoder_dim=256,
        decoder_dim=256, # This is kept for legacy reasons, but not used by the encoder itself
        position_embedding="fourier", # Not used here, but kept for legacy
        mlp_dropout=0.3,
        num_queries=256, # Not used here, but kept for legacy
    ):
        super().__init__()
        self.pre_encoder = pre_encoder
        self.encoder = encoder

    def _break_up_pc(self, pc):
        """ Separates coordinates from features. """
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def run_encoder(self, point_clouds):
        """
        This is the main function called by your `Encoder` class.
        It processes the point cloud and returns encoded features and coordinates.
        """
        xyz, features = self._break_up_pc(point_clouds)
        
        # Pre-encoder (PointNet++ set abstraction)
        pre_enc_xyz, pre_enc_features, pre_enc_inds = self.pre_encoder(xyz, features)
        
        # The transformer encoder expects features in (Seq_len, Batch, Dim) format
        pre_enc_features = pre_enc_features.permute(2, 0, 1)

        # Transformer Encoder
        enc_xyz, enc_features, enc_inds = self.encoder(
            pre_enc_features, xyz=pre_enc_xyz
        )
        
        if enc_inds is None:
            enc_inds = pre_enc_inds
        else:
            enc_inds = torch.gather(pre_enc_inds, 1, enc_inds.type(torch.int64))
            
        return enc_xyz, enc_features, enc_inds

    def forward(self, inputs):
        """ 
        A simple forward pass that wraps run_encoder.
        In your architecture, you call run_encoder directly, but this makes the module runnable.
        """
        point_clouds = inputs["point_clouds"]
        enc_xyz, enc_features, enc_inds = self.run_encoder(point_clouds)
        
        # Return a dictionary consistent with the rest of the project
        return {
            "encoder_features": enc_features,
            "encoder_xyz": enc_xyz,
            "encoder_inds": enc_inds,
        }

def build_preencoder(args):
    mlp_dims = [3 * int(args.use_color), 64, 128, args.enc_dim]
    preencoder = PointnetSAModuleVotes(
        radius=0.2,
        nsample=64,
        npoint=args.preenc_npoints,
        mlp=mlp_dims,
        normalize_xyz=True,
    )
    return preencoder

def build_encoder(args):
    if args.enc_type == "vanilla":
        encoder_layer = TransformerEncoderLayer(
            d_model=args.enc_dim,
            nhead=args.enc_nhead,
            dim_feedforward=args.enc_ffn_dim,
            dropout=args.enc_dropout,
            activation=args.enc_activation,
        )
        encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=args.enc_nlayers
        )
    elif args.enc_type in ["masked"]:
        # This part of your code remains unchanged, providing a different encoder option
        encoder_layer = TransformerEncoderLayer(
            d_model=args.enc_dim, nhead=args.enc_nhead, dim_feedforward=args.enc_ffn_dim,
            dropout=args.enc_dropout, activation=args.enc_activation,
        )
        interim_downsampling = PointnetSAModuleVotes(
            radius=0.4, nsample=32, npoint=args.preenc_npoints // 2,
            mlp=[args.enc_dim, 256, 256, args.enc_dim], normalize_xyz=True,
        )
        masking_radius = [math.pow(x, 2) for x in [0.4, 0.8, 1.2]]
        encoder = MaskedTransformerEncoder(
            encoder_layer=encoder_layer, num_layers=3,
            interim_downsampling=interim_downsampling, masking_radius=masking_radius,
        )
    else:
        raise ValueError(f"Unknown encoder type {args.enc_type}")
    return encoder

def build_pointnetencoder(args):
    """ The main builder function called by your `encoder.py` """
    pre_encoder = build_preencoder(args)
    encoder = build_encoder(args)
    model = PointNetEncoder(
        pre_encoder,
        encoder,
        encoder_dim=args.enc_dim,
        mlp_dropout=args.mlp_dropout,
        num_queries=args.nqueries,
    )
    return model
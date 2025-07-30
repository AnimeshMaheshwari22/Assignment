import torch
import torch.nn as nn
import timm
import numpy as np
from PIL import Image

from models.pointnet_enc import build_pointnetencoder
from models.attention_modules import (
    MultiHeadDeformableAttention2D,
    MultiHeadDeformableAttention3D,
    get_reference_points_2d,
    ImageToDepthCrossAttention,
    DepthToImageCrossAttention,
)

class Encoder(nn.Module):
    def __init__(self, pointnet_args, swin_model_name='swin_base_patch4_window7_224', attn_2d_heads=8, attn_3d_heads=4, attn_sampling_points=4, fusion_dim=256):
        super().__init__()
        self.swin_encoder = timm.create_model(swin_model_name, pretrained=True, features_only=True)
        swin_embed_dim = self.swin_encoder.feature_info.channels()[-1]
        self.deformable_attn_2d = MultiHeadDeformableAttention2D(
            embed_dim=swin_embed_dim, num_heads=attn_2d_heads, num_sampling_points=attn_sampling_points
        )
        self.pointnet_encoder = build_pointnetencoder(pointnet_args)
        self.deformable_attn_3d = MultiHeadDeformableAttention3D(
            embed_dim=pointnet_args.enc_dim, num_heads=attn_3d_heads, num_sampling_points=attn_sampling_points
        )
        self.img_proj = nn.Linear(swin_embed_dim, fusion_dim)
        self.pc_proj = nn.Linear(pointnet_args.enc_dim, fusion_dim)
        self.img_to_pc_fusion = ImageToDepthCrossAttention(fusion_dim=fusion_dim)
        self.pc_to_img_fusion = DepthToImageCrossAttention(fusion_dim=fusion_dim)

    def forward(self, rgb_image, point_cloud):
        image_feature_maps = self.swin_encoder(rgb_image)
        image_features = image_feature_maps[-1]
        N, H, W, E = image_features.shape
        ref_points_2d = get_reference_points_2d(H, W, device=image_features.device).unsqueeze(0).expand(N, -1, -1, -1)
        refined_image_features = self.deformable_attn_2d(image_features, ref_points_2d)
        enc_xyz, enc_features, enc_inds = self.pointnet_encoder.run_encoder(point_cloud)
        query_features_3d = enc_features.permute(1, 0, 2)
        refined_pc_features = self.deformable_attn_3d(query_features_3d, enc_xyz)
        img_feats_flat = refined_image_features.view(N, H * W, -1)
        img_feats_proj = self.img_proj(img_feats_flat)
        pc_feats_proj = self.pc_proj(refined_pc_features)
        updated_img_feats_flat = self.img_to_pc_fusion(img_feats_proj, pc_feats_proj)
        updated_pc_feats = self.pc_to_img_fusion(pc_feats_proj, updated_img_feats_flat)
        updated_img_feats = updated_img_feats_flat.view(N, H, W, -1)
        return {
            "updated_image_features": updated_img_feats,
            "updated_depth_features": updated_pc_feats,
            "point_cloud_xyz": enc_xyz,
        }

if __name__ == '__main__':
    class PointNetArgs:
        use_color = False
        enc_dim = 256
        preenc_npoints = 2048
        enc_type = "vanilla"
        enc_nhead = 4
        enc_ffn_dim = 128
        enc_dropout = 0.1
        enc_activation = "relu"
        enc_nlayers = 3
        mlp_dropout = 0.3
        nqueries = 256

    args = PointNetArgs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(pointnet_args=args, fusion_dim=args.enc_dim).to(device)
    encoder.eval()
    IMAGE_PATH = r"D:\Assignment\processed_dataset\rgb_img_1.png"
    PC_PATH = r"D:\Assignment\scripts\pc_1.npy"

    try:
        image = Image.open(IMAGE_PATH).convert("RGB")
        data_config = timm.data.resolve_model_data_config(encoder.swin_encoder)
        transforms = timm.data.create_transform(**data_config, is_training=False)
        image_tensor = transforms(image).unsqueeze(0).to(device)
        pc_np = np.load(PC_PATH)
        if pc_np.shape[-1] != 3:
            pc_np = pc_np.reshape(-1, 3)
        pc_tensor = torch.tensor(pc_np, dtype=torch.float32).unsqueeze(0).to(device)
    except FileNotFoundError as e:
        print(f"❌ ERROR: Could not find a file. Please check your paths.\n{e}")
        exit()

    with torch.no_grad():
        output = encoder(image_tensor, pc_tensor)

    print("Output dictionary keys:", output.keys())
    img_feat = output["updated_image_features"]
    pc_feat = output["updated_depth_features"]
    pc_xyz = output["point_cloud_xyz"]
    print(f"Shape of updated image features: {img_feat.shape}")
    print(f"Shape of updated depth features: {pc_feat.shape}")
    print(f"Shape of final point cloud coordinates: {pc_xyz.shape}")
    assert img_feat.shape == (1, 7, 7, args.enc_dim)
    assert pc_feat.shape == (1, args.preenc_npoints, args.enc_dim)
    assert pc_xyz.shape == (1, args.preenc_npoints, 3)
    print("All output shapes match expected values. The two-output fusion structure works! 👍")

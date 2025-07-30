import os
import argparse
import torch
import numpy as np
from PIL import Image
import timm
import open3d as o3d
from typing import Optional, Union
import numpy.typing as npt

from main_model import Detector3D
from encoder import Encoder
from decoder import FusionDecoder, FusionDecoderLayer


def display_scene(
    points: Union[str, npt.NDArray],
    detected_boxes: Optional[Union[str, npt.NDArray]] = None,
    reference_boxes: Optional[Union[str, npt.NDArray]] = None,
    rgb: Optional[Union[str, npt.NDArray]] = None
) -> None:
    if isinstance(points, str):
        pts = np.load(points)
    elif isinstance(points, np.ndarray):
        pts = points
    else:
        raise TypeError

    if isinstance(rgb, str):
        img = Image.open(rgb).convert('RGB')
    elif isinstance(rgb, Image.Image):
        img = rgb
    else:
        img = None

    if pts.ndim == 3 and pts.shape[0] == 3:
        pts = pts.transpose(1, 2, 0).reshape(-1, 3)
    else:
        pts = pts.reshape(-1, 3)

    geom = o3d.geometry.PointCloud()
    geom.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

    if img is not None:
        arr = np.array(img).astype(np.float64) / 255.0
        geom.colors = o3d.utility.Vector3dVector(arr.reshape(-1, 3))

    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    viewer.add_geometry(geom)

    if reference_boxes is not None:
        ref = np.load(reference_boxes) if isinstance(reference_boxes, str) else reference_boxes
        for bb in ref.astype(np.float64):
            cube = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(bb))
            cube.color = (1.0, 0.0, 0.0)
            viewer.add_geometry(cube)

    if detected_boxes is not None:
        pred = np.load(detected_boxes) if isinstance(detected_boxes, str) else detected_boxes
        for bb in pred.astype(np.float64):
            cube = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(bb))
            cube.color = (0.0, 0.0, 0.5)
            viewer.add_geometry(cube)

    viewer.run()
    viewer.destroy_window()


def preprocess_pointcloud(path):
    arr = np.load(path)
    flat = arr.transpose(1, 2, 0).reshape(-1, 3)
    return arr, torch.tensor(flat, dtype=torch.float32)


def forward_model(model, img_path, cloud_tensor, transform, device, threshold=0.25):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x, cloud_tensor.unsqueeze(0))

    logits = pred['pred_logits'].squeeze(0)
    boxes = pred['pred_corners'].squeeze(0)
    scores = logits.sigmoid().squeeze(-1)

    keep = scores > threshold
    return boxes[keep].cpu().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--num', type=str, required=True)
    parser.add_argument('--model_path', type=str, default="results/detector3d_final.pth")
    parser.add_argument('--score_thresh', type=float, default=0.6)
    args = parser.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    p_path = os.path.join(args.data_root, f"pc_{args.num}.npy")
    i_path = os.path.join(args.data_root, f"rgb_img_{args.num}.png")
    gt_path = os.path.join(args.data_root, f"bbox3d_{args.num}.npy")

    if not os.path.exists(p_path) or not os.path.exists(i_path):
        raise FileNotFoundError("Missing input files")

    class Args:
        use_color = False
        enc_dim = 256
        preenc_npoints = 2048
        enc_type = "vanilla"
        enc_nhead = 4
        enc_ffn_dim = 128
        enc_dropout = 0.1
        enc_activation = "relu"
        enc_nlayers = 5
        mlp_dropout = 0.3
        nqueries = 16

    enc = Encoder(pointnet_args=Args(), fusion_dim=256)
    dec_layer = FusionDecoderLayer(d_model=256)
    dec = FusionDecoder(dec_layer, num_layers=6, d_model=256, num_classes=1, num_queries=Args.nqueries)
    net = Detector3D(enc, dec, criterion=None).to(dev)

    print(f"Model: {args.model_path}")
    net.load_state_dict(torch.load(args.model_path, map_location=dev))
    net.eval()

    dummy = timm.create_model('swin_base_patch4_window7_224', pretrained=False)
    transform = timm.data.create_transform(**timm.data.resolve_model_data_config(dummy), is_training=False)

    pc_np, pc_tensor = preprocess_pointcloud(p_path)
    prediction = forward_model(net, i_path, pc_tensor.to(dev), transform, dev, args.score_thresh)

    display_scene(
        points=pc_np,
        detected_boxes=prediction,
        reference_boxes=gt_path if os.path.exists(gt_path) else None,
        rgb=i_path
    )

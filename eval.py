import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np
import timm
from tqdm import tqdm
import random
import argparse
from scipy.spatial.transform import Rotation as R

from main_model import Detector3D
from encoder import Encoder
from decoder import FusionDecoder, FusionDecoderLayer
from train import Custom3DDataset, custom_collate_fn

def box_8_param_to_corners(box_8_param):
    center = box_8_param[:3]
    dims = np.exp(box_8_param[3:6])
    yaw = np.arctan2(box_8_param[7], box_8_param[6])
    
    rotation = R.from_euler('z', yaw).as_matrix()
    
    l, w, h = dims[0], dims[1], dims[2]
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    
    corners = np.vstack([x_corners, y_corners, z_corners])
    
    corners = rotation @ corners
    corners[0, :] += center[0]
    corners[1, :] += center[1]
    corners[2, :] += center[2]
    
    return corners.T

def polygon_clip(subject_polygon, clip_polygon):
    def is_inside(p1, p2, q):
        return (p2[0] - p1[0]) * (q[1] - p1[1]) > (p2[1] - p1[1]) * (q[0] - p1[0])

    def compute_intersection(p1, p2, s, e):
        dc = [p1[0] - p2[0], p1[1] - p2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        
        n1 = p1[0] * p2[1] - p1[1] * p2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        
        denominator = dc[0] * dp[1] - dc[1] * dp[0]
        if abs(denominator) < 1e-6:
            return None
        
        n3 = 1.0 / denominator
        x = (n1 * dp[0] - n2 * dc[0]) * n3
        y = (n1 * dp[1] - n2 * dc[1]) * n3
        return [x, y]

    final_polygon = list(subject_polygon)
    for i in range(len(clip_polygon)):
        clip_p1 = clip_polygon[i - 1]
        clip_p2 = clip_polygon[i]

        input_list = final_polygon
        final_polygon = []
        
        if not input_list:
            return []

        s = input_list[-1]
        for e in input_list:
            if is_inside(clip_p1, clip_p2, e):
                if not is_inside(clip_p1, clip_p2, s):
                    intersection = compute_intersection(clip_p1, clip_p2, s, e)
                    if intersection: final_polygon.append(intersection)
                final_polygon.append(list(e))
            elif is_inside(clip_p1, clip_p2, s):
                intersection = compute_intersection(clip_p1, clip_p2, s, e)
                if intersection: final_polygon.append(intersection)
            s = e
            
    return final_polygon

def get_3d_iou(box1_corners, box2_corners):
    box1_bev = box1_corners[:4, :2]
    box2_bev = box2_corners[:4, :2]
    
    clipped_polygon = polygon_clip(box1_bev.tolist(), box2_bev.tolist())
    if len(clipped_polygon) < 3:
        return 0.0

    area_clipped = 0.0
    for i in range(len(clipped_polygon)):
        p1 = clipped_polygon[i]
        p2 = clipped_polygon[(i + 1) % len(clipped_polygon)]
        area_clipped += p1[0] * p2[1] - p2[0] * p1[1]
    area_clipped = abs(area_clipped) / 2.0

    area1 = np.linalg.norm(box1_corners[0, :2] - box1_corners[1, :2]) * np.linalg.norm(box1_corners[1, :2] - box1_corners[2, :2])
    area2 = np.linalg.norm(box2_corners[0, :2] - box2_corners[1, :2]) * np.linalg.norm(box2_corners[1, :2] - box2_corners[2, :2])
    
    bev_union = area1 + area2 - area_clipped
    bev_iou = area_clipped / bev_union if bev_union > 0 else 0.0
    
    z_min1, z_max1 = box1_corners[:, 2].min(), box1_corners[:, 2].max()
    z_min2, z_max2 = box2_corners[:, 2].min(), box2_corners[:, 2].max()
    
    z_intersection = max(0, min(z_max1, z_max2) - max(z_min1, z_min2))
    z_union = (z_max1 - z_min1) + (z_max2 - z_min2) - z_intersection
    
    iou_3d = bev_iou * (z_intersection / z_union if z_union > 0 else 0.0)
    
    return iou_3d

def evaluate(model, dataloader, device, iou_threshold=0.25, score_threshold=0.25):
    model.eval()
    
    all_predictions = []
    all_ground_truths = []
    
    print(f"Gathering predictions with score > {score_threshold} from the test set...")
    for i, (images, point_clouds, targets_list) in enumerate(tqdm(dataloader, desc="Evaluating")):
        images = images.to(device)
        point_clouds = point_clouds.to(device)
        
        with torch.no_grad():
            outputs = model(images, point_clouds)
        
        pred_logits = outputs['pred_logits'].cpu()
        pred_boxes = outputs['pred_boxes'].cpu()
        
        for sample_idx in range(len(targets_list)):
            scores = torch.sigmoid(pred_logits[sample_idx, :, 0])
            boxes = pred_boxes[sample_idx]
            
            mask = scores > score_threshold
            confident_scores = scores[mask]
            confident_boxes = boxes[mask]
            
            for score, box in zip(confident_scores, confident_boxes):
                all_predictions.append({
                    "sample_idx": i * dataloader.batch_size + sample_idx,
                    "score": score.item(),
                    "box_params": box.numpy()
                })
            
            gt_boxes = targets_list[sample_idx]['boxes'].numpy()
            for gt_box in gt_boxes:
                all_ground_truths.append({
                    "sample_idx": i * dataloader.batch_size + sample_idx,
                    "box_params": gt_box,
                    "used": False
                })

    all_predictions.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"Calculating AP @ {iou_threshold} IoU for {len(all_predictions)} filtered predictions...")
    tp = np.zeros(len(all_predictions))
    fp = np.zeros(len(all_predictions))
    num_gt_boxes = len(all_ground_truths)
    
    ate_list, ase_list, aoe_list = [], [], []

    for pred_idx, pred in enumerate(tqdm(all_predictions, desc="Matching Preds/GTs")):
        sample_gts = [gt for gt in all_ground_truths if gt['sample_idx'] == pred['sample_idx']]
        if not sample_gts:
            fp[pred_idx] = 1
            continue

        pred_corners = box_8_param_to_corners(pred['box_params'])
        
        ious = []
        for gt in sample_gts:
            gt_corners = box_8_param_to_corners(gt['box_params'])
            ious.append(get_3d_iou(pred_corners, gt_corners))
        
        max_iou = max(ious) if ious else 0
        best_gt_idx = np.argmax(ious) if ious else -1

        if max_iou >= iou_threshold:
            matched_gt = sample_gts[best_gt_idx]
            if not matched_gt['used']:
                tp[pred_idx] = 1
                matched_gt['used'] = True
                
                pred_params = pred['box_params']
                gt_params = matched_gt['box_params']
                
                ate_list.append(np.linalg.norm(pred_params[:3] - gt_params[:3]))
                
                aligned_pred_params = np.copy(pred_params)
                aligned_pred_params[:3] = gt_params[:3]
                ase_list.append(1 - get_3d_iou(box_8_param_to_corners(aligned_pred_params), box_8_param_to_corners(gt_params)))

                pred_yaw = np.arctan2(pred_params[7], pred_params[6])
                gt_yaw = np.arctan2(gt_params[7], gt_params[6])
                aoe = abs(pred_yaw - gt_yaw)
                if aoe > np.pi: aoe = 2 * np.pi - aoe
                aoe_list.append(aoe)
                
            else:
                fp[pred_idx] = 1
        else:
            fp[pred_idx] = 1

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recalls = tp_cumsum / (num_gt_boxes + 1e-6)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    
    ap = 0.0
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.

    mean_ate = np.mean(ate_list) if ate_list else 0
    mean_ase = np.mean(ase_list) if ase_list else 0
    mean_aoe = np.mean(aoe_list) if aoe_list else 0
    
    return {
        f"mAP_@{iou_threshold}": ap,
        "ATE": mean_ate,
        "ASE": mean_ase,
        "AOE": mean_aoe
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate 3D Object Detector")
    parser.add_argument('--data_root', type=str, required=True, help="Path to the root of the processed dataset directory.")
    parser.add_argument('--model_path', type=str, default="./results/detector3d_final.pth", help="Path to the trained model weights (.pth file).")
    parser.add_argument('--iou_thresh', type=float, default=0.25, help="IoU threshold for considering a detection a True Positive.")
    parser.add_argument('--score_thresh', type=float, default=0.25, help="Score threshold to filter predictions before evaluation.")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    JSONL_PATH = os.path.join(args.data_root, "dataset.jsonl")
    if not os.path.exists(JSONL_PATH):
        raise FileNotFoundError(f"Dataset JSONL file not found at: {JSONL_PATH}")

    print("Loading test data split...")
    with open(JSONL_PATH, 'r') as f:
        all_samples = [json.loads(line) for line in f]
    
    num_samples = len(all_samples)
    indices = list(range(num_samples))
    random.seed(42)
    random.shuffle(indices)
    
    test_split = int(np.floor(0.2 * num_samples))
    test_indices = indices[:test_split]
    test_samples = [all_samples[i] for i in test_indices]
    
    print(f"Loaded {len(test_samples)} samples for evaluation.")

    dummy_swin = timm.create_model('swin_base_patch4_window7_224', pretrained=False)
    data_config = timm.data.resolve_model_data_config(dummy_swin)
    val_test_transforms = timm.data.create_transform(**data_config, is_training=False)

    test_dataset = Custom3DDataset(test_samples, args.data_root, val_test_transforms, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=custom_collate_fn)

    print("Loading trained model...")
    class PointNetArgs:
        use_color=False; enc_dim=256; preenc_npoints=2048; enc_type="vanilla"
        enc_nhead=4; enc_ffn_dim=128; enc_dropout=0.1; enc_activation="relu"
        enc_nlayers=5; mlp_dropout=0.3; nqueries=16
    
    encoder = Encoder(pointnet_args=PointNetArgs(), fusion_dim=256)
    decoder_layer = FusionDecoderLayer(d_model=256)
    decoder = FusionDecoder(decoder_layer, num_layers=6, d_model=256, num_classes=1, num_queries=PointNetArgs.nqueries)
    model = Detector3D(encoder, decoder, criterion=None).to(device)
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model weights not found at: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    results = evaluate(model, test_loader, device, 
                       iou_threshold=args.iou_thresh, 
                       score_threshold=args.score_thresh)
    
    print("\n--- 📊 Evaluation Results ---")
    print(f"Score Threshold: {args.score_thresh}")
    print(f"IoU Threshold:   {args.iou_thresh}")
    print("---------------------------------")
    print(f"mAP: {results[f'mAP_@{args.iou_thresh}']:.4f}")
    print("\n--- Diagnostic Metrics (on True Positives) ---")
    print(f"Average Translation Error (ATE): {results['ATE']:.4f} meters")
    print(f"Average Scale Error (ASE):       {results['ASE']:.4f} (1 - 3D IoU)")
    print(f"Average Orientation Error (AOE): {results['AOE']:.4f} radians")
    print("------------------------------------------------\n")
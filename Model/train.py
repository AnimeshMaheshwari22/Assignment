import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import json
import os
import numpy as np
from PIL import Image
import timm
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import random

from main_model import Detector3D
from encoder import Encoder
from decoder import FusionDecoder, FusionDecoderLayer
from main_model import HungarianMatcher3D, SetCriterion3D

def gt_corners_to_8_params(corners):
    center = corners.mean(axis=0)
    edge_vector = corners[1, :2] - corners[0, :2]
    yaw = np.arctan2(edge_vector[1], edge_vector[0])
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    rotation_matrix = np.array([[cos_yaw, sin_yaw], [-sin_yaw, cos_yaw]])
    aligned_corners_bev = corners[:, :2] @ rotation_matrix.T
    bev_min = aligned_corners_bev.min(axis=0)
    bev_max = aligned_corners_bev.max(axis=0)
    size_bev = bev_max - bev_min
    z_min, z_max = corners[:, 2].min(), corners[:, 2].max()
    size_z = z_max - z_min
    size = np.array([size_bev[0], size_bev[1], size_z])
    log_dims = np.log(size + 1e-6)
    return np.concatenate([center, log_dims, [cos_yaw, sin_yaw]])

def random_flip_scene(point_cloud, corners):
    if np.random.rand() > 0.5:
        point_cloud[:, 1] *= -1
        corners[:, :, 1] *= -1
    return point_cloud, corners

def random_rotate_scene(point_cloud, corners):
    angle = np.random.uniform(-np.pi / 4, np.pi / 4)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0, 0, 1]
    ])
    point_cloud = point_cloud @ rot_matrix.T
    num_boxes = corners.shape[0]
    corners_flat = corners.reshape(num_boxes * 8, 3)
    rotated_corners_flat = corners_flat @ rot_matrix.T
    corners = rotated_corners_flat.reshape(num_boxes, 8, 3)
    return point_cloud, corners

def random_scale_scene(point_cloud, corners):
    scale = np.random.uniform(0.9, 1.1)
    point_cloud *= scale
    corners *= scale
    return point_cloud, corners

class Custom3DDataset(Dataset):
    def __init__(self, samples, data_dir, image_transforms, augment=False):
        self.data_dir = data_dir
        self.image_transforms = image_transforms
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        image = Image.open(os.path.join(self.data_dir, sample_info['rgb_path'])).convert("RGB")
        image_tensor = self.image_transforms(image)
        point_cloud = np.load(os.path.join(self.data_dir, sample_info['pc_path']))
        if point_cloud.shape[-1] != 3: point_cloud = point_cloud.reshape(-1, 3)
        bbox_corners = np.load(os.path.join(self.data_dir, sample_info['bbox3d_path']))
        if self.augment:
            point_cloud, bbox_corners = random_flip_scene(point_cloud, bbox_corners)
            point_cloud, bbox_corners = random_rotate_scene(point_cloud, bbox_corners)
            point_cloud, bbox_corners = random_scale_scene(point_cloud, bbox_corners)
        bboxes_8param = np.array([gt_corners_to_8_params(c) for c in bbox_corners])
        labels = torch.zeros(bboxes_8param.shape[0], dtype=torch.long)
        targets = {
            'labels': labels,
            'boxes': torch.tensor(bboxes_8param, dtype=torch.float32),
            'corners': torch.tensor(bbox_corners, dtype=torch.float32)
        }
        return image_tensor, torch.tensor(point_cloud, dtype=torch.float32), targets

def custom_collate_fn(batch):
    images, point_clouds, targets = zip(*batch)
    batched_images = torch.stack(images, 0)
    max_points = max(pc.shape[0] for pc in point_clouds)
    padded_pcs = []
    for pc in point_clouds:
        padding_needed = max_points - pc.shape[0]
        if padding_needed > 0:
            padding = torch.zeros(padding_needed, 3, dtype=pc.dtype, device=pc.device)
            padded_pc = torch.cat([pc, padding], dim=0)
            padded_pcs.append(padded_pc)
        else:
            padded_pcs.append(pc)
    batched_point_clouds = torch.stack(padded_pcs, 0)
    batched_targets = list(targets)
    return batched_images, batched_point_clouds, batched_targets

if __name__ == '__main__':
    DATA_DIR = "/home/animeshmaheshwari/Documents/ImageSegmentation/ObjectDetection/Assignment/processed_dataset"
    JSONL_PATH = os.path.join(DATA_DIR, "dataset.jsonl")
    RESULTS_DIR = "results"
    EPOCHS = 50
    BATCH_SIZE = 4
    LEARNING_RATE = 5e-5
    NUM_CLASSES = 1
    NUM_QUERIES = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Splitting dataset...")
    with open(JSONL_PATH, 'r') as f:
        all_samples = [json.loads(line) for line in f]
    num_samples = len(all_samples)
    indices = list(range(num_samples))
    random.shuffle(indices)
    test_split = int(np.floor(0.2 * num_samples))
    val_split = int(np.floor(0.1 * (num_samples - test_split)))
    test_indices = indices[:test_split]
    train_val_indices = indices[test_split:]
    val_indices = train_val_indices[:val_split]
    train_indices = train_val_indices[val_split:]
    train_samples = [all_samples[i] for i in train_indices]
    val_samples = [all_samples[i] for i in val_indices]
    test_samples = [all_samples[i] for i in test_indices]

    print(f"Total samples: {num_samples}")
    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")

    class PointNetArgs:
        use_color = False; enc_dim = 256; preenc_npoints = 2048; enc_type = "vanilla"
        enc_nhead = 4; enc_ffn_dim = 128; enc_dropout = 0.1; enc_activation = "relu"
        enc_nlayers = 5; mlp_dropout = 0.3; nqueries = NUM_QUERIES

    encoder = Encoder(pointnet_args=PointNetArgs(), fusion_dim=256)
    decoder_layer = FusionDecoderLayer(d_model=256)
    decoder = FusionDecoder(decoder_layer, num_layers=6, d_model=256, num_classes=NUM_CLASSES, num_queries=NUM_QUERIES)
    matcher = HungarianMatcher3D(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)
    weight_dict = {'loss_ce': 2.0, 'loss_bbox': 5.0, 'loss_giou': 10.0}
    criterion = SetCriterion3D(num_classes=NUM_CLASSES, matcher=matcher, weight_dict=weight_dict).to(device)
    model = Detector3D(encoder, decoder, criterion).to(device)

    dummy_swin = timm.create_model('swin_base_patch4_window7_224', pretrained=False)
    data_config = timm.data.resolve_model_data_config(dummy_swin)
    train_transforms = timm.data.create_transform(**data_config, is_training=True)
    val_test_transforms = timm.data.create_transform(**data_config, is_training=False)

    train_dataset = Custom3DDataset(train_samples, DATA_DIR, train_transforms, augment=True)
    val_dataset = Custom3DDataset(val_samples, DATA_DIR, val_test_transforms, augment=False)
    test_dataset = Custom3DDataset(test_samples, DATA_DIR, val_test_transforms, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=custom_collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    lr_scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

    os.makedirs(os.path.join(RESULTS_DIR, 'logs'), exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(RESULTS_DIR, 'logs'))

    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Training")
        for i, (images, point_clouds, targets_list) in enumerate(loop):
            images = images.to(device)
            point_clouds = point_clouds.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets_list]
            loss, loss_dict = model(images, point_clouds, targets=targets)
            if torch.isnan(loss): 
                print(f"NaN loss detected at epoch {epoch+1}, batch {i}. Skipping batch.")
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            global_step = epoch * len(train_loader) + i
            writer.add_scalar('Train/loss_total', loss.item(), global_step)
            for k, v in loss_dict.items():
                writer.add_scalar(f'Train/{k}', v.item(), global_step)
            loop.set_postfix(loss=loss.item(), giou=loss_dict['loss_giou'].item())

        model.eval()
        total_val_loss = 0
        val_loop = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Validation", leave=False)
        with torch.no_grad():
            for images, point_clouds, targets_list in val_loop:
                images = images.to(device)
                point_clouds = point_clouds.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets_list]
                outputs = model(images, point_clouds, targets=None)
                loss, loss_dict_details = criterion(outputs, targets)
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        writer.add_scalar('Validation/avg_loss_epoch', avg_val_loss, epoch)
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Avg Validation Loss: {avg_val_loss:.4f}")
        lr_scheduler.step()

    writer.close()
    print("\nTraining complete!")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, "detector3d_final.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    print("\n--- Test Set Sample Indices ---")
    print(f"The following {len(test_indices)} indices (from the original dataset.jsonl) form the test set:")
    print(sorted(test_indices))

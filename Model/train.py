import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np
from PIL import Image
import timm

# --- Import the main model and its components ---
# The user has specified their main model file is named 'main_model.py'
from main_model import Detector3D, FusionDecoder, FusionDecoderLayer, build_matcher_3d, Encoder

# ==============================================================================
# == Bounding Box Conversion Helper (Unchanged)                               ==
# ==============================================================================

def corners_to_center_size_yaw(corners):
    """
    Converts 8-corner 3D bounding box representation to the model's 8-parameter format.
    
    Args:
        corners (np.ndarray): Bounding box corners, shape (Num_Objects, 8, 3)
    
    Returns:
        np.ndarray: Bounding boxes in (center_x, center_y, center_z, w, h, l, sin(yaw), cos(yaw))
                    format, shape (Num_Objects, 8)
    """
    processed_boxes = []
    for box_corners in corners:
        # Center is the mean of the 8 corner points
        center = box_corners.mean(axis=0)
        
        # To find yaw, we assume the "front" of the box is defined by the first two corners.
        # This might need adjustment based on your specific dataset's corner ordering.
        # We calculate the vector of one of the base edges.
        edge_vector = box_corners[1] - box_corners[0]
        yaw = np.arctan2(edge_vector[1], edge_vector[0])
        
        # To find size, we create a rotation matrix to align the box with the axes.
        # This makes finding width, height, and length trivial.
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        rotation_matrix = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw,  cos_yaw, 0],
            [0,        0,       1]
        ])
        
        # Rotate the box corners to be axis-aligned
        aligned_corners = box_corners @ rotation_matrix.T
        min_coords = aligned_corners.min(axis=0)
        max_coords = aligned_corners.max(axis=0)
        size = max_coords - min_coords
        
        # Assemble the 8-parameter representation
        processed_box = np.concatenate([center, size, [np.sin(yaw), np.cos(yaw)]])
        processed_boxes.append(processed_box)
        
    return np.array(processed_boxes, dtype=np.float32)


# ==============================================================================
# == 1. Custom Dataset for 3D Object Detection (Unchanged)                    ==
# ==============================================================================

class Custom3DDataset(Dataset):
    """
    Custom PyTorch Dataset for loading 3D detection data as described.
    It reads a .jsonl file where each line points to the data for one frame.
    """
    def __init__(self, jsonl_path, data_dir, image_transforms):
        """
        Args:
            jsonl_path (str): Path to the .jsonl manifest file.
            data_dir (str): Path to the root directory where data files are stored.
            image_transforms: PyTorch transforms to be applied to the RGB images.
        """
        self.data_dir = data_dir
        self.image_transforms = image_transforms
        self.samples = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        # --- Load RGB Image ---
        rgb_path = os.path.join(self.data_dir, sample_info['rgb_path'])
        image = Image.open(rgb_path).convert("RGB")
        image_tensor = self.image_transforms(image)
        
        # --- Load Point Cloud ---
        pc_path = os.path.join(self.data_dir, sample_info['pc_path'])
        point_cloud = np.load(pc_path)
        # Ensure point cloud has shape (N, 3) - this handles the (3, H, W) case
        if point_cloud.shape[-1] != 3:
            point_cloud = point_cloud.reshape(-1, 3)
        pc_tensor = torch.tensor(point_cloud, dtype=torch.float32)
        
        # --- Load and Process 3D Bounding Boxes ---
        bbox_path = os.path.join(self.data_dir, sample_info['bbox3d_path'])
        bbox_corners = np.load(bbox_path) # Shape (Num_Objects, 8, 3)
        
        # Convert the corners to the model's expected 8-parameter format
        bboxes_8param = corners_to_center_size_yaw(bbox_corners)
        
        # Since there is only one class ("Object"), the label for all boxes is 0.
        labels = torch.zeros(bboxes_8param.shape[0], dtype=torch.long)
        
        # Prepare the target dictionary in the format expected by the model
        targets = {
            'labels': labels,
            'boxes': torch.tensor(bboxes_8param, dtype=torch.float32)
        }
        
        return image_tensor, pc_tensor, targets

def collate_fn(batch):
    """
    Custom collate function for a batch size of 1.
    It returns point clouds and targets as lists, which the training loop will handle.
    """
    images, point_clouds, targets = zip(*batch)
    batched_images = torch.stack(images, 0)
    # Return point_clouds and targets as lists
    return batched_images, list(point_clouds), list(targets)


# ==============================================================================
# == 2. Main Training Loop (Updated)                                          ==
# ==============================================================================

if __name__ == '__main__':
    # ✅ CHANGE: Enable anomaly detection to get a more detailed traceback
    torch.autograd.set_detect_anomaly(True)

    # --- Configuration ---
    DATA_DIR = "D:\Assignment\processed_dataset" # Root directory containing rgb, pc, and bbox files
    JSONL_PATH = os.path.join(DATA_DIR, "dataset.jsonl")
    EPOCHS = 10
    # ✅ CHANGE: Set batch size to 1 to process one sample at a time
    BATCH_SIZE = 1
    LEARNING_RATE = 1e-4
    
    # --- Model Configuration ---
    class PointNetArgs:
        use_color = False; enc_dim = 256; preenc_npoints = 2048; enc_type = "vanilla"; enc_nhead = 4; enc_ffn_dim = 128
        enc_dropout = 0.1; enc_activation = "relu"; enc_nlayers = 3; mlp_dropout = 0.3; nqueries = 256
    
    class MatcherArgs:
        set_cost_class = 2.0; set_cost_bbox = 5.0; focal_alpha = 0.25

    pointnet_args = PointNetArgs()
    matcher_args = MatcherArgs()
    
    NUM_CLASSES = 1
    NUM_QUERIES = 256
    FUSION_DIM = 256
    DECODER_LAYERS = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Training on {device} ---")

    # --- 1. Build Model ---
    print("Building model...")
    encoder = Encoder(pointnet_args=pointnet_args, fusion_dim=FUSION_DIM)
    decoder_layer = FusionDecoderLayer(d_model=FUSION_DIM)
    decoder = FusionDecoder(decoder_layer, num_layers=DECODER_LAYERS, d_model=FUSION_DIM, num_classes=NUM_CLASSES + 1, num_queries=NUM_QUERIES)
    matcher = build_matcher_3d(args=matcher_args)
    model = Detector3D(encoder, decoder, matcher, num_classes=NUM_CLASSES).to(device)
    print("✅ Model built successfully!")

    # --- 2. Prepare Dataset ---
    print("Loading dataset...")
    dummy_swin = timm.create_model('swin_base_patch4_window7_224', pretrained=False)
    data_config = timm.data.resolve_model_data_config(dummy_swin)
    image_transforms = timm.data.create_transform(**data_config, is_training=True)
    
    dataset = Custom3DDataset(jsonl_path=JSONL_PATH, data_dir=DATA_DIR, image_transforms=image_transforms)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=collate_fn)
    print(f"✅ Dataset loaded with {len(dataset)} samples.")

    # --- 3. Setup Optimizer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # --- 4. Training Loop ---
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for i, (images, point_clouds, targets) in enumerate(dataloader):
            images = images.to(device)
            # ✅ CHANGE: Since batch size is 1, `point_clouds` is a list with one tensor.
            # We stack it to create a single tensor of shape (1, N, 3) for the model.
            point_clouds_tensor = torch.stack(point_clouds, 0).to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss = model(images, point_clouds_tensor, targets=targets)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️ Invalid loss detected at epoch {epoch+1}, batch {i+1}. Skipping batch.")
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"--- Epoch {epoch+1} Finished --- Average Loss: {avg_loss:.4f} ---")

    print("\n🎉 Training complete!")

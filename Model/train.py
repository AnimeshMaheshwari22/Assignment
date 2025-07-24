# =================================================================================
# File: train.py
# Description: Training loop for the Vision-Language 3D Box Model.
# =================================================================================
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import numpy as np
import json
import os
from PIL import Image
from torchvision import transforms
import math
from functools import partial

# --- Imports from other project files ---
# These imports assume your other python files are in the same directory.
from main_model import VisionLanguage3DBoxModel
from transformer_blocks import VisionLanguageEncoder, TransformerDecoder, TransformerDecoderLayer
from matcher_and_criterion import HungarianMatcher3D, SetCriterion

# =================================================================================
# SECTION 1: DATASET PREPARATION
# =================================================================================

class ThreeDObjectDataset(Dataset):
    """
    Custom PyTorch Dataset to load data from a .jsonl file and corresponding data files.
    """
    def __init__(self, jsonl_path, data_root, text_prompt, image_transform=None):
        self.data_root = data_root
        self.text_prompt = text_prompt
        self.image_transform = image_transform
        
        self.samples = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        # --- Load RGB Image ---
        rgb_path = os.path.join(self.data_root, sample_info['rgb_path'])
        rgb_image = Image.open(rgb_path).convert("RGB")
        if self.image_transform:
            rgb_image = self.image_transform(rgb_image)

        # --- Load Point Cloud (as pseudo-image) ---
        pc_path = os.path.join(self.data_root, sample_info['pc_path'])
        point_cloud = np.load(pc_path)
        # Convert to tensor and apply same transforms as RGB
        point_cloud_tensor = torch.from_numpy(point_cloud).permute(2, 0, 1).float()
        if self.image_transform:
             # Normalize pc data similarly to images if needed
            point_cloud_tensor = self.image_transform(transforms.ToPILImage()(point_cloud_tensor))


        # --- Load 3D Bounding Box Targets ---
        bbox_path = os.path.join(self.data_root, sample_info['bbox3d_path'])
        # .item() is used to extract the dictionary from the numpy array
        bbox_data = np.load(bbox_path, allow_pickle=True).item()
        
        # Convert target numpy arrays to tensors
        target = {
            'labels': torch.tensor(bbox_data['labels'], dtype=torch.long),
            'center_unnormalized': torch.tensor(bbox_data['center_unnormalized'], dtype=torch.float),
            'size_unnormalized': torch.tensor(bbox_data['size_unnormalized'], dtype=torch.float),
            'angle_continuous': torch.tensor(bbox_data['angle_continuous'], dtype=torch.float)
        }

        # --- Assemble Sample ---
        sample = {
            'rgb_input': rgb_image,
            'pc_input': point_cloud_tensor,
            'text_prompts': self.text_prompt,
            'target': target
        }
        
        return sample

def collate_fn(batch):
    """Custom collate function to handle batches of dictionaries."""
    batched_data = {}
    
    # Batch standard tensors
    batched_data['rgb_input'] = torch.stack([item['rgb_input'] for item in batch])
    batched_data['pc_input'] = torch.stack([item['pc_input'] for item in batch])
    
    # Group text prompts and targets into lists
    batched_data['text_prompts'] = [item['text_prompts'] for item in batch]
    batched_data['targets'] = [item['target'] for item in batch]
    
    # Add dummy point cloud dimensions (in a real scenario, these would come from the dataset)
    batched_data['point_cloud_dims_min'] = torch.tensor([0.0, 0.0, 0.0])
    batched_data['point_cloud_dims_max'] = torch.tensor([1.0, 1.0, 1.0])
    
    return batched_data

# =================================================================================
# SECTION 2: TRAINING SCRIPT
# =================================================================================

def train():
    # --- Configuration ---
    DATA_DIR = "D:\Assignment\processed_dataset"
    JSONL_FILE = "D:\Assignment\scripts\dataset.jsonl"
    NUM_EPOCHS = 10
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # --- Setup Device ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- Initialize Model Components ---
    dataset_config = DummyDatasetConfig() # Using a placeholder for dataset properties
    
    matcher = HungarianMatcher3D(cost_class=1, cost_center=1, cost_size=1, cost_angle=1)
    weight_dict = {'loss_ce': 1, 'loss_center': 1, 'loss_size': 1, 'loss_angle': 1}
    losses = ['labels', 'boxes']
    criterion = SetCriterion(dataset_config.num_semcls, matcher=matcher, weight_dict=weight_dict, losses=losses).to(device)

    model = VisionLanguage3DBoxModel(
        dataset_config=dataset_config
    ).to(device)

    # --- Setup Optimizer ---
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # --- Setup Dataset and DataLoader ---
    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ThreeDObjectDataset(
        jsonl_path=JSONL_FILE,
        data_root=DATA_DIR,
        text_prompt="Objects", # Fixed text prompt
        image_transform=image_transforms
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0 # Set to > 0 for faster loading if not on Windows or in a notebook
    )

    # --- Training Loop ---
    print("\n--- Starting Training ---")
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        for i, batch in enumerate(data_loader):
            # Move data to device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items() if k != 'targets'}
            targets = [{k: v.to(device) for k, v in t.items()} for t in batch['targets']]

            # Forward pass
            optimizer.zero_grad()
            predictions, loss_dict = model(inputs, targets)
            
            # Calculate total weighted loss
            total_loss = sum(loss_dict[k] * weight_dict.get(k.rsplit('_', 1)[0], 1.0) for k in loss_dict)

            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            
            if (i + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(data_loader)}], Loss: {total_loss.item():.4f}")

        print(f"--- End of Epoch [{epoch+1}/{NUM_EPOCHS}], Average Loss: {epoch_loss / len(data_loader):.4f} ---\n")

    print("--- Training Finished ---")

if __name__ == '__main__':
    # This is a placeholder for a configuration class, as defined in main_model.py
    # In a real scenario, you would load this from a config file.
    class DummyDatasetConfig:
        def __init__(self): self.num_semcls, self.num_angle_bin = 18, 12
        def box_parametrization_to_corners(self, center, size, angle):
            B, Q, _ = center.shape; l, w, h = size[..., 0], size[..., 1], size[..., 2]
            corners_body = torch.tensor([ [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2], [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2], [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2] ], device=center.device)
            cos_a, sin_a = torch.cos(angle), torch.sin(angle); z = torch.zeros_like(cos_a); o = torch.ones_like(cos_a)
            rot_mat = torch.stack([torch.stack([cos_a,-sin_a,z]), torch.stack([sin_a,cos_a,z]), torch.stack([z,z,o])], dim=-2)
            rotated_corners = torch.einsum('bqij,jkq->bqik', rot_mat, corners_body)
            return rotated_corners + center.unsqueeze(-1)
            
    train()

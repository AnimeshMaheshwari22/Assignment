import os
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt

def visualize_sample(idx, data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    rgb_path = os.path.join(data_dir, f"rgb_img_{idx}.png")
    mask_path = os.path.join(data_dir, f"mask_{idx}.npy")
    pc_path = os.path.join(data_dir, f"pc_{idx}.npy")

    # Load data
    try:
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        mask = np.load(mask_path)  # (N_instances, H, W)
        pc = np.load(pc_path)      # (3, H, W)
    except Exception as e:
        print(f"[!] Skipping index {idx}: {e}")
        return

    # Generate mask overlay
    mask_overlay = rgb.copy()
    num_instances = mask.shape[0]
    colors = plt.cm.get_cmap("tab10", num_instances)

    for i in range(num_instances):
        instance_mask = mask[i].astype(np.uint8) * 255
        contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = (np.array(colors(i)[:3]) * 255).astype(np.uint8).tolist()
        cv2.drawContours(mask_overlay, contours, -1, color, 2)

    # Compute norm of point cloud
    x, y, z = pc[0], pc[1], pc[2]
    norm = np.sqrt(x**2 + y**2 + z**2)
    norm_valid = np.where(np.isfinite(norm), norm, np.nan)
    norm_min, norm_max = np.nanmin(norm_valid), np.nanmax(norm_valid)
    norm_normalized = (norm_valid - norm_min) / (norm_max - norm_min + 1e-6)
    pc_colormap = plt.cm.viridis(norm_normalized)
    pc_colormap = (pc_colormap[:, :, :3] * 255).astype(np.uint8)

    # Create figure and save
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    axs[0].imshow(rgb)
    axs[0].set_title(f"RGB Image {idx}")
    axs[0].axis("off")

    axs[1].imshow(mask_overlay)
    axs[1].set_title(f"Mask Overlay {idx}")
    axs[1].axis("off")

    axs[2].imshow(pc_colormap)
    axs[2].set_title(f"PC Norm Map {idx}")
    axs[2].axis("off")

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"visualization_{idx}.png")
    plt.savefig(out_path)
    plt.close(fig)

def get_all_indices(data_dir):
    pattern = re.compile(r"rgb_img_(\d+)\.png")
    indices = []
    for fname in os.listdir(data_dir):
        match = pattern.match(fname)
        if match:
            indices.append(int(match.group(1)))
    return sorted(indices)

if __name__ == "__main__":
    data_dir = "D:\Assignment\processed_dataset"       # <-- UPDATE THIS
    output_dir = "visualization"    # <-- UPDATE THIS

    all_indices = get_all_indices(data_dir)
    print(f"Found {len(all_indices)} samples.")

    for idx in all_indices:
        print(f"Processing index {idx}")
        visualize_sample(idx, data_dir, output_dir)

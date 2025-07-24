import os
import shutil

src_root = "D:\Assignment\dl_challenge"       # Change as needed
dst_root = "processed_dataset"      # Change as needed
os.makedirs(dst_root, exist_ok=True)

counter = 1
for folder in os.listdir(src_root):
    folder_path = os.path.join(src_root, folder)
    if not os.path.isdir(folder_path):
        continue
    
    rgb_src = os.path.join(folder_path, "rgb.jpg")
    bbox_src = os.path.join(folder_path, "bbox3d.npy")
    mask_src = os.path.join(folder_path, "mask.npy")
    pc_src = os.path.join(folder_path, "pc.npy")
    
    shutil.copy(rgb_src, os.path.join(dst_root, f"rgb_img_{counter}.png"))
    shutil.copy(bbox_src, os.path.join(dst_root, f"bbox3d_{counter}.npy"))
    shutil.copy(mask_src, os.path.join(dst_root, f"mask_{counter}.npy"))
    shutil.copy(pc_src, os.path.join(dst_root, f"pc_{counter}.npy"))
    
    counter += 1

print("✅ Files renamed and copied.")

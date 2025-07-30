import onnxruntime as ort
import numpy as np
from PIL import Image
import timm
import torch

def preprocess_image(image_path: str):
    dummy_swin = timm.create_model('swin_base_patch4_window7_224', pretrained=False)
    data_config = timm.data.resolve_model_data_config(dummy_swin)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    image = Image.open(image_path).convert("RGB")
    image_tensor = transforms(image)
    return image_tensor.unsqueeze(0).numpy()

def preprocess_point_cloud(pc_path: str, num_points: int):
    point_cloud = np.load(pc_path)
    if point_cloud.shape[-1] != 3:
        point_cloud = point_cloud.reshape(-1, 3)

    current_points = point_cloud.shape[0]
    if current_points > num_points:
        indices = np.random.choice(current_points, num_points, replace=False)
        point_cloud = point_cloud[indices, :]
    elif current_points < num_points:
        padding_needed = num_points - current_points
        padding = np.zeros((padding_needed, 3), dtype=point_cloud.dtype)
        point_cloud = np.vstack((point_cloud, padding))

    return np.expand_dims(point_cloud, axis=0).astype(np.float32)

def main():
    PC_NUM_POINTS = 2048
    IMAGE_FILE_PATH = "/home/animeshmaheshwari/Documents/ImageSegmentation/ObjectDetection/Assignment/processed_dataset/rgb_img_1.png"
    PC_FILE_PATH = "/home/animeshmaheshwari/Documents/ImageSegmentation/ObjectDetection/Assignment/processed_dataset/pc_1.npy"
    ONNX_MODEL_PATH = "detector3d.onnx"

    session = ort.InferenceSession(ONNX_MODEL_PATH)
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]

    if len(input_names) != 2:
        print(f"ERROR: Expected 2 inputs, but the ONNX model has {len(input_names)}.")
        print(f"   Inputs found: {input_names}")
        print("   Please re-run the export_onnx.py script.")
        return

    print("ONNX model loaded successfully.")
    print(f"   Input Names: {input_names}")
    print(f"   Output Names: {output_names}")

    print(f"Preprocessing files...")
    try:
        input_image = preprocess_image(IMAGE_FILE_PATH)
        input_pc = preprocess_point_cloud(PC_FILE_PATH, num_points=PC_NUM_POINTS)
        print(f"  > Image shape: {input_image.shape}")
        print(f"  > Point Cloud shape: {input_pc.shape}")
    except FileNotFoundError as e:
        print(f"ERROR: Could not find a file. Please check your paths.\n{e}")
        return

    model_inputs = {input_names[0]: input_image, input_names[1]: input_pc}
    results = session.run(output_names, model_inputs)
    print("\nInference executed successfully.")

    pred_logits, pred_boxes, pred_corners, _ = results
    print("\n--- Output Shapes ---")
    print(f"Predicted Logits Shape: {pred_logits.shape}")
    print(f"Predicted Boxes Shape: {pred_boxes.shape}")
    print(f"Predicted Corners Shape: {pred_corners.shape}")

if __name__ == '__main__':
    main()

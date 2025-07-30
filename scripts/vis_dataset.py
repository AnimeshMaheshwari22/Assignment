import os
import sys
import numpy as np
from typing import Union, Optional
import open3d as o3d
from PIL import Image
import numpy.typing as npt


def load_array(source: Union[str, npt.NDArray]) -> np.ndarray:
    if isinstance(source, str):
        if not os.path.isfile(source):
            raise FileNotFoundError(f"File not found: {source}")
        return np.load(source)
    elif isinstance(source, np.ndarray):
        return source
    else:
        raise ValueError(f"Invalid input type: {type(source)}")


def prepare_point_cloud(raw_pc: np.ndarray, image: Optional[np.ndarray] = None) -> o3d.geometry.PointCloud:
    if raw_pc.ndim == 3 and raw_pc.shape[0] == 3:
        raw_pc = raw_pc.transpose(1, 2, 0).reshape(-1, 3)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(raw_pc.astype(np.float64))

    if image is not None:
        if image.shape[0] != raw_pc.shape[1] or image.shape[1] != raw_pc.shape[2]:
            image = np.resize(image, (raw_pc.shape[1], raw_pc.shape[2], 3))
        color_vals = image.reshape(-1, 3).astype(np.float64) / 255.0
        cloud.colors = o3d.utility.Vector3dVector(color_vals)

    return cloud


def draw_bounding_boxes(bbox_array: np.ndarray) -> list:
    boxes = []
    for box_coords in bbox_array:
        box = o3d.geometry.OrientedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(box_coords.astype(np.float64))
        )
        box.color = (0.1, 0.2, 1.0)  # blue-ish
        boxes.append(box)
    return boxes


def read_rgb_image(path: str) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.array(image)


def render_scene(cloud: o3d.geometry.PointCloud, boxes: list):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Viewer", width=1024, height=768)
    vis.add_geometry(cloud)
    for box in boxes:
        vis.add_geometry(box)
    vis.run()
    vis.destroy_window()


def run_viewer(data_root: str, sample_id: str):
    pc_file = os.path.join(data_root, f"pc_{sample_id}.npy")
    rgb_file = os.path.join(data_root, f"rgb_img_{sample_id}.png")
    bbox_file = os.path.join(data_root, f"bbox3d_{sample_id}.npy")

    point_cloud = load_array(pc_file)
    bounding_boxes = load_array(bbox_file)
    rgb_image = read_rgb_image(rgb_file)

    pointcloud_o3d = prepare_point_cloud(point_cloud, rgb_image)
    bbox_o3d_list = draw_bounding_boxes(bounding_boxes)
    render_scene(pointcloud_o3d, bbox_o3d_list)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {os.path.basename(__file__)} <data_folder> <sample_number>")
        sys.exit(1)

    folder = sys.argv[1]
    number = sys.argv[2]
    run_viewer(folder, number)

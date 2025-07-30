import torch
import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Model')))
from main_model import Detector3D
from encoder import Encoder
from decoder import FusionDecoder, FusionDecoderLayer

class DetectorONNX(Detector3D):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder, criterion=None)

    def forward(self, rgb_image, point_cloud):
        processed_outputs = super().forward(rgb_image, point_cloud, targets=None)
        return (
            processed_outputs['pred_logits'],
            processed_outputs['pred_boxes'],
            processed_outputs['pred_corners'],
            point_cloud
        )

def main():
    NUM_CLASSES = 1
    NUM_QUERIES = 16
    device = torch.device("cuda")

    class PointNetArgs:
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
        nqueries = NUM_QUERIES

    encoder = Encoder(pointnet_args=PointNetArgs(), fusion_dim=256)
    decoder_layer = FusionDecoderLayer(d_model=256)
    decoder = FusionDecoder(decoder_layer, num_layers=6, d_model=256, num_classes=NUM_CLASSES, num_queries=NUM_QUERIES)

    model = DetectorONNX(encoder, decoder).to(device)

    model_weights_path = "/home/animeshmaheshwari/Documents/ImageSegmentation/ObjectDetection/Assignment/Model/results/detector3d_final.pth"
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()
    print("Model loaded and in evaluation mode.")

    dummy_image = torch.randn(1, 3, 224, 224, device=device)
    dummy_pc = torch.randn(1, 2048, 3, device=device)

    onnx_output_path = "detector3d.onnx"
    input_names = ["rgb_image", "point_cloud"]
    output_names = ["pred_logits", "pred_boxes", "pred_corners", "dummy_pc_output"]

    torch.onnx.export(
        model,
        (dummy_image, dummy_pc),
        onnx_output_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        do_constant_folding=True,
        verbose=False
    )
    print(f"Model successfully exported to {onnx_output_path}")

if __name__ == '__main__':
    main()

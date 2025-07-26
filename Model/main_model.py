import torch
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

# --- Import all the modules you have created ---
# This script assumes all these files are in your project directory or accessible in your python path.
from encoder import Encoder
from decoder import FusionDecoder, FusionDecoderLayer

# ==============================================================================
# == ✅ NEW: 3D-Compatible Hungarian Matcher (Corrected) ✅                      ==
# ==============================================================================
# This version uses a standard classification cost calculation that is compatible
# with the model's output and removes the need for the complex 'label_map'.

class HungarianMatcher3D(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network.
    It is adapted for 3D boxes by using only L1 and classification costs for matching.
    """
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, focal_alpha = 0.25):
        """
        Creates the matcher.
        
        Args:
            cost_class: This is the relative weight of the classification error in the matching cost.
            cost_bbox: This is the relative weight of the L1 error of the 3D bounding box parameters.
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        assert cost_class != 0 or cost_bbox != 0, "all costs can't be 0"
        self.focal_alpha = focal_alpha

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching. """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        # Use softmax for probabilities as it's standard for cross-entropy-based costs
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes + 1]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 8]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. This is the negative log-probability of the target class.
        # This is a standard approach in DETR-like models.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes.
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

def build_matcher_3d(args):
    return HungarianMatcher3D(
        cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox,
        focal_alpha=args.focal_alpha
    )

class SetCriterion(nn.Module):
    """
    This class computes the loss for the 3D detector.
    It comprises of:
        1) a matcher that finds the best match between predictions and targets.
        2) the loss calculation for the matched pairs.
    """
    def __init__(self, num_classes, matcher, weight_dict, losses):
        """
        Create the criterion.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """Classification loss (e.g., Focal Loss or Cross-entropy)."""
        assert 'pred_logits' in outputs
        pred_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(pred_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=pred_logits.device)
        target_classes[idx] = target_classes_o
        
        loss_ce = F.cross_entropy(pred_logits.transpose(1, 2), target_classes)
        losses = {'loss_ce': loss_ce}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """L1 loss for 3D bounding box regression."""
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets):
        """
        This performs the loss computation.
        """
        # --- Step 1: Match predictions with ground truth ---
        indices = self.matcher(outputs, targets)

        # --- Step 2: Compute losses for the matched pairs ---
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        return losses

class Detector3D(nn.Module):
    """ The complete 3D Object Detector model """
    def __init__(self, encoder, decoder, matcher, num_classes=10):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        weight_dict = {'loss_ce': 1.0, 'loss_bbox': 5.0}
        losses = ['labels', 'boxes']
        self.criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict, losses=losses)

    def forward(self, rgb_image, point_cloud, targets=None):
        """
        The forward pass of the model.
        """
        encoder_out = self.encoder(rgb_image, point_cloud)
        
        pc_features = encoder_out["updated_depth_features"]
        img_features = encoder_out["updated_image_features"]
        xyz_coords = encoder_out["point_cloud_xyz"]
        
        decoder_out = self.decoder(pc_features, img_features, xyz_coords)
        
        if self.training:
            assert targets is not None, "targets should be provided for training"
            losses = self.criterion(decoder_out, targets)
            final_loss = sum(losses[k] * self.criterion.weight_dict[k] for k in losses.keys() if k in self.criterion.weight_dict)
            return final_loss
            
        return decoder_out

# ==============================================================================
# == Main Execution Block to Assemble and Test the Full Model                 ==
# ==============================================================================
if __name__ == '__main__':
    # --- 1. Define Configuration Arguments ---
    class PointNetArgs:
        use_color = False; enc_dim = 256; preenc_npoints = 2048; enc_type = "vanilla"; enc_nhead = 4; enc_ffn_dim = 128
        enc_dropout = 0.1; enc_activation = "relu"; enc_nlayers = 3; mlp_dropout = 0.3; nqueries = 256
    
    class MatcherArgs:
        set_cost_class = 2.0
        set_cost_bbox = 5.0
        focal_alpha = 0.25

    pointnet_args = PointNetArgs()
    matcher_args = MatcherArgs()
    
    NUM_CLASSES = 10
    NUM_QUERIES = 256
    FUSION_DIM = 256
    DECODER_LAYERS = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Assembling End-to-End Model on {device} ---")

    # --- 2. Build the Main Components ---
    encoder = Encoder(pointnet_args=pointnet_args, fusion_dim=FUSION_DIM)
    decoder_layer = FusionDecoderLayer(d_model=FUSION_DIM)
    decoder = FusionDecoder(decoder_layer, num_layers=DECODER_LAYERS, d_model=FUSION_DIM, num_classes=NUM_CLASSES + 1, num_queries=NUM_QUERIES)
    matcher = build_matcher_3d(args=matcher_args)

    # --- 3. Create the Final Detector3D Model ---
    model = Detector3D(encoder, decoder, matcher, num_classes=NUM_CLASSES).to(device)
    model.train()
    print("\n✅ Model assembled successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- 4. Create Dummy Data and Run a Forward Pass ---
    B = 2
    dummy_image = torch.randn(B, 3, 224, 224).to(device)
    dummy_pc = torch.randn(B, 4096, 3).to(device)
    
    dummy_targets = [
        {'labels': torch.tensor([1, 3]).long().to(device), 'boxes': torch.randn(2, 8).to(device)},
        {'labels': torch.tensor([5]).long().to(device), 'boxes': torch.randn(1, 8).to(device)}
    ]
    
    print("\n--- Running a dummy training step ---")
    loss = model(dummy_image, dummy_pc, targets=dummy_targets)
    print(f"Calculated loss: {loss.item()}")
    
    # Test inference mode
    model.eval()
    with torch.no_grad():
        predictions = model(dummy_image, dummy_pc)
    print("\n--- Running a dummy inference step ---")
    print("Prediction keys:", predictions.keys())
    print("Logits shape:", predictions['pred_logits'].shape)
    print("Boxes shape:", predictions['pred_boxes'].shape)
    
    assert predictions['pred_logits'].shape == (B, NUM_QUERIES, NUM_CLASSES + 1)
    assert predictions['pred_boxes'].shape == (B, NUM_QUERIES, 8)
    print("\n✅ End-to-end model flow is correct!")

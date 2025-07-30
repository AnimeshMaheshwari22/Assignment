import torch
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from utils.box_util import get_3d_box_batch_tensor, generalized_box3d_iou

class HungarianMatcher3D(nn.Module):
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_bbox_params = outputs["pred_boxes"].flatten(0, 1)
        out_bbox_corners = outputs["pred_corners"].flatten(0, 1)

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox_params = torch.cat([v["boxes"]for v in targets])
        tgt_bbox_corners = torch.cat([v["corners"] for v in targets])

        cost_class = -out_prob[:, tgt_ids]
        cost_bbox = torch.cdist(out_bbox_params, tgt_bbox_params, p=1)

        gious = generalized_box3d_iou(
            out_bbox_corners.unsqueeze(0),
            tgt_bbox_corners.unsqueeze(0),
            nums_k2=torch.tensor([len(tgt_bbox_corners)], device=out_bbox_corners.device)
        )
        cost_giou = -gious.squeeze(0)

        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

class SetCriterion3D(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_labels(self, outputs, targets, indices):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.zeros_like(src_logits)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes[idx] = F.one_hot(target_classes_o, num_classes=self.num_classes).float()
        loss = F.binary_cross_entropy_with_logits(src_logits, target_classes, reduction="mean")
        return {'loss_ce': loss}

    def loss_boxes(self, outputs, targets, indices):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        tgt_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction='mean')

        src_corners = outputs['pred_corners'][idx]
        tgt_corners = torch.cat([t['corners'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        gious = generalized_box3d_iou(
            src_corners.unsqueeze(0), 
            tgt_corners.unsqueeze(0),
            nums_k2=torch.tensor([len(tgt_corners)], device=src_corners.device),
            needs_grad=True
        ).squeeze(0)

        loss_giou = (1 - torch.diag(gious)).mean()
        return {'loss_bbox': loss_bbox, 'loss_giou': loss_giou}

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        loss_dict = {}
        loss_dict.update(self.loss_labels(outputs, targets, indices))
        loss_dict.update(self.loss_boxes(outputs, targets, indices))
        final_loss = sum(loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys())
        return final_loss, loss_dict

class Detector3D(nn.Module):
    def __init__(self, encoder, decoder, criterion):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.criterion = criterion

    def process_outputs(self, outputs_raw):
        sem_cls_logits = outputs_raw['sem_cls_logits']
        box_params = outputs_raw['bbox_params']
        center = box_params[..., :3]
        log_dims = box_params[..., 3:6]
        yaw_params = F.normalize(box_params[..., 6:8], p=2, dim=-1)
        final_box_params = torch.cat([center, log_dims, yaw_params], dim=-1)
        dims = torch.exp(log_dims)
        yaw = torch.atan2(yaw_params[..., 1], yaw_params[..., 0])
        corners = get_3d_box_batch_tensor(dims, yaw, center)
        return {
            'pred_logits': sem_cls_logits,
            'pred_boxes': final_box_params,
            'pred_corners': corners,
        }

    def forward(self, rgb_image, point_cloud, targets=None):
        encoder_out = self.encoder(rgb_image, point_cloud)
        pc_features = encoder_out["updated_depth_features"]
        img_features = encoder_out["updated_image_features"]
        outputs_raw = self.decoder(pc_features, img_features, None)
        final_outputs = self.process_outputs(outputs_raw['outputs'])
        if self.training:
            assert targets is not None, "targets should be provided for training"
            loss, loss_dict = self.criterion(final_outputs, targets)
            return loss, loss_dict
        return final_outputs

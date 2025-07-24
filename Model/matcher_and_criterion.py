import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np

class HungarianMatcher3D(nn.Module):
    """
    Computes an assignment between predictions and ground truth targets for 3D boxes.
    """
    def __init__(self, cost_class: float = 1.0, cost_center: float = 1.0, cost_size: float = 1.0, cost_angle: float = 1.0):
        """
        Creates the matcher.
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost.
            cost_center: This is the relative weight of the L1 error of the box center.
            cost_size: This is the relative weight of the L1 error of the box size.
            cost_angle: This is the relative weight of the L1 error of the box angle.
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_center = cost_center
        self.cost_size = cost_size
        self.cost_angle = cost_angle
        assert cost_class != 0 or cost_center != 0 or cost_size != 0 or cost_angle != 0, "all costs cannot be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ 
        Performs the matching.
        
        Params:
            outputs: This is a dict that contains at least these entries:
                 "sem_cls_logits": Tensor of dim [batch_size, num_queries, num_classes]
                 "center_unnormalized": Tensor of dim [batch_size, num_queries, 3]
                 "size_unnormalized": Tensor of dim [batch_size, num_queries, 3]
                 "angle_continuous": Tensor of dim [batch_size, num_queries]

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes]
                 "center_unnormalized": Tensor of dim [num_target_boxes, 3]
                 "size_unnormalized": Tensor of dim [num_target_boxes, 3]
                 "angle_continuous": Tensor of dim [num_target_boxes]

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["sem_cls_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["sem_cls_logits"].flatten(0, 1).softmax(-1)
        out_center = outputs["center_unnormalized"].flatten(0, 1)
        out_size = outputs["size_unnormalized"].flatten(0, 1)
        out_angle = outputs["angle_continuous"].flatten(0, 1)

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_center = torch.cat([v["center_unnormalized"] for v in targets])
        tgt_size = torch.cat([v["size_unnormalized"] for v in targets])
        tgt_angle = torch.cat([v["angle_continuous"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use focal loss here.
        # The cost is the negative log-likelihood of the target class.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_center = torch.cdist(out_center, tgt_center, p=1)
        cost_size = torch.cdist(out_size, tgt_size, p=1)
        cost_angle = torch.cdist(out_angle.unsqueeze(-1), tgt_angle.unsqueeze(-1), p=1)

        # Final cost matrix
        C = (self.cost_class * cost_class + self.cost_center * cost_center + 
             self.cost_size * cost_size + self.cost_angle * cost_angle)
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["labels"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

class SetCriterion(nn.Module):
    """
    This class computes the loss for the 3D object detection model.
    The process happens in two steps:
    1) We compute hungarian assignment between ground truth boxes and the model's outputs.
    2) We supervise each pair of matched ground-truth / prediction.
    """
    def __init__(self, num_classes, matcher, weight_dict, losses):
        """ 
        Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category.
            matcher: module able to compute a matching between targets and proposals.
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_loss = FocalLoss()

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """Classification loss (Focal Loss)."""
        assert 'sem_cls_logits' in outputs
        src_logits = outputs['sem_cls_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1] # remove 'no object' class
        loss_ce = self.focal_loss(src_logits, target_classes_onehot)
        losses = {'loss_ce': loss_ce}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes: L1 regression loss."""
        assert 'center_unnormalized' in outputs
        idx = self._get_src_permutation_idx(indices)
        
        # Center Loss
        src_center = outputs['center_unnormalized'][idx]
        target_center = torch.cat([t['center_unnormalized'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_center = F.l1_loss(src_center, target_center, reduction='none')

        # Size Loss
        src_size = outputs['size_unnormalized'][idx]
        target_size = torch.cat([t['size_unnormalized'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_size = F.l1_loss(src_size, target_size, reduction='none')

        # Angle Loss
        src_angle = outputs['angle_continuous'][idx]
        target_angle = torch.cat([t['angle_continuous'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_angle = F.l1_loss(src_angle, target_angle, reduction='none')

        losses = {}
        losses['loss_center'] = loss_center.sum() / num_boxes
        losses['loss_size'] = loss_size.sum() / num_boxes
        losses['loss_angle'] = loss_angle.sum() / num_boxes
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
        assert loss in loss_map, f'Unsupported loss: {loss}'
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets):
        """ 
        This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format.
             targets: list of dicts, where each dict corresponds to a single image and contains ground truth info.
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if num_boxes == 0: # Avoid division by zero
            num_boxes = torch.as_tensor([1.0], dtype=torch.float, device=num_boxes.device)
        
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

class FocalLoss(nn.Module):
    """
    Focal Loss for dense object detection.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        p = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.sum()

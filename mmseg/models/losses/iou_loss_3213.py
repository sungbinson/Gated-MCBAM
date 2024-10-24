import torch
import torch.nn as nn
from mmseg.registry import MODELS


@MODELS.register_module()
class IoULoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(IoULoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, avg_factor=None):
        # Flatten predictions and targets
        pred = pred.view(-1)
        target = target.view(-1)

        # Calculate intersection and union
        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target) - intersection

        # Compute IoU loss
        iou = (intersection + 1e-6) / (union + 1e-6)
        iou_loss = 1 - iou

        # Apply avg_factor if provided
        if avg_factor is not None:
            if self.reduction == 'mean':
                return self.loss_weight * iou_loss / avg_factor
            elif self.reduction == 'sum':
                return self.loss_weight * iou_loss.sum()
            else:
                return self.loss_weight * iou_loss
        else:
            if self.reduction == 'mean':
                return self.loss_weight * iou_loss.mean()
            elif self.reduction == 'sum':
                return self.loss_weight * iou_loss.sum()
            else:
                return self.loss_weight * iou_loss
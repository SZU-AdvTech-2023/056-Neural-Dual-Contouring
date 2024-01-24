import torch
from torch import nn
import torch.nn.functional as F
from utils.geometry_utils import edge_acc


class CornerCriterion(nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.loss_rate = 9

    def forward(self, outputs_s1, targets, gauss_targets, epoch=0):
        # Compute the acc first, use the acc to guide the setup of loss weight
        preds_s1 = (outputs_s1 >= 0.5).float()
        pos_target_ids = torch.where(targets == 1)
        correct = (preds_s1[pos_target_ids] == targets[pos_target_ids]).float().sum()
        recall_s1 = correct / len(pos_target_ids[0])

        rate = self.loss_rate

        loss_weight = (gauss_targets > 0.5).float() * rate + 1
        loss_s1 = F.binary_cross_entropy (outputs_s1.double(), gauss_targets.double(), weight=loss_weight, reduction='none')
        loss_s1 = loss_s1.sum(-1).sum(-1).mean()
        # loss_s1 = torch.nn.BCELoss()(preds_s1.double(), targets)
        return loss_s1, recall_s1


class EdgeCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.edge_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.33, 1.0]).cuda(), reduction='none')

    def forward(self, logits_s1, logits_s2_hybrid, logits_s2_rel, s2_ids, s2_edge_mask, edge_labels, edge_lengths,
                edge_mask, s2_gt_values):
        # loss for edge filtering
        s1_losses = self.edge_loss(logits_s1, edge_labels)
        s1_losses[torch.where(edge_mask == True)] = 0
        s1_losses = s1_losses[torch.where(s1_losses > 0)].sum() / edge_mask.shape[0]
        gt_values = torch.ones_like(edge_mask).long() * 2
        s1_acc = edge_acc(logits_s1, edge_labels, edge_lengths, gt_values)

        # loss for stage-2
        s2_labels = torch.gather(edge_labels, 1, s2_ids)

        # the image-aware decoder
        s2_losses_hybrid = self.edge_loss(logits_s2_hybrid, s2_labels)
        s2_losses_hybrid[torch.where((s2_edge_mask == True) | (s2_gt_values != 2))] = 0
        # aggregate the loss into the final scalar
        s2_losses_hybrid = s2_losses_hybrid[torch.where(s2_losses_hybrid > 0)].sum() / s2_edge_mask.shape[0]
        s2_edge_lengths = (s2_edge_mask == 0).sum(dim=-1)
        # compute edge-level acc
        s2_acc_hybrid = edge_acc(logits_s2_hybrid, s2_labels, s2_edge_lengths, s2_gt_values)

        # the geom-only decoder
        s2_losses_rel = self.edge_loss(logits_s2_rel, s2_labels)
        s2_losses_rel[torch.where((s2_edge_mask == True) | (s2_gt_values != 2))] = 0
        # aggregate the loss into the final scalar
        s2_losses_rel = s2_losses_rel[torch.where(s2_losses_rel > 0)].sum() / s2_edge_mask.shape[0]
        s2_edge_lengths = (s2_edge_mask == 0).sum(dim=-1)
        # compute edge-level f1-score
        s2_acc_rel = edge_acc(logits_s2_rel, s2_labels, s2_edge_lengths, s2_gt_values)

        return s1_losses, s1_acc, s2_losses_hybrid, s2_acc_hybrid, s2_losses_rel, s2_acc_rel


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


sigmoid_focal_loss_jit: "torch.jit.ScriptModule" = torch.jit.script(sigmoid_focal_loss)
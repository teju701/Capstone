import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation Loss (identical to baseline)
# ─────────────────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    def __init__(self, ignore_index: int = 255, smooth: float = 1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, H, W = logits.shape
        prob  = F.softmax(logits, dim=1)
        valid = targets != self.ignore_index
        t     = targets.clone(); t[~valid] = 0

        one_hot    = F.one_hot(t, C).permute(0, 3, 1, 2).float()
        valid_mask = valid.unsqueeze(1).float()
        prob       = prob    * valid_mask
        one_hot    = one_hot * valid_mask

        inter = (prob * one_hot).sum(dim=(0, 2, 3))
        union = prob.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
        return 1.0 - ((2.0 * inter + self.smooth) / (union + self.smooth)).mean()


class SegLoss(nn.Module):
    """CE (70%) + Dice (30%) with label smoothing — identical to baseline."""
    def __init__(self, ignore_index: int = 255, label_smoothing: float = 0.1):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss(ignore_index=ignore_index,
                                        label_smoothing=label_smoothing)
        self.dice = DiceLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return 0.7 * self.ce(logits, targets) + 0.3 * self.dice(logits, targets)


# ─────────────────────────────────────────────────────────────────────────────
# Depth Loss (identical to baseline)
# ─────────────────────────────────────────────────────────────────────────────

class BerHuLoss(nn.Module):
    """Reverse Huber loss for normalised depth targets in [0, 1]."""
    def __init__(self, threshold: float = 0.2):
        super().__init__()
        self.threshold = threshold

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred   = pred.squeeze(1)
        mask   = target > 0
        p, t   = pred[mask], target[mask]
        diff   = torch.abs(p - t)
        c      = self.threshold * diff.max().detach()
        loss   = torch.where(diff <= c, diff,
                             (diff ** 2 + c ** 2) / (2.0 * c + 1e-8))
        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Uncertainty-Weighted MTL Loss  (Kendall et al., NeurIPS 2018)
#
# Formula:
#   L_total = exp(-log_σ_seg)   × L_seg   + log_σ_seg
#           + exp(-log_σ_depth) × L_depth + log_σ_depth
#
# Intuition:
#   - Each task has a learnable log-uncertainty parameter (log_σ).
#   - Higher uncertainty → lower effective weight for that task's loss.
#   - The log(σ) regularisation term prevents uncertainty from going to ∞.
#   - At initialisation (log_σ = 0, σ = 1): both tasks are equally weighted.
#   - After training: model learns that seg loss (~0.60) needs lower weight
#     and depth loss (~0.04) needs higher weight to balance gradients.
#
# Why not fixed weights?
#   Seg loss ≈ 0.60 at convergence, BerHu loss ≈ 0.04 at convergence.
#   A naive sum completely drowns depth gradients. Tuning fixed weights
#   requires multiple training runs. Uncertainty weighting adapts automatically.
# ─────────────────────────────────────────────────────────────────────────────

class MTLLoss(nn.Module):
    """
    Joint MTL loss with automatic uncertainty-based task weighting.

    Contains three sub-losses and two learnable parameters (log_sigma_seg,
    log_sigma_depth). These parameters MUST be included in the optimizer —
    see train_mtl.py for the correct setup.
    """

    def __init__(self, ignore_index: int = 255, label_smoothing: float = 0.1,
                 berhu_threshold: float = 0.2):
        super().__init__()
        self.seg_loss   = SegLoss(ignore_index=ignore_index,
                                  label_smoothing=label_smoothing)
        self.depth_loss = BerHuLoss(threshold=berhu_threshold)

        # Learnable log-uncertainty parameters — initialised to 0 (σ=1)
        self.log_sigma_seg   = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_depth = nn.Parameter(torch.tensor(0.0))

    def forward(self,
                seg_pred:   torch.Tensor,
                seg_gt:     torch.Tensor,
                depth_pred: torch.Tensor,
                depth_gt:   torch.Tensor):
        """
        Args:
            seg_pred   : (B, 19, H, W) logits
            seg_gt     : (B, H, W)     long, ignore=255
            depth_pred : (B, 1,  H, W) sigmoid in [0,1]
            depth_gt   : (B, H, W)     float, 0=invalid

        Returns:
            total_loss      : scalar tensor (backprop through this)
            l_seg           : float — raw seg loss (for logging)
            l_depth         : float — raw depth loss (for logging)
            w_seg           : float — effective seg weight (for logging)
            w_depth         : float — effective depth weight (for logging)
        """
        l_seg   = self.seg_loss(seg_pred, seg_gt)
        l_depth = self.depth_loss(depth_pred, depth_gt)

        # Uncertainty weights: w = exp(-log_sigma) = 1/sigma
        w_seg   = torch.exp(-self.log_sigma_seg)
        w_depth = torch.exp(-self.log_sigma_depth)

        # Combined loss with regularisation
        total = (w_seg   * l_seg   + self.log_sigma_seg   +
                 w_depth * l_depth + self.log_sigma_depth)

        return (total,
                l_seg.item(),
                l_depth.item(),
                w_seg.item(),
                w_depth.item())
"""
Loss Functions for Imbalanced Medical Image Classification
==========================================================
Implements advanced loss functions designed for the severe class imbalance
in HAM10000 (nv: 67%, vasc: 1.4%, df: 1.1%):

  1. Focal Loss          — down-weights easy examples, focuses on hard ones
  2. Label Smoothing CE  — regularises overconfident predictions
  3. Asymmetric Loss     — separate gamma for positive/negative samples
  4. Class-Weighted CE   — classical inverse-frequency weighting
  5. Soft Label CE       — compatible with MixUp/CutMix soft targets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1. Focal Loss
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    - gamma=0: reduces to cross-entropy
    - gamma=2: standard focal loss (Lin et al., 2017)
    - alpha:  per-class weight tensor (or None for uniform)

    Reference: Lin et al. (2017) - "Focal Loss for Dense Object Detection"
    """

    def __init__(self, gamma: float = 2.0,
                 alpha: torch.Tensor = None,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma     = gamma
        self.alpha     = alpha       # shape [num_classes] or None
        self.reduction = reduction

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, C] raw model outputs
            targets: [B] integer class indices  OR  [B, C] soft labels
        """
        # Handle soft labels (MixUp/CutMix)
        if targets.dim() == 2:
            return self._soft_forward(logits, targets)

        log_p = F.log_softmax(logits, dim=-1)
        p     = torch.exp(log_p)

        # Gather per-sample probabilities
        log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt     = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha.gather(0, targets)
            focal_weight = alpha_t * focal_weight

        loss = -focal_weight * log_pt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    def _soft_forward(self, logits: torch.Tensor,
                      soft_targets: torch.Tensor) -> torch.Tensor:
        """Focal loss with soft targets (MixUp/CutMix)."""
        log_p = F.log_softmax(logits, dim=-1)
        p     = torch.exp(log_p)

        # Expected probability under soft labels
        pt = (soft_targets * p).sum(dim=1, keepdim=True)
        focal_weight = (1 - pt) ** self.gamma

        ce = -(soft_targets * log_p).sum(dim=1)
        loss = focal_weight.squeeze(1) * ce

        return loss.mean() if self.reduction == "mean" else loss.sum()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Label Smoothing Cross-Entropy
# ─────────────────────────────────────────────────────────────────────────────
class LabelSmoothingCE(nn.Module):
    """
    Cross-entropy with label smoothing.
    Prevents overconfident predictions; improves calibration.

    Handles both hard labels and soft labels (MixUp/CutMix).

    Reference: Szegedy et al. (2016) - "Rethinking the Inception Architecture"
    """

    def __init__(self, smoothing: float = 0.1, num_classes: int = 7,
                 weight: torch.Tensor = None):
        super().__init__()
        self.smoothing   = smoothing
        self.num_classes = num_classes
        self.weight      = weight

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=-1)

        if targets.dim() == 2:
            # Already soft labels from MixUp/CutMix
            smooth_targets = (1 - self.smoothing) * targets + \
                             self.smoothing / self.num_classes
        else:
            # Hard labels → convert to one-hot
            one_hot = torch.zeros_like(log_p).scatter_(
                1, targets.unsqueeze(1), 1.0)
            smooth_targets = (1 - self.smoothing) * one_hot + \
                             self.smoothing / self.num_classes

        if self.weight is not None:
            w = self.weight.to(logits.device)
            loss = -(smooth_targets * log_p * w.unsqueeze(0)).sum(dim=1)
        else:
            loss = -(smooth_targets * log_p).sum(dim=1)

        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Asymmetric Loss (ASL)
# ─────────────────────────────────────────────────────────────────────────────
class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss with separate focusing parameters for
    positive (gamma_pos) and negative (gamma_neg) samples.

    Especially effective for multi-label & imbalanced multi-class settings.

    Reference: Ridnik et al. (2021) - "Asymmetric Loss for Multi-Label Classification"
    """

    def __init__(self, gamma_pos: float = 0.0, gamma_neg: float = 4.0,
                 clip: float = 0.05):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip      = clip

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)

        # Convert to one-hot if hard labels
        if targets.dim() == 1:
            one_hot = torch.zeros_like(logits).scatter_(
                1, targets.unsqueeze(1), 1.0)
        else:
            one_hot = targets.float()

        # Probability shift (negative clipping)
        p_m = torch.clamp(p - self.clip, min=0)

        p_pos = one_hot * p
        p_neg = (1 - one_hot) * p_m

        loss_pos = one_hot       * torch.log(p_pos + 1e-8) * (1 - p_pos) ** self.gamma_pos
        loss_neg = (1 - one_hot) * torch.log(1 - p_neg + 1e-8) * p_neg ** self.gamma_neg

        return -((loss_pos + loss_neg).sum(dim=1)).mean()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Soft-label compatible cross entropy (for MixUp/CutMix)
# ─────────────────────────────────────────────────────────────────────────────
class SoftCrossEntropy(nn.Module):
    """Standard CE extended to accept soft probability targets."""

    def __init__(self, weight: torch.Tensor = None):
        super().__init__()
        self.weight = weight

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=-1)

        if targets.dim() == 1:
            if self.weight is not None:
                return F.cross_entropy(logits, targets,
                                       weight=self.weight.to(logits.device))
            return F.cross_entropy(logits, targets)

        # Soft targets
        loss = -(targets * log_p)
        if self.weight is not None:
            loss = loss * self.weight.to(logits.device).unsqueeze(0)
        return loss.sum(dim=1).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Loss factory
# ─────────────────────────────────────────────────────────────────────────────
def build_loss(
    name: str,
    class_weights: torch.Tensor = None,
    num_classes: int = 7,
    gamma: float = 2.0,
    label_smoothing: float = 0.1,
) -> nn.Module:
    """
    Build loss function by name.

    Args:
        name:           'focal', 'label_smoothing', 'asymmetric',
                        'weighted_ce', 'soft_ce'
        class_weights:  Tensor of shape [num_classes], computed from
                        dataset.compute_class_weights()
        gamma:          Focal loss gamma
        label_smoothing: Label smoothing epsilon

    Returns:
        nn.Module loss function
    """
    name = name.lower()

    if name == "focal":
        return FocalLoss(gamma=gamma, alpha=class_weights)

    elif name in ("label_smoothing", "label_smoothing_ce"):
        return LabelSmoothingCE(smoothing=label_smoothing,
                                num_classes=num_classes,
                                weight=class_weights)

    elif name in ("asymmetric", "asl"):
        return AsymmetricLoss()

    elif name in ("weighted_ce", "cross_entropy"):
        return SoftCrossEntropy(weight=class_weights)

    elif name == "soft_ce":
        return SoftCrossEntropy()

    else:
        raise ValueError(f"Unknown loss: {name}. "
                         f"Choose from: focal, label_smoothing, "
                         f"asymmetric, weighted_ce, soft_ce")


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge Distillation Loss (for ensemble → single model compression)
# ─────────────────────────────────────────────────────────────────────────────
class KDLoss(nn.Module):
    """
    Knowledge Distillation Loss.
    Combines student cross-entropy with teacher KL divergence.

    L = alpha * CE(student, hard_labels)
      + (1-alpha) * T^2 * KL(soft_student || soft_teacher)

    Reference: Hinton et al. (2015) - "Distilling the Knowledge in a Neural Network"
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.1,
                 num_classes: int = 7):
        super().__init__()
        self.T     = temperature
        self.alpha = alpha
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss()

    def forward(self, student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                hard_targets: torch.Tensor) -> torch.Tensor:
        # Hard target loss
        ce_loss = self.ce(student_logits, hard_targets)

        # Soft target loss (KL divergence)
        soft_student = F.log_softmax(student_logits / self.T, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.T, dim=1)
        kd_loss = F.kl_div(soft_student, soft_teacher,
                           reduction="batchmean") * (self.T ** 2)

        return self.alpha * ce_loss + (1 - self.alpha) * kd_loss

"""Compound focal + multiclass Dice loss for the MARIDA deployment pipeline.

                Total = Focal_Loss(gamma=2.0, alpha=class_weights)
                      + lambda_dice * Multiclass_Dice_Loss(smooth=1.0)
                      + lambda_lovasz * Lovasz_Softmax_Loss

Both terms mask out unlabeled pixels (IGNORE_INDEX). The Focal loss alpha
weights give Marine Debris (class index 0) an explicit boost (>= 8.0) relative
to majority classes so the extreme pixel imbalance does not drown the target
class.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.encoding import IGNORE_INDEX, N_CLASSES

# Class index 0 in the training target space corresponds to 11-class value 1,
# i.e. Marine Debris (see CONTEXT.md class mapping / src/data/encoding.py).
DEBRIS_INDEX = 0
DEBRIS_ALPHA_MIN = 4.0

# Class 6 in the training target space corresponds to 11-class value 7,
# i.e. Marine Water. Relative to this majority class debris carries the boost.
MARINE_WATER_INDEX = 6


def build_focal_alphas(class_weights: torch.Tensor | None, debris_alpha: float = DEBRIS_ALPHA_MIN) -> torch.Tensor:
    """Return an alpha vector for the Focal loss (length N_CLASSES).

    Uses the inverted-frequency class weights premultiplied on top of the
    base probability, then enforces that Marine Debris has alpha >= 4.0 and
    is at least 3x Marine Water. Unlabeled pixels are not a model class, so
    the vector lives in the 11-class training target space {0..10}.
    """
    if class_weights is None:
        alpha = torch.ones(N_CLASSES, dtype=torch.float32)
    else:
        alpha = class_weights.detach().float().clone()
    alpha[DEBRIS_INDEX] = max(float(alpha[DEBRIS_INDEX]), DEBRIS_ALPHA_MIN)
    water_alpha = float(alpha[MARINE_WATER_INDEX])
    if alpha[DEBRIS_INDEX] < 3.0 * water_alpha:
        alpha[DEBRIS_INDEX] = 3.0 * water_alpha
    return alpha


class FocalLoss(nn.Module):
    """Multiclass focal loss with ignore indexing.

    loss = -alpha_c * (1 - p_c)^gamma * log(p_c) over labeled pixels only.

    Args:
        gamma: focusing parameter (default 2.0).
        alpha: per-class weight vector (length N_CLASSES) or None for uniform.
        ignore_index: training target value for unlabeled pixels.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        ignore_index: int = IGNORE_INDEX,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha.clone().float())
        else:
            self.register_buffer("alpha", torch.ones(N_CLASSES, dtype=torch.float32))
        self.alpha: torch.Tensor

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4 or target.ndim != 3:
            raise ValueError(f"focal loss expects logits (B,C,H,W) and target (B,H,W), got {tuple(logits.shape)} and {tuple(target.shape)}")
        if logits.shape[0] != target.shape[0]:
            raise ValueError("batch dimension mismatch between logits and target")
        if self.alpha.shape[0] != logits.shape[1]:
            raise ValueError(
                f"alpha length {self.alpha.shape[0]} does not match logit channels {logits.shape[1]}"
            )

        log_probs = F.log_softmax(logits.float(), dim=1)
        probs = log_probs.exp()
        alpha = self.alpha.to(logits.device)

        valid = target != self.ignore_index
        # Target class channel for vectorized indexing. Keep values clipped so
        # out-of-range channels never index into log_probs.
        cls = torch.clamp(target, min=0, max=log_probs.shape[1] - 1)
        one_hot = F.one_hot(cls, num_classes=log_probs.shape[1]).to(
            dtype=log_probs.dtype, device=logits.device
        )  # (B, H, W, C)

        log_pt = (one_hot * log_probs.permute(0, 2, 3, 1)).sum(dim=-1)  # (B, H, W)
        pt = (one_hot * probs.permute(0, 2, 3, 1)).sum(dim=-1)  # (B, H, W)
        alpha_t = (one_hot * alpha).sum(dim=-1)  # (B, H, W)

        focused = -alpha_t * ((1.0 - pt) ** self.gamma) * log_pt  # (B, H, W)

        if not valid.any():
            return focused.sum() * 0.0
        focused = focused[valid]
        if self.reduction == "sum":
            return focused.sum()
        if self.reduction == "none":
            return focused
        return focused.mean()


class MulticlassDiceLoss(nn.Module):
    """Multiclass soft Dice loss flattened over label channel and batch.

    loss = 1 - 2*|x*y| / (|x|+|y|+smooth), averaged over the N_CLASSES label
    channels. Unlabeled pixels (masked target == IGNORE_INDEX) are zeroed in
    the one-hot so they contribute no target mass. In an all-background sample
    the channel becomes a no-op and every class for which the model predicts
    nothing stays at channel score 1 (zero loss).
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits.float(), dim=1)  # (B, C, H, W)
        valid = target != IGNORE_INDEX
        cls = torch.clamp(target, min=0, max=logits.shape[1] - 1)
        target_one_hot = F.one_hot(cls, num_classes=logits.shape[1]).to(
            dtype=probs.dtype, device=logits.device
        )  # (B, H, W, C)
        target_one_hot = target_one_hot * valid.unsqueeze(-1)  # zero unlabeled
        target_one_hot = target_one_hot.permute(0, 3, 1, 2)  # (B, C, H, W)

        intersection = (probs * target_one_hot).sum(dim=(0, 2, 3))
        cardinality = probs.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3))
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return (1.0 - dice).mean()


class LovaszSoftmaxLoss(nn.Module):
    """Per-class Lovász-Softmax surrogate for the mean Jaccard index.
    Directly optimises the IoU for each class, making it far superior to
    Dice for heavily imbalanced segmentation. Unlabeled pixels (IGNORE_INDEX)
    are masked out before the per-class computation.
    Reference: Berman et al. (2018) "The Lovász-Softmax loss: A tractable
    surrogate for the optimization of the intersection-over-union measure
    in neural networks." CVPR 2018.
    """
    def __init__(
        self,
        ignore_index: int = IGNORE_INDEX,
        per_image: bool = False,
        classes: str = "present",  # "all" | "present" | list[int]
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.per_image = per_image
        self.classes = classes

    @staticmethod
    def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
        """Compute Lovász gradient for a sorted binary label vector."""
        p = gt_sorted.shape[0]
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union
        if p > 1:
            jaccard[1:] = jaccard[1:] - jaccard[:-1]
        return jaccard

    def _lovasz_softmax_flat(
        self,
        probs: torch.Tensor,   # (P, C) – labeled pixels only
        labels: torch.Tensor,  # (P,)   – ground-truth class indices
    ) -> torch.Tensor:
        if probs.numel() == 0:
            return probs * 0.0
        C = probs.shape[1]
        if isinstance(self.classes, list):
            classes_iter = self.classes
        elif self.classes == "present":
            classes_iter = labels.unique().tolist()
        else:
            classes_iter = range(C)
        losses = []
        for c in classes_iter:
            fg = (labels == c).float()
            if fg.sum() == 0 and self.classes == "present":
                continue
            errors = (fg - probs[:, c]).abs()
            errors_sorted, perm = torch.sort(errors, descending=True)
            perm = perm.data
            fg_sorted = fg[perm]
            losses.append(torch.dot(errors_sorted, self._lovasz_grad(fg_sorted)))
        if not losses:
            return probs.sum() * 0.0
        return torch.stack(losses).mean()

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        probs = torch.softmax(logits.float(), dim=1)  # (B, C, H, W)
        if self.per_image:
            loss = torch.stack([
                self._lovasz_softmax_flat(
                    probs[b].permute(1, 2, 0).reshape(-1, probs.shape[1])[target[b].reshape(-1) != self.ignore_index],
                    target[b].reshape(-1)[target[b].reshape(-1) != self.ignore_index],
                )
                for b in range(probs.shape[0])
            ]).mean()
        else:
            B, C, H, W = probs.shape
            probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, C)
            target_flat = target.reshape(-1)
            valid = target_flat != self.ignore_index
            loss = self._lovasz_softmax_flat(probs_flat[valid], target_flat[valid])
        return loss


class CompoundFocalDiceLoss(nn.Module):
    """Total = Focal_Loss(gamma, alpha) + lambda_dice * Dice_Loss(smooth) + lambda_lovasz * Lovasz_Softmax_Loss."""

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        lambda_dice: float = 1.0,
        lambda_lovasz: float = 0.0,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        if alpha is None:
            alpha = build_focal_alphas(class_weights)
        self.focal = FocalLoss(gamma=gamma, alpha=alpha)
        self.dice = MulticlassDiceLoss(smooth=smooth)
        self._lovasz = LovaszSoftmaxLoss()
        self.lambda_dice = float(lambda_dice)
        self.lambda_lovasz = float(lambda_lovasz)
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        focal_loss = self.focal(logits, target)
        dice_loss = self.dice(logits, target)
        lovasz_loss = self._lovasz(logits, target)
        if not torch.isfinite(focal_loss):
            raise FloatingPointError(f"non-finite focal loss: {float(focal_loss)}")
        if not torch.isfinite(dice_loss):
            raise FloatingPointError(f"non-finite dice loss: {float(dice_loss)}")
        if not torch.isfinite(lovasz_loss):
            raise FloatingPointError(f"non-finite lovasz loss: {float(lovasz_loss)}")
        return focal_loss + self.lambda_dice * dice_loss + self.lambda_lovasz * lovasz_loss
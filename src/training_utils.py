"""Utility functions for training and evaluating the PyTorch U-Net."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


def flatten_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten tensor to (batch_size, -1) for metric computations."""
    return tensor.reshape(tensor.size(0), -1)


def soft_dice_coefficient(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Continuous Dice coefficient used for differentiable loss computation."""

    preds_flat = flatten_tensor(preds)
    targets_flat = flatten_tensor(targets)

    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean()


def dice_coefficient(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Binary Dice coefficient for evaluation metrics."""

    preds_bin = (preds >= threshold).float()
    targets_bin = (targets >= threshold).float()

    preds_flat = flatten_tensor(preds_bin)
    targets_flat = flatten_tensor(targets_bin)

    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)

    # Where both prediction and target are empty, define Dice as 1.
    empty_mask = (union <= eps).float()
    dice = dice * (1 - empty_mask) + empty_mask

    return dice.mean()


def binary_iou(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    preds_bin = (preds >= threshold).float()
    targets_bin = (targets >= threshold).float()

    preds_flat = flatten_tensor(preds_bin)
    targets_flat = flatten_tensor(targets_bin)

    intersection = (preds_flat * targets_flat).sum(dim=1)
    total = preds_flat.sum(dim=1) + targets_flat.sum(dim=1) - intersection
    iou = (intersection + eps) / (total + eps)

    empty_mask = (total <= eps).float()
    iou = iou * (1 - empty_mask) + empty_mask

    return iou.mean()


class DiceBCELoss(nn.Module):
    """Combination of Binary Cross Entropy and Dice loss."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.bce = nn.BCELoss()
        self.eps = eps

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:  # noqa: D401
        bce_loss = self.bce(preds, targets)
        dice_loss = 1.0 - soft_dice_coefficient(preds, targets, eps=self.eps)
        return bce_loss + dice_loss


def numpy_to_tensors(images: np.ndarray, masks: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert NHWC numpy arrays to NCHW torch tensors."""

    images_tensor = torch.from_numpy(np.transpose(images, (0, 3, 1, 2))).float()
    masks_tensor = torch.from_numpy(np.transpose(masks, (0, 3, 1, 2))).float()
    return images_tensor, masks_tensor

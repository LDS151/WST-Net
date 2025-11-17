"""Evaluate the trained PyTorch model on train and test splits."""

from __future__ import annotations

import os
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import TensorDataset

from data_preparation import DataLoader, split_data_train_test
from training_utils import DiceBCELoss, binary_iou, dice_coefficient, numpy_to_tensors
from unet_model_recipe import unet_model

SCRIPT_PATH = os.path.dirname(__file__)
IMAGE_SIZE: Tuple[int, int] = (224, 224)
BATCH_SIZE = 16
MODEL_PATH = os.path.join(SCRIPT_PATH, "..", "models", "unet_brain_segmentation.pt")


def evaluate_split(
    model: torch.nn.Module,
    dataloader: TorchDataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, masks)

            total_loss += loss.item() * batch_size
            total_dice += dice_coefficient(outputs, masks).item() * batch_size
            total_iou += binary_iou(outputs, masks).item() * batch_size
            total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "dice": total_dice / total_samples,
        "iou": total_iou / total_samples,
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_loader = DataLoader()
    images, masks = data_loader.load_data(img_size=IMAGE_SIZE)
    images_train, images_test, masks_train, masks_test = split_data_train_test(images, masks)

    train_dataset = TensorDataset(*numpy_to_tensors(images_train, masks_train))
    test_dataset = TensorDataset(*numpy_to_tensors(images_test, masks_test))

    train_loader = TorchDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = TorchDataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = unet_model(starting_filters=32, in_channels=3, out_channels=1, device=device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)

    criterion = DiceBCELoss()

    train_metrics = evaluate_split(model, train_loader, criterion, device)
    test_metrics = evaluate_split(model, test_loader, criterion, device)

    print("Train metrics:")
    print(train_metrics)
    print("Test metrics:")
    print(test_metrics)


if __name__ == "__main__":
    main()

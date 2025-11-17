#!/usr/bin/env python3
"""
Run segmentation + clinical classification inference on brain MRI slices.

Usage example:
python predict.py \
  --images /path/to/case_001 \
  --checkpoint /path/to/best_fold1.pt \
  --patient-id case_001
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from clinical_model import ClinicalMLP
from slice_mamba import SliceMamba
from unet_model_recipe import unet_model

try:
    # _load_image keeps preprocessing identical to the training pipeline.
    from data_preparation import _load_image  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("data_preparation.py must be available in PYTHONPATH") from exc


IMG_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def natural_key(path: Path) -> Tuple[int, str]:
    name = path.name.lower()
    match = re.search(r"(\d+)", name)
    index = int(match.group(1)) if match else -1
    return index, name


def list_images(input_path: Path, include_masks: bool = False) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Path not found: {input_path}")
    imgs: List[Path] = []
    for entry in input_path.iterdir():
        if not entry.is_file():
            continue
        ext = entry.suffix.lower()
        if ext not in IMG_EXTS:
            continue
        name_lower = entry.name.lower()
        if (not include_masks) and "_mask" in name_lower:
            continue
        imgs.append(entry)
    if not imgs:
        raise RuntimeError(f"No images found under {input_path}")
    imgs.sort(key=natural_key)
    return imgs


def load_and_stack_images(paths: Sequence[Path], image_size: Tuple[int, int]) -> torch.Tensor:
    tensors: List[torch.Tensor] = []
    for path in paths:
        array = _load_image(str(path), image_size, mode="RGB").astype(np.float32)
        array /= 255.0
        tensor = torch.from_numpy(np.transpose(array, (2, 0, 1)))
        tensors.append(tensor)
    return torch.stack(tensors, dim=0)


def infer_patient_id(path: Path) -> Optional[str]:
    name = path.name if path.is_dir() else path.parent.name
    name = name.strip().lower()
    if name.startswith("case"):
        return name
    return None


def load_clinical_from_metadata(patient_id: str, metadata_path: Path) -> Optional[np.ndarray]:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pandas is required to read metadata Excel files") from exc

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata Excel not found: {metadata_path}")

    table = pd.read_excel(metadata_path, sheet_name="Sheet1", header=1)
    table = table.dropna(subset=["id"])
    table[["入院NIHSS", "出院NIHSS"]] = table[["入院NIHSS", "出院NIHSS"]].fillna(5)

    extra_columns = [
        "甘油三酯",
        "总胆固醇",
        "高密度脂蛋白",
        "低密度脂蛋白",
        "椎基底动脉狭窄（1大于50%，0小于50%）",
    ]
    for column in extra_columns:
        if column not in table.columns:
            raise KeyError(f"lack: {column}")
        fill_value = table[column].mean()
        if np.isnan(fill_value):
            fill_value = 0.0
        table[column] = table[column].fillna(fill_value)

    normalized_id = patient_id.strip().lower()
    for _, row in table.iterrows():
        key = f"case_{int(row['id'])}".lower()
        if key != normalized_id:
            continue
        clinical = np.asarray(
            [
                row["入院NIHSS"],
                row["出院NIHSS"],
                row["甘油三酯"],
                row["总胆固醇"],
                row["高密度脂蛋白"],
                row["低密度脂蛋白"],
                row["椎基底动脉狭窄（1大于50%，0小于50%）"],
            ],
            dtype=np.float32,
        )
        return clinical
    return None


def resolve_clinical_features(
    args: argparse.Namespace,
    image_root: Path,
) -> np.ndarray:
    if args.clinical is not None:
        values = np.asarray(args.clinical, dtype=np.float32)
        if values.ndim != 1:
            raise ValueError("Clinical features must be a 1-D list of floats.")
        return values

    source_id = args.patient_id or infer_patient_id(image_root)
    if source_id:
        metadata_path = Path(args.metadata_xlsx)
        clinical = load_clinical_from_metadata(source_id, metadata_path)
        if clinical is not None:
            return clinical
        print(f"[WARN] Failed to locate metadata for {source_id}, falling back to zeros.")
    if args.fallback_clinical_dim < 2:
        raise ValueError("Fallback clinical dimension must be >= 2.")
    print("[WARN] Clinical features not provided; using zeros.")
    return np.zeros(args.fallback_clinical_dim, dtype=np.float32)


def instantiate_models(
    device: torch.device,
    conditioning_dim: int,
    slice_feature_dim: int,
    clinical_input_dim: int,
    slice_mamba_patch: int,
) -> Tuple[torch.nn.Module, Optional[torch.nn.Module], Optional[torch.nn.Module]]:
    unet = unet_model(
        starting_filters=32,
        in_channels=3,
        out_channels=1,
        device=device,
        conditioning_dim=conditioning_dim,
    )

    slice_encoder: Optional[torch.nn.Module] = None
    clinical_head: Optional[torch.nn.Module] = None

    if slice_feature_dim > 0:
        slice_encoder = SliceMamba(
            in_channels=512,
            out_channels=slice_feature_dim,
            patch_size=slice_mamba_patch,
            merge_batch_as_sequence=True,
        ).to(device)

        from dynamic_residual_fusion import DRFWrapper
        clinical_head = DRFWrapper(
            clinical_dim=clinical_input_dim,
            image_dim=slice_feature_dim,
            fusion_dim=256
        ).to(device)

    return unet, slice_encoder, clinical_head


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    unet: torch.nn.Module,
    slice_encoder: Optional[torch.nn.Module],
    clinical_head: Optional[torch.nn.Module],
) -> Dict[str, bool]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    flags = {"unet": False, "slice": False, "clinical": False}

    if isinstance(checkpoint, dict) and "unet" in checkpoint:
        unet.load_state_dict(checkpoint["unet"])
        flags["unet"] = True
        if slice_encoder is not None and "slice_mamba" in checkpoint:
            slice_encoder.load_state_dict(checkpoint["slice_mamba"])
            flags["slice"] = True
        if clinical_head is not None and "clinical" in checkpoint:
            clinical_head.load_state_dict(checkpoint["clinical"])
            flags["clinical"] = True
    else:
        unet.load_state_dict(checkpoint)
        flags["unet"] = True

    return flags


def save_mask(mask: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(mask.astype(np.uint8), mode="L")
    img.save(out_path)


def save_probability_map(prob: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, prob.astype(np.float32))


def parse_args() -> argparse.Namespace:
    default_metadata = Path(__file__).resolve().parent.parent / "../information.xlsx"
    parser = argparse.ArgumentParser(description="MRI segmentation + classification inference.")
    parser.add_argument("--images", required=True, help="Path to an image file or a directory of slices.")
    parser.add_argument("--checkpoint", required=True, help="Path to the .pt checkpoint.")
    parser.add_argument("--output-dir", default="output/predictions", help="Directory to store predicted masks.")
    parser.add_argument("--image-size", nargs=2, type=int, default=(224, 224), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--seg-threshold", type=float, default=0.5, help="Threshold for binary masks.")
    parser.add_argument("--classification-threshold", type=float, default=0.5, help="Positive threshold for clinical prediction.")
    parser.add_argument("--device", type=str, default=None, help="Device string, e.g. cuda:0 or cpu.")
    parser.add_argument("--conditioning-dim", type=int, default=1, help="Conditioning dimension used when training the U-Net.")
    parser.add_argument("--slice-feature-dim", type=int, default=128, help="Dimensionality of SliceMamba output.")
    parser.add_argument("--slice-patch-size", type=int, default=2, help="Patch size used by SliceMamba.")
    parser.add_argument("--clinical", nargs="+", type=float, help="Override clinical feature values (floats).")
    parser.add_argument("--patient-id", type=str, help="Patient identifier (e.g., case_001) to fetch clinical features.")
    parser.add_argument("--metadata-xlsx", type=str, default=str(default_metadata), help="Excel file with clinical metadata.")
    parser.add_argument("--fallback-clinical-dim", type=int, default=7, help="Length of zero vector if no clinical data is found.")
    parser.add_argument("--save-prob-maps", action="store_true", help="Also save raw probability maps (.npy).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_root = Path(args.images).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    image_paths = list_images(image_root)
    tensors = load_and_stack_images(image_paths, tuple(args.image_size))

    clinical_features = resolve_clinical_features(args, image_root)
    clinical_dim = int(clinical_features.shape[0])

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    unet, slice_encoder, clinical_head = instantiate_models(
        device=device,
        conditioning_dim=args.conditioning_dim,
        slice_feature_dim=args.slice_feature_dim,
        clinical_input_dim=clinical_dim,
        slice_mamba_patch=args.slice_patch_size,
    )

    flags = load_checkpoint(checkpoint_path, device, unet, slice_encoder, clinical_head)
    if not flags["unet"]:
        raise RuntimeError("Failed to load U-Net weights from checkpoint.")
    classification_enabled = bool(slice_encoder and clinical_head and flags["slice"] and flags["clinical"])
    if not classification_enabled:
        if slice_encoder and clinical_head:
            print("[WARN] Slice encoder or clinical head weights missing; classification output disabled.")
        else:
            print("[WARN] Classification modules not instantiated; skipping classification.")

    unet.eval()
    if slice_encoder:
        slice_encoder.eval()
    if clinical_head:
        clinical_head.eval()

    with torch.no_grad():
        images_tensor = tensors.to(device=device, dtype=torch.float32)
        clinical_vector = torch.from_numpy(clinical_features).to(device=device, dtype=torch.float32).unsqueeze(0)
        clinical_batch = clinical_vector.repeat(images_tensor.size(0), 1)

        seg_probs, bottleneck = unet(images_tensor, clinical_batch)
        seg_probs = seg_probs.squeeze(1).cpu().numpy()  # [N, H, W]

        cls_prob: Optional[float] = None
        if classification_enabled and slice_encoder and clinical_head:
            slice_feat = slice_encoder(bottleneck)
            slice_feat = F.normalize(slice_feat.float(), dim=1)
            patient_clinical = F.normalize(clinical_vector.float(), dim=1)
            logit = clinical_head(patient_clinical, slice_feat)
            cls_prob = torch.sigmoid(logit).item()

    bin_masks = (seg_probs >= float(args.seg_threshold)).astype(np.uint8) * 255

    for path, mask, prob in zip(image_paths, bin_masks, seg_probs):
        rel_name = path.name
        mask_name = f"{os.path.splitext(rel_name)[0]}_pred_mask.png"
        mask_path = output_dir / mask_name
        save_mask(mask, mask_path)
        if args.save_prob_maps:
            prob_name = f"{os.path.splitext(rel_name)[0]}_prob.npy"
            prob_path = output_dir / prob_name
            save_probability_map(prob, prob_path)

    print(f"Saved {len(image_paths)} predicted masks to {output_dir}")

    if classification_enabled and cls_prob is not None:
        prediction = int(cls_prob >= args.classification_threshold)
        print(
            f"Clinical progression probability: {cls_prob:.4f} "
            f"(threshold={args.classification_threshold} -> label={prediction})"
        )
    else:
        print("Classification result unavailable.")


if __name__ == "__main__":
    main()

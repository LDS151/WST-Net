"""Training pipeline for the PyTorch U-Net model with K-Fold + CSV logging (Dice + HD95 + ASSD)."""

from __future__ import annotations

import os
import csv
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader as TorchDataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from clinical_model import ClinicalMLP
from slice_mamba import SliceMamba
from data_preparation import BrainMRIDataset, DataLoader
from training_utils import DiceBCELoss, dice_coefficient
from unet_model_recipe import unet_model
import torch.nn as nn
 
# Only KDTree and erosion from SciPy are used
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree


class DynamicLossWeighting(nn.Module):
    def __init__(self, initial_weights=[1.0, 0.5]):
        super().__init__()
        init = [-torch.log(torch.tensor(w, dtype=torch.float32)) for w in initial_weights]
        self.log_vars = nn.Parameter(torch.tensor(init, dtype=torch.float32))

    def forward(self, losses):
        weights = torch.exp(-self.log_vars)           # w = exp(-log_var)
        weighted = [w * l for w, l in zip(weights, losses)]
        return sum(weighted) + torch.sum(self.log_vars)


SCRIPT_PATH = os.path.dirname(__file__)
EPOCHS = 30
BATCH_SIZE = 20
LEARNING_RATE = 1e-5
IMAGE_SIZE: Tuple[int, int] = (224, 224)
MODEL_DIR = os.path.join(SCRIPT_PATH, "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "unet_brain_segmentation.pt")
LAMBDA_CLASSIFICATION = 1.0
DYNAMIC_WEIGHTS = None

# K-fold and selection strategy
KFOLDS = 5
SEED = 2025
EPS = 1.5e-2  # Scheme B: Use HD95/ASSD to determine the winner when Dice is approximate

# Log file (single file) configuration
LOG_DIR = os.path.join(SCRIPT_PATH, "..", "logs")
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_CSV = os.path.join(LOG_DIR, f"metrics_{RUN_ID}.csv")
CSV_HEADER = [
    "fold", "epoch", "split", "best", "best_model_path",
    "seg_loss", "cls_loss", "total_loss",
    "dice", "hd95", "assd",
    "cls_acc", "recall", "precision", "f1", "auc",
]


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _ensure_dirs() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

def _csv_append(row: Dict) -> None:
    """Write a row in a unified header order; if the file does not exist, write the header first."""
    file_exists = os.path.exists(LOG_CSV)
    with open(LOG_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(CSV_HEADER)
        writer.writerow([row.get(k, "") for k in CSV_HEADER])




def _mask_to_surface(mask: np.ndarray) -> np.ndarray:
    """Binary mask -> surface pixels (boundary)."""
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if not mask.any():
        return mask
    eroded = binary_erosion(mask)
    return mask ^ eroded  # XOR: boundary = original - erosion

def _coords_from_mask(mask: np.ndarray, spacing=(1.0, 1.0)) -> np.ndarray:
    """Convert True pixel coordinates to physical coordinates (mm). Returns [N,2], columns are (y_mm, x_mm)."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    sy, sx = float(spacing[0]), float(spacing[1])
    coords = np.stack([ys * sy, xs * sx], axis=1).astype(np.float32)
    return coords

def hd95_and_assd_binary(pred: np.ndarray, gt: np.ndarray, spacing=(1.0, 1.0)) -> Tuple[float, float]:
    """
    pred/gt: 2D binary array {0,1} or {False,True}
    Returns (HD95, ASSD), unit is consistent with spacing (default is pixels; if physical spacing is passed in, it is mm).
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    # Both empty => completely consistent (no region), defined as 0
# Both sides are empty: evaluation is meaningless -> return NaN (will be skipped later)
    if not pred.any() and not gt.any():
        return np.nan, np.nan

    # Take surface points
    psurf = _mask_to_surface(pred)
    gsurf = _mask_to_surface(gt)
    P = _coords_from_mask(psurf, spacing)
    G = _coords_from_mask(gsurf, spacing)

    # If one is empty and the other is not => return a large value (stable training/logging; also convenient for early judgment of inferiority)
    if P.shape[0] == 0 or G.shape[0] == 0:
        return np.nan, np.nan


    # KD-Tree nearest neighbor distance
    tree_G = cKDTree(G)
    dists_P2G, _ = tree_G.query(P, k=1, workers=-1)
    tree_P = cKDTree(P)
    dists_G2P, _ = tree_P.query(G, k=1, workers=-1)

    # HD95: 95th percentile of bidirectional distance merge
    all_d = np.hstack([dists_P2G, dists_G2P])
    hd95 = float(np.percentile(all_d, 95))

    # ASSD: bidirectional average
    assd = float(0.5 * (dists_P2G.mean() + dists_G2P.mean()))
    return hd95, assd

def batch_hd95_assd_from_logits(seg_probs, masks, thresh=0.5, spacing=(1.0, 1.0)):
    probs = seg_probs.detach().cpu().numpy()
    gts   = masks.detach().cpu().numpy()
    B = probs.shape[0]
    hd_list, assd_list = [], []
    for i in range(B):
        pred = (probs[i, 0] >= thresh).astype(np.uint8)
        gt   = (gts[i, 0] > 0.5).astype(np.uint8)
        hd, asd = hd95_and_assd_binary(pred, gt, spacing)
        hd_list.append(hd)
        assd_list.append(asd)

    hd_valid   = [h for h in hd_list if not np.isnan(h)]
    assd_valid = [a for a in assd_list if not np.isnan(a)]
    if len(hd_valid) == 0 or len(assd_valid) == 0:
        return float("nan"), float("nan"), 0
    return float(np.mean(hd_valid)), float(np.mean(assd_valid)), len(hd_valid)



class PatientBatchSampler:
    def __init__(
        self,
        dataset: BrainMRIDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._patient_to_indices: Dict[str, List[int]] = defaultdict(list)

        for index in range(len(dataset)):
            patient_id = dataset.get_patient_id(index)
            self._patient_to_indices[patient_id].append(index)

    def __iter__(self):
        patient_ids = list(self._patient_to_indices.keys())
        if self.shuffle:
            np.random.shuffle(patient_ids)

        for patient_id in patient_ids:
            indices = self._patient_to_indices[patient_id]
            if len(indices) < self.batch_size and self.drop_last:
                continue
            yield indices

    def __len__(self) -> int:
        return sum(
            1 for indices in self._patient_to_indices.values()
            if not self.drop_last or len(indices) >= self.batch_size
        )


def build_dataset_by_patients(data_loader: DataLoader, patients: List[str]) -> BrainMRIDataset:
    patient_set = set(patients)
    paired_paths = [
        (image_path, mask_path)
        for image_path, mask_path in zip(data_loader.image_files, data_loader.mask_files)
        if os.path.basename(os.path.dirname(image_path)) in patient_set
    ]
    image_files = [pair[0] for pair in paired_paths]
    mask_files = [pair[1] for pair in paired_paths]
    return BrainMRIDataset(
        image_files=image_files,
        mask_files=mask_files,
        img_size=IMAGE_SIZE,
        normalize=True,
        patient_metadata=data_loader._patient_metadata,
    )


def better(a: Dict[str, float], b: Dict[str, float] | None, eps: float = EPS) -> bool:
    if b is None:
        return True

    # 1) First look at Dice (high priority)
    a_dice = float(a.get("dice", 0.0))
    b_dice = float(b.get("dice", 0.0))
    if a_dice > b_dice + eps:
        return True

    # 2) If Dice is close, look at HD95 (low priority)
    if abs(a_dice - b_dice) <= eps:
        a_hd95 = float(a.get("hd95", np.inf))
        b_hd95 = float(b.get("hd95", np.inf))
        if a_hd95 < b_hd95 - eps:
            return True

        # 3) HD95 is also close: look at ASSD (low priority)
        if abs(a_hd95 - b_hd95) <= eps:
            a_assd = float(a.get("assd", np.inf))
            b_assd = float(b.get("assd", np.inf))
            if a_assd < b_assd - eps:
                return True

            # 4) Then fall back to F1 (high priority)
            a_f1 = float(a.get("f1", 0.0))
            b_f1 = float(b.get("f1", 0.0))
            return a_f1 > b_f1

    return False


def train_one_epoch(
    model: torch.nn.Module,
    slice_encoder: torch.nn.Module,
    clinical_model: torch.nn.Module,
    dataloader: TorchDataLoader,
    seg_loss_fn: torch.nn.Module,
    cls_loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_cls: float = LAMBDA_CLASSIFICATION,
) -> Dict[str, float]:
    model.train()
    clinical_model.train()
    slice_encoder.train()

    running_seg_loss = 0.0
    running_cls_loss = 0.0
    running_dice = 0.0
    running_hd95 = 0.0
    running_assd = 0.0
    running_cls_acc = 0.0
    running_total_loss = 0.0
    slice_count = 0
    patient_count = 0

    # Accumulation of patient-level classification indicators
    all_labels, all_probs, all_preds = [], [], []

    hd_sum = 0.0
    assd_sum = 0.0
    hd_n = 0
    assd_n = 0

    for images, masks, labels, clinical in dataloader:

        images = images.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        clinical = clinical.to(device)
        batch_size = images.size(0)

        patient_clinical = clinical[:1]
        if clinical.size(0) > 1 and not torch.allclose(clinical, patient_clinical.expand_as(clinical)):
            raise ValueError("Clinical features within the same patient batch are inconsistent and cannot be aggregated by patient for classification output.")

        optimizer.zero_grad()

        # U-Net forward (with clinical conditions)
        seg_logits, bottleneck = model(images, clinical)

       
        bottleneck_for_cls = bottleneck
        slice_feat = slice_encoder(bottleneck_for_cls)
        slice_feat = torch.nn.functional.normalize(slice_feat.float(), dim=1)
        patient_clinical = torch.nn.functional.normalize(patient_clinical.float(), dim=1)
        # Patient-level classification: clinical + slice features
        cls_logits = clinical_model(patient_clinical, slice_feat)

        seg_loss = seg_loss_fn(seg_logits, masks)
        patient_logit = cls_logits.view(1)
        patient_label = labels[:1]
        cls_loss = cls_loss_fn(patient_logit, patient_label)

        loss = DYNAMIC_WEIGHTS([seg_loss, cls_loss])
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            running_seg_loss += seg_loss.item() * batch_size
            running_cls_loss += cls_loss.item()
            running_total_loss += loss.item()
            running_dice += dice_coefficient(seg_logits.detach(), masks).item() * batch_size

        _b_hd95, _b_assd, valid_n = batch_hd95_assd_from_logits(seg_logits, masks, thresh=0.5, spacing=(1.0, 1.0))
        if valid_n > 0:
            hd_sum   += _b_hd95 * valid_n
            hd_n     += valid_n
            assd_sum += _b_assd * valid_n
            assd_n   += valid_n



        prob = torch.sigmoid(patient_logit).view(-1)
        pred = (prob >= 0.5).float()
        running_cls_acc += (pred == patient_label).sum().item()

        all_labels.append(patient_label.item())
        all_probs.append(float(prob.item()))
        all_preds.append(int(pred.item()))

        slice_count += batch_size
        patient_count += 1

    # Handle empty samples
    if patient_count == 0 or slice_count == 0 or len(all_labels) == 0:
        return {
            "seg_loss": float("nan"),
            "cls_loss": float("nan"),
            "total_loss": float("nan"),
            "dice": float("nan"),
            "hd95": float("nan"),
            "assd": float("nan"),
            "cls_acc": float("nan"),
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "auc": float("nan"),
        }

    # Four classification indicators
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float("nan")
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return {
        "seg_loss": running_seg_loss / slice_count,
        "cls_loss": running_cls_loss / patient_count,
        "total_loss": running_total_loss / patient_count,
        "dice": running_dice / slice_count,
        "hd95": (hd_sum / hd_n) if hd_n > 0 else float("nan"),
        "assd": (assd_sum / assd_n) if assd_n > 0 else float("nan"),

        "cls_acc": running_cls_acc / patient_count,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "auc": auc,
    }

def evaluate(
    model: torch.nn.Module,
    slice_encoder: torch.nn.Module,
    clinical_model: torch.nn.Module,
    dataloader: TorchDataLoader,
    seg_loss_fn: torch.nn.Module,
    cls_loss_fn: torch.nn.Module,
    device: torch.device,
    lambda_cls: float = LAMBDA_CLASSIFICATION,
) -> Dict[str, float]:
    model.eval()
    clinical_model.eval()
    slice_encoder.eval()

    running_seg_loss = 0.0
    running_cls_loss = 0.0
    running_dice = 0.0
    running_hd95 = 0.0
    running_assd = 0.0
    running_cls_acc = 0.0
    running_total_loss = 0.0
    slice_count = 0
    patient_count = 0
    hd_sum = 0.0
    assd_sum = 0.0
    hd_n = 0
    assd_n = 0

    all_labels, all_probs, all_preds = [], [], []

    with torch.no_grad():
        for images, masks, labels, clinical in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            clinical = clinical.to(device)
            batch_size = images.size(0)

            patient_clinical = clinical[:1]
            if clinical.size(0) > 1 and not torch.allclose(clinical, patient_clinical.expand_as(clinical)):
                raise ValueError("Clinical features within the same patient batch are inconsistent and cannot be aggregated by patient for classification output.")

            seg_logits, bottleneck = model(images, clinical)

            # Keep consistent during validation (use detach, does not affect no_grad)
            slice_feat = slice_encoder(bottleneck.detach())
            slice_feat = torch.nn.functional.normalize(slice_feat.float(), dim=1)
            patient_clinical = torch.nn.functional.normalize(patient_clinical.float(), dim=1)
            cls_logits = clinical_model(patient_clinical, slice_feat)

            seg_loss = seg_loss_fn(seg_logits, masks)
            patient_logit = cls_logits.view(1)
            patient_label = labels[:1]
            cls_loss = cls_loss_fn(patient_logit, patient_label)

            running_seg_loss += seg_loss.item() * batch_size
            running_cls_loss += cls_loss.item()
            running_total_loss += DYNAMIC_WEIGHTS([seg_loss, cls_loss]).item()
            running_dice += dice_coefficient(seg_logits, masks).item() * batch_size

            _b_hd95, _b_assd, valid_n = batch_hd95_assd_from_logits(seg_logits, masks, thresh=0.5, spacing=(1.0, 1.0))
            if valid_n > 0:
                hd_sum   += _b_hd95 * valid_n
                hd_n     += valid_n
                assd_sum += _b_assd * valid_n
                assd_n   += valid_n


            prob = torch.sigmoid(patient_logit).view(-1)
            pred = (prob >= 0.5).float()
            running_cls_acc += (pred == patient_label).sum().item()

            all_labels.append(patient_label.item())
            all_probs.append(float(prob.item()))
            all_preds.append(int(pred.item()))

            slice_count += batch_size
            patient_count += 1

    if patient_count == 0 or slice_count == 0 or len(all_labels) == 0:
        return {
            "seg_loss": float("nan"),
            "cls_loss": float("nan"),
            "total_loss": float("nan"),
            "dice": float("nan"),
            "hd95": float("nan"),
            "assd": float("nan"),
            "cls_acc": float("nan"),
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "auc": float("nan"),
        }

    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float("nan")
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return {
        "seg_loss": running_seg_loss / slice_count,
        "cls_loss": running_cls_loss / patient_count,
        "total_loss": running_total_loss / patient_count,
        "dice": running_dice / slice_count,
        "hd95": (hd_sum / hd_n) if hd_n > 0 else float("nan"),
        "assd": (assd_sum / assd_n) if assd_n > 0 else float("nan"),

        "cls_acc": running_cls_acc / patient_count,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "auc": auc,
    }


def main() -> None:
    _ensure_dirs()
    set_seed(SEED)   # Fix randomness
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

    # Use patient ID for K-fold
    data_loader = DataLoader()
    all_patient_ids = np.array(data_loader.get_patient_ids())
    y = np.array(
        [int(data_loader._patient_metadata[pid]["label"]) for pid in all_patient_ids],
        dtype=np.int32 
    )
    kf = StratifiedKFold(n_splits=KFOLDS, shuffle=True, random_state=SEED)
    fold_results: List[Dict[str, float]] = []

    print(f"[INFO] CSV log: {LOG_CSV}")

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_patient_ids, y), 1):
        train_patients = all_patient_ids[train_idx].tolist()
        val_patients   = all_patient_ids[val_idx].tolist()

        train_dataset = build_dataset_by_patients(data_loader, train_patients)
        val_dataset   = build_dataset_by_patients(data_loader, val_patients)

        train_loader = TorchDataLoader(
            train_dataset,
            batch_sampler=PatientBatchSampler(train_dataset, batch_size=BATCH_SIZE, shuffle=True),
        )
        val_loader   = TorchDataLoader(
            val_dataset,
            batch_sampler=PatientBatchSampler(val_dataset, batch_size=BATCH_SIZE, shuffle=False),
        )

        # Reinitialize model/optimizer for each fold
        sample_clinical = train_loader.dataset[0][3]
      # conditioning_dim = sample_clinical.numel()
        conditioning_dim = 1
        slice_feature_dim = 128
        model = unet_model(
            starting_filters=32,
            in_channels=3,
            out_channels=1,
            device=device,
            conditioning_dim=conditioning_dim,
        )
        slice_encoder = SliceMamba(
            in_channels=512,
            out_channels=slice_feature_dim,
            patch_size=2,
            merge_batch_as_sequence=True,
        ).to(device)
        from dynamic_residual_fusion import DRFWrapper
        clinical_model = DRFWrapper(
            clinical_dim=7, # Based on data_preparation.py
            image_dim=slice_feature_dim,
            fusion_dim=256
        ).to(device)

        seg_loss_fn = DiceBCELoss()
        cls_loss_fn = torch.nn.BCEWithLogitsLoss()
        global DYNAMIC_WEIGHTS
        DYNAMIC_WEIGHTS = DynamicLossWeighting(initial_weights=[1.0, 0.5]).to(device)

        optimizer = torch.optim.Adam(
            list(model.parameters())
            + list(slice_encoder.parameters())
            + list(clinical_model.parameters())
            + list(DYNAMIC_WEIGHTS.parameters()),
            lr=LEARNING_RATE,
        )

        best_metrics = None
        best_state_dict = None
        best_epoch = -1

        print(f"\n===== Fold {fold}/{KFOLDS} =====")
        for epoch in range(1, EPOCHS + 1):
            train_metrics = train_one_epoch(
                model, slice_encoder, clinical_model,
                train_loader, seg_loss_fn, cls_loss_fn, optimizer, device,
            )
            val_metrics = evaluate(
                model, slice_encoder, clinical_model,
                val_loader, seg_loss_fn, cls_loss_fn, device,
            )

            # Console print
            print(
                f"Fold {fold} | Epoch {epoch}/{EPOCHS} - "
                f"train(seg_loss={train_metrics['seg_loss']:.4f}, cls_loss={train_metrics['cls_loss']:.4f}, "
                f"dice={train_metrics['dice']:.4f}, hd95={train_metrics['hd95']:.4f}, assd={train_metrics['assd']:.4f}, "
                f"cls_acc={train_metrics['cls_acc']:.4f}, recall={train_metrics['recall']:.4f}, "
                f"precision={train_metrics['precision']:.4f}, f1={train_metrics['f1']:.4f}, auc={train_metrics['auc']:.4f}) | "
                f"val(seg_loss={val_metrics['seg_loss']:.4f}, cls_loss={val_metrics['cls_loss']:.4f}, "
                f"dice={val_metrics['dice']:.4f}, hd95={val_metrics['hd95']:.4f}, assd={val_metrics['assd']:.4f}, "
                f"cls_acc={val_metrics['cls_acc']:.4f}, recall={val_metrics['recall']:.4f}, "
                f"precision={val_metrics['precision']:.4f}, f1={val_metrics['f1']:.4f}, auc={val_metrics['auc']:.4f})"
            )

            # —— Write to CSV: train/val results for the current round ——
            _csv_append({
                "fold": fold, "epoch": epoch, "split": "train", "best": 0, "best_model_path": "",
                **train_metrics,
            })
            _csv_append({
                "fold": fold, "epoch": epoch, "split": "val", "best": 0, "best_model_path": "",
                **val_metrics,
            })

            # Scheme B: Dice has high priority; if Dice is approximate, compare HD95 (low priority), ASSD (low priority), and then F1 (high priority)
            if better(val_metrics, best_metrics, EPS):
                best_metrics = val_metrics
                best_epoch = epoch
                best_state_dict = {
                    "unet": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "slice_mamba": {k: v.detach().cpu().clone() for k, v in slice_encoder.state_dict().items()},
                    "clinical": {k: v.detach().cpu().clone() for k, v in clinical_model.state_dict().items()},
                }

        # Save the best model for each fold
        best_model_path = ""
        if best_state_dict is not None:
            best_model_path = os.path.join(MODEL_DIR, f"best_fold{fold}.pt")
            torch.save(best_state_dict, best_model_path)

        # —— Write to CSV: the best record of this fold (split=best_val, best=1) ——
        if best_metrics is not None:
            _csv_append({
                "fold": fold, "epoch": best_epoch, "split": "best_val", "best": 1,
                "best_model_path": best_model_path, **best_metrics,
            })

        # Record the best validation indicators for each fold
        fold_results.append(best_metrics)

    # Console summary (mean ± standard deviation; NaN ignored)
    keys = ["dice", "hd95", "assd", "recall", "precision", "f1", "auc", "cls_acc", "seg_loss", "cls_loss", "total_loss"]
    print("\n===== K-Fold Summary =====")
    for k in keys:
        arr = np.array([r[k] for r in fold_results if r is not None], dtype=np.float32)
        if arr.size == 0:
            continue
        mean = np.nanmean(arr)
        std = np.nanstd(arr, ddof=1) if arr.size > 1 else 0.0
        print(f"{k}: {mean:.4f} ± {std:.4f}")

if __name__ == "__main__":
    main()

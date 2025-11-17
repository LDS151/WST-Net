"""Data loading and preprocessing utilities for MRI segmentation.

Kaggle dataset: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
"""

import glob
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

SCRIPT_PATH = os.path.dirname(__file__)
DATASET_PATH = os.path.join(SCRIPT_PATH, "../..", "data", "kaggle_3m")
INFO_PATH = os.path.join(SCRIPT_PATH, "..", "information.xlsx")


def _load_image(path: str, target_size: Tuple[int, int], mode: str) -> np.ndarray:
    """Load an image using Pillow and return as numpy array."""
    image = Image.open(path).convert(mode)
    if image.size != target_size:
        image = image.resize(target_size)
    return np.asarray(image)



class DataLoader:
    """
    Load Brain Mri Segmentation data, preprocess and return target input pairs.

    Dataset should be in same convention as original kaggle one, it is:
    Patient_folder/
        file_1_mask.tif
        file_1.tif
        ...
    """
    def __init__(self, dataset_path: str = DATASET_PATH):
        self._dataset_path = dataset_path
        self.__prepare_filenames()
        self._patient_metadata = self._load_patient_metadata()

    def _load_patient_metadata(self) -> Dict[str, Dict[str, np.ndarray]]:
            table = pd.read_excel(INFO_PATH, sheet_name="Sheet1", header=1)
            table = table.dropna(subset=["黄如月序号"])
            table[["入院NIHSS", "出院NIHSS"]] = table[["入院NIHSS", "出院NIHSS"]].fillna(5)
            extra_columns = [
                "甘油三酯",
                "总胆固醇",
                "高密度脂蛋白",
                "低密度脂蛋白",
                "椎基底动脉狭窄（1大于50%，0小于50%）",
            ]
            for column in extra_columns:
                if column in table.columns:
                    fill_value = table[column].mean()
                    if np.isnan(fill_value):
                        fill_value = 0.0
                    table[column] = table[column].fillna(fill_value)
                else:
                    raise KeyError(f"信息表缺少所需列: {column}")
            metadata: Dict[str, Dict[str, np.ndarray]] = {}
            for _, row in table.iterrows():
                patient_key = f"case_{int(row['黄如月序号'])}"
                label = float(row["是否进展"])
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
                metadata[patient_key] = {"label": label, "clinical": clinical}
            return metadata


    def __prepare_filenames(self):
        """
        Go through files in dataset path and split them into mask_files and image_files.
        Only files with masks included.
        """
        file_names = list(sorted(glob.glob(self._dataset_path + "/**/*mask.tif")))
        self.image_files: List[str] = []
        self.mask_files: List[str] = []
        for path in file_names:
            self.mask_files.append(path)
            self.image_files.append(path[: -9] + ".tif")

        self._patient_ids = sorted({os.path.basename(os.path.dirname(path)) for path in self.mask_files})

    def load_data(
            self,
            normalize: bool = True,
            img_size: Tuple[int, int] = (255, 255),
            specific_patient: str = None) -> Tuple[np.ndarray, np.ndarray]:

        images_files = self.image_files
        masks_files = self.mask_files
        if specific_patient is not None:
            images_files = [
                image_path for image_path in images_files if specific_patient in image_path
            ]
            masks_files = [
                mask_path for mask_path in masks_files if specific_patient in mask_path
            ]

        inputs = np.zeros((len(images_files), *img_size, 3), dtype=np.float32)
        targets = np.zeros((len(masks_files), *img_size, 1), dtype=np.uint8)

        for index, (image_path, mask_path) in enumerate(zip(images_files, masks_files)):
            image = _load_image(image_path, img_size, mode="RGB")
            mask = _load_image(mask_path, img_size, mode="L")

            if normalize:
                image = image.astype(np.float32) / 255.0
                mask = (mask.astype(np.uint8) / 255).astype(np.uint8)

            inputs[index] = image
            targets[index, ..., 0] = mask

        return inputs, targets

    def get_patient_ids(self) -> List[str]:
        """Return sorted patient identifiers discovered in the dataset."""

        return list(self._patient_ids)

    def get_dataset(
        self,
        normalize: bool = True,
        img_size: Tuple[int, int] = (255, 255),
        specific_patient: Optional[str] = None,
        augmentations: Optional[Iterable] = None,
    ) -> "BrainMRIDataset":
        """Create a PyTorch dataset with the same preprocessing logic."""

        images_files = self.image_files
        masks_files = self.mask_files
        if specific_patient is not None:
            images_files = [
                image_path for image_path in images_files if specific_patient in image_path
            ]
            masks_files = [
                mask_path for mask_path in masks_files if specific_patient in mask_path
            ]

        return BrainMRIDataset(
            image_files=images_files,
            mask_files=masks_files,
            img_size=img_size,
            normalize=normalize,
            augmentations=augmentations,
            patient_metadata=self._patient_metadata,

        )


def split_data_train_test(
        inputs: np.ndarray, targets: np.ndarray,
        random_state=2022, test_size=0.2):
    """
    Function for spliting data into train and test set (based on sklearn train_test_split).
    Default input shape (number_of_images, img_height, img_width, 3).
    Keeping random_state frozen for reproducible outcomes.
    """
    input_train, input_test, target_train, target_test = train_test_split(
        inputs,
        targets,
        test_size=test_size,
        random_state=random_state
    )

    return input_train, input_test, target_train, target_test


class BrainMRIDataset(Dataset):
    """Torch dataset wrapper for the Brain MRI segmentation task."""

    def __init__(
        self,
        image_files: List[str],
        mask_files: List[str],
        img_size: Tuple[int, int] = (255, 255),
        normalize: bool = True,
        augmentations: Optional[Iterable] = None,
        patient_metadata: Optional[Dict[str, Dict[str, np.ndarray]]] = None
    ) -> None:
        if len(image_files) != len(mask_files):
            raise ValueError("Image and mask file lists must have the same length.")

        self.patient_metadata = patient_metadata or {}
        self.image_files = image_files
        self.mask_files = mask_files
        self.img_size = img_size
        self.normalize = normalize
        self.augmentations = augmentations
        self.patient_ids = [os.path.basename(os.path.dirname(path)) for path in self.image_files]
    def get_patient_id(self, index: int) -> str:
        return self.patient_ids[index]

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image_path = self.image_files[index]
        mask_path = self.mask_files[index]

        image = _load_image(image_path, self.img_size, mode="RGB").astype(np.float32)
        mask = _load_image(mask_path, self.img_size, mode="L").astype(np.uint8)

        if self.normalize:
            image /= 255.0
            mask = (mask / 255).astype(np.float32)
        else:
            mask = mask.astype(np.float32)

        if self.augmentations is not None:
            augmented = self.augmentations(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1)))
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        patient_id = self.patient_ids[index]
        info = self.patient_metadata.get(patient_id)
        if info is None:
            raise KeyError(f"No metadata for patient {patient_id}")

        label_tensor = torch.tensor(info["label"], dtype=torch.float32)
        clinical_tensor = torch.from_numpy(info["clinical"]).float()
        return image_tensor.float(), mask_tensor.float(), label_tensor, clinical_tensor

        #return image_tensor.float(), mask_tensor.float()

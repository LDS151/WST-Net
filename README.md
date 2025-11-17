# Brain MRI Segmentation and Clinical Outcome Prediction

## Overview

This is a unified framework for medical image analysis that leverages both multi-modal and multi-task learning.
![Model Architecture](fig/main.png)


## Key Features

- **Segmentation Model**: A **UNetWavelet** architecture is used for the segmentation stage.  
It integrates Wavelet transforms to enhance multi-scale feature extraction and improve segmentation quality.

- **SliceMamba**: The **SliceMamba** module (a State Space Model, SSM) encodes sequential 2D MRI slices into a unified, patient-level representation.

- **ClinicalMLP**: **ClinicalMLP** performs deep multimodal fusion, combining imaging-derived embeddings with clinical data for downstream classification tasks.
  
## Dataset Setup

1.  **Image Data**: The image data should follow the Kaggle LGG MRI Segmentation dataset format. Place the data in a structure like this:
    ```
    data/image/
    ├── case_001/
    │   ├── case_001_slice_001.tif
    │   └── case_001_slice_001_mask.tif
    │   └── ...
    └── case_002/
        └── ...
    ```

2.  **Clinical Data**: A file named `information.xlsx` containing clinical metadata is required at the project root. It should include columns for patient ID, outcomes, and clinical measurements (e.g., NIHSS scores).

## Usage

### Training

To start the K-fold cross-validation training process, run:

```bash

pip install -r requirements.txt

python src/unet_training.py
```

- Trained models for each fold will be saved in the `models/` directory.
- Detailed logs, including performance metrics for each epoch, will be written to a CSV file in the `logs/` directory.

### Inference

To run inference on a set of MRI slices for a single patient, use the `predict.py` script:

```bash
python src/predict.py \
  --images /path/to/patient_slice_directory \
  --checkpoint models/best_fold1.pt \
  --patient-id case_001
```

- **`--images`**: Path to the directory containing the patient's MRI slices.
- **`--checkpoint`**: Path to the trained model file (`.pt`).
- **`--patient-id`**: The patient identifier used to look up clinical data from `information.xlsx`.



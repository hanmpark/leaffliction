# leaffliction

<p align="center">
  <img src="https://img.shields.io/badge/Project-Leaffliction-1f6feb?style=for-the-badge" alt="Project Leaffliction">
  <img src="https://img.shields.io/badge/Model-SVC%20(RBF)-2ea44f?style=for-the-badge" alt="Model SVC RBF">
  <img src="https://img.shields.io/badge/Pipeline-Analysis%20%C2%B7%20Augmentation%20%C2%B7%20Transformation%20%C2%B7%20Classification-111111?style=for-the-badge" alt="Pipeline">
</p>

<p align="center">
  A practical end-to-end plant leaf disease workflow using handcrafted features and classical machine learning.<br>
  The repository covers dataset analysis, augmentation/balancing, transformation, training, and prediction.
</p>

---

## Project Overview

`leaffliction` is organized as a reproducible pipeline:

1. Analyze class distribution per plant.
2. Augment or rebalance image classes.
3. Run transformation and morphology-oriented image inspection.
4. Train a `StandardScaler + SVC(kernel="rbf")` classifier.
5. Predict from a single image or a directory tree.

The implementation is split into focused modules in `src/`.

---

## Quick Start

### 1) Install dependencies

```bash
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Analyze dataset distribution

```bash
python3 src/analysis/Distribution.py <dataset_dir> --out charts
```

### 3) Augment or balance data

Augment originals in-place:

```bash
python3 src/augmentation/Augmentation.py <input_path>
```

Create a balanced output copy:

```bash
python3 src/augmentation/balance_dataset.py \
  --src <dataset_dir> \
  --out augmented_directory \
  --augmentation-script src/augmentation/Augmentation.py
```

Balance source in-place:

```bash
python3 src/augmentation/balance_dataset.py \
  --src <dataset_dir> \
  --in-place \
  --augmentation-script src/augmentation/Augmentation.py
```

### 4) Run transformation pipeline

Single image visualization:

```bash
python3 src/transformation/Transformation.py <image_path>
```

Batch export mode:

```bash
python3 src/transformation/Transformation.py -src <input_dir> -dst <output_dir> --mask
```

### 5) Train classifier

```bash
python3 src/classification/train.py <dataset_dir>
```

Optional automation flags:

- `--val-split <float>`
- `--auto-balance true|false`
- `--output <path>`
- `--generate-zip`

### 6) Predict

Single image:

```bash
python3 src/classification/predict.py artifacts/model <image_path>
```

Directory mode:

```bash
python3 src/classification/predict.py artifacts/model <images_dir> --max=100
```

---

## Dataset Expectations

Preferred structure:

```text
<dataset_root>/
├── Apple/
│   ├── Apple_Black_rot/
│   ├── Apple_healthy/
│   ├── Apple_rust/
│   └── Apple_scab/
└── Grape/
    ├── Grape_Black_rot/
    ├── Grape_Esca/
    ├── Grape_healthy/
    └── Grape_spot/
```

Flat class layout (`<root>/<class>`) is also supported by analysis/augmentation scripts.

When training data has no split folders, `train.py` creates:

- `training_data/`
- `validation_data/`

inside the working dataset directory.

---

## Feature Space

The classifier uses handcrafted features (`src/classification/features.py`):

- HSV color histogram: `8 x 8 x 8 = 512`
- HOG descriptor: `8100`
- Texture features (GLCM + image stats): `10`
- Total per-image feature vector: `8622`

Model:

- `Pipeline(StandardScaler -> SVC(kernel="rbf"))`

---

## Artifacts

Main outputs:

- `artifacts/model/model.pkl`
- `prediction_summary.txt` (directory prediction mode)

If `--generate-zip` is enabled in training:

- `artifacts/learnings.zip` includes `model/` and `input_images/`

`model.pkl` payload contains:

- `estimator`
- `feature_config`
- `class_to_idx`
- `idx_to_class`
- `training_summary`

---

## Repository Layout

```text
leaffliction/
├── .github/workflows/flake8.yml
├── artifacts/
├── augmented_directory/
├── dataset/
├── src/
│   ├── analysis/
│   ├── augmentation/
│   ├── transformation/
│   └── classification/
├── test_images/
├── requirements.txt
└── README.md
```

---

## Module Guides

| Module | Purpose | Guide |
|---|---|---|
| Analysis | Per-plant class-distribution charts | [`src/analysis/README.md`](src/analysis/README.md) |
| Augmentation | Geometric augmentation + balancing | [`src/augmentation/README.md`](src/augmentation/README.md) |
| Transformation | Segmentation/ROI/histogram visual pipeline | [`src/transformation/README.md`](src/transformation/README.md) |
| Classification | Handcrafted features, training, prediction | [`src/classification/README.md`](src/classification/README.md) |

---

## License

Distributed under the terms of [`LICENSE`](LICENSE).

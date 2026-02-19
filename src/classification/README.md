# Classification Module

This folder contains the classical (non-deep-learning) image classification pipeline:

- `train.py`: trains an SVM model from a dataset.
- `predict.py`: predicts one class for one image using a saved model.
- `features.py`: handcrafted feature extraction (color histogram + HOG + texture).
- `console_output.py`: CLI output formatting helpers.

## What This Pipeline Does

At a high level:

1. Analyze input dataset balance per class.
2. If unbalanced (class gap > 6), prompt:
   - balance in-place then train, or
   - train directly without balancing.
3. Detect `training_data` / `validation_data` split folders in input data.
4. If missing, prompt for validation split ratio (default `0.2`) and create them.
5. Extract handcrafted features from training split images.
6. Train `Pipeline(StandardScaler -> SVC(kernel="rbf"))`.
7. Save artifacts (model + metadata) in `artifacts`.
8. Run prediction from `artifacts/model/model.pkl`.

## Dataset Assumptions

- Images are discovered recursively.
- Class label = image parent path relative to dataset root.
- Supported image extensions are defined in `features.py` (`.jpg`, `.png`, `.bmp`, `.tif`, `.webp`, etc.).

## Training Program (`train.py`)

### Command

From repository root:

```bash
python3 src/classification/train.py <dataset_dir> [--output=<path>] [--generate-zip] [--val-split=0.2] [--auto-balance=true]
```

### Training Flow

1. Validates `<dataset_dir>`.
2. If `artifacts/model/model.pkl` already exists, prompts:
   - erase and retrain, or
   - use existing model (skip retraining).
3. If `--output=<path>` is set, duplicates `<dataset_dir>` into that path and
   uses the copied dataset as the working/training dataset.
4. Analyzes per-class counts on working dataset.
5. If class imbalance exceeds `+/- 6`, asks user whether to balance in-place.
   - Non-interactive override: `--auto-balance=true|false`
6. Optional in-place balancing uses:
   - `src/augmentation/balance_dataset.py --in-place`
   - `src/augmentation/Augmentation.py`
7. Detects split folders in working dataset:
   - `<dataset_dir>/training_data`
   - `<dataset_dir>/validation_data`
8. If split folders are missing, warns and creates them from a validation
   split ratio:
   - prompted value (default `0.2`), or
   - `--val-split=<float>` if provided.
   - source images are moved into `training_data/` and `validation_data/`.
9. Recreates `artifacts/model`.
10. Extracts features for training split samples:
   - HSV 3D color histogram
   - HOG descriptor
   - Texture (GLCM statistics + Laplacian/gray stats)
11. Trains `SVC(kernel="rbf")` inside a sklearn `Pipeline` with `StandardScaler`.
12. Saves `artifacts/model/model.pkl`.
13. Optionally builds `artifacts/learnings.zip` when `--generate-zip` is used.

### Main Defaults

- Output dir: `./artifacts`
- Image size: `128x128`
- Seed: `42`
- Balance tolerance: `+/- 6` images per class
- Default validation split: `0.2` (used when creating split folders)
- Classifier: `SVC(kernel="rbf")`

### Automation Flags

- `--val-split=<float>`: set split ratio directly (`0 < value < 1`), no prompt.
- `--auto-balance=true`: if dataset is unbalanced, balance in-place automatically.
- `--auto-balance=false`: if dataset is unbalanced, skip balancing automatically.

### Generated Artifacts

- `artifacts/model/model.pkl`
- `artifacts/learnings.zip` (only with `--generate-zip`)

## Model Payload (`model.pkl`)

Saved object is a dictionary with keys:

- `estimator`: fitted sklearn pipeline (`StandardScaler` + `SVC`)
- `feature_config`: feature extraction config used for training
- `class_to_idx`: mapping class name -> integer index
- `idx_to_class`: mapping index (string key) -> class name
- `training_summary`: training metadata (input dataset, feature size, and config)

When `--generate-zip` is enabled, `learnings.zip` contains:

- `model/` (trained model artifacts)
- `input_images/` (supported input images from `<dataset_dir>`)

## Prediction Program (`predict.py`)

### Command

From repository root:

```bash
python3 src/classification/predict.py <model_dir> <input_path> [--max=100]
```

Example:

```bash
python3 src/classification/predict.py artifacts/model test_images/Unit_test2/Grape_healthy.JPG
```

Directory mode example:

```bash
python3 src/classification/predict.py artifacts/model ./images --max=100
```

### Prediction Flow

1. Validates `<model_dir>` and `<input_path>`.
2. Loads `model.pkl`.
3. Reads class mapping from `model.pkl` (`idx_to_class`).
   - Fallback for older artifacts: `classes.json` in model dir.
4. If `<input_path>` is an image:
   - recomputes features using saved `feature_config`
   - runs `estimator.predict(...)`
   - prints predicted class and displays original + resized image
5. If `<input_path>` is a directory:
   - recursively discovers images up to 5 directory levels deep
   - shuffles all discovered images, samples unique images once (`--max` optional)
   - shows a live progress bar (`predicted/total`)
   - shows running success percentage
   - prints a confusion matrix (rows=expected labels, cols=predicted labels)
   - writes detailed report to `prediction_summary.txt`:
     - summary (input dir, processed, success, failed)
     - confusion matrix
     - failed files (`FilePath`, `Prediction`, `Correct Output`)
     - succeeded files (`FilePath`, `Prediction`)

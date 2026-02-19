# Classification Module

This folder contains the classical (non-deep-learning) image classification pipeline:

- `train.py`: trains an SVM model from a dataset.
- `predict.py`: predicts one class for one image using a saved model.
- `features.py`: handcrafted feature extraction (color histogram + HOG + texture).
- `console_output.py`: CLI output formatting helpers.

## What This Pipeline Does

At a high level:

1. Build a balanced augmented dataset in `artifacts/augmented_directory`.
2. Split samples into train/validation sets.
3. Extract handcrafted features from every image.
4. Train `Pipeline(StandardScaler -> SVC(kernel="rbf"))`.
5. Save artifacts (model + metadata) in `artifacts`.
6. Run prediction for a single image from `artifacts/model/model.pkl`.

## Dataset Assumptions

- Images are discovered recursively.
- Class label = image parent path relative to dataset root.
- Supported image extensions are defined in `features.py` (`.jpg`, `.png`, `.bmp`, `.tif`, `.webp`, etc.).

## Training Program (`train.py`)

### Command

From repository root:

```bash
python3 src/classification/train.py <dataset_dir>
```

### Training Flow

1. Validates `<dataset_dir>`.
2. Recreates output folders:
   - `artifacts/model`
   - `artifacts/augmented_directory`
3. Runs augmentation/balancing script:
   - `src/augmentation/balance_dataset.py`
   - `src/augmentation/Augmentation.py`
4. Collects samples + class mapping from the augmented directory.
5. Performs stratified split (`val_split = 0.2` by default).
6. Writes split manifest to `artifacts/dataset_split_paths.json`.
7. Extracts features for all samples:
   - HSV 3D color histogram
   - HOG descriptor
   - Texture (GLCM statistics + Laplacian/gray stats)
8. Trains `SVC(kernel="rbf")` inside a sklearn `Pipeline` with `StandardScaler`.
9. Evaluates train/validation accuracy + confusion matrix.
10. Saves `artifacts/model/model.pkl`.
11. Builds `artifacts/learnings.zip` containing:
    - `model/`
    - `augmented_directory/`

### Main Defaults

- Output dir: `./artifacts`
- Image size: `128x128`
- Seed: `42`
- Validation split: `0.2`
- Classifier: `SVC(kernel="rbf")`

### Generated Artifacts

- `artifacts/model/model.pkl`
- `artifacts/dataset_split_paths.json`
- `artifacts/augmented_directory/...`
- `artifacts/learnings.zip`

## Model Payload (`model.pkl`)

Saved object is a dictionary with keys:

- `estimator`: fitted sklearn pipeline (`StandardScaler` + `SVC`)
- `feature_config`: feature extraction config used for training
- `class_to_idx`: mapping class name -> integer index
- `idx_to_class`: mapping index (string key) -> class name
- `training_summary`: training metadata (accuracy, split counts, confusion matrix, etc.)

## Split Manifest (`dataset_split_paths.json`)

Structure:

```json
{
  "relative_to": "/absolute/path/to/artifacts/augmented_directory",
  "train_set": ["class_a/img1.jpg", "..."],
  "validation_set": ["class_b/img9.jpg", "..."]
}
```

Paths are relative to `relative_to`.

## Prediction Program (`predict.py`)

### Command

From repository root:

```bash
python3 src/classification/predict.py <model_dir> <image_path>
```

Example:

```bash
python3 src/classification/predict.py artifacts/model test_images/Unit_test2/Grape_healthy.JPG
```

### Prediction Flow

1. Validates `<model_dir>` and `<image_path>`.
2. Loads `model.pkl`.
3. Reads class mapping from `model.pkl` (`idx_to_class`).
   - Fallback for older artifacts: `classes.json` in model dir.
4. Recomputes features on input image using saved `feature_config`.
5. Runs `estimator.predict(...)` and prints predicted class.
6. Displays:
   - original image
   - resized model-input image

# Augmentation Module

This folder contains two scripts:

- `Augmentation.py`: applies geometric augmentations.
- `balance_dataset.py`: balances class counts by running augmentation where needed.

## 1) `Augmentation.py`

`Augmentation.py` creates augmented copies for each original image.

### How It Works

For each original image, it generates six transformed files:

- `Flip`
- `Rotate`
- `Skew`
- `Shear`
- `Crop`
- `Distortion`

The output names use this pattern:

- `<name>_Flip.<ext>`
- `<name>_Rotate.<ext>`
- `<name>_Skew.<ext>`
- `<name>_Shear.<ext>`
- `<name>_Crop.<ext>`
- `<name>_Distortion.<ext>`

It only processes original images and skips files that already end with known augmentation suffixes.

Supported image formats:

- `.jpg`
- `.jpeg`
- `.png`

### Supported Input Layouts

- Single image path.
- Dataset root in `<plant>/<class>` layout.
- Dataset root in `<class>` layout.
- A single class directory.

### Usage

From repository root:

```bash
python3 src/augmentation/Augmentation.py <input_path>
```

Examples:

```bash
python3 src/augmentation/Augmentation.py test_images/Unit_test1/Apple_healthy1.JPG
python3 src/augmentation/Augmentation.py dataset
```

The script writes augmented images next to the originals.

## 2) `balance_dataset.py`

`balance_dataset.py` creates a balanced dataset copy by augmenting underrepresented classes.

### How It Works

1. Copies `--src` to `--out` (recreates `--out` from scratch).
2. Detects class directories in `<plant>/<class>`, `<class>`, or single-class layout.
3. Counts images in each class.
4. Sets target count = largest class count.
5. For each class below target:
   - randomly picks an original image that still has missing augmentation outputs,
   - runs `--augmentation-script` on that image,
   - repeats until class count reaches or passes target.
6. Prints before/after class counts.

### Usage

From repository root:

```bash
python3 src/augmentation/balance_dataset.py \
  --src dataset \
  --out augmented_directory \
  --augmentation-script src/augmentation/Augmentation.py
```

Arguments:

- `--src`: source dataset directory.
- `--out`: output directory (default: `augmented_directory`).
- `--augmentation-script`: path to augmentation script to execute.

## Notes

- `--out` is deleted and recreated each run.
- Because one augmentation call may create multiple images, a class can exceed the exact target count.
- The script stops with an error if no eligible originals remain for a class that is still below target.

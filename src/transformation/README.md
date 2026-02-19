# Transformation Module

This folder contains the image transformation and visualization pipeline.

Main files:

- `Transformation.py`: entry point.
- `cli.py`: command-line orchestration.
- `transforms.py`: transformation primitives and analysis helpers.
- `io.py`: image load/save helpers.

## What It Does

Given an image, the pipeline:

1. Loads RGB image data.
2. Applies Gaussian blur.
3. Builds a foreground mask (HSV threshold with Otsu fallback + morphology cleanup).
4. Finds the largest contour.
5. Creates ROI visualization (bounding box).
6. Generates object analysis overlay.
7. Generates pseudolandmarks overlay.
8. Builds channel histogram figure (RGB, HSV, LAB channels).
9. Prints contour metrics (`area`, `perimeter`, `width`, `height`, `centroid`).

## Usage

Run from repository root.

### Single image mode

```bash
python3 src/transformation/Transformation.py <image_path>
```

Example:

```bash
python3 src/transformation/Transformation.py test_images/Unit_test2/Grape_healthy.JPG
```

Behavior:

- Opens a matplotlib window with the generated visualizations.
- Does not write files in single-image mode.

### Directory (batch) mode

```bash
python3 src/transformation/Transformation.py -src <input_dir> -dst <output_dir> [--mask]
```

Example:

```bash
python3 src/transformation/Transformation.py \
  -src test_images/Unit_test2 \
  -dst artifacts/transformation_out \
  --mask
```

Behavior:

- Processes all supported image files in `-src`.
- Skips non-image files.
- Saves output files to `-dst`.

Supported input extensions:

- `jpg`
- `jpeg`
- `png`
- `bmp`
- `tif`
- `tiff`

## Batch Output Files

For each input image `<base>.<ext>`, batch mode writes:

- `<base>_orig.png`
- `<base>_blur.png`
- `<base>_roi.png`
- `<base>_analyze.png`
- `<base>_pseudolandmarks.png`
- `<base>_hist.png`

If `--mask` is used, it also writes:

- `<base>_mask.png`
- `<base>_mask_raw.png`

## Notes

- `-dst` is required when using `-src`.
- You must provide either one image path or `-src`, but not both.
- This module depends on `plantcv`, `opencv-python`, `numpy`, and `matplotlib`.

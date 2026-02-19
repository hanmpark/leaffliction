# Analysis Module

This folder contains `Distribution.py`, a CLI tool that analyzes class counts and generates chart images for each plant.

## What It Does

`Distribution.py`:

1. Reads a dataset directory.
2. Detects class folders.
3. Groups classes by plant name.
4. Counts images per class.
5. Saves one pie chart and one bar chart per plant.

Supported image formats:

- `.jpg`
- `.jpeg`
- `.png`
- `.bmp`
- `.gif`
- `.tif`
- `.tiff`

## Supported Folder Layouts

The script supports:

1. Two-level dataset:
   - `<root>/<plant>/<class>/<image>`
2. One-level dataset:
   - `<root>/<class>/<image>`
   - Plant name is inferred from class prefix before `_`.
3. Single plant folder:
   - `<plant>/<class>/<image>`

## Usage

From repository root:

```bash
python3 src/analysis/Distribution.py <dataset_dir> [--out charts] [--no-show]
```

Example:

```bash
python3 src/analysis/Distribution.py dataset --out charts
```

Arguments:

- `dataset_dir`: dataset root, plant folder, or flat class folder.
- `--out`: directory where chart images are saved (default: `charts`).
- `--no-show`: disable GUI display (save files only).

## Output

For each plant, the script creates:

- `<plant>_pie.png`
- `<plant>_bar.png`

Example output:

- `charts/Apple_pie.png`
- `charts/Apple_bar.png`
- `charts/Grape_pie.png`
- `charts/Grape_bar.png`

## Notes

- The script uses a non-interactive matplotlib backend, so it is safe in headless environments.
- Empty folders are ignored with warnings.
- If no valid class folders with images are found, the script exits with an error.

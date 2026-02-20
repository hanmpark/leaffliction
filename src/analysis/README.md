# Analysis Module

This folder contains `Distribution.py`, a CLI tool that analyzes class counts and displays charts for each plant.

## What It Does

`Distribution.py`:

1. Reads a dataset directory.
2. Detects class folders.
3. Groups classes by plant name.
4. Counts images per class.
5. Displays one window per plant with both a pie chart and a bar chart.

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
python3 src/analysis/Distribution.py <dataset_dir>
```

Example:

```bash
python3 src/analysis/Distribution.py dataset
```

Arguments:

- `dataset_dir`: dataset root, plant folder, or flat class folder.

## Output

For each plant, the script opens a single figure window containing:

- Pie chart (class distribution ratio)
- Bar chart (class image counts)

## Notes

- The script requires an interactive matplotlib backend for GUI display.
- Empty folders are ignored with warnings.
- If no valid class folders with images are found, the script exits with an error.

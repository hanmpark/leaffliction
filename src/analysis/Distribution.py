#!/usr/bin/env python3
"""Generate per-plant class distribution charts from image folders.

Usage:
  python src/analysis/Distribution.py <input_dir>

Supported layouts:
- Two levels: input_dir/<plant>/<class>/*.jpg
- One level: input_dir/<class>/*.jpg

For one-level layouts, plants are inferred from class names
(prefix before "_").
"""

import argparse
import sys
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def count_images(directory: Path) -> int:
    return sum(1 for p in directory.rglob("*") if is_image_file(p))


def has_images(directory: Path) -> bool:
    return any(is_image_file(p) for p in directory.rglob("*"))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a dataset directory and display "
            "pie/bar charts for each plant "
            "type."
        )
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help=(
            "Input directory. Can be a dataset root, "
            "a single plant directory, or a flat set "
            "of class directories."
        ),
    )
    return parser.parse_args(argv)


def get_subdirs(path: Path) -> list[Path]:
    return sorted([p for p in path.iterdir() if p.is_dir()])


def infer_plant_from_class_name(class_name: str, fallback: str) -> str:
    if "_" not in class_name:
        return fallback
    plant = class_name.split("_", 1)[0].strip()
    return plant or fallback


def simplify_class_label(class_name: str, plant_name: str) -> str:
    prefix = f"{plant_name}_"
    if class_name.startswith(prefix) and len(class_name) > len(prefix):
        return class_name[len(prefix):]
    return class_name


def input_looks_like_single_plant(
    input_dir: Path,
    class_dirs: list[Path],
) -> bool:
    prefix = f"{input_dir.name}_"
    return bool(class_dirs) and all(
        class_dir.name.startswith(prefix)
        for class_dir in class_dirs
    )


def build_charts_for_plant(
    plant_name: str,
    class_dirs: list[Path],
) -> None:
    import matplotlib.pyplot as plt

    labels: list[str] = []
    counts: list[int] = []
    for class_dir in sorted(class_dirs):
        labels.append(simplify_class_label(class_dir.name, plant_name))
        counts.append(count_images(class_dir))

    total = sum(counts)
    if total == 0:
        print(f"[warn] No images found under {plant_name}", file=sys.stderr)
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    pie_ax, bar_ax = axes

    # Pie chart on the left.
    pie_ax.pie(
        counts,
        labels=labels,
        autopct=lambda pct: f"{pct:.1f}%",
        startangle=90,
    )
    pie_ax.set_title(f"{plant_name} - Class Distribution (Pie)")

    # Bar chart on the right.
    bar_ax.bar(labels, counts)
    bar_ax.set_title(f"{plant_name} - Class Distribution (Bar)")
    bar_ax.set_xlabel("Class")
    bar_ax.set_ylabel("Image Count")
    bar_ax.set_ylim(0, max(counts) * 1.1)
    bar_ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()

    if plt.get_backend().lower() == "agg":
        print(
            "[warn] Matplotlib is using a non-interactive backend; "
            "cannot display charts.",
            file=sys.stderr,
        )
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)
        print(f"[ok] {plant_name}: displayed pie and bar charts")


def group_classes_by_plant(
    input_dir: Path,
    top_dirs: list[Path],
) -> dict[str, list[Path]]:
    plant_to_classes: dict[str, list[Path]] = {}
    flat_class_dirs: list[Path] = []

    for top_dir in top_dirs:
        nested = get_subdirs(top_dir)
        if nested:
            plant_to_classes.setdefault(top_dir.name, []).extend(nested)
            continue

        if has_images(top_dir):
            flat_class_dirs.append(top_dir)
            continue

        print(
            f"[warn] Ignoring empty directory: {top_dir}",
            file=sys.stderr,
        )

    if not flat_class_dirs:
        return plant_to_classes

    if (
        not plant_to_classes
        and input_looks_like_single_plant(input_dir, flat_class_dirs)
    ):
        plant_to_classes[input_dir.name] = list(flat_class_dirs)
        return plant_to_classes

    for class_dir in flat_class_dirs:
        plant_name = infer_plant_from_class_name(
            class_dir.name,
            input_dir.name,
        )
        plant_to_classes.setdefault(plant_name, []).append(class_dir)

    return plant_to_classes


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    dataset_dir: Path = args.dataset_dir

    if not dataset_dir.exists() or not dataset_dir.is_dir():
        print(
            f"[error] Dataset directory not found: {dataset_dir}",
            file=sys.stderr,
        )
        return 1

    top_dirs = get_subdirs(dataset_dir)
    if not top_dirs:
        print(
            f"[error] No subdirectories found in {dataset_dir}",
            file=sys.stderr,
        )
        return 1

    plant_to_classes = group_classes_by_plant(dataset_dir, top_dirs)
    if not plant_to_classes:
        print(
            f"[error] No class directories with images found in {dataset_dir}",
            file=sys.stderr,
        )
        return 1

    for plant_name in sorted(plant_to_classes):
        class_dirs = plant_to_classes[plant_name]
        build_charts_for_plant(plant_name, class_dirs)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

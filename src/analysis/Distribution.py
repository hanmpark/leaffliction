#!/usr/bin/env python3
"""Generate per-plant class distribution charts from image folders.

Usage:
  python src/analysis/Distribution.py <input_dir> [--out charts] [--no-show]

Supported layouts:
- Two levels: input_dir/<plant>/<class>/*.jpg
- One level: input_dir/<class>/*.jpg

For one-level layouts, plants are inferred from class names (prefix before "_").
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
            "Analyze a dataset directory and produce "
            "pie/bar charts for each plant "
            "type."
        )
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help=(
            "Input directory. Can be a dataset root, a single plant directory, "
            "or a flat set of class directories."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("charts"),
        help="Output directory for generated charts (default: charts).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display charts (save only).",
    )
    return parser.parse_args(argv)


def get_subdirs(path: Path) -> list[Path]:
    return sorted([p for p in path.iterdir() if p.is_dir()])


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)


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


def input_looks_like_single_plant(input_dir: Path, class_dirs: list[Path]) -> bool:
    prefix = f"{input_dir.name}_"
    return bool(class_dirs) and all(class_dir.name.startswith(prefix) for class_dir in class_dirs)


def build_charts_for_plant(
    plant_name: str,
    class_dirs: list[Path],
    out_dir: Path,
    no_show: bool,
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

    out_dir.mkdir(parents=True, exist_ok=True)
    base = safe_name(plant_name)

    figures = []

    # Pie chart
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        counts,
        labels=labels,
        autopct=lambda pct: f"{pct:.1f}%",
        startangle=90,
    )
    ax.set_title(f"{plant_name} - Class Distribution (Pie)")
    pie_path = out_dir / f"{base}_pie.png"
    fig.tight_layout()
    fig.savefig(pie_path, dpi=150)
    figures.append(fig)

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, counts)
    ax.set_title(f"{plant_name} - Class Distribution (Bar)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Image Count")
    ax.set_ylim(0, max(counts) * 1.1)
    fig.tight_layout()
    bar_path = out_dir / f"{base}_bar.png"
    fig.savefig(bar_path, dpi=150)
    figures.append(fig)

    if not no_show and plt.get_backend().lower() != "agg":
        plt.show()

    for fig in figures:
        plt.close(fig)

    print(f"[ok] {plant_name}: saved {pie_path} and {bar_path}")


def group_classes_by_plant(input_dir: Path, top_dirs: list[Path]) -> dict[str, list[Path]]:
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

    if not plant_to_classes and input_looks_like_single_plant(input_dir, flat_class_dirs):
        plant_to_classes[input_dir.name] = list(flat_class_dirs)
        return plant_to_classes

    for class_dir in flat_class_dirs:
        plant_name = infer_plant_from_class_name(class_dir.name, input_dir.name)
        plant_to_classes.setdefault(plant_name, []).append(class_dir)

    return plant_to_classes


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    dataset_dir: Path = args.dataset_dir

    import matplotlib

    matplotlib.use("Agg")

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
        build_charts_for_plant(plant_name, class_dirs, args.out, args.no_show)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

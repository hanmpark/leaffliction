#!/usr/bin/env python3
"""Dataset distribution analysis.

Usage:
  python src/analysis/distribution.py /path/to/dataset [--out charts]
  [--no-show]

The dataset directory is expected to contain plant-type directories. Each plant
directory contains class subdirectories (e.g., diseases) with images inside.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def count_images(directory: Path) -> int:
    return sum(1 for p in directory.rglob("*") if is_image_file(p))


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
        help="Root dataset directory containing plant-type subdirectories.",
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


def build_charts(plant_dir: Path, out_dir: Path, no_show: bool) -> None:
    import matplotlib.pyplot as plt

    class_dirs = get_subdirs(plant_dir)
    if not class_dirs:
        print(
            f"[warn] No class subdirectories in {plant_dir}",
            file=sys.stderr,
        )
        return

    labels: list[str] = []
    counts: list[int] = []
    for class_dir in class_dirs:
        labels.append(class_dir.name)
        counts.append(count_images(class_dir))

    total = sum(counts)
    if total == 0:
        print(f"[warn] No images found under {plant_dir}", file=sys.stderr)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    plant_name = plant_dir.name
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

    if not no_show:
        plt.show()

    for fig in figures:
        plt.close(fig)

    print(f"[ok] {plant_name}: saved {pie_path} and {bar_path}")


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    dataset_dir: Path = args.dataset_dir
    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")

    if not dataset_dir.exists() or not dataset_dir.is_dir():
        print(
            f"[error] Dataset directory not found: {dataset_dir}",
            file=sys.stderr,
        )
        return 1

    plant_dirs = get_subdirs(dataset_dir)
    if not plant_dirs:
        print(
            f"[error] No plant subdirectories found in {dataset_dir}",
            file=sys.stderr,
        )
        return 1

    for plant_dir in plant_dirs:
        build_charts(plant_dir, args.out, args.no_show)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

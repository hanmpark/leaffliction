#!/usr/bin/env python3
import argparse
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
AUG_OUTPUT_SUFFIXES = (
    "Flip",
    "Rotate",
    "Skew",
    "Shear",
    "Crop",
    "Distortion",
)
LEGACY_AUG_OUTPUT_SUFFIXES = (
    "Blur",
    "Contrast",
    "Scale",
    "Illumination",
    "Projective",
)
AUG_SUFFIXES = tuple(
    f"_{s}" for s in AUG_OUTPUT_SUFFIXES + LEGACY_AUG_OUTPUT_SUFFIXES
)


def is_augmented(path: Path) -> bool:
    """Return True if file stem ends with a known augmentation suffix."""
    stem = path.stem
    return any(stem.endswith(suffix) for suffix in AUG_SUFFIXES)


def has_images(directory: Path) -> bool:
    """Return True if the directory has at least one supported image file."""
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            return True
    return False


def list_class_dirs(root: Path) -> List[Path]:
    """Support <root>/<plant>/<class>, <root>/<class>, and class-only root."""
    class_dirs: List[Path] = []
    top_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not top_dirs and has_images(root):
        return [root]

    for top_dir in top_dirs:
        nested = sorted([p for p in top_dir.iterdir() if p.is_dir()])
        if nested:
            class_dirs.extend(nested)
            continue
        if has_images(top_dir):
            class_dirs.append(top_dir)
    return sorted(class_dirs)


def count_images(class_dir: Path) -> int:
    """Count image files (jpg/jpeg/png, any case) in a class directory."""
    count = 0
    for p in class_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            count += 1
    return count


def list_original_images(class_dir: Path) -> List[Path]:
    """List original (non-augmented) images in a class directory."""
    originals: List[Path] = []
    for p in class_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        if is_augmented(p):
            continue
        originals.append(p)
    return originals


def expected_output_paths(original: Path) -> List[Path]:
    """Return the augmentation output paths for a given original image."""
    return [
        original.with_name(f"{original.stem}_{suffix}{original.suffix}")
        for suffix in AUG_OUTPUT_SUFFIXES
    ]


def has_missing_aug_outputs(original: Path) -> bool:
    """Return True if any expected augmentation output is missing."""
    return any(not p.exists() for p in expected_output_paths(original))


def run_aug(aug_script: Path, image_path: Path) -> None:
    """Run augmentation script on the given image."""
    if aug_script.suffix.lower() == ".py":
        cmd = [sys.executable, str(aug_script), str(image_path), "--quiet"]
    else:
        cmd = [str(aug_script), str(image_path)]
    subprocess.run(cmd, check=True)


def format_progress_line(
    label: str,
    current: int,
    target: int,
    label_width: int,
    width: int = 28,
) -> str:
    """Format one class progress bar line."""
    ratio = (current / target) if target > 0 else 0.0
    bounded_ratio = min(1.0, max(0.0, ratio))
    filled = int(round(width * bounded_ratio))
    bar = "#" * filled + "-" * (width - filled)
    padded_label = label.ljust(label_width)
    return (
        f"{padded_label}: [{bar}] {current}/{target} "
        f"({ratio * 100.0:6.2f}%)"
    )


def print_progress_bars(
    class_dirs: List[Path],
    out: Path,
    current_counts: Dict[Path, int],
    target: int,
    rendered_lines: int,
) -> int:
    """Render or refresh all class progress bars in-place when possible."""
    ordered_dirs = sorted(
        class_dirs,
        key=lambda p: (
            -(
                (current_counts[p] / target)
                if target > 0
                else 0.0
            ),
            str(p.relative_to(out)).replace("\\", "/"),
        ),
    )
    labels = [
        str(class_dir.relative_to(out)).replace("\\", "/")
        for class_dir in ordered_dirs
    ]
    label_width = max((len(label) for label in labels), default=0)
    lines = [
        format_progress_line(
            label,
            current_counts[class_dir],
            target,
            label_width=label_width,
        )
        for class_dir, label in zip(ordered_dirs, labels)
    ]

    if sys.stdout.isatty():
        if rendered_lines:
            print(f"\033[{rendered_lines}F", end="")
        for line in lines:
            print(f"{line}\033[K")
    else:
        for line in lines:
            print(line)
    return len(lines)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Balance image dataset by augmentation."
    )
    parser.add_argument(
        "--src",
        required=True,
        help=(
            "Source dataset directory. Supports <plant>/<class>, <class>, "
            "or a single class directory."
        ),
    )
    parser.add_argument(
        "--out",
        default="augmented_directory",
        help="Output dataset directory (default: augmented_directory).",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        dest="in_place",
        help="Balance the source dataset in-place (no output copy).",
    )
    parser.add_argument(
        "--augmentation-script",
        required=True,
        help="Path to augmentation script.",
    )
    args = parser.parse_args(argv)

    src = Path(args.src)
    out = Path(args.out)
    aug_script = Path(args.augmentation_script)

    if not src.exists() or not src.is_dir():
        print(
            f"Error: --src '{src}' is missing or not a directory.",
            file=sys.stderr,
        )
        return 1
    if not aug_script.exists() or not aug_script.is_file():
        print(
            (
                "Error: --augmentation-script "
                f"'{aug_script}' is missing or not a file."
            ),
            file=sys.stderr,
        )
        return 1

    if args.in_place:
        out = src
    else:
        try:
            if out.exists():
                shutil.rmtree(out)
            shutil.copytree(src, out)
        except (OSError, shutil.Error) as exc:
            print(
                f"Error: failed to prepare output directory '{out}': {exc}",
                file=sys.stderr,
            )
            return 1

    try:
        class_dirs = list_class_dirs(out)
    except OSError as exc:
        print(
            f"Error: failed to list class directories: {exc}",
            file=sys.stderr,
        )
        return 1

    if not class_dirs:
        print(
            "Error: no class directories found under --src.",
            file=sys.stderr,
        )
        return 1

    before_counts: Dict[Path, int] = {}
    for class_dir in class_dirs:
        try:
            before_counts[class_dir] = count_images(class_dir)
        except OSError as exc:
            print(
                f"Error: unreadable class directory '{class_dir}': {exc}",
                file=sys.stderr,
            )
            return 1

    target = max(before_counts.values())
    class_dirs = sorted(
        class_dirs,
        key=lambda p: (
            -(
                (before_counts[p] / target)
                if target > 0
                else 0.0
            ),
            str(p.relative_to(out)).replace("\\", "/"),
        ),
    )
    rng = random.Random(42)
    current_counts: Dict[Path, int] = dict(before_counts)
    rendered_lines = print_progress_bars(
        class_dirs,
        out,
        current_counts,
        target,
        rendered_lines=0,
    )

    for class_dir in class_dirs:
        rel = class_dir.relative_to(out)
        current = before_counts[class_dir]
        if current >= target:
            continue

        originals = list_original_images(class_dir)
        eligible = [p for p in originals if has_missing_aug_outputs(p)]
        if not originals:
            print(
                f"Error: no original images to augment in '{rel}'.",
                file=sys.stderr,
            )
            return 1
        if not eligible:
            print(
                f"Error: no eligible originals left to augment in '{rel}'.",
                file=sys.stderr,
            )
            return 1

        while current < target:
            if not eligible:
                print(
                    (
                        f"Error: ran out of eligible originals in '{rel}' "
                        "before reaching target "
                        f"(current {current}, target {target})."
                    ),
                    file=sys.stderr,
                )
                return 1
            chosen = rng.choice(eligible)
            before = count_images(class_dir)
            try:
                run_aug(aug_script, chosen)
            except subprocess.CalledProcessError as exc:
                print(
                    (
                        f"Error: augmentation failed for class '{rel}' "
                        f"with image '{chosen}': {exc}"
                    ),
                    file=sys.stderr,
                )
                return 1
            after = count_images(class_dir)
            if after <= before:
                print(
                    (
                        "Error: augmentation produced no new images for class "
                        f"'{rel}' using '{chosen}'."
                    ),
                    file=sys.stderr,
                )
                return 1
            if not has_missing_aug_outputs(chosen):
                if chosen in eligible:
                    eligible.remove(chosen)
            current = after
            current_counts[class_dir] = current
            rendered_lines = print_progress_bars(
                class_dirs,
                out,
                current_counts,
                target,
                rendered_lines=rendered_lines,
            )

    after_counts: Dict[Path, int] = {}
    for class_dir in class_dirs:
        after_counts[class_dir] = count_images(class_dir)

    print(f"Target count: {target}")
    print("Per-class counts (before -> after):")
    for class_dir in class_dirs:
        rel = class_dir.relative_to(out)
        print(
            f"{rel}: {before_counts[class_dir]} -> {after_counts[class_dir]}"
        )
    if args.in_place:
        print(f"Output dataset: {out.resolve()} (in-place)")
    else:
        print(f"Output dataset: {out.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

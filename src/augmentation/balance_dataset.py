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
    "Rotate",
    "Blur",
    "Contrast",
    "Scale",
    "Illumination",
    "Projective",
)
AUG_SUFFIXES = tuple(f"_{s}" for s in AUG_OUTPUT_SUFFIXES) + (
    "_Flip",
    "_Skew",
    "_Shear",
    "_Crop",
    "_Distortion",
)


def is_augmented(path: Path) -> bool:
    """Return True if file stem ends with a known augmentation suffix."""
    stem = path.stem
    return any(stem.endswith(suffix) for suffix in AUG_SUFFIXES)


def list_class_dirs(root: Path) -> List[Path]:
    """Return all class directories under <root>/<Plant>/<Class>."""
    class_dirs: List[Path] = []
    for plant_dir in root.iterdir():
        if not plant_dir.is_dir():
            continue
        for class_dir in plant_dir.iterdir():
            if class_dir.is_dir():
                class_dirs.append(class_dir)
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
        cmd = [sys.executable, str(aug_script), str(image_path)]
    else:
        cmd = [str(aug_script), str(image_path)]
    subprocess.run(cmd, check=True)


def _parse_target(value: str, max_count: int) -> int:
    if value == "max":
        return max_count
    try:
        target = int(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid --target '{value}'. Use 'max' or an integer."
        ) from exc
    if target < 0:
        raise ValueError("--target must be non-negative.")
    return target


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Balance image dataset by augmentation."
    )
    parser.add_argument("--src", required=True, help="Source dataset root.")
    parser.add_argument(
        "--out",
        default="augmented_directory",
        help="Output dataset directory (default: augmented_directory).",
    )
    parser.add_argument(
        "--augmentation-script",
        required=True,
        help="Path to augmentation script.",
    )
    parser.add_argument(
        "--target",
        default="max",
        help="Target count per class: 'max' or an integer.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
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
            "Error: no class directories found under the dataset root.",
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

    max_count = max(before_counts.values())
    try:
        target = _parse_target(args.target, max_count)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)

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
    print(f"Output dataset: {out.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

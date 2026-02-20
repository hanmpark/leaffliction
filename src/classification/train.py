#!/usr/bin/env python3
"""Train a lightweight plant disease classifier with handcrafted features."""

from __future__ import annotations

import argparse
import pickle
import random
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from .features import (
        IMAGE_EXTENSIONS,
        build_hog_descriptor,
        default_feature_config,
        extract_feature_vector_from_path,
    )
    from .console_output import (
        print_dataset_summary,
        print_progress,
        print_train_intro,
        print_train_outro,
        run_with_spinner,
    )
except ImportError:  # pragma: no cover
    from features import (
        IMAGE_EXTENSIONS,
        build_hog_descriptor,
        default_feature_config,
        extract_feature_vector_from_path,
    )
    from console_output import (
        print_dataset_summary,
        print_progress,
        print_train_intro,
        print_train_outro,
        run_with_spinner,
    )


DEFAULT_OUT_DIR = Path("./artifacts")
DEFAULT_IMG_SIZE = 128
DEFAULT_SEED = 42
DEFAULT_VAL_SPLIT = 0.2
BALANCE_TOLERANCE = 6
TRAINING_DIRNAME = "training_data"
VALIDATION_DIRNAME = "validation_data"


def parse_bool_flag(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        "Expected true/false."
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an image classifier using manual features "
            "(color histogram + HOG + texture) with SVC."
        )
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help=(
            "Dataset directory; each image parent folder is treated "
            "as its class label."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional working dataset path. When set, source dataset is "
            "copied there and training uses that copy."
        ),
    )
    parser.add_argument(
        "--generate-zip",
        action="store_true",
        help=(
            "Generate artifacts/learnings.zip containing model artifacts "
            "and input images."
        ),
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=None,
        help=(
            "Validation split ratio (0 < value < 1). "
            "If set, no validation-split prompt is shown."
        ),
    )
    parser.add_argument(
        "--auto-balance",
        type=parse_bool_flag,
        default=None,
        help=(
            "Auto-balance unbalanced datasets without prompt "
            "(true/false)."
        ),
    )
    return parser.parse_args(argv)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def prepare_working_dataset(
    source_dir: Path,
    output_dir: Path | None,
) -> Path:
    """Return training dataset root (copied output dir if requested)."""
    if output_dir is None:
        return source_dir

    src_resolved = source_dir.resolve()
    out_resolved = output_dir.resolve()
    if src_resolved == out_resolved:
        raise SystemExit(
            "Error: --output must be different from input dataset directory."
        )

    try:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(source_dir, output_dir)
    except (OSError, shutil.Error) as exc:
        raise SystemExit(
            f"Error: failed to prepare output dataset '{output_dir}': {exc}"
        ) from exc

    print(f"[info] Copied input dataset to: {output_dir}")
    return output_dir


def collect_dataset_samples(
    dataset_dir: Path,
) -> tuple[list[tuple[Path, int]], list[str], dict[str, int]]:
    # Collect supported image files under the dataset directory.
    image_paths = sorted(
        path
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    class_keys = sorted(
        str(path.parent.relative_to(dataset_dir)).replace("\\", "/")
        for path in image_paths
    )
    classes = sorted(set(class_keys))

    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    samples = [
        (
            path,
            class_to_idx[
                str(path.parent.relative_to(dataset_dir)).replace("\\", "/")
            ],
        )
        for path in image_paths
    ]

    return samples, classes, class_to_idx


def list_split_source_images(dataset_dir: Path) -> list[Path]:
    """List source images while excluding training/validation split folders."""
    images: list[Path] = []
    for path in sorted(dataset_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rel = path.relative_to(dataset_dir)
        if (
            rel.parts
            and rel.parts[0] in {TRAINING_DIRNAME, VALIDATION_DIRNAME}
        ):
            continue
        images.append(path)
    return images


def extract_feature_matrix(
    samples: list[tuple[Path, int]],
    feature_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    hog_descriptor = build_hog_descriptor(feature_config)
    feature_rows: list[np.ndarray] = []
    labels: list[int] = []
    total = len(samples)
    step = max(1, total // 100) if total else 1

    for index, (image_path, label) in enumerate(samples, start=1):
        feature_vector, _ = extract_feature_vector_from_path(
            image_path,
            feature_config,
            hog_descriptor=hog_descriptor,
        )

        feature_rows.append(feature_vector)
        labels.append(label)
        if index % step == 0 or index == total:
            print_progress("Feature extraction", index, total)

    X = np.vstack(feature_rows).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    return X, y


def build_svc_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(kernel="rbf"),
            ),
        ]
    )


def train_svc_model(
    model: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Pipeline:
    run_with_spinner("Training svm", model.fit, X_train, y_train)
    return model


def collect_class_counts(dataset_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for image_path in list_split_source_images(dataset_dir):
        class_key = image_path.parent.relative_to(dataset_dir).as_posix()
        counts[class_key] = counts.get(class_key, 0) + 1
    return counts


def is_balanced(counts: dict[str, int], tolerance: int) -> bool:
    if not counts:
        return True
    values = counts.values()
    return (max(values) - min(values)) <= tolerance


def print_balance_report(counts: dict[str, int], tolerance: int) -> None:
    if not counts:
        print("[warn] No images found during balance analysis.")
        return
    min_count = min(counts.values())
    max_count = max(counts.values())
    print("[info] Input dataset balance analysis:")
    print(f"  classes: {len(counts)}")
    print(f"  min images/class: {min_count}")
    print(f"  max images/class: {max_count}")
    print(f"  allowed gap (+/-): {tolerance}")
    print("  per-class counts:")
    for class_name in sorted(counts):
        print(f"    {class_name}: {counts[class_name]}")


def choose_unbalanced_action() -> str:
    prompt = (
        "\nDataset is not balanced.\n"
        "Choose an action:\n"
        "  1. Balance dataset in-place, then train\n"
        "  2. Train directly on current dataset\n"
        "Enter 1 or 2: "
    )
    if not sys.stdin.isatty():
        print(
            "[warn] Non-interactive mode detected. "
            "Continuing without balancing."
        )
        return "train"

    while True:
        try:
            choice = input(prompt).strip()
        except EOFError:
            print(
                "[warn] No interactive input available. "
                "Continuing without balancing."
            )
            return "train"
        if choice == "1":
            return "balance"
        if choice == "2":
            return "train"
        print("Invalid choice. Enter 1 or 2.")


def get_existing_split_dirs(dataset_dir: Path) -> tuple[Path, Path] | None:
    train_dir = dataset_dir / TRAINING_DIRNAME
    val_dir = dataset_dir / VALIDATION_DIRNAME
    if train_dir.is_dir() and val_dir.is_dir():
        return train_dir, val_dir
    return None


def choose_validation_split(default_split: float) -> float:
    if not sys.stdin.isatty():
        print(
            "[warn] Non-interactive mode detected. "
            f"Using default validation split {default_split:.2f}."
        )
        return default_split

    prompt = (
        "\nSelect validation split ratio (0 < value < 1).\n"
        f"Press Enter for default ({default_split:.2f}): "
    )
    while True:
        try:
            raw = input(prompt).strip()
        except EOFError:
            print(
                "[warn] No interactive input available. "
                f"Using default validation split {default_split:.2f}."
            )
            return default_split
        if raw == "":
            return default_split
        try:
            value = float(raw)
        except ValueError:
            print("Invalid value. Enter a float like 0.2.")
            continue
        if 0.0 < value < 1.0:
            return value
        print("Invalid value. Expected 0 < value < 1.")


def create_split_dirs(
    dataset_dir: Path,
    validation_split: float,
    seed: int,
) -> tuple[Path, Path]:
    source_images = list_split_source_images(dataset_dir)
    if not source_images:
        raise SystemExit("Error: no source images found to split.")
    if len(source_images) < 2:
        raise SystemExit("Error: need at least 2 images to create a split.")

    labels = [
        str(path.parent.relative_to(dataset_dir)).replace("\\", "/")
        for path in source_images
    ]
    indices = np.arange(len(source_images))
    try:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=validation_split,
            random_state=seed,
            shuffle=True,
            stratify=labels,
        )
    except ValueError:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=validation_split,
            random_state=seed,
            shuffle=True,
        )

    train_root = dataset_dir / TRAINING_DIRNAME
    val_root = dataset_dir / VALIDATION_DIRNAME
    for split_root in (train_root, val_root):
        if split_root.exists():
            shutil.rmtree(split_root)
        split_root.mkdir(parents=True, exist_ok=True)

    def _move(indices_to_move: np.ndarray, destination_root: Path) -> None:
        for idx in sorted(int(i) for i in indices_to_move):
            src = source_images[idx]
            rel = src.relative_to(dataset_dir)
            dst = destination_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))

    _move(train_idx, train_root)
    _move(val_idx, val_root)

    # Remove now-empty directories left after moving files.
    for directory in sorted(dataset_dir.rglob("*"), reverse=True):
        if not directory.is_dir():
            continue
        rel = directory.relative_to(dataset_dir)
        if (
            rel.parts
            and rel.parts[0] in {TRAINING_DIRNAME, VALIDATION_DIRNAME}
        ):
            continue
        try:
            directory.rmdir()
        except OSError:
            pass
    return train_root, val_root


def get_augmentation_scripts() -> tuple[Path, Path]:
    src_dir = Path(__file__).resolve().parents[1]
    balance_script = src_dir / "augmentation" / "balance_dataset.py"
    augment_script = src_dir / "augmentation" / "Augmentation.py"
    return balance_script, augment_script


def balance_dataset_in_place(dataset_dir: Path) -> None:
    balance_script, augment_script = get_augmentation_scripts()
    if not balance_script.exists():
        raise SystemExit(f"Error: augmentation tool missing: {balance_script}")
    if not augment_script.exists():
        raise SystemExit(f"Error: augmentation tool missing: {augment_script}")

    cmd = [
        sys.executable,
        str(balance_script),
        "--src",
        str(dataset_dir),
        "--augmentation-script",
        str(augment_script),
        "--in-place",
        "--augmentation-no-show",
    ]
    print("[info] Balancing dataset in-place...")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            "Error: dataset augmentation failed with exit code "
            f"{exc.returncode}."
        ) from exc


def save_artifacts(
    model_dir: Path,
    estimator: Pipeline,
    class_to_idx: dict[str, int],
    classes: list[str],
    feature_config: dict[str, Any],
    training_summary: dict[str, Any],
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)

    idx_to_class = {str(i): class_name for i, class_name in enumerate(classes)}
    model_payload = {
        "estimator": estimator,
        "feature_config": feature_config,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "training_summary": training_summary,
    }
    with (model_dir / "model.pkl").open("wb") as f:
        pickle.dump(model_payload, f)


def make_zip(out_dir: Path, dataset_dir: Path) -> Path:
    zip_path = out_dir / "learnings.zip"
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zf:
        for top in ("model",):
            top_dir = out_dir / top
            if not top_dir.exists():
                continue
            for path in sorted(top_dir.rglob("*")):
                if path.is_file():
                    arcname = path.relative_to(out_dir)
                    zf.write(path, arcname=str(arcname))

        for path in sorted(dataset_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            arcname = Path("input_images") / path.relative_to(dataset_dir)
            zf.write(path, arcname=str(arcname))
    return zip_path


def prepare_output_dirs(out_dir: Path) -> Path:
    model_dir = out_dir / "model"
    out_dir.mkdir(parents=True, exist_ok=True)

    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    zip_path = out_dir / "learnings.zip"
    if zip_path.exists():
        zip_path.unlink()

    return model_dir


def choose_existing_model_action() -> str:
    prompt = (
        "\nA trained model already exists at artifacts/model/model.pkl.\n"
        "Choose an action:\n"
        "  1. Erase existing model and retrain\n"
        "  2. Use existing model (skip retraining)\n"
        "Enter 1 or 2: "
    )
    if not sys.stdin.isatty():
        print("[warn] Non-interactive mode detected. Using existing model.")
        return "use"

    while True:
        try:
            choice = input(prompt).strip()
        except EOFError:
            print(
                "[warn] No interactive input available. "
                "Using existing model."
            )
            return "use"
        if choice == "1":
            return "retrain"
        if choice == "2":
            return "use"
        print("Invalid choice. Enter 1 or 2.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.dataset_dir.exists():
        raise SystemExit(
            f"Error: dataset directory not found: {args.dataset_dir}"
        )
    if not args.dataset_dir.is_dir():
        raise SystemExit(
            f"Error: dataset path is not a directory: {args.dataset_dir}"
        )
    if args.val_split is not None and not (0.0 < args.val_split < 1.0):
        raise SystemExit("Error: --val-split must satisfy 0 < value < 1.")

    out_dir = DEFAULT_OUT_DIR
    img_size = DEFAULT_IMG_SIZE
    seed = DEFAULT_SEED
    val_split_used: float | None = None
    existing_model_path = out_dir / "model" / "model.pkl"

    if existing_model_path.exists():
        action = choose_existing_model_action()
        if action == "use":
            if args.output is not None:
                print(
                    "[info] --output is ignored when using existing model."
                )
            zip_path: Path | None = None
            if args.generate_zip:
                zip_path = make_zip(out_dir, args.dataset_dir)
            print_train_outro(
                model_name="existing_model",
                best_val_accuracy=None,
                confusion_matrix=None,
                model_dir=out_dir / "model",
                zip_path=zip_path,
            )
            if zip_path is not None:
                print(
                    "Signature command:\n"
                    f"  sha1sum {shlex.quote(str(zip_path))} > signature.txt"
                )
            return 0

    working_dataset_dir = prepare_working_dataset(
        args.dataset_dir,
        args.output,
    )

    # Print a run header and keep training deterministic.
    print_train_intro(
        dataset_dir=working_dataset_dir,
        out_dir=out_dir,
        img_size=img_size,
        seed=seed,
        val_split=DEFAULT_VAL_SPLIT,
    )

    seed_everything(seed)

    feature_config = default_feature_config(img_size)

    split_dirs = get_existing_split_dirs(working_dataset_dir)
    if split_dirs is None:
        class_counts = collect_class_counts(working_dataset_dir)
        print_balance_report(class_counts, BALANCE_TOLERANCE)
        if not is_balanced(class_counts, BALANCE_TOLERANCE):
            if args.auto_balance is True:
                action = "balance"
                print(
                    "[info] Auto-balance enabled: "
                    "balancing dataset in-place."
                )
            elif args.auto_balance is False:
                action = "train"
                print(
                    "[info] Auto-balance disabled: "
                    "training without balancing."
                )
            else:
                action = choose_unbalanced_action()
            if action == "balance":
                balance_dataset_in_place(working_dataset_dir)
                class_counts = collect_class_counts(working_dataset_dir)
                print_balance_report(class_counts, BALANCE_TOLERANCE)
        else:
            print("[info] Input dataset is already balanced within tolerance.")

        print("[warn] input data doesn't have any training/validation split")
        if args.val_split is not None:
            val_split_used = args.val_split
            print(
                "[info] Using validation split from --val-split: "
                f"{val_split_used:.2f}"
            )
        else:
            val_split_used = choose_validation_split(DEFAULT_VAL_SPLIT)
        train_dir, val_dir = create_split_dirs(
            dataset_dir=working_dataset_dir,
            validation_split=val_split_used,
            seed=seed,
        )
        print(f"[info] Created training split: {train_dir}")
        print(f"[info] Created validation split: {val_dir}")
        print(f"[info] Validation split ratio: {val_split_used:.2f}")
    else:
        train_dir, val_dir = split_dirs
        print(f"[info] Using existing training split: {train_dir}")
        print(f"[info] Using existing validation split: {val_dir}")

    model_dir = prepare_output_dirs(out_dir)
    samples, classes, class_to_idx = collect_dataset_samples(train_dir)
    if not samples:
        raise SystemExit(
            "Error: no supported images found in training split."
        )
    val_samples, _, _ = collect_dataset_samples(val_dir)
    if val_split_used is None:
        total_split = len(samples) + len(val_samples)
        if total_split > 0:
            val_split_used = len(val_samples) / float(total_split)
    if val_split_used is not None:
        print(f"[info] Effective validation split ratio: {val_split_used:.2f}")
    print_dataset_summary(
        num_classes=len(classes),
        total_samples=len(samples) + len(val_samples),
        train_samples=len(samples),
        val_samples=len(val_samples),
    )
    print(f"Training dataset: {train_dir}")
    print(f"Validation dataset: {val_dir}")

    # Compute handcrafted features for the full training dataset.
    X_all, y_all = extract_feature_matrix(samples, feature_config)

    feature_dim = int(X_all.shape[1])
    print(f"Feature vector dimension: {feature_dim}")

    X_train = X_all
    y_train = y_all

    model = build_svc_model()

    # Train only. Evaluation is handled separately on split folders.
    best_model = train_svc_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
    )

    best_name = "svm"

    # Save model metadata needed for reproducible prediction.
    timestamp = datetime.now(timezone.utc).isoformat()
    training_summary: dict[str, Any] = {
        "algorithm": best_name,
        "img_size": img_size,
        "seed": seed,
        "val_split": val_split_used,
        "dataset_dir": str(train_dir.resolve()),
        "source_dataset_dir": str(args.dataset_dir.resolve()),
        "timestamp_utc": timestamp,
        "best_val_accuracy": None,
        "best_train_accuracy": None,
        "feature_dimension": feature_dim,
        "train_sample_count": int(X_train.shape[0]),
        "val_sample_count": int(len(val_samples)),
        "train_split_dir": str(train_dir.resolve()),
        "validation_split_dir": str(val_dir.resolve()),
        "best_confusion_matrix": [],
        "evaluation_skipped_in_training": True,
        "hyperparameters": {
            "kernel": "rbf",
        },
    }

    save_artifacts(
        model_dir=model_dir,
        estimator=best_model,
        class_to_idx=class_to_idx,
        classes=classes,
        feature_config=feature_config,
        training_summary=training_summary,
    )

    zip_path: Path | None = None
    if args.generate_zip:
        zip_path = make_zip(out_dir, working_dataset_dir)

    print_train_outro(
        model_name=best_name,
        best_val_accuracy=None,
        confusion_matrix=None,
        model_dir=model_dir,
        zip_path=zip_path,
    )
    if zip_path is not None:
        print(
            "Signature command:\n"
            f"  sha1sum {shlex.quote(str(zip_path))} > signature.txt"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Train a lightweight plant disease classifier with handcrafted features."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
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
        print_train_metrics,
        print_train_outro,
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
        print_train_metrics,
        print_train_outro,
    )


DEFAULT_OUT_DIR = Path("./artifacts")
DEFAULT_IMG_SIZE = 128
DEFAULT_SEED = 42
DEFAULT_VAL_SPLIT = 0.2
DEFAULT_SVM_C = 8.0


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
    return parser.parse_args(argv)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def collect_dataset_samples(
    dataset_dir: Path,
) -> tuple[list[tuple[Path, int]], list[str], dict[str, int]]:
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


def split_indices(
    targets: np.ndarray,
    val_split: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    all_indices = np.arange(len(targets))
    val_count = int(round(len(targets) * val_split))
    val_count = max(1, min(len(targets) - 1, val_count))
    train_idx, val_idx = train_test_split(
        all_indices,
        test_size=val_count,
        random_state=seed,
        shuffle=True,
        stratify=targets,
    )

    return train_idx.tolist(), val_idx.tolist()


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


def build_svc_model(seed: int, svm_c: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(
                    C=svm_c,
                    kernel="rbf",
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
                    random_state=seed,
                ),
            ),
        ]
    )


def train_svc_model(
    model: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[Pipeline, float, float, np.ndarray]:
    num_classes = int(np.unique(y_train).size)
    print("Training svm...")
    start = time.monotonic()
    model.fit(X_train, y_train)
    elapsed = time.monotonic() - start
    print(f"Training svm done in {elapsed:.1f}s")
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    train_acc = float(accuracy_score(y_train, train_pred))
    val_acc = float(accuracy_score(y_val, val_pred))
    print_train_metrics(train_acc, val_acc)

    labels = list(range(num_classes))
    cm = confusion_matrix(y_val, val_pred, labels=labels)
    return model, train_acc, val_acc, cm


def get_augmentation_scripts() -> tuple[Path, Path]:
    src_dir = Path(__file__).resolve().parents[1]
    balance_script = src_dir / "augmentation" / "balance_dataset.py"
    augment_script = src_dir / "augmentation" / "Augmentation.py"
    return balance_script, augment_script


def build_augmented_dataset(dataset_dir: Path, augmented_dir: Path) -> None:
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
        "--out",
        str(augmented_dir),
        "--augmentation-script",
        str(augment_script),
    ]
    print("[info] Building balanced augmented dataset...")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            "Error: dataset augmentation failed with exit code "
            f"{exc.returncode}."
        ) from exc


def save_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    csv_path: Path,
    png_path: Path,
) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred", *class_names])
        for class_name, row in zip(class_names, cm.tolist()):
            writer.writerow([class_name, *row])

    fig_w = max(6, min(16, len(class_names) * 0.8))
    fig_h = fig_w
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Validation Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    max_value = cm.max() if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            color = (
                "white"
                if val > (max_value / 2 if max_value else 0)
                else "black"
            )
            ax.text(
                j,
                i,
                str(val),
                va="center",
                ha="center",
                color=color,
                fontsize=8,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def save_artifacts(
    model_dir: Path,
    estimator: Pipeline,
    class_to_idx: dict[str, int],
    classes: list[str],
    config: dict[str, Any],
    feature_config: dict[str, Any],
    metrics_payload: dict[str, Any],
    best_confusion_matrix: np.ndarray,
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)

    model_payload = {
        "estimator": estimator,
        "feature_config": feature_config,
    }
    with (model_dir / "model.pkl").open("wb") as f:
        pickle.dump(model_payload, f)

    idx_to_class = {str(i): class_name for i, class_name in enumerate(classes)}
    classes_payload = {
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
    }
    (model_dir / "classes.json").write_text(
        json.dumps(classes_payload, indent=2),
        encoding="utf-8",
    )

    (model_dir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    transforms_info = {
        "preprocessing": {
            "resize": {
                "width": feature_config["img_size"],
                "height": feature_config["img_size"],
            },
        },
        "features": {
            "color_histogram": {
                "space": "HSV",
                "bins": feature_config["color_hist_bins"],
            },
            "hog": feature_config["hog"],
            "texture": feature_config["texture"],
        },
    }
    (model_dir / "transforms.json").write_text(
        json.dumps(transforms_info, indent=2),
        encoding="utf-8",
    )

    transforms_doc = (
        "Preprocessing:\n"
        f"- Resize to {feature_config['img_size']}x"
        f"{feature_config['img_size']}\n\n"
        "Feature extraction:\n"
        "- HSV color histogram\n"
        "- HOG (Histogram of Oriented Gradients)\n"
        "- Texture statistics (GLCM + Laplacian variance)\n"
    )
    (model_dir / "transforms.txt").write_text(transforms_doc, encoding="utf-8")

    metrics_payload = {
        **metrics_payload,
        "best_confusion_matrix": best_confusion_matrix.tolist(),
    }
    (model_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )

    save_confusion_matrix(
        best_confusion_matrix,
        classes,
        model_dir / "confusion_matrix.csv",
        model_dir / "confusion_matrix.png",
    )


def make_zip(out_dir: Path) -> Path:
    zip_path = out_dir / "learnings.zip"
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zf:
        for top in ("model", "augmented_directory"):
            top_dir = out_dir / top
            if not top_dir.exists():
                continue
            for path in sorted(top_dir.rglob("*")):
                if path.is_file():
                    arcname = path.relative_to(out_dir)
                    zf.write(path, arcname=str(arcname))
    return zip_path


def prepare_output_dirs(out_dir: Path) -> tuple[Path, Path]:
    model_dir = out_dir / "model"
    augmented_dir = out_dir / "augmented_directory"
    out_dir.mkdir(parents=True, exist_ok=True)

    for directory in (model_dir, augmented_dir):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    zip_path = out_dir / "learnings.zip"
    if zip_path.exists():
        zip_path.unlink()

    return model_dir, augmented_dir


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

    out_dir = DEFAULT_OUT_DIR
    img_size = DEFAULT_IMG_SIZE
    seed = DEFAULT_SEED
    val_split = DEFAULT_VAL_SPLIT
    svm_c = DEFAULT_SVM_C

    print_train_intro(
        dataset_dir=args.dataset_dir,
        out_dir=out_dir,
        img_size=img_size,
        seed=seed,
        val_split=val_split,
        svm_c=svm_c,
    )

    seed_everything(seed)

    feature_config = default_feature_config(img_size)

    model_dir, augmented_dir = prepare_output_dirs(out_dir)
    build_augmented_dataset(args.dataset_dir, augmented_dir)
    samples, classes, class_to_idx = collect_dataset_samples(augmented_dir)

    targets = np.asarray([label for _, label in samples], dtype=np.int64)
    train_indices, val_indices = split_indices(
        targets,
        val_split,
        seed,
    )

    print_dataset_summary(
        num_classes=len(classes),
        total_samples=len(samples),
        train_samples=len(train_indices),
        val_samples=len(val_indices),
    )
    print(f"Training dataset: {augmented_dir}")

    X_all, y_all = extract_feature_matrix(samples, feature_config)

    feature_dim = int(X_all.shape[1])
    print(f"Feature vector dimension: {feature_dim}")

    X_train = X_all[train_indices]
    y_train = y_all[train_indices]
    X_val = X_all[val_indices]
    y_val = y_all[val_indices]

    model = build_svc_model(
        seed=seed,
        svm_c=svm_c,
    )

    (
        best_model,
        best_train_accuracy,
        best_val_accuracy,
        best_cm,
    ) = train_svc_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    best_name = "svm"

    timestamp = datetime.now(timezone.utc).isoformat()
    config = {
        "algorithm": best_name,
        "img_size": img_size,
        "feature_config": feature_config,
        "seed": seed,
        "val_split": val_split,
        "dataset_dir": str(augmented_dir.resolve()),
        "source_dataset_dir": str(args.dataset_dir.resolve()),
        "timestamp_utc": timestamp,
        "best_val_accuracy": float(best_val_accuracy),
        "best_train_accuracy": float(best_train_accuracy),
        "hyperparameters": {
            "svm_c": svm_c,
        },
    }

    metrics_payload: dict[str, Any] = {
        "model_scores": [
            {
                "model": best_name,
                "train_accuracy": float(best_train_accuracy),
                "val_accuracy": float(best_val_accuracy),
            }
        ],
        "feature_dimension": feature_dim,
        "train_sample_count": int(X_train.shape[0]),
        "val_sample_count": int(X_val.shape[0]),
    }

    save_artifacts(
        model_dir=model_dir,
        estimator=best_model,
        class_to_idx=class_to_idx,
        classes=classes,
        config=config,
        feature_config=feature_config,
        metrics_payload=metrics_payload,
        best_confusion_matrix=best_cm,
    )

    zip_path = make_zip(out_dir)

    print_train_outro(
        model_name=best_name,
        best_val_accuracy=best_val_accuracy,
        confusion_matrix=best_cm,
        model_dir=model_dir,
        zip_path=zip_path,
    )
    print(
        "Signature command:\n"
        f"  sha1sum {shlex.quote(str(zip_path))} > signature.txt"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
        read_image_rgb,
        resize_rgb_image,
    )
except ImportError:  # pragma: no cover
    from features import (
        IMAGE_EXTENSIONS,
        build_hog_descriptor,
        default_feature_config,
        extract_feature_vector_from_path,
        read_image_rgb,
        resize_rgb_image,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an image classifier using manual features "
            "(color histogram + HOG + texture) and sklearn models."
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
        "--out",
        type=Path,
        default=Path("./artifacts"),
        help="Output directory (default: ./artifacts).",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=128,
        help="Square resize used for feature extraction (default: 128).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic split/training (default: 42).",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio in (0, 1) (default: 0.2).",
    )
    parser.add_argument(
        "--augment-save-per-image",
        type=int,
        default=1,
        help=(
            "Number of saved augmented previews per "
            "train image (default: 1)."
        ),
    )
    parser.add_argument(
        "--model",
        choices=("auto", "svm", "random_forest", "logistic_regression"),
        default="auto",
        help=(
            "Model to train. 'auto' trains SVM, RandomForest, and "
            "LogisticRegression and picks the best validation score."
        ),
    )
    parser.add_argument(
        "--svm-c",
        type=float,
        default=8.0,
        help="SVM C parameter (default: 8.0).",
    )
    parser.add_argument(
        "--rf-trees",
        type=int,
        default=400,
        help="RandomForest number of trees (default: 400).",
    )
    parser.add_argument(
        "--logreg-c",
        type=float,
        default=3.0,
        help="LogisticRegression inverse regularization C (default: 3.0).",
    )
    return parser.parse_args(argv)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def collect_dataset_samples(
    dataset_dir: Path,
) -> tuple[list[tuple[Path, int]], list[str], dict[str, int]]:
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise ValueError(f"Dataset directory not found: {dataset_dir}")

    image_paths = sorted(
        path
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if len(image_paths) < 2:
        raise ValueError("Dataset must contain at least 2 images.")

    class_keys = sorted(
        str(path.parent.relative_to(dataset_dir)).replace("\\", "/")
        for path in image_paths
    )
    classes = sorted(set(class_keys))
    if len(classes) < 2:
        raise ValueError(
            "Dataset must contain at least 2 class directories with images."
        )

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

    unreadable: list[str] = []
    for image_path, _ in samples:
        if cv2.imread(str(image_path), cv2.IMREAD_COLOR) is None:
            unreadable.append(str(image_path))

    if unreadable:
        preview = "\n".join(unreadable[:10])
        suffix = "" if len(unreadable) <= 10 else "\n..."
        raise ValueError(
            "Unreadable image files found "
            f"({len(unreadable)}):\n{preview}{suffix}"
        )

    return samples, classes, class_to_idx


def split_indices(
    targets: np.ndarray,
    val_split: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    all_indices = np.arange(len(targets))
    val_count = int(round(len(targets) * val_split))
    val_count = max(1, min(len(targets) - 1, val_count))

    try:
        train_idx, val_idx = train_test_split(
            all_indices,
            test_size=val_count,
            random_state=seed,
            shuffle=True,
            stratify=targets,
        )
    except ValueError:
        rng = np.random.default_rng(seed)
        shuffled = all_indices.copy()
        rng.shuffle(shuffled)
        val_idx = shuffled[:val_count]
        train_idx = shuffled[val_count:]
        print(
            "[warn] Stratified split unavailable for this dataset; "
            "used deterministic random split."
        )

    return train_idx.tolist(), val_idx.tolist()


def random_preview_augment(
    rgb_img: np.ndarray,
    img_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    img = resize_rgb_image(rgb_img, img_size)

    if rng.random() < 0.5:
        img = cv2.flip(img, 1)

    h, w = img.shape[:2]
    angle = float(rng.uniform(-15.0, 15.0))
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    img = cv2.warpAffine(
        img,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    alpha = float(rng.uniform(0.9, 1.15))
    beta = float(rng.uniform(-18.0, 18.0))
    img = np.clip(
        img.astype(np.float32) * alpha + beta,
        0,
        255,
    ).astype(np.uint8)
    return img


def save_augmented_images(
    samples: list[tuple[Path, int]],
    train_indices: list[int],
    augmented_dir: Path,
    class_names: list[str],
    per_image: int,
    img_size: int,
    seed: int,
) -> int:
    if per_image <= 0:
        return 0

    rng = np.random.default_rng(seed)
    saved = 0

    for sample_idx in train_indices:
        image_path, class_idx = samples[sample_idx]
        class_dir = augmented_dir / class_names[class_idx]
        class_dir.mkdir(parents=True, exist_ok=True)

        rgb = read_image_rgb(image_path)
        stem = image_path.stem

        for i in range(per_image):
            augmented = random_preview_augment(rgb, img_size=img_size, rng=rng)
            unique = uuid.uuid4().hex[:8]
            out_name = f"{stem}_aug-cv_k{i + 1}_{unique}.jpg"
            out_path = class_dir / out_name

            bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_path), bgr)
            saved += 1

    return saved


def extract_feature_matrix(
    samples: list[tuple[Path, int]],
    feature_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    hog_descriptor = build_hog_descriptor(feature_config)
    feature_rows: list[np.ndarray] = []
    labels: list[int] = []

    for index, (image_path, label) in enumerate(samples, start=1):
        try:
            feature_vector, _ = extract_feature_vector_from_path(
                image_path,
                feature_config,
                hog_descriptor=hog_descriptor,
            )
        except ValueError as exc:
            raise ValueError(
                f"Failed to featurize '{image_path}': {exc}"
            ) from exc

        feature_rows.append(feature_vector)
        labels.append(label)

        if index % 200 == 0:
            print(
                f"[info] Extracted features for {index}/{len(samples)} "
                "images..."
            )

    if not feature_rows:
        raise ValueError("No features were extracted from the dataset.")

    X = np.vstack(feature_rows).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    return X, y


def build_model_candidates(
    seed: int,
    svm_c: float,
    rf_trees: int,
    logreg_c: float,
) -> dict[str, Pipeline]:
    models: dict[str, Pipeline] = {
        "svm": Pipeline(
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
        ),
        "random_forest": Pipeline(
            steps=[
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=rf_trees,
                        max_depth=None,
                        min_samples_leaf=1,
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                )
            ]
        ),
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=logreg_c,
                        class_weight="balanced",
                        max_iter=5000,
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
    }
    return models


def train_and_select_model(
    model_mode: str,
    model_candidates: dict[str, Pipeline],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[Pipeline, str, list[dict[str, float]], np.ndarray]:
    num_classes = int(np.unique(y_train).size)
    available = sorted(model_candidates.keys())

    if model_mode == "auto":
        selected_names = available
    elif model_mode in model_candidates:
        selected_names = [model_mode]
    else:
        raise ValueError(f"Unsupported --model value: {model_mode}")

    best_name = ""
    best_model: Pipeline | None = None
    best_val_pred: np.ndarray | None = None
    best_val_acc = -1.0
    model_scores: list[dict[str, float]] = []

    for name in selected_names:
        model = model_candidates[name]
        print(f"[info] Training {name}...")
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        train_acc = float(accuracy_score(y_train, train_pred))
        val_acc = float(accuracy_score(y_val, val_pred))
        model_scores.append(
            {
                "model": name,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
            }
        )

        print(
            f"[info] {name}: train_acc={train_acc * 100:.2f}% "
            f"val_acc={val_acc * 100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_name = name
            best_model = model
            best_val_pred = val_pred

    if best_model is None or best_val_pred is None:
        raise RuntimeError("Model training failed to produce a best model.")

    labels = list(range(num_classes))
    cm = confusion_matrix(y_val, best_val_pred, labels=labels)
    return best_model, best_name, model_scores, cm


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

    if not (0.0 < args.val_split < 1.0):
        raise SystemExit("--val-split must be in (0, 1).")
    if args.augment_save_per_image < 0:
        raise SystemExit("--augment-save-per-image must be >= 0.")
    if args.rf_trees < 10:
        raise SystemExit("--rf-trees must be >= 10.")
    if args.svm_c <= 0 or args.logreg_c <= 0:
        raise SystemExit("--svm-c and --logreg-c must be > 0.")

    seed_everything(args.seed)

    try:
        feature_config = default_feature_config(args.img_size)
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from exc

    model_dir, augmented_dir = prepare_output_dirs(args.out)

    try:
        samples, classes, class_to_idx = collect_dataset_samples(
            args.dataset_dir
        )
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from exc

    targets = np.asarray([label for _, label in samples], dtype=np.int64)
    train_indices, val_indices = split_indices(
        targets,
        args.val_split,
        args.seed,
    )
    if not train_indices or not val_indices:
        raise SystemExit(
            "Error: invalid train/validation split produced empty split. "
            "Adjust --val-split or dataset size."
        )

    print(f"Number of classes: {len(classes)}")
    print(f"Total samples: {len(samples)}")
    print(f"Train samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")

    saved_aug_count = save_augmented_images(
        samples=samples,
        train_indices=train_indices,
        augmented_dir=augmented_dir,
        class_names=classes,
        per_image=args.augment_save_per_image,
        img_size=args.img_size,
        seed=args.seed,
    )
    print(f"Saved augmented previews: {saved_aug_count}")

    try:
        X_all, y_all = extract_feature_matrix(samples, feature_config)
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from exc

    feature_dim = int(X_all.shape[1])
    print(f"Feature vector dimension: {feature_dim}")

    X_train = X_all[train_indices]
    y_train = y_all[train_indices]
    X_val = X_all[val_indices]
    y_val = y_all[val_indices]

    candidates = build_model_candidates(
        seed=args.seed,
        svm_c=args.svm_c,
        rf_trees=args.rf_trees,
        logreg_c=args.logreg_c,
    )

    best_model, best_name, model_scores, best_cm = train_and_select_model(
        model_mode=args.model,
        model_candidates=candidates,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    best_val_accuracy = max(score["val_accuracy"] for score in model_scores)
    best_train_accuracy = next(
        score["train_accuracy"]
        for score in model_scores
        if score["model"] == best_name
    )

    timestamp = datetime.now(timezone.utc).isoformat()
    config = {
        "algorithm": best_name,
        "model_selection_mode": args.model,
        "img_size": args.img_size,
        "feature_config": feature_config,
        "seed": args.seed,
        "val_split": args.val_split,
        "dataset_dir": str(args.dataset_dir.resolve()),
        "timestamp_utc": timestamp,
        "best_val_accuracy": float(best_val_accuracy),
        "best_train_accuracy": float(best_train_accuracy),
        "hyperparameters": {
            "svm_c": args.svm_c,
            "rf_trees": args.rf_trees,
            "logreg_c": args.logreg_c,
        },
    }

    metrics_payload: dict[str, Any] = {
        "model_scores": model_scores,
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

    zip_path = make_zip(args.out)

    print(f"Selected model: {best_name}")
    print(f"Final validation accuracy (best): {best_val_accuracy * 100:.2f}%")
    print("Best validation confusion matrix:")
    print(best_cm)
    print(f"Model artifacts: {model_dir}")
    print(f"Zip path: {zip_path}")
    print(
        "Signature command:\n"
        f"  sha1sum {shlex.quote(str(zip_path))} > signature.txt"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

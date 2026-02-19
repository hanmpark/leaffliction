#!/usr/bin/env python3
"""Predict class probabilities from a saved handcrafted-feature classifier."""

from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

try:
    from .features import (
        IMAGE_EXTENSIONS,
        build_hog_descriptor,
        extract_feature_vector_from_path,
        read_image_rgb,
    )
    from .console_output import (
        format_disease_label,
        print_prediction_report,
        print_progress,
    )
except ImportError:  # pragma: no cover
    from features import (
        IMAGE_EXTENSIONS,
        build_hog_descriptor,
        extract_feature_vector_from_path,
        read_image_rgb,
    )
    from console_output import (
        format_disease_label,
        print_prediction_report,
        print_progress,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict leaf disease from one image, or from a directory of "
            "images (recursive depth: 5)."
        )
    )
    parser.add_argument(
        "model_dir",
        type=Path,
        help=(
            "Directory containing model.pkl "
            "(classes.json is optional fallback)."
        ),
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to one image, or a directory of images.",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        dest="max_images",
        help=(
            "Maximum number of random unique images to predict in directory "
            "mode (example: --max=100)."
        ),
    )
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_idx_to_class(idx_to_class: dict[str, Any]) -> list[str]:
    if not isinstance(idx_to_class, dict) or not idx_to_class:
        raise ValueError("empty or invalid idx_to_class mapping")

    classes: list[str] = []
    for i in range(len(idx_to_class)):
        if str(i) in idx_to_class:
            classes.append(str(idx_to_class[str(i)]))
            continue
        if i in idx_to_class:
            classes.append(str(idx_to_class[i]))
            continue
        raise ValueError(f"missing class index {i}")
    return classes


def load_idx_to_class(
    model_payload: dict[str, Any],
    model_dir: Path,
) -> list[str]:
    idx_to_class = model_payload.get("idx_to_class")
    if isinstance(idx_to_class, dict):
        try:
            return parse_idx_to_class(idx_to_class)
        except ValueError:
            pass

    classes_path = model_dir / "classes.json"
    if not classes_path.exists():
        raise SystemExit(
            "Error: model.pkl missing 'idx_to_class' "
            "and classes.json not found."
        )

    classes_payload = load_json(classes_path)
    idx_to_class = classes_payload.get("idx_to_class")
    if not isinstance(idx_to_class, dict):
        raise SystemExit("Error: classes.json missing 'idx_to_class'.")

    try:
        return parse_idx_to_class(idx_to_class)
    except ValueError as exc:
        raise SystemExit(f"Error: invalid class mapping: {exc}") from exc


def show_images(
    original_img: np.ndarray,
    transformed_img: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(transformed_img)
    axes[1].set_title("Model Input (Resized)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def collect_images_from_directory(
    root: Path,
    max_depth: int = 5,
) -> list[Path]:
    """Collect supported images recursively up to max directory depth."""
    images: list[Path] = []
    stack: list[tuple[Path, int]] = [(root, 0)]

    while stack:
        current_dir, depth = stack.pop()
        try:
            entries = sorted(current_dir.iterdir())
        except OSError as exc:
            raise SystemExit(
                f"Error: failed to read directory '{current_dir}': {exc}"
            ) from exc

        for entry in entries:
            if entry.is_dir():
                if depth < max_depth:
                    stack.append((entry, depth + 1))
                continue
            if not entry.is_file():
                continue
            if entry.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(entry)

    return images


def predict_single_image(
    estimator: Any,
    feature_config: dict[str, Any],
    idx_to_class: list[str],
    image_path: Path,
) -> int:
    """Predict one image and show side-by-side visualization."""
    hog_descriptor = build_hog_descriptor(feature_config)
    feature_vector, transformed_img = extract_feature_vector_from_path(
        image_path,
        feature_config,
        hog_descriptor=hog_descriptor,
    )

    X = feature_vector.reshape(1, -1)
    predicted_idx = int(estimator.predict(X)[0])
    if predicted_idx < 0 or predicted_idx >= len(idx_to_class):
        raise SystemExit(
            "Error: prediction output class index is out of range."
        )
    print_prediction_report(
        image_path=image_path,
        idx_to_class=idx_to_class,
        predicted_idx=predicted_idx,
    )

    original_img = read_image_rgb(image_path)
    show_images(
        original_img=original_img,
        transformed_img=transformed_img,
    )
    return 0


def predict_directory(
    estimator: Any,
    idx_to_class: list[str],
    feature_config: dict[str, Any],
    input_dir: Path,
    max_images: int | None,
) -> int:
    """Predict a random unique subset of images from a directory tree."""
    if max_images is not None and max_images <= 0:
        raise SystemExit("Error: --max must be greater than 0.")

    all_images = collect_images_from_directory(input_dir, max_depth=5)
    total_found = len(all_images)
    if total_found == 0:
        raise SystemExit(
            f"Error: no supported images found in '{input_dir}' (depth <= 5)."
        )

    print(f"[info] Found {total_found} image(s) in directory mode.")
    selected = list(all_images)
    random.shuffle(selected)
    if max_images is not None:
        selected = selected[:max_images]
    total_selected = len(selected)
    print(f"[info] Predicting {total_selected} image(s).")

    hog_descriptor = build_hog_descriptor(feature_config)
    success_entries: list[tuple[Path, str]] = []
    failure_entries: list[tuple[Path, str, str]] = []
    evaluated_pairs: list[tuple[str, str]] = []

    def expected_label_from_path(path: Path) -> str:
        return format_disease_label(path.parent.name)

    for index, image_path in enumerate(selected, start=1):
        expected_label = expected_label_from_path(image_path)
        try:
            feature_vector, _ = extract_feature_vector_from_path(
                image_path,
                feature_config,
                hog_descriptor=hog_descriptor,
            )
            X = feature_vector.reshape(1, -1)
            predicted_idx = int(estimator.predict(X)[0])
            if predicted_idx < 0 or predicted_idx >= len(idx_to_class):
                raise ValueError("predicted class index out of range")
            predicted_label = format_disease_label(idx_to_class[predicted_idx])
            evaluated_pairs.append((expected_label, predicted_label))

            if predicted_label == expected_label:
                success_entries.append((image_path, predicted_label))
            else:
                failure_entries.append(
                    (image_path, predicted_label, expected_label)
                )
        except Exception as exc:  # pragma: no cover
            failure_entries.append(
                (image_path, f"<error: {exc}>", expected_label)
            )
            print(
                f"\n[warn] Skipping '{image_path}': {exc}",
                flush=True,
            )

        success_rate = (len(success_entries) / index) * 100.0 if index else 0.0
        print_progress(
            "Batch prediction",
            index,
            total_selected,
            suffix=f"success={success_rate:6.2f}%",
        )

    success_count = len(success_entries)
    failure_count = len(failure_entries)
    success_rate = (success_count / total_selected) * 100.0
    known_labels = [format_disease_label(label) for label in idx_to_class]
    confusion_labels = list(dict.fromkeys(known_labels))
    label_to_idx = {label: i for i, label in enumerate(confusion_labels)}
    confusion = np.zeros(
        (len(confusion_labels), len(confusion_labels)),
        dtype=np.int64,
    )
    for expected_label, predicted_label in evaluated_pairs:
        if (
            expected_label not in label_to_idx
            or predicted_label not in label_to_idx
        ):
            continue
        confusion[
            label_to_idx[expected_label],
            label_to_idx[predicted_label],
        ] += 1

    summary_path = Path("prediction_summary.txt")
    lines: list[str] = [
        "Prediction Summary",
        f"Input directory: {input_dir.resolve()}",
        f"Files processed: {total_selected}",
        f"Success: {success_count}",
        f"Failed: {failure_count}",
        f"Evaluated for confusion matrix: {len(evaluated_pairs)}",
        "",
        "Confusion Matrix",
        "Rows=Expected, Columns=Predicted",
        f"Labels: {', '.join(confusion_labels)}",
        np.array2string(confusion),
        "",
        "Failed files",
        "FilePath\tPrediction\tCorrect Output",
    ]
    for file_path, prediction, correct_output in failure_entries:
        lines.append(f"{file_path}\t{prediction}\t{correct_output}")
    lines.extend(["", "Succeeded files", "FilePath\tPrediction"])
    for file_path, prediction in success_entries:
        lines.append(f"{file_path}\t{prediction}")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Directory prediction summary:")
    print(f"  processed: {total_selected}")
    print(f"  success: {success_count}")
    print(f"  failed: {failure_count}")
    print(f"  success percentage: {success_rate:.2f}%")
    print("  confusion matrix (rows=expected, cols=predicted):")
    print(f"    labels: {confusion_labels}")
    print(confusion)
    print(f"  details file: {summary_path.resolve()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.model_dir.exists():
        raise SystemExit(f"Error: model directory not found: {args.model_dir}")
    if not args.model_dir.is_dir():
        raise SystemExit(
            f"Error: model path is not a directory: {args.model_dir}"
        )
    if not args.input_path.exists():
        raise SystemExit(f"Error: input path not found: {args.input_path}")

    model_path = args.model_dir / "model.pkl"
    if not model_path.exists():
        raise SystemExit(f"Error: required artifact missing: {model_path}")

    with model_path.open("rb") as f:
        model_payload = pickle.load(f)

    estimator = model_payload.get("estimator")
    feature_config = model_payload.get("feature_config")
    idx_to_class = load_idx_to_class(model_payload, args.model_dir)
    if estimator is None:
        raise SystemExit("Error: model.pkl missing 'estimator'.")
    if not isinstance(feature_config, dict):
        raise SystemExit("Error: model.pkl missing 'feature_config'.")

    if args.input_path.is_file():
        return predict_single_image(
            estimator=estimator,
            feature_config=feature_config,
            idx_to_class=idx_to_class,
            image_path=args.input_path,
        )
    if args.input_path.is_dir():
        return predict_directory(
            estimator=estimator,
            idx_to_class=idx_to_class,
            feature_config=feature_config,
            input_dir=args.input_path,
            max_images=args.max_images,
        )

    raise SystemExit(f"Error: unsupported input path: {args.input_path}")


if __name__ == "__main__":
    raise SystemExit(main())

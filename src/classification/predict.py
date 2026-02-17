#!/usr/bin/env python3
"""Predict class probabilities from a saved handcrafted-feature classifier."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

try:
    from .features import (
        build_hog_descriptor,
        extract_feature_vector_from_path,
        read_image_rgb,
    )
    from .console_output import print_prediction_report
except ImportError:  # pragma: no cover
    from features import (
        build_hog_descriptor,
        extract_feature_vector_from_path,
        read_image_rgb,
    )
    from console_output import print_prediction_report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict a leaf disease class from one image."
    )
    parser.add_argument(
        "model_dir",
        type=Path,
        help="Directory containing model.pkl/classes.json/config.json.",
    )
    parser.add_argument(
        "image_path",
        type=Path,
        help="Path to input image.",
    )
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_idx_to_class(classes_payload: dict[str, Any]) -> list[str]:
    idx_to_class = classes_payload["idx_to_class"]
    return [
        str(idx_to_class[str(i)])
        for i in range(len(idx_to_class))
    ]


def show_images(
    original_img: np.ndarray,
    transformed_img: np.ndarray,
    image_path: Path,
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.model_dir.exists():
        raise SystemExit(f"Error: model directory not found: {args.model_dir}")
    if not args.model_dir.is_dir():
        raise SystemExit(
            f"Error: model path is not a directory: {args.model_dir}"
        )
    if not args.image_path.exists():
        raise SystemExit(f"Error: image not found: {args.image_path}")
    if not args.image_path.is_file():
        raise SystemExit(f"Error: image path is not a file: {args.image_path}")

    model_path = args.model_dir / "model.pkl"
    classes_path = args.model_dir / "classes.json"
    config_path = args.model_dir / "config.json"

    for required_path in (model_path, classes_path, config_path):
        if not required_path.exists():
            raise SystemExit(
                f"Error: required artifact missing: {required_path}"
            )

    classes_payload = load_json(classes_path)
    load_json(config_path)

    idx_to_class = parse_idx_to_class(classes_payload)
    with model_path.open("rb") as f:
        model_payload = pickle.load(f)

    estimator = model_payload.get("estimator")
    feature_config = model_payload.get("feature_config")
    if estimator is None:
        raise SystemExit("Error: model.pkl missing 'estimator'.")
    if not isinstance(feature_config, dict):
        raise SystemExit("Error: model.pkl missing 'feature_config'.")

    hog_descriptor = build_hog_descriptor(feature_config)
    feature_vector, transformed_img = extract_feature_vector_from_path(
        args.image_path,
        feature_config,
        hog_descriptor=hog_descriptor,
    )

    X = feature_vector.reshape(1, -1)
    probs = estimator.predict_proba(X)[0]

    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 1 or probs.size != len(idx_to_class):
        raise SystemExit(
            "Error: prediction output shape does not match class count."
        )

    top_k = min(3, len(idx_to_class))
    top_indices = np.argsort(probs)[::-1][:top_k]
    top_index_list = [int(idx) for idx in top_indices]
    print_prediction_report(
        image_path=args.image_path,
        idx_to_class=idx_to_class,
        probs=probs,
        top_indices=top_index_list,
    )

    original_img = read_image_rgb(args.image_path)
    show_images(
        original_img=original_img,
        transformed_img=transformed_img,
        image_path=args.image_path,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

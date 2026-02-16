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
except ImportError:  # pragma: no cover
    from features import (
        build_hog_descriptor,
        extract_feature_vector_from_path,
        read_image_rgb,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict a leaf disease class from one image."
    )
    parser.add_argument(
        "image_path",
        type=Path,
        help="Path to input image.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("./artifacts/model"),
        help="Directory containing model.pkl/classes.json/config.json.",
    )
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to read JSON '{path}': {exc}") from exc


def parse_idx_to_class(classes_payload: dict[str, Any]) -> list[str]:
    idx_to_class = classes_payload.get("idx_to_class")
    if isinstance(idx_to_class, list):
        return [str(x) for x in idx_to_class]
    if isinstance(idx_to_class, dict):
        try:
            return [
                str(idx_to_class[str(i)])
                for i in range(len(idx_to_class))
            ]
        except KeyError as exc:
            raise ValueError(
                "Invalid idx_to_class mapping in classes.json."
            ) from exc
    raise ValueError("Missing or invalid idx_to_class in classes.json.")


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
    backend = plt.get_backend().lower()
    if backend.endswith("agg"):
        preview_path = image_path.with_name(
            f"{image_path.stem}_predict_preview.png"
        )
        fig.savefig(preview_path, dpi=150, bbox_inches="tight")
        print(
            f"[INFO] Non-interactive backend '{backend}': "
            f"saved preview to {preview_path}"
        )
    else:
        plt.show()
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.image_path.exists() or not args.image_path.is_file():
        raise SystemExit(f"Error: image not found: {args.image_path}")
    if not args.model_dir.exists() or not args.model_dir.is_dir():
        raise SystemExit(f"Error: model directory not found: {args.model_dir}")

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
    if not idx_to_class:
        raise SystemExit("Error: no classes found in classes.json")

    try:
        with model_path.open("rb") as f:
            model_payload = pickle.load(f)
    except Exception as exc:
        raise SystemExit(f"Error: failed to load model pickle: {exc}") from exc

    estimator = model_payload.get("estimator")
    feature_config = model_payload.get("feature_config")

    if estimator is None:
        raise SystemExit("Error: model.pkl missing 'estimator'.")
    if not isinstance(feature_config, dict):
        raise SystemExit("Error: model.pkl missing 'feature_config'.")

    try:
        hog_descriptor = build_hog_descriptor(feature_config)
        feature_vector, transformed_img = extract_feature_vector_from_path(
            args.image_path,
            feature_config,
            hog_descriptor=hog_descriptor,
        )
    except ValueError as exc:
        raise SystemExit(f"Error: failed to preprocess image: {exc}") from exc

    X = feature_vector.reshape(1, -1)

    try:
        probs = estimator.predict_proba(X)[0]
    except Exception as exc:
        raise SystemExit(f"Error: failed to run prediction: {exc}") from exc

    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 1 or probs.size != len(idx_to_class):
        raise SystemExit(
            "Error: prediction output shape does not match class count."
        )

    top_k = min(3, len(idx_to_class))
    top_indices = np.argsort(probs)[::-1][:top_k]

    best_idx = int(top_indices[0])
    best_class = idx_to_class[best_idx]
    print(f"Predicted class: {best_class}")
    print("Top-3 probabilities:")
    for rank, idx in enumerate(top_indices, start=1):
        prob = float(probs[idx]) * 100.0
        print(f"{rank}. {idx_to_class[int(idx)]}: {prob:.2f}%")

    original_img = read_image_rgb(args.image_path)
    show_images(
        original_img=original_img,
        transformed_img=transformed_img,
        image_path=args.image_path,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

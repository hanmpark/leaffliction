#!/usr/bin/env python3
"""Console output formatting utilities for classification programs."""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


RESET = "\033[0m"
BOLD = "\033[1m"
FG_CYAN = "\033[36m"
FG_GREEN = "\033[32m"
FG_YELLOW = "\033[33m"
FG_BLUE = "\033[34m"
FG_MAGENTA = "\033[35m"


SPINNER_FRAMES = "|/-\\"


def supports_color() -> bool:
    """Return True when ANSI color output is supported."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return bool(sys.stdout.isatty())


def style(text: str, *codes: str, enabled: bool) -> str:
    """Apply ANSI style codes to text when enabled."""
    if not enabled:
        return text
    active = "".join(code for code in codes if code)
    return f"{active}{text}{RESET}"


def format_disease_label(label: str) -> str:
    """Normalize display label for console output."""
    normalized = label.replace("\\", "/")
    leaf = normalized.split("/")[-1]
    return leaf.strip().lower().replace(" ", "_")


def print_progress(
    label: str,
    current: int,
    total: int,
    width: int = 28,
) -> None:
    """Render a one-line progress bar."""
    if total <= 0:
        return
    ratio = min(1.0, max(0.0, current / float(total)))
    filled = int(round(width * ratio))
    bar = "#" * filled + "-" * (width - filled)
    line = f"{label}: [{bar}] {current}/{total} ({ratio * 100.0:6.2f}%)"
    end = "\n" if current >= total else "\r"
    print(line, end=end, flush=True)


def run_with_spinner(
    label: str,
    fn: Callable[..., object],
    *args: object,
    **kwargs: object,
) -> object:
    """Run a blocking function while showing a spinner and elapsed time."""
    result: dict[str, object] = {}
    error: dict[str, BaseException] = {}
    done = threading.Event()

    def _target() -> None:
        try:
            result["value"] = fn(*args, **kwargs)
        except BaseException as exc:  # pragma: no cover
            error["value"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()

    start = time.monotonic()
    frame_index = 0
    while not done.wait(0.2):
        elapsed = time.monotonic() - start
        frame = SPINNER_FRAMES[frame_index % len(SPINNER_FRAMES)]
        print(
            f"{label} {frame} {elapsed:5.1f}s",
            end="\r",
            flush=True,
        )
        frame_index += 1

    thread.join()
    elapsed = time.monotonic() - start
    print(f"{label} done in {elapsed:.1f}s" + " " * 8)

    if "value" in error:
        raise error["value"]
    return result.get("value")


def print_train_intro(
    dataset_dir: Path,
    out_dir: Path,
    img_size: int,
    seed: int,
    val_split: float,
    svm_c: float,
) -> None:
    """Print a styled train-session intro block."""
    color_enabled = supports_color()
    banner = "=" * 40

    print(style(banner, BOLD, FG_CYAN, enabled=color_enabled))
    print(style("Training Session", BOLD, FG_CYAN, enabled=color_enabled))
    print(style(banner, BOLD, FG_CYAN, enabled=color_enabled))
    print(f"{style('Dataset:', BOLD, enabled=color_enabled)} {dataset_dir}")
    print(f"{style('Output:', BOLD, enabled=color_enabled)} {out_dir}")
    print(f"{style('Image size:', BOLD, enabled=color_enabled)} {img_size}")
    print(f"{style('Seed:', BOLD, enabled=color_enabled)} {seed}")
    print(
        f"{style('Validation split:', BOLD, enabled=color_enabled)} "
        f"{val_split:.2f}"
    )
    print(f"{style('SVC C:', BOLD, enabled=color_enabled)} {svm_c}")
    print()


def print_dataset_summary(
    num_classes: int,
    total_samples: int,
    train_samples: int,
    val_samples: int,
) -> None:
    """Print dataset split summary."""
    color_enabled = supports_color()
    print(style("Dataset Summary", BOLD, FG_MAGENTA, enabled=color_enabled))
    print(f"  classes: {num_classes}")
    print(f"  total samples: {total_samples}")
    print(f"  train samples: {train_samples}")
    print(f"  validation samples: {val_samples}")
    print()


def print_train_metrics(train_acc: float, val_acc: float) -> None:
    """Print train/validation accuracy metrics."""
    color_enabled = supports_color()
    print(style("Model Metrics", BOLD, FG_MAGENTA, enabled=color_enabled))
    print(f"  train accuracy: {train_acc * 100.0:.2f}%")
    print(f"  validation accuracy: {val_acc * 100.0:.2f}%")
    print()


def print_train_outro(
    model_name: str,
    best_val_accuracy: float,
    confusion_matrix: np.ndarray,
    model_dir: Path,
    zip_path: Path,
) -> None:
    """Print final training summary and artifact paths."""
    color_enabled = supports_color()
    print(style("Training Result", BOLD, FG_CYAN, enabled=color_enabled))
    print(f"  selected model: {model_name}")
    print(f"  best validation accuracy: {best_val_accuracy * 100.0:.2f}%")
    print("  confusion matrix:")
    print(confusion_matrix)
    print(f"  model artifacts: {model_dir}")
    print(f"  zip path: {zip_path}")


def print_prediction_report(
    image_path: Path,
    idx_to_class: Sequence[str],
    probs: np.ndarray,
    top_indices: Sequence[int],
) -> None:
    """Print a styled prediction summary block."""
    best_idx = int(top_indices[0])
    confidence = float(probs[best_idx]) * 100.0
    top_labels = [
        format_disease_label(idx_to_class[int(idx)])
        for idx in top_indices
    ]
    width = max(len(name) for name in top_labels)
    color_enabled = supports_color()

    banner = "=" * 30
    print(style(banner, BOLD, FG_CYAN, enabled=color_enabled))
    print(style("Prediction Result", BOLD, FG_CYAN, enabled=color_enabled))
    print(style(banner, BOLD, FG_CYAN, enabled=color_enabled))
    print()

    image_path_key = style("Image path:", BOLD, enabled=color_enabled)
    predicted_key = style(
        "Predicted disease:",
        BOLD,
        FG_GREEN,
        enabled=color_enabled,
    )
    confidence_key = style("Confidence:", BOLD, enabled=color_enabled)

    predicted_value = style(
        top_labels[0],
        BOLD,
        FG_GREEN,
        enabled=color_enabled,
    )
    confidence_value = style(
        f"{confidence:.2f}%",
        BOLD,
        FG_GREEN,
        enabled=color_enabled,
    )

    print(f"{image_path_key} {image_path}")
    print(f"{predicted_key} {predicted_value}")
    print(f"{confidence_key} {confidence_value}")
    print()

    print(style("Top probabilities:", BOLD, FG_MAGENTA, enabled=color_enabled))
    arrow = style("->", FG_BLUE, enabled=color_enabled)

    for rank, (label, idx) in enumerate(zip(top_labels, top_indices), start=1):
        prob = float(probs[int(idx)]) * 100.0
        label_text = f"{label:<{width}}"
        if rank == 1:
            label_text = style(
                label_text,
                BOLD,
                FG_GREEN,
                enabled=color_enabled,
            )
            prob_text = style(
                f"{prob:.2f}%",
                BOLD,
                FG_GREEN,
                enabled=color_enabled,
            )
        else:
            label_text = style(label_text, FG_CYAN, enabled=color_enabled)
            prob_text = style(f"{prob:.2f}%", FG_YELLOW, enabled=color_enabled)
        print(f"  {label_text} {arrow} {prob_text}")

#!/usr/bin/env python3
"""Feature extraction utilities for classical image classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np


IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def default_feature_config(img_size: int) -> dict[str, Any]:
    """Return a validated default feature configuration."""
    if img_size < 32:
        raise ValueError("--img-size must be >= 32.")
    if img_size % 8 != 0:
        raise ValueError(
            "--img-size must be a multiple of 8 for HOG extraction."
        )

    return {
        "img_size": int(img_size),
        "color_hist_bins": [8, 8, 8],
        "hog": {
            "block_size": 16,
            "block_stride": 8,
            "cell_size": 8,
            "bins": 9,
        },
        "texture": {
            "levels": 16,
            "offsets": [[0, 1], [1, 0], [1, 1], [-1, 1]],
        },
    }


def build_hog_descriptor(
    feature_config: Mapping[str, Any],
) -> cv2.HOGDescriptor:
    """Construct an OpenCV HOG descriptor from configuration."""
    img_size = int(feature_config["img_size"])
    hog_cfg = feature_config["hog"]

    block_size = int(hog_cfg["block_size"])
    block_stride = int(hog_cfg["block_stride"])
    cell_size = int(hog_cfg["cell_size"])
    bins = int(hog_cfg["bins"])

    if block_size < cell_size:
        raise ValueError("HOG block_size must be >= cell_size.")
    if block_size % cell_size != 0:
        raise ValueError("HOG block_size must be divisible by cell_size.")
    if block_stride > block_size:
        raise ValueError("HOG block_stride must be <= block_size.")
    if block_stride % cell_size != 0:
        raise ValueError("HOG block_stride must be divisible by cell_size.")
    if (img_size - block_size) % block_stride != 0:
        raise ValueError(
            "Invalid HOG geometry: (img_size - block_size) must be divisible "
            "by block_stride."
        )

    return cv2.HOGDescriptor(
        _winSize=(img_size, img_size),
        _blockSize=(block_size, block_size),
        _blockStride=(block_stride, block_stride),
        _cellSize=(cell_size, cell_size),
        _nbins=bins,
    )


def read_image_rgb(path: Path) -> np.ndarray:
    """Read an image path as RGB uint8."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_rgb_image(rgb_img: np.ndarray, img_size: int) -> np.ndarray:
    """Resize to square size using area/interlinear interpolation."""
    interpolation = (
        cv2.INTER_AREA
        if rgb_img.shape[0] > img_size or rgb_img.shape[1] > img_size
        else cv2.INTER_LINEAR
    )
    return cv2.resize(
        rgb_img,
        (img_size, img_size),
        interpolation=interpolation,
    )


def extract_color_histogram(
    rgb_img: np.ndarray,
    bins: tuple[int, int, int],
) -> np.ndarray:
    """Extract normalized HSV 3D color histogram."""
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).reshape(-1)
    return hist.astype(np.float32)


def _offset_slices(
    height: int,
    width: int,
    dx: int,
    dy: int,
) -> tuple[slice, slice, slice, slice]:
    if abs(dx) >= height or abs(dy) >= width:
        raise ValueError("Texture offset is too large for the resized image.")

    if dx >= 0:
        src_x = slice(0, height - dx)
        dst_x = slice(dx, height)
    else:
        src_x = slice(-dx, height)
        dst_x = slice(0, height + dx)

    if dy >= 0:
        src_y = slice(0, width - dy)
        dst_y = slice(dy, width)
    else:
        src_y = slice(-dy, width)
        dst_y = slice(0, width + dy)

    return src_x, src_y, dst_x, dst_y


def _glcm_features(glcm: np.ndarray) -> np.ndarray:
    eps = 1e-12
    p = glcm.astype(np.float64)
    total = p.sum()
    if total <= 0:
        return np.zeros(7, dtype=np.float32)
    p /= total

    i, j = np.indices(p.shape)
    delta = i - j
    abs_delta = np.abs(delta)

    contrast = np.sum((delta ** 2) * p)
    dissimilarity = np.sum(abs_delta * p)
    homogeneity = np.sum(p / (1.0 + delta ** 2))
    asm = np.sum(p * p)
    energy = float(np.sqrt(asm))

    mu_i = np.sum(i * p)
    mu_j = np.sum(j * p)
    sigma_i = float(np.sqrt(np.sum(((i - mu_i) ** 2) * p)))
    sigma_j = float(np.sqrt(np.sum(((j - mu_j) ** 2) * p)))
    if sigma_i < eps or sigma_j < eps:
        correlation = 0.0
    else:
        correlation = float(
            np.sum((i - mu_i) * (j - mu_j) * p) / (sigma_i * sigma_j)
        )

    entropy = float(-np.sum(p * np.log2(p + eps)))

    return np.array(
        [
            contrast,
            dissimilarity,
            homogeneity,
            energy,
            correlation,
            entropy,
            asm,
        ],
        dtype=np.float32,
    )


def extract_texture_features(
    gray_img: np.ndarray,
    levels: int,
    offsets: list[tuple[int, int]],
) -> np.ndarray:
    """Extract texture features using directional GLCM statistics."""
    quantized = np.floor(
        gray_img.astype(np.float32) * levels / 256.0
    ).astype(np.int32)
    quantized = np.clip(quantized, 0, levels - 1)

    h, w = quantized.shape
    directional_features: list[np.ndarray] = []

    for dx, dy in offsets:
        src_x, src_y, dst_x, dst_y = _offset_slices(h, w, dx, dy)
        a = quantized[src_x, src_y].reshape(-1)
        b = quantized[dst_x, dst_y].reshape(-1)

        glcm = np.zeros((levels, levels), dtype=np.float64)
        np.add.at(glcm, (a, b), 1)
        glcm += glcm.T

        directional_features.append(_glcm_features(glcm))

    if not directional_features:
        raise ValueError("Texture extraction requires at least one offset.")

    glcm_stats = np.mean(np.stack(directional_features, axis=0), axis=0)

    lap = cv2.Laplacian(gray_img, cv2.CV_32F)
    extras = np.array(
        [
            float(gray_img.mean() / 255.0),
            float(gray_img.std() / 255.0),
            float(lap.var() / (255.0 ** 2)),
        ],
        dtype=np.float32,
    )

    return np.concatenate([glcm_stats.astype(np.float32), extras], axis=0)


def extract_feature_vector(
    rgb_img: np.ndarray,
    feature_config: Mapping[str, Any],
    hog_descriptor: cv2.HOGDescriptor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the combined color/HOG/texture feature vector."""
    img_size = int(feature_config["img_size"])
    color_hist_bins = tuple(int(v) for v in feature_config["color_hist_bins"])
    texture_cfg = feature_config["texture"]
    levels = int(texture_cfg["levels"])
    offsets = [tuple(int(v) for v in pair) for pair in texture_cfg["offsets"]]

    resized = resize_rgb_image(rgb_img, img_size)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

    color_features = extract_color_histogram(resized, bins=color_hist_bins)

    descriptor = hog_descriptor or build_hog_descriptor(feature_config)
    hog_features = descriptor.compute(gray)
    if hog_features is None:
        raise ValueError("OpenCV HOG extraction failed for image.")
    hog_features = hog_features.reshape(-1).astype(np.float32)

    texture_features = extract_texture_features(
        gray_img=gray,
        levels=levels,
        offsets=offsets,
    )

    feature_vector = np.concatenate(
        [color_features, hog_features, texture_features],
        axis=0,
    ).astype(np.float32)

    return feature_vector, resized


def extract_feature_vector_from_path(
    image_path: Path,
    feature_config: Mapping[str, Any],
    hog_descriptor: cv2.HOGDescriptor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and featurize one image path."""
    rgb = read_image_rgb(image_path)
    return extract_feature_vector(
        rgb,
        feature_config,
        hog_descriptor=hog_descriptor,
    )

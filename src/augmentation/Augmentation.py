#!/usr/bin/env python3

import sys
from pathlib import Path

import cv2
import numpy as np


def save(img, path, suffix):
    out = path.with_name(f"{path.stem}_{suffix}{path.suffix}")
    cv2.imwrite(str(out), img)


def flip(img):
    return cv2.flip(img, 1)


def rotate(img, angle=30):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    return cv2.warpAffine(
        img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )


def skew(img, x_ratio=0.12):
    h, w = img.shape[:2]
    offset = int(w * x_ratio)
    if offset <= 0:
        return img.copy()
    offset = min(offset, max(1, w // 3))
    src = np.float32(
        [
            [0, 0],
            [w - 1, 0],
            [0, h - 1],
            [w - 1, h - 1],
        ]
    )
    dst = np.float32(
        [
            [offset, 0],
            [w - 1 - offset, 0],
            [0, h - 1],
            [w - 1, h - 1],
        ]
    )
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )


def shear(img, shear_factor=0.2):
    h, w = img.shape[:2]
    a = np.array([[1.0, shear_factor], [0.0, 1.0]], dtype=np.float32)
    cx, cy = w / 2.0, h / 2.0
    tx = cx - (a[0, 0] * cx + a[0, 1] * cy)
    ty = cy - (a[1, 0] * cx + a[1, 1] * cy)
    m = np.array([[a[0, 0], a[0, 1], tx], [a[1, 0], a[1, 1], ty]], dtype=np.float32)
    return cv2.warpAffine(
        img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )


def crop(img, ratio=0.9):
    if ratio >= 1.0 or ratio <= 0.0:
        return img.copy()
    h, w = img.shape[:2]
    new_w = max(1, int(w * ratio))
    new_h = max(1, int(h * ratio))
    x0 = max(0, (w - new_w) // 2)
    y0 = max(0, (h - new_h) // 2)
    cropped = img[y0 : y0 + new_h, x0 : x0 + new_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def distortion(img, k1=-0.2):
    h, w = img.shape[:2]
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    x = (xs - w / 2.0) / (w / 2.0)
    y = (ys - h / 2.0) / (h / 2.0)
    r2 = x * x + y * y
    factor = 1 + k1 * r2
    map_x = (x * factor * (w / 2.0) + w / 2.0).astype(np.float32)
    map_y = (y * factor * (h / 2.0) + h / 2.0).astype(np.float32)
    return cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: ./Augmentation.py image_path")
        return 1

    path = Path(sys.argv[1])
    img = cv2.imread(str(path))

    if img is None:
        print("Error: cannot load image")
        return 1

    save(flip(img), path, "Flip")
    save(rotate(img), path, "Rotate")
    save(skew(img), path, "Skew")
    save(shear(img), path, "Shear")
    save(crop(img), path, "Crop")
    save(distortion(img), path, "Distortion")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3

import sys
from pathlib import Path

import cv2
import numpy as np


def save(img, path, suffix):
    out = path.with_name(f"{path.stem}_{suffix}{path.suffix}")
    cv2.imwrite(str(out), img)


def rotate(img, angle=30):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    return cv2.warpAffine(img, m, (w, h))


def blur(img, ksize=9):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def contrast(img, alpha=1.6):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)


def scaling(img, scale=1.2):
    h, w = img.shape[:2]
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    x0 = max(0, (new_w - w) // 2)
    y0 = max(0, (new_h - h) // 2)
    return resized[y0 : y0 + h, x0 : x0 + w]


def illumination(img, beta=40):
    return cv2.convertScaleAbs(img, alpha=1.0, beta=beta)


def projective(img):
    h, w = img.shape[:2]
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
            [w * 0.08, h * 0.04],
            [w * 0.92, h * 0.12],
            [w * 0.04, h * 0.96],
            [w * 0.96, h * 0.86],
        ]
    )
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, (w, h))


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: ./Augmentation.py image_path")
        return 1

    path = Path(sys.argv[1])
    img = cv2.imread(str(path))

    if img is None:
        print("Error: cannot load image")
        return 1

    save(rotate(img), path, "Rotate")
    save(blur(img), path, "Blur")
    save(contrast(img), path, "Contrast")
    save(scaling(img), path, "Scale")
    save(illumination(img), path, "Illumination")
    save(projective(img), path, "Projective")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""I/O helpers for Leaffliction transformations."""

import cv2


def load_image(path):
    """Load image from path using OpenCV; return BGR and RGB."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to read image")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return bgr, rgb


def save_image(path, img):
    """Save an RGB or grayscale image to disk."""
    if img.ndim == 2:
        cv2.imwrite(path, img)
        return
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)

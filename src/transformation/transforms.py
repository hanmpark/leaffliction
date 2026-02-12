"""Image transformation primitives for Leaffliction."""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from plantcv import plantcv as pcv


def apply_blur(rgb_img):
    return pcv.gaussian_blur(img=rgb_img, ksize=(11, 11), sigma_x=0, sigma_y=None)


def build_mask(rgb_img):
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    mask_hsv = cv2.inRange(hsv, (25, 40, 40), (90, 255, 255))

    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    _, mask_otsu = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    ratio_hsv = np.count_nonzero(mask_hsv) / mask_hsv.size
    mask = mask_hsv if 0.01 <= ratio_hsv <= 0.95 else mask_otsu

    ratio = np.count_nonzero(mask) / mask.size
    if ratio < 0.01 or ratio > 0.95:
        mask = cv2.bitwise_not(mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for op in (cv2.MORPH_OPEN, cv2.MORPH_CLOSE):
        mask = cv2.morphologyEx(mask, op, kernel)
    return (mask > 0).astype(np.uint8) * 255


def largest_contour(mask):
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[
        -2
    ]
    return max(contours, key=cv2.contourArea) if contours else None


def make_roi_image(rgb_img, contour):
    roi = rgb_img.copy()
    if contour is None:
        return roi
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 255), 2)
    return roi


def contour_metrics(contour):
    if contour is None:
        return None
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    x, y, w, h = cv2.boundingRect(contour)
    m = cv2.moments(contour)
    cx = int(m["m10"] / m["m00"]) if m["m00"] else None
    cy = int(m["m01"] / m["m00"]) if m["m00"] else None
    return {
        "area": area,
        "perimeter": perimeter,
        "width": int(w),
        "height": int(h),
        "centroid": (cx, cy),
    }


def analyze_object_image(rgb_img, obj_mask, contour=None, label="leaf"):
    try:
        labeled, n_labels = pcv.create_labels(mask=obj_mask)
        if n_labels < 1:
            return rgb_img.copy()
        return pcv.analyze.size(
            img=rgb_img, labeled_mask=labeled, n_labels=1, label=label
        )
    except Exception:
        overlay = rgb_img.copy()
        if contour is not None:
            cv2.drawContours(overlay, [contour], -1, (255, 0, 255), 2)
            metrics = contour_metrics(contour)
            if metrics and metrics["centroid"][0] is not None:
                cv2.circle(overlay, metrics["centroid"], 5, (0, 255, 255), -1)
        return overlay


def draw_points(img, points, color, radius=4):
    if points is None:
        return 0
    h, w = img.shape[:2]
    count = 0
    for pt in points:
        if pt is None:
            continue
        arr = np.asarray(pt).reshape(-1)
        if arr.size < 2:
            continue
        x, y = int(arr[0]), int(arr[1])
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        cv2.circle(img, (x, y), radius + 1, (0, 0, 0), -1)
        cv2.circle(img, (x, y), radius, color, -1)
        count += 1
    return count


def _fallback_pseudolandmarks(mask, n=12):
    if mask is None or mask.ndim != 2:
        return [], [], []
    ys = np.where(mask > 0)[0]
    if ys.size == 0:
        return [], [], []
    y_min, y_max = int(ys.min()), int(ys.max())
    sample_ys = (
        [y_min]
        if y_min == y_max
        else np.linspace(y_min, y_max, n, dtype=int)
    )
    left, right, center = [], [], []
    for y in sample_ys:
        xs = np.where(mask[y] > 0)[0]
        if xs.size == 0:
            continue
        x_left, x_right = int(xs.min()), int(xs.max())
        x_center = int((x_left + x_right) / 2)
        left.append((x_left, y))
        right.append((x_right, y))
        center.append((x_center, y))
    return left, right, center


def pseudolandmarks_image(rgb_img, mask, label="leaf"):
    overlay = rgb_img.copy()
    radius = max(4, min(overlay.shape[:2]) // 150)
    try:
        left, right, center = pcv.homology.y_axis_pseudolandmarks(
            img=np.copy(rgb_img), mask=mask, label=label
        )
    except Exception:
        left, right, center = [], [], []

    count = 0
    count += draw_points(overlay, left, (255, 0, 0), radius=radius)
    count += draw_points(overlay, right, (0, 0, 255), radius=radius)
    count += draw_points(overlay, center, (255, 255, 0), radius=radius)
    if count == 0:
        left, right, center = _fallback_pseudolandmarks(mask)
        draw_points(overlay, left, (255, 0, 0), radius=radius)
        draw_points(overlay, right, (0, 0, 255), radius=radius)
        draw_points(overlay, center, (255, 255, 0), radius=radius)
    return overlay


def histogram_figure(rgb_img, mask, title="Histogram"):
    with plt.style.context("ggplot"):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)

        if mask is None:
            mask_idx = slice(None)
            total = rgb_img.shape[0] * rgb_img.shape[1]
        else:
            mask_idx = mask > 0
            total = int(np.count_nonzero(mask_idx))
            if total == 0:
                mask_idx = slice(None)
                total = rgb_img.shape[0] * rgb_img.shape[1]

        channels = {
            "red": (rgb_img[..., 0], "red"),
            "green": (rgb_img[..., 1], "green"),
            "blue": (rgb_img[..., 2], "blue"),
            "hue": (hsv[..., 0] * (255.0 / 179.0), "#7b68ee"),
            "saturation": (hsv[..., 1], "#00bcd4"),
            "value": (hsv[..., 2], "#ff9800"),
            "lightness": (lab[..., 0], "#666666"),
            "green-magenta": (lab[..., 1], "#ff00ff"),
            "blue-yellow": (lab[..., 2], "#d4d000"),
        }

        for label, (channel, color) in channels.items():
            data = channel[mask_idx].astype(np.float32)
            hist, bins = np.histogram(data, bins=256, range=(0, 255))
            ax.plot(bins[:-1], (hist / float(total)) * 100.0, color=color, lw=1, label=label)

        ax.set_title(title)
        ax.set_xlabel("Pixel intensity")
        ax.set_ylabel("Proportion of pixels (%)")
        ax.set_xlim(0, 255)
        ax.legend(title="color Channel", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        return fig

"""CLI orchestration for Leaffliction transformations."""

import argparse
import importlib.util
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from plantcv import plantcv as pcv

from transforms import (
    analyze_object_image,
    apply_blur,
    build_mask,
    contour_metrics,
    histogram_figure,
    largest_contour,
    make_roi_image,
    pseudolandmarks_image,
)

EXTS = {"jpg", "jpeg", "png", "bmp", "tif", "tiff"}
pcv.params.debug = None
pcv.params.verbose = False


def _load_io():
    spec = importlib.util.spec_from_file_location(
        "leaffliction_io", Path(__file__).with_name("io.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_io = _load_io()
load_image, save_image = _io.load_image, _io.save_image


def _figure_to_rgb(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    canvas = fig.canvas
    if hasattr(canvas, "buffer_rgba"):
        buf = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
        return buf.reshape(h, w, 4)[:, :, :3]
    if hasattr(canvas, "tostring_rgb"):
        buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        return buf.reshape(h, w, 3)
    buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
    return buf.reshape(h, w, 4)[:, :, 1:4]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Leaffliction Part 3: Image Transformation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  ./Transformation.py path/to/image.jpg\n"
            "  ./Transformation.py -src path/to/images_dir -dst path/to/output_dir --mask\n"
        ),
    )
    parser.add_argument("image", nargs="?", help="Path to a single image")
    parser.add_argument("-src", dest="src", help="Source directory")
    parser.add_argument("-dst", dest="dst", help="Destination directory")
    parser.add_argument(
        "--mask",
        action="store_true",
        help="Also save *_mask.png and *_mask_raw.png",
    )
    return parser


def _outputs(rgb, label):
    blur = apply_blur(rgb)
    mask = build_mask(rgb)
    contour = largest_contour(mask)

    obj_mask = mask.copy()
    if contour is not None:
        obj_mask = np.zeros_like(mask)
        cv2.drawContours(obj_mask, [contour], -1, 255, -1)

    masked_rgb = rgb.copy()
    masked_rgb[obj_mask == 0] = (255, 255, 255)

    out = {
        "rgb": rgb,
        "blur": blur,
        "mask_rgb": masked_rgb,
        "obj_mask": obj_mask,
        "roi": make_roi_image(rgb, contour),
        "analyze": analyze_object_image(rgb, obj_mask, contour, label=label),
        "pseudo": pseudolandmarks_image(rgb, obj_mask, label=label),
        "hist_fig": histogram_figure(rgb, obj_mask, title="Histogram"),
    }
    return out, contour


def _print_metrics(name, contour):
    metrics = contour_metrics(contour)
    if not metrics:
        print(f"{name} | No object detected for metrics")
        return
    cx, cy = metrics["centroid"]
    print(
        f"{name} | area={metrics['area']:.2f} perimeter={metrics['perimeter']:.2f} "
        f"width={metrics['width']} height={metrics['height']} centroid=({cx},{cy})"
    )


def process_single(image_path):
    _, rgb = load_image(image_path)
    base = os.path.splitext(os.path.basename(image_path))[0]
    out, contour = _outputs(rgb, base)
    _print_metrics(base, contour)

    hist_img = None
    if out["hist_fig"] is not None:
        hist_img = _figure_to_rgb(out["hist_fig"])
        plt.close(out["hist_fig"])

    items = [
        ("Original", out["rgb"]),
        ("Gaussian Blur", out["blur"]),
        ("Mask", out["mask_rgb"]),
        ("ROI", out["roi"]),
        ("Analyze", out["analyze"]),
        ("Pseudolandmarks", out["pseudo"]),
        ("Histogram", hist_img),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(14, 8))
    axes = axes.ravel()
    for ax, (title, img) in zip(axes, items):
        if img is None:
            ax.axis("off")
            continue
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    for ax in axes[len(items) :]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def process_directory(src_dir, dst_dir, save_mask=False):
    os.makedirs(dst_dir, exist_ok=True)

    for name in sorted(os.listdir(src_dir)):
        src_path = os.path.join(src_dir, name)
        if not os.path.isfile(src_path):
            continue
        if name.lower().rsplit(".", 1)[-1] not in EXTS:
            print(f"[SKIP] {name} (not an image)")
            continue

        try:
            _, rgb = load_image(src_path)
            base = os.path.splitext(name)[0]
            out, contour = _outputs(rgb, base)
            _print_metrics(f"[OK] {name}", contour)

            for suffix, img in (
                ("orig", out["rgb"]),
                ("blur", out["blur"]),
                ("roi", out["roi"]),
                ("analyze", out["analyze"]),
                ("pseudolandmarks", out["pseudo"]),
            ):
                save_image(os.path.join(dst_dir, f"{base}_{suffix}.png"), img)

            if out["hist_fig"] is not None:
                out["hist_fig"].savefig(
                    os.path.join(dst_dir, f"{base}_hist.png"),
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close(out["hist_fig"])

            if save_mask:
                save_image(os.path.join(dst_dir, f"{base}_mask.png"), out["mask_rgb"])
                cv2.imwrite(os.path.join(dst_dir, f"{base}_mask_raw.png"), out["obj_mask"])

        except Exception as exc:
            print(f"[ERR] {name} | {exc}")


def main():
    args = build_parser().parse_args()

    if args.image and args.src:
        raise SystemExit("Provide either a single image or -src, not both.")
    if not args.image and not args.src:
        raise SystemExit("Provide a single image or -src directory.")

    if args.src:
        if not args.dst:
            raise SystemExit("-dst is required when using -src.")
        if not os.path.isdir(args.src):
            raise SystemExit(f"Source directory not found: {args.src}")
        process_directory(args.src, args.dst, save_mask=args.mask)
        return

    if not os.path.isfile(args.image):
        raise SystemExit(f"Image not found: {args.image}")
    process_single(args.image)


if __name__ == "__main__":
    main()

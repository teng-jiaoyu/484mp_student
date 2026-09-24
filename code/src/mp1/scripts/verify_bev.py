"""Render captured images/masks through a BEV config without ROS or a model.

Example (from code/src/mp1):
    python scripts/verify_bev.py --config data/bev_config.local.json \
        --capture-dir /path/to/capture --output-dir data/bev_validation/local-gem
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from generate_bev_config import BEV_IMAGE_HEIGHT, BEV_IMAGE_WIDTH
from line_fit import perspective_transform
from lane_geometry import compute_lane_error, fit_lane_image, render_lane_bev


def load_config(path):
    config = json.loads(Path(path).read_text())
    src = np.asarray(config["src"], dtype=np.float32)
    if src.shape != (4, 2) or not np.isfinite(src).all():
        raise ValueError("src must contain four finite pixel coordinates")
    scale = np.asarray(config["unit_conversion_factor"], dtype=float)
    dimensions = np.asarray(config["bev_world_dim"], dtype=float)
    if scale.shape != (2,) or dimensions.shape != (2,):
        raise ValueError("BEV dimensions and scale must each contain two values")
    if not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
        raise ValueError("BEV dimensions must be finite and positive")
    if not np.allclose(scale, dimensions / [BEV_IMAGE_HEIGHT, BEV_IMAGE_WIDTH]):
        raise ValueError("BEV scale does not match the fixed 800x600 output")
    return config, src


def panel(image, title, scale=None):
    result = image.copy()
    if result.ndim == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    if scale is not None:
        # Grid is drawn on previews only, at one metre spacing.
        for x in np.arange(0, 800, 1 / scale[1]).astype(int):
            cv2.line(result, (x, 0), (x, 599), (65, 65, 65), 1)
        for y in np.arange(0, 600, 1 / scale[0]).astype(int):
            cv2.line(result, (0, y), (799, y), (65, 65, 65), 1)
        cv2.drawMarker(result, (400, 599), (255, 0, 255), cv2.MARKER_TRIANGLE_UP, 20, 2)
    result = cv2.copyMakeBorder(result, 36, 0, 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(result, title, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not save {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True, help="Contains images/ and masks/")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fit-lanes", action="store_true", help="Also fit lanes and compute XTE/HE")
    args = parser.parse_args()
    config, src = load_config(args.config)
    images = sorted((args.capture_dir / "images").glob("*.png"))
    if not images:
        raise ValueError(f"No PNG images under {args.capture_dir / 'images'}")
    expected_shape = (BEV_IMAGE_HEIGHT, BEV_IMAGE_WIDTH)
    samples = []
    for path in images:
        image = cv2.imread(str(path))
        mask_path = args.capture_dir / "masks" / path.name
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
        if image is None or image.shape[:2] != expected_shape:
            raise ValueError(f"Expected an 800x600 image: {path}; do not resize with unchanged K")
        if mask is None or mask.shape != expected_shape:
            raise ValueError(f"Expected a matching 800x600 mask: {mask_path}")
        if not np.isin(mask, [0, 1, 255]).all():
            raise ValueError(f"Expected binary 0/1 or 0/255 labels: {mask_path}")
        samples.append((path, image, (mask > 0).astype(np.uint8) * 255))

    # Validate all input pairs before creating output files.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"config": str(args.config.resolve()), "config_values": config, "samples": []}
    for path, image, mask in samples:
        bev, M, Minv = perspective_transform(image, src)
        if not np.isfinite(M).all() or not np.isfinite(Minv).all():
            raise ValueError("Invalid perspective matrix")
        # Preserve categorical labels; RGB interpolation in the supplied helper
        # is intentionally left unchanged.
        mask_bev = cv2.warpPerspective(mask, M, (800, 600), flags=cv2.INTER_NEAREST)
        visible = cv2.warpPerspective(np.full(expected_shape, 255, np.uint8), M, (800, 600),
                                      flags=cv2.INTER_NEAREST)
        # Independent image points, not only the four construction corners.
        probes = np.float32([[[50, 350], [400, 450], [750, 590]]])
        roundtrip = cv2.perspectiveTransform(cv2.perspectiveTransform(probes, M), Minv)
        error = float(np.max(np.linalg.norm(roundtrip - probes, axis=-1)))
        if error >= .1:
            raise ValueError(f"Round-trip error is {error:.6f}px")
        preview = np.vstack([
            np.hstack([panel(image, "Camera RGB | 800 x 600"),
                       panel(bev, "BEV RGB | 1 m grid | verify orientation", config["unit_conversion_factor"])]),
            np.hstack([panel(mask, "Camera lane mask"),
                       panel(mask_bev, "BEV mask | nearest-neighbour | black may be unseen")]),
        ])
        write_image(args.output_dir / f"{path.stem}_bev.png", bev)
        write_image(args.output_dir / f"{path.stem}_mask_bev.png", mask_bev)
        write_image(args.output_dir / f"{path.stem}_visible.png", visible)
        write_image(args.output_dir / f"{path.stem}_comparison.png", preview)
        sample = {
            "image": str(path.resolve()), "roundtrip_max_error_px": error,
            "visible_fraction": float(np.mean(visible > 0)),
            "mask_bev_values": np.unique(mask_bev).tolist(),
        }
        if args.fit_lanes:
            overlay, fitted_bev, ret = fit_lane_image(image, mask, config)
            metrics = None
            if ret is not None:
                metrics = compute_lane_error((ret["left_fit"] + ret["right_fit"]) / 2, config)
            sample["lane_geometry"] = {
                "detected": ret is not None,
                "xte_m": None if metrics is None else metrics[0],
                "he_rad": None if metrics is None else metrics[1],
                "he_deg": None if metrics is None else float(np.degrees(metrics[1])),
            }
            write_image(args.output_dir / f"{path.stem}_lanes.png", render_lane_bev(fitted_bev, ret, metrics))
            write_image(args.output_dir / f"{path.stem}_overlay.png", image if overlay is None else overlay)
        summary["samples"].append(sample)

    summary["limitations"] = [
        "Round-trip consistency does not prove physical calibration.",
        "Black regions outside visible.png are outside the source camera view.",
        "Sparse real samples do not validate lane width or full-track accuracy.",
    ]
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Validated and rendered {len(samples)} sample(s) to {args.output_dir}.")


if __name__ == "__main__":
    main()

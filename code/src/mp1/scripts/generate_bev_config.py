"""Generate the course BEV JSON, or an explicitly selected local GEM calibration."""
import argparse
import json
from pathlib import Path
import warnings

import numpy as np


BEV_IMAGE_HEIGHT, BEV_IMAGE_WIDTH = 600, 800
BEV_HEIGHT, BEV_WIDTH = 15, 20


def calibration(profile="template"):
    """Return K, R, t with the explicit convention P_optical = R @ P_base + t.

    The template constants are retained verbatim, not certified as calibrated.
    local-gem is derived from gem.urdf.xacro and gem.gazebo on this machine.
    """
    K = np.array([
        [476.7030836014194, 0.0, 400.0],
        [0.0, 476.7030836014194, 300.0],
        [0.0, 0.0, 1.0],
    ])
    R = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], dtype=float)
    t = np.array([0.160, -0.110, 1.546])
    if profile == "local-gem":
        # Base: forward/left/up. Optical: right/down/forward.
        # Camera origin: base_link joint (z=.44) + camera joint (.394,0,1.19).
        R = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=float)
        camera_in_base = np.array([0.394, 0.0, 0.44 + 1.19])
        t = -R @ camera_in_base
    elif profile != "template":
        raise ValueError(f"Unknown calibration profile: {profile}")
    return K, R, t


def project_points(points, K, R, t):
    """Project Nx3 base-frame points; keep off-image/behind-camera plane corners.

    Homography control points need not be visible. Zero optical depth, however,
    has no finite image coordinate and cannot be passed to OpenCV.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected an Nx3 array of base-frame points")
    camera_points = points @ R.T + t
    homogeneous = camera_points @ K.T
    if not np.isfinite(homogeneous).all():
        raise ValueError("Projection contains non-finite values")
    if np.any(np.abs(homogeneous[:, 2]) < 1e-9):
        raise ValueError("A control point has zero optical depth")
    return homogeneous[:, :2] / homogeneous[:, 2, None]


def build_config(profile="template"):
    K, R, t = calibration(profile)
    # Preserve the original template order and dimensions for the default profile.
    world_corners = np.array([
        [BEV_HEIGHT, -BEV_WIDTH / 2, 0],
        [0, -BEV_WIDTH / 2, 0],
        [0, BEV_WIDTH / 2, 0],
        [BEV_HEIGHT, BEV_WIDTH / 2, 0],
    ], dtype=float)
    if profile == "local-gem":
        # Physical left has positive base Y. Map far-left, near-left, near-right,
        # far-right to (0,0), (0,600), (800,600), (800,0), without a mirror.
        world_corners[:, 1] *= -1
    src = project_points(world_corners, K, R, t).astype(np.float32)
    return {
        "bev_world_dim": (BEV_HEIGHT, BEV_WIDTH),
        "unit_conversion_factor": (
            BEV_HEIGHT / BEV_IMAGE_HEIGHT, BEV_WIDTH / BEV_IMAGE_WIDTH
        ),
        "src": src.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("template", "local-gem"), default="template")
    parser.add_argument("--output", type=Path, help="JSON path, relative to current directory")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing generated config")
    args = parser.parse_args()
    output = build_config(args.profile)
    if args.profile == "template":
        warnings.warn(
            "Template R/t retained as P_camera = R @ P_base + t. "
            "Forward X=15 m has negative optical depth; this is NOT a verified "
            "calibration for the local simulator. Use --profile local-gem for local BEV.",
            stacklevel=1,
        )
    filename = "bev_config.json" if args.profile == "template" else "bev_config.local.json"
    save_fn = args.output or Path("data") / filename
    if save_fn.exists() and not args.force:
        if input(f"{save_fn} already exists. Overwrite? (y/n): ").lower() != "y":
            print("Exiting without changes.")
            return
    save_fn.parent.mkdir(parents=True, exist_ok=True)
    save_fn.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.profile} BEV config to {save_fn}.")


if __name__ == "__main__":
    main()

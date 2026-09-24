"""Model/ROS-independent lane geometry shared by live and offline processing."""
import cv2
import numpy as np

from line_fit import closest_point_on_polynomial, final_viz, lane_fit, perspective_transform


def _dimensions_and_scale(config):
    dimensions = np.asarray(config["bev_world_dim"], dtype=float)
    scale_yx = np.asarray(config["unit_conversion_factor"], dtype=float)
    if (dimensions.shape != (2,) or scale_yx.shape != (2,)
            or not np.isfinite(dimensions).all() or not np.isfinite(scale_yx).all()
            or np.any(dimensions <= 0) or np.any(scale_yx <= 0)):
        raise ValueError("BEV dimensions and metres-per-pixel scales must be positive and finite")
    return dimensions, scale_yx


def compute_lane_error(poly_px, config):
    """Return (signed XTE metres, HE radians, reference pixels, nearest pixels).

    BEV u points right; v points down. XTE is positive to the right of the
    forward-directed centreline. HE is path heading minus vehicle heading,
    positive for a path turning left. This matches -WorldGT.XTE and WorldGT.HE.
    The reference is the base origin at the BEV lower-edge centre, as in the
    course skeleton; it is not the physical camera mounting position.
    """
    dimensions, scale_yx = _dimensions_and_scale(config)
    coeffs = np.asarray(poly_px, dtype=float)
    if coeffs.shape != (3,) or not np.isfinite(coeffs).all():
        raise ValueError("Expected three finite coefficients of x = A*y^2 + B*y + C")
    height_m, width_m = dimensions
    Sy, Sx = scale_yx
    scale = np.array([Sx, Sy])
    reference_m = np.array([width_m / 2, height_m])
    # Solve in metres, so the nearest point is correct even if Sx != Sy.
    poly_m = coeffs * Sx / Sy ** np.array([2, 1, 0])
    closest_m = closest_point_on_polynomial(reference_m, poly_m)
    displacement = reference_m - closest_m
    slope = float(np.polyval(np.polyder(poly_m), closest_m[1]))
    xte = float(np.sign(displacement[0]) * np.linalg.norm(displacement))
    he = float(np.arctan(slope))
    if not np.isfinite([xte, he, *closest_m]).all():
        raise ValueError("Non-finite lane error")
    return xte, he, reference_m / scale, closest_m / scale


def fit_lane_image(raw_img, binary_img, config):
    """Camera RGB/mask -> (overlay or None, binary BEV, lane_fit result or None).

    Bad input raises ValueError. Valid frames without a credible pair of lanes
    return None for the overlay/result and never synthesize a missing lane.
    """
    dimensions, scale = _dimensions_and_scale(config)
    pixels = dimensions / scale
    shape = tuple(np.rint(pixels).astype(int))
    if not np.allclose(pixels, shape) or min(shape) <= 0:
        raise ValueError("BEV dimensions/scales must describe an integer image size")
    raw_img = np.asarray(raw_img)
    mask = np.asarray(binary_img)
    if raw_img.shape != (*shape, 3) or raw_img.dtype != np.uint8:
        raise ValueError(f"Expected uint8 BGR image of shape {(*shape, 3)}; keep calibration resolution")
    if mask.shape != shape or not np.isin(mask, [0, 1, 255]).all():
        raise ValueError(f"Expected binary 0/1 or 0/255 mask of shape {shape}")
    src = np.asarray(config["src"], dtype=np.float32)
    if src.shape != (4, 2) or not np.isfinite(src).all():
        raise ValueError("Expected four finite BEV source points")
    binary = (mask > 0).astype(np.uint8) * 255
    warped, _, inverse = perspective_transform(binary, src, interpolation=cv2.INTER_NEAREST)
    ret = lane_fit(warped)
    if ret is None:
        return None, warped, None
    overlay = final_viz(raw_img, ret["left_fit"], ret["right_fit"], inverse)
    return overlay, warped, ret


def render_lane_bev(binary_bev, ret=None, metrics=None):
    """Draw current-frame geometry, or an explicit N/A, without opening a window."""
    canvas = cv2.cvtColor(np.pad(binary_bev, ((0, 100), (0, 0))), cv2.COLOR_GRAY2BGR)
    label = "XTE: N/A   HE: N/A"
    if ret is not None and metrics is not None:
        xte, he, reference, closest = metrics
        centre = (ret["left_fit"] + ret["right_fit"]) / 2
        for coeffs, colour in [(ret["left_fit"], (255, 0, 0)),
                               (centre, (0, 255, 255)), (ret["right_fit"], (0, 0, 255))]:
            points = np.column_stack((np.polyval(coeffs, ret["ploty"]), ret["ploty"]))
            points = np.rint(np.clip(points, -1_000_000, 1_000_000)).astype(np.int32)
            cv2.polylines(canvas, [points], False, colour, 3)
        reference = tuple(np.rint(np.clip(reference, -1_000_000, 1_000_000)).astype(int))
        closest = tuple(np.rint(np.clip(closest, -1_000_000, 1_000_000)).astype(int))
        cv2.circle(canvas, closest, 7, (0, 255, 0), -1)
        cv2.line(canvas, reference, closest, (0, 255, 0), 3)
        for delta in [-20, 20]:
            cv2.line(canvas, reference, (reference[0] + delta, reference[1] + 20), (255, 0, 255), 3)
        label = f"XTE: {xte:+.3f} m   HE: {np.degrees(he):+.2f} deg"
    cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1] - 1, 590), 34), (0, 0, 0), -1)
    cv2.putText(canvas, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas

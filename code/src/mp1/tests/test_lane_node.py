"""Exercise actual LaneVisualizer methods with ROS messages and a stub model output.

Run in the configured ECE484 environment. Only model inference and GUI calls are
stubbed; conversion, geometry, rendering, and the image callback are production code.
"""
import contextlib
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    from lane_detect import LaneVisualizer
    from cv_bridge import CvBridge
    ROS_AVAILABLE = True
except ModuleNotFoundError:
    ROS_AVAILABLE = False


@unittest.skipUnless(ROS_AVAILABLE, "Source /home/jerry/Data/ece484/env.zsh for ROS callback tests")
class LaneNodeTests(unittest.TestCase):
    def make_context(self):
        # Identity camera->BEV config isolates callback behavior from calibration.
        config = {"bev_world_dim": [15, 20], "unit_conversion_factor": [.025, .025],
                  "src": [[0, 0], [0, 600], [800, 600], [800, 0]]}
        logger = Mock()
        ctx = SimpleNamespace(_bev_cfg=config, _cv_bridge=CvBridge(), _model=object(),
                              _dev="cpu", _odom_msg=None, _show_windows=False,
                              get_logger=lambda: logger)
        ctx.fit_poly_lanes = lambda raw, mask: LaneVisualizer.fit_poly_lanes(ctx, raw, mask)
        ctx.compute_error = lambda coeffs: LaneVisualizer.compute_error(ctx, coeffs)
        return ctx, logger

    def test_compute_error_wrapper(self):
        ctx, _ = self.make_context()
        xte, he, reference, closest = LaneVisualizer.compute_error(ctx, [0, 0, 360])
        self.assertAlmostEqual(xte, 1)
        self.assertAlmostEqual(he, 0)
        np.testing.assert_allclose(reference, [400, 600])
        np.testing.assert_allclose(closest, [360, 600])

    def test_callback_valid_then_missing_frame_has_no_stale_metrics(self):
        ctx, logger = self.make_context()
        raw = np.zeros((600, 800, 3), np.uint8)
        message = ctx._cv_bridge.cv2_to_imgmsg(raw, encoding="bgr8")
        mask = np.zeros((600, 800), np.uint8)
        mask[:, 291:298] = 1
        mask[:, 423:430] = 1  # boundaries=294/426, centre=360 -> XTE=+1 m
        text = io.StringIO()
        with patch("lane_detect.inference", side_effect=[mask, np.zeros_like(mask)]), \
                patch("lane_detect.cv2.imshow") as show, contextlib.redirect_stdout(text):
            LaneVisualizer._on_image(ctx, message)
            LaneVisualizer._on_image(ctx, message)
        lines = text.getvalue().splitlines()
        self.assertIn("EST XTE: 1.00 m", lines[0])
        self.assertIn("EST XTE: N/A m - HE: N/A", lines[1])
        show.assert_not_called()
        logger.warning.assert_not_called()

    def test_callback_bad_inference_is_reported_and_does_not_crash(self):
        ctx, logger = self.make_context()
        message = ctx._cv_bridge.cv2_to_imgmsg(np.zeros((600, 800, 3), np.uint8), encoding="bgr8")
        text = io.StringIO()
        with patch("lane_detect.inference", return_value=None), contextlib.redirect_stdout(text):
            LaneVisualizer._on_image(ctx, message)
        self.assertIn("EST XTE: N/A", text.getvalue())
        logger.warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()

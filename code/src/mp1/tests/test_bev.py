"""Geometry tests without ROS, model weights, or a running simulator."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from generate_bev_config import build_config, calibration, project_points
from line_fit import perspective_transform


class BEVTests(unittest.TestCase):
    def setUp(self):
        self.config = build_config("local-gem")
        self.src = np.float32(self.config["src"])
        _, self.M, self.Minv = perspective_transform(np.zeros((600, 800), np.uint8), self.src)

    def test_template_is_preserved_but_not_silently_certified(self):
        config = build_config()
        self.assertEqual(set(config), {"src", "bev_world_dim", "unit_conversion_factor"})
        self.assertEqual(config["bev_world_dim"], (15, 20))
        self.assertEqual(config["unit_conversion_factor"], (.025, .025))
        _, R, t = calibration()
        np.testing.assert_array_equal(t, [.160, -.110, 1.546])
        self.assertLess((R @ [15, 0, 0] + t)[2], 0)
        self.assertTrue(np.isfinite(config["src"]).all())

    def test_local_axes_and_metric_scale_independent_pinhole(self):
        # Independent optical model: u=cx-f*Y/(X-Cx), v=cy+f*Cz/(X-Cx).
        ground = np.array([[5, 0], [5, 1], [5, -1], [6, 0], [12, 3.]])
        f = 800 / (2 * np.tan(1.3962634 / 2))  # from Gazebo horizontal FOV
        pixel = np.column_stack((400 - f * ground[:, 1] / (ground[:, 0] - .394),
                                 300 + f * 1.63 / (ground[:, 0] - .394)))
        actual = cv2.perspectiveTransform(pixel.astype(np.float32)[None], self.M)[0]
        expected = np.column_stack((400 - 40 * ground[:, 1], 600 - 40 * ground[:, 0]))
        np.testing.assert_allclose(actual, expected, atol=.01, rtol=0)

    def test_corners_and_roundtrip(self):
        expected = np.float32([[0, 0], [0, 600], [800, 600], [800, 0]])
        actual = cv2.perspectiveTransform(self.src[None], self.M)[0]
        np.testing.assert_allclose(actual, expected, atol=.01)
        probes = np.float32([[[150, 380], [400, 490], [700, 590]]])
        recovered = cv2.perspectiveTransform(cv2.perspectiveTransform(probes, self.M), self.Minv)
        np.testing.assert_allclose(recovered, probes, atol=.01)

    def test_invalid_projection(self):
        K, R, t = calibration("local-gem")
        for point in [[[.394, 0, 0]], [[np.nan, 0, 0]], [[np.inf, 0, 0]]]:
            with self.subTest(point=point), self.assertRaises(ValueError):
                with np.errstate(invalid="ignore"):
                    project_points(point, K, R, t)
        with self.assertRaises(ValueError):
            project_points([1, 2, 3], K, R, t)

    def test_offscreen_and_behind_camera_control_points_are_kept(self):
        # The BEV lower edge is behind the camera mount, not an observed obstacle.
        self.assertGreater(np.max(np.abs(self.src)), 800)
        self.assertTrue(np.isfinite(self.src).all())

    def test_profile_rejects_typo(self):
        with self.assertRaises(ValueError):
            build_config("unknown")

    def test_cli_synthetic_parallel_lanes_and_mask_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            (root / "masks").mkdir()
            config_path = root / "config.json"
            subprocess.run([sys.executable, str(SCRIPTS / "generate_bev_config.py"),
                            "--profile", "local-gem", "--output", str(config_path)],
                           check=True, capture_output=True, text=True)
            mask = np.zeros((600, 800), np.uint8)
            X = np.linspace(3.2, 15, 300)
            f = 800 / (2 * np.tan(1.3962634 / 2))
            for Y in [-1.65, 1.65]:
                pixels = np.column_stack((400 - f * Y / (X - .394), 300 + f * 1.63 / (X - .394)))
                cv2.polylines(mask, [np.rint(pixels).astype(np.int32)], False, 255, 1)
            cv2.imwrite(str(root / "masks" / "0.png"), mask)
            cv2.imwrite(str(root / "images" / "0.png"), cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
            result = root / "result"
            cmd = [sys.executable, str(SCRIPTS / "verify_bev.py"), "--config", str(config_path),
                   "--capture-dir", str(root), "--output-dir", str(result)]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            bev = cv2.imread(str(result / "0_mask_bev.png"), cv2.IMREAD_GRAYSCALE)
            self.assertEqual(bev.shape, (600, 800))
            self.assertEqual(np.unique(bev).tolist(), [0, 255])
            for row in [150, 250, 350, 400]:
                # A one-pixel source line can skip a row under nearest-neighbour
                # resampling. Measure a narrow band, not a single raster row.
                _, xs = np.nonzero(bev[row - 2:row + 3])
                self.assertTrue(np.any(xs < 400) and np.any(xs > 400))
                self.assertAlmostEqual(float(np.mean(xs[xs < 400])), 334, delta=2)
                self.assertAlmostEqual(float(np.mean(xs[xs > 400])), 466, delta=2)
            summary = json.loads((result / "summary.json").read_text())
            self.assertLess(summary["samples"][0]["roundtrip_max_error_px"], .1)
            self.assertTrue((result / "0_comparison.png").is_file())
            # Wrong input resolution must fail, not silently invalidate calibration.
            cv2.imwrite(str(root / "images" / "0.png"), np.zeros((384, 640, 3), np.uint8))
            failed = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("Expected an 800x600 image", failed.stderr)


if __name__ == "__main__":
    unittest.main()

"""Analytic, synthetic-image, and ground-truth sign checks for lane geometry."""
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np
from scipy.optimize import minimize_scalar

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from generate_bev_config import build_config
from lane_geometry import compute_lane_error, fit_lane_image, render_lane_bev
from line_fit import closest_point_on_polynomial, lane_fit, perspective_transform
from worldgt import WorldGT


def draw_lanes(centre=(0, 0, 400), width=132, thickness=5):
    image = np.zeros((600, 800), np.uint8)
    y = np.arange(600)
    for delta in [-width / 2, width / 2]:
        points = np.column_stack((np.polyval(centre, y) + delta, y))
        cv2.polylines(image, [np.rint(points).astype(np.int32)], False, 255, thickness)
    return image


class ErrorTests(unittest.TestCase):
    def setUp(self):
        self.cfg = build_config("local-gem")

    def test_straight_signed_distance_and_reference(self):
        for c, expected in [(400, 0), (360, 1), (440, -1)]:
            with self.subTest(c=c):
                xte, he, reference, closest = compute_lane_error([0, 0, c], self.cfg)
                self.assertAlmostEqual(xte, expected)
                self.assertEqual(he, 0)
                np.testing.assert_allclose(reference, [400, 600])
                np.testing.assert_allclose(closest, [c, 600])

    def test_heading_both_directions(self):
        for slope in [-.1, .1]:
            xte, he, _, closest = compute_lane_error([0, slope, 400 - 600 * slope], self.cfg)
            self.assertAlmostEqual(xte, 0)
            self.assertAlmostEqual(he, np.arctan(slope))
            np.testing.assert_allclose(closest, [400, 600])

    def test_metric_nearest_point_with_unequal_scales(self):
        cfg = {"bev_world_dim": [15, 20], "unit_conversion_factor": [.05, .025]}
        poly = [0, .25, 320]
        m, c = .125, 8
        y = (15 + m * (10 - c)) / (1 + m*m)
        xte, he, reference, closest = compute_lane_error(poly, cfg)
        np.testing.assert_allclose(reference, [400, 300])
        np.testing.assert_allclose(closest * [.025, .05], [m*y + c, y])
        self.assertAlmostEqual(xte, (10 - m*15 - c) / np.sqrt(1 + m*m))
        self.assertAlmostEqual(he, np.arctan(m))

    def test_curve_matches_independent_distance_minimization(self):
        poly = [.001, -.6, 460]
        xte, he, _, closest = compute_lane_error(poly, self.cfg)
        def distance(y_m):
            x_m = .025 * np.polyval(poly, y_m / .025)
            return (x_m - 10)**2 + (y_m - 15)**2
        result = minimize_scalar(distance, bounds=(-50, 50), method="bounded",
                                 options={"xatol": 1e-10})
        self.assertAlmostEqual(abs(xte), np.sqrt(result.fun), places=7)
        self.assertAlmostEqual(closest[1] * .025, result.x, places=6)
        derivative = .002 * closest[1] - .6
        self.assertAlmostEqual(he, np.arctan(derivative))

    def test_nearest_point_chooses_global_minimum(self):
        closest = closest_point_on_polynomial([2, 0], [1, 0, 0])
        self.assertAlmostEqual(closest[0], 1.5)
        self.assertAlmostEqual(float(np.sum((closest - [2, 0])**2)), 1.75)
        np.testing.assert_allclose(closest_point_on_polynomial([7, 5], [0, 0, 3]), [3, 5])

    def test_signs_match_worldgt_including_yaw_wrap(self):
        for yaw in [-3.05, 0, 3.05]:
            for theta in [-.2, .2]:
                for offset in [-.8, .8]:
                    with self.subTest(yaw=yaw, theta=theta, offset=offset):
                        m = np.tan(theta)
                        intercept = offset / np.cos(theta)
                        local = np.array([[-100, -100*m + intercept], [100, 100*m + intercept]])
                        rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
                        world = WorldGT.__new__(WorldGT)
                        world._data = {"test": local @ rotation.T + [3, 4]}
                        _, _, raw_gt_xte, gt_he = world.get_metrics(3, 4, yaw)
                        poly = [0, m, (10 - 15*m - intercept) / .025]
                        xte, he, _, _ = compute_lane_error(poly, self.cfg)
                        self.assertAlmostEqual(xte, -raw_gt_xte, places=7)
                        self.assertAlmostEqual(he, gt_he, places=7)

    def test_invalid_coefficients_and_scale(self):
        for poly in [[1, 2], [0, np.nan, 400], [0, 0, np.inf]]:
            with self.subTest(poly=poly), self.assertRaises(ValueError):
                compute_lane_error(poly, self.cfg)
        for scale in [[0, .025], [-.025, .025], [.025, np.nan]]:
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                compute_lane_error([0, 0, 400], {**self.cfg, "unit_conversion_factor": scale})


class LaneFitTests(unittest.TestCase):
    def test_straights_offsets_turns_and_disjoint_pixels(self):
        for poly in [[0, 0, 400], [0, 0, 496], [0, 0, 304],
                     [0, .1, 340], [0, -.1, 460], [.0004, -.24, 436], [-.0004, .24, 364]]:
            with self.subTest(poly=poly):
                ret = lane_fit(draw_lanes(poly))
                self.assertIsNotNone(ret)
                y = np.arange(600)
                np.testing.assert_allclose(ret["left_fitx"], np.polyval(poly, y) - 66, atol=1)
                np.testing.assert_allclose(ret["right_fitx"], np.polyval(poly, y) + 66, atol=1)
                self.assertEqual(np.intersect1d(ret["left_lane_inds"], ret["right_lane_inds"]).size, 0)
                for key in ["left_lane_inds", "right_lane_inds"]:
                    self.assertEqual(np.unique(ret[key]).size, ret[key].size)

    def test_gaps_noise_and_empty_near_field(self):
        poly = [.0003, -.18, 427]
        mask = draw_lanes(poly)
        mask[260:350] = 0
        mask[480:] = 0
        rng = np.random.default_rng(484)
        mask[rng.integers(0, 480, 400), rng.integers(0, 800, 400)] = 255
        ret = lane_fit(mask)
        self.assertIsNotNone(ret)
        centre = (ret["left_fit"] + ret["right_fit"]) / 2
        np.testing.assert_allclose(np.polyval(centre, np.arange(600)),
                                   np.polyval(poly, np.arange(600)), atol=2)

    def test_missing_single_crossing_short_and_noise_are_rejected(self):
        blank = np.zeros((600, 800), np.uint8)
        single = blank.copy()
        cv2.line(single, (400, 0), (400, 599), 255, 5)
        diagonal = blank.copy()
        cv2.line(diagonal, (0, 0), (799, 599), 255, 5)
        crossing = blank.copy()
        cv2.line(crossing, (300, 0), (500, 599), 255, 5)
        cv2.line(crossing, (500, 0), (300, 599), 255, 5)
        short = draw_lanes()
        short[:550] = 0
        noise = blank.copy()
        rng = np.random.default_rng(0)
        noise[rng.integers(0, 600, 500), rng.integers(0, 800, 500)] = 255
        for name, mask in [("blank", blank), ("single", single), ("diagonal", diagonal),
                           ("crossing", crossing), ("short", short), ("noise", noise)]:
            with self.subTest(name=name):
                self.assertIsNone(lane_fit(mask))

    def test_end_to_end_camera_mask_to_errors_and_overlay(self):
        cfg = build_config("local-gem")
        _, _, inv = perspective_transform(np.zeros((600, 800), np.uint8), np.float32(cfg["src"]))
        for poly in [[0, 0, 400], [0, 0, 440], [0, .1, 340], [.0003, -.18, 427]]:
            with self.subTest(poly=poly):
                camera_mask = cv2.warpPerspective(draw_lanes(poly), inv, (800, 600), flags=cv2.INTER_NEAREST)
                raw = np.zeros((600, 800, 3), np.uint8)
                for encoding in [camera_mask, camera_mask // 255]:
                    overlay, binary, ret = fit_lane_image(raw, encoding, cfg)
                    self.assertIsNotNone(ret)
                    self.assertEqual(overlay.shape, raw.shape)
                    self.assertEqual(overlay.dtype, np.uint8)
                    self.assertTrue(np.any(overlay[:, :, 1] > 0))
                    self.assertTrue(np.isin(binary, [0, 255]).all())
                    estimated = compute_lane_error((ret["left_fit"] + ret["right_fit"]) / 2, cfg)
                    expected = compute_lane_error(poly, cfg)
                    self.assertAlmostEqual(estimated[0], expected[0], delta=.06)
                    self.assertAlmostEqual(estimated[1], expected[1], delta=.02)
                    self.assertEqual(render_lane_bev(binary, ret, estimated).shape, (700, 800, 3))

    def test_bad_inputs_and_empty_frame_contract(self):
        cfg = build_config("local-gem")
        raw = np.zeros((600, 800, 3), np.uint8)
        for mask in [None, np.zeros((384, 640)), np.full((600, 800), .5), np.full((600, 800), np.nan)]:
            with self.subTest(shape=np.shape(mask)), self.assertRaises(ValueError):
                fit_lane_image(raw, mask, cfg)
        overlay, binary, ret = fit_lane_image(raw, np.zeros((600, 800), np.uint8), cfg)
        self.assertIsNone(overlay)
        self.assertIsNone(ret)
        self.assertFalse(binary.any())
        self.assertEqual(render_lane_bev(binary).shape, (700, 800, 3))


if __name__ == "__main__":
    unittest.main()

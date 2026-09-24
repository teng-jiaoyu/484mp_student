import os

import torch
import json
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
from rclpy.parameter import Parameter

from worldgt import WorldGT
from lane_geometry import compute_lane_error, fit_lane_image, render_lane_bev
from model_utils import load_model, inference

import rich
import cv2
from scipy.spatial.transform import Rotation as R


class LaneVisualizer(Node):
    def __init__(self):
        super().__init__("lane_visualizer")

        sim_time_param = Parameter(
            "use_sim_time",
            Parameter.Type.BOOL,
            True
        )
        self.set_parameters([sim_time_param])
        self.declare_parameter("bev_config", os.path.join("data", "bev_config.json"))
        self.declare_parameter("show_windows", True)
        self._show_windows = self.get_parameter("show_windows").value

        self._dev = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        try:
            self._model = load_model()

            if self._model is not None:
                self._model = self._model.to(self._dev)
                self._model = self._model.eval()
                rich.print("[green]loaded SimpleEnet :o")
            else:
                self.get_logger().error(
                    "could not load SimpleEnet model x_X"
                )
                exit(1)

        except Exception as e:
            self.get_logger().error(
                f"could not load SimpleEnet model x_X: {e}"
            )
            exit(1)

        try:
            with open(
                self.get_parameter("bev_config").value
            ) as f:
                self._bev_cfg = json.load(f)

        except FileNotFoundError as e:
            self.get_logger().error(
                f"could not load bev config x_X: {e}"
            )
            exit(1)

        self._world = WorldGT("Silverstone")

        self._image_msg = None
        self._odom_msg = None

        self._cv_bridge = CvBridge()

        self.create_subscription(
            Image,
            "/camera/image_raw",
            self._on_image,
            10
        )

        self.create_subscription(
            Odometry,
            "/odom",
            self._on_odom,
            10
        )

    def _on_odom(self, msg) -> None:
        self._odom_msg = msg

    def _on_image(self, msg) -> None:
        self._image_msg = msg

        if self._model is None:
            return

        try:
            image = self._cv_bridge.imgmsg_to_cv2(self._image_msg, "bgr8")
        except CvBridgeError as exc:
            self.get_logger().warning(f"Could not decode camera image: {exc}")
            return

        ret = None
        metrics = None
        combine_fit_img = None
        binary_BEV = np.zeros(image.shape[:2], dtype=np.uint8)
        try:
            # Accept either 0/1 or 0/255 masks; normalize once in the shared core.
            mask = inference(self._model, image, self._dev)
            combine_fit_img, binary_BEV, ret = self.fit_poly_lanes(image, mask)
            if ret is not None:
                poly_px = (ret["left_fit"] + ret["right_fit"]) / 2
                metrics = self.compute_error(poly_px)
        except (ValueError, TypeError, KeyError, RuntimeError, cv2.error,
                np.linalg.LinAlgError, FloatingPointError) as exc:
            self.get_logger().warning(f"Lane geometry unavailable for this frame: {exc}")
            ret = None
            metrics = None
            combine_fit_img = None

        binary_BEV = render_lane_bev(binary_BEV, ret, metrics)
        if metrics is None:
            XTE = HE = "N/A"
        else:
            XTE = f"{metrics[0]:.2f}"
            HE = f"{np.degrees(metrics[1]):.2f}"

        # Ground-truth metrics from odometry
        try:
            if self._odom_msg is None:
                raise RuntimeError(
                    "No odometry received yet"
                )

            pos = self._odom_msg.pose.pose.position
            q = self._odom_msg.pose.pose.orientation

            rotation = R.from_quat([
                q.x,
                q.y,
                q.z,
                q.w
            ])

            euler_angles = rotation.as_euler(
                "xyz",
                degrees=False
            )

            yaw = euler_angles[2]

            lane, _, gt_XTE, gt_HE = self._world.get_metrics(
                pos.x,
                pos.y,
                yaw
            )

            gt_XTE = f"{-gt_XTE:.2f}"
            gt_HE = f"{np.degrees(gt_HE):.2f}"

        except Exception as e:
            self.get_logger().debug(
                f"Could not compute ground truth: {e}"
            )

            lane = "unknown"
            gt_XTE = "N/A"
            gt_HE = "N/A"

        print(
            f"EST XTE: {XTE} m - "
            f"HE: {HE}° -- "
            f"GT XTE: {gt_XTE} m "
            f"HE: {gt_HE}° - "
            f"lane: {lane}"
        )

        if combine_fit_img is None:
            combine_fit_img = image

        if self._show_windows:
            cv2.imshow("render_view", combine_fit_img)
            cv2.imshow("binary_BEV", binary_BEV)
            cv2.waitKey(1)

    def compute_error(self, poly_px):
        """Return XTE (m), HE (rad), reference and closest point (BEV pixels).

        Shared with offline verification: BEV right/down coordinates, signed
        distance to the centreline, heading from the tangent at its nearest point.
        """
        return compute_lane_error(poly_px, self._bev_cfg)

    def fit_poly_lanes(self, raw_img, binary_img):
        result = fit_lane_image(raw_img, binary_img, self._bev_cfg)
        if result[2] is None:
            self.get_logger().debug("No reliable pair of lane boundaries in this frame.")
        return result


def main(args=None):
    rclpy.init(args=args)

    node = LaneVisualizer()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if node._show_windows:
            cv2.destroyAllWindows()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

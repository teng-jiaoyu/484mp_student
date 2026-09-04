import argparse
import os
from pynput import keyboard
import torch
import rich
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from ackermann_msgs.msg import AckermannDrive
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy
from rclpy.parameter import Parameter

from dataset import CaptureDataset
from model_utils import load_model, inference


class DriveControl(Node):
    def __init__(self, args) -> None:
        super().__init__("DriveControl")

        sim_time_param = Parameter(
            "use_sim_time",
            Parameter.Type.BOOL,
            True
        )
        self.set_parameters([sim_time_param])

        self._image_msg = None
        self._odom_msg = None

        self._cap_ds = CaptureDataset(args.dataset_path)
        self._keypresses = set()
        self._cv_bridge = CvBridge()
        self._max_speed = args.max_speed
        self._max_steer = args.max_steer

        if args.run_model:
            self._dev = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

            try:
                self._model = load_model()

                if self._model is not None:
                    self._model = self._model.to(self._dev)
                    rich.print("[green]loaded SimpleEnet :o")

            except Exception as e:
                self.get_logger().error(
                    f"could not load SimpleEnet model x_X: {e}"
                )
                self._model = None
        else:
            self._model = None

        streaming_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE
        )

        self._mask_pub = self.create_publisher(
            Image,
            args.mask_topic,
            10
        )

        self.create_subscription(
            Image,
            args.camera_topic,
            self._on_image,
            10
        )

        self.create_subscription(
            Odometry,
            args.odom_topic,
            self._on_odom,
            10
        )

        self._drive_pub = self.create_publisher(
            AckermannDrive,
            args.ackermann_topic,
            streaming_qos
        )

        self.create_timer(
            1 / 50,
            self._timer_callback
        )

        self._keylogger = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self._keylogger.start()

    def _timer_callback(self) -> None:
        speed = 0.0
        steer = 0.0

        if "w" in self._keypresses:
            speed = self._max_speed
        elif "s" in self._keypresses:
            speed = -self._max_speed

        if "a" in self._keypresses:
            steer = self._max_steer
        elif "d" in self._keypresses:
            steer = -self._max_steer

        msg = AckermannDrive()
        msg.speed = speed
        msg.steering_angle = steer
        msg.steering_angle_velocity = 0.0
        msg.acceleration = 0.0

        self._drive_pub.publish(msg)

    def _on_press(self, key):
        try:
            c = key.char

            if c is None:
                return

        except Exception:
            return

        if c in "wasd":
            self._keypresses.add(c)

        else:
            match c:
                case "e":
                    self._capture()

                case "q":
                    self._shutdown()

    def _on_release(self, key):
        try:
            c = key.char

            if c is None:
                return

        except Exception:
            return

        if c in "wasd":
            self._keypresses.discard(c)

    def _on_image(self, msg) -> None:
        self._image_msg = msg

        if self._model is not None:
            image = self._cv_bridge.imgmsg_to_cv2(
                self._image_msg,
                "bgr8"
            )

            mask = inference(
                self._model,
                image,
                self._dev
            )

            m = mask.astype(np.uint8) * 255

            mask_msg = self._cv_bridge.cv2_to_imgmsg(
                m,
                "mono8"
            )

            self._mask_pub.publish(mask_msg)

    def _on_odom(self, msg) -> None:
        self._odom_msg = msg

    def _capture(self) -> None:
        if self._image_msg is None:
            self.get_logger().error(
                "Cannot capture: no camera image received yet."
            )
            return

        if self._odom_msg is None:
            self.get_logger().error(
                "Cannot capture: no odometry received yet."
            )
            return

        try:
            image = self._cv_bridge.imgmsg_to_cv2(
                self._image_msg,
                "bgr8"
            )

            pos = self._odom_msg.pose.pose.position
            rot = self._odom_msg.pose.pose.orientation

            pose = {
                "x": pos.x,
                "y": pos.y,
                "z": pos.z,
                "qx": rot.x,
                "qy": rot.y,
                "qz": rot.z,
                "qw": rot.w
            }

            self._cap_ds.capture(
                image,
                pose
            )

            print(
                f"\nCapturing image "
                f"(total: {len(self._cap_ds)}) "
                f"at "
                f"({pose['x']:.3f}, "
                f"{pose['y']:.3f}, "
                f"{pose['z']:.3f})"
            )

        except Exception as e:
            self.get_logger().error(
                f"Could not capture image/odometry: {e}"
            )

    def _shutdown(self):
        print("Exiting ...")
        self._keylogger.stop()
        rclpy.shutdown()


def print_instructions(args):
    print(
        f"""
Ackermann Drive Keyboard Teleop Control
----------------------------------------
Control Keys:
  w - Move forward (hold)
  s - Move backward (hold)
  a - Steer left (hold)
  d - Steer right (hold)
  q - Quit
  e - Capture and save current camera image

Current Settings:
  Max Speed: {args.max_speed} m/s
  Max Steering Angle: {args.max_steer} rad
  Camera Topic: {args.camera_topic}
  Odometry Topic: {args.odom_topic}
  Ackermann Topic: {args.ackermann_topic}
"""
    )


def main():
    parser = argparse.ArgumentParser(
        description="Teleop control of GEM"
    )

    parser.add_argument(
        "--dataset_path",
        default=os.path.join(
            "data",
            "capture"
        ),
        help="Path to store captured data"
    )

    parser.add_argument(
        "--camera_topic",
        default="/camera/image_raw",
        help="Topic name of camera stream"
    )

    parser.add_argument(
        "--odom_topic",
        default="/odom",
        help="Topic name of odometry"
    )

    parser.add_argument(
        "--ackermann_topic",
        default="/ackermann_cmd",
        help="Topic name of ackermann cmd topic"
    )

    parser.add_argument(
        "--max_speed",
        type=float,
        default=5.0,
        help="Max speed of GEM"
    )

    parser.add_argument(
        "--max_steer",
        type=float,
        default=0.8,
        help="Max steering angle of GEM"
    )

    parser.add_argument(
        "--mask_topic",
        default="/lane_mask",
        help="Publishing topic of lane line mask"
    )

    parser.add_argument(
        "--run_model",
        action="store_true",
        help="Run SimpleEnet inference"
    )

    args = parser.parse_args()

    print_instructions(args)

    rclpy.init()

    node = DriveControl(args)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
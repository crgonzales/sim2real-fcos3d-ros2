#!/usr/bin/env python3
"""Replay nuScenes camera samples onto ROS 2 topics.

Publishes one camera's keyframes as sensor_msgs/Image + sensor_msgs/CameraInfo
so the FCOS3D detector node consumes real data through exactly the same
interface it would use for Isaac Sim output.

Also publishes the ground-truth 3D boxes for the current sample on a separate
topic, which gives us a live visual check that predictions and GT share a
coordinate frame -- the failure mode that silently corrupts 3D evaluation.

Usage:
    ros2 run fcos3d_ros2 nuscenes_publisher --ros-args \
        -p dataroot:=/workspace/data/nuscenes \
        -p version:=v1.0-mini \
        -p rate_hz:=2.0

Contributor: Carlos Gonzales
"""

import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Vector3
from cv_bridge import CvBridge

import cv2
from pyquaternion import Quaternion


class NuScenesPublisher(Node):

    def __init__(self):
        super().__init__('nuscenes_publisher')

        self.declare_parameter('dataroot', '/workspace/data/nuscenes')
        self.declare_parameter('version', 'v1.0-mini')
        self.declare_parameter('camera', 'CAM_FRONT')
        self.declare_parameter('rate_hz', 2.0)
        self.declare_parameter('loop', True)
        self.declare_parameter('frame_id', 'camera_front_optical')
        self.declare_parameter('image_topic', '/camera/front/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/front/camera_info')
        self.declare_parameter('gt_topic', '/nuscenes/gt_markers')
        self.declare_parameter('publish_gt', True)
        # Photometric sweep: emulate lighting change without a simulator.
        # 1.0 = unmodified. See eval/lighting_sweep.md for why this exists.
        self.declare_parameter('brightness_gain', 1.0)
        self.declare_parameter('gamma', 1.0)

        gp = self.get_parameter
        self.camera = gp('camera').value
        self.frame_id = gp('frame_id').value
        self.loop = gp('loop').value
        self.publish_gt = gp('publish_gt').value
        self.gain = float(gp('brightness_gain').value)
        self.gamma = float(gp('gamma').value)

        from nuscenes.nuscenes import NuScenes
        self.get_logger().info('Loading nuScenes index (this takes a moment)...')
        self.nusc = NuScenes(
            version=gp('version').value,
            dataroot=gp('dataroot').value,
            verbose=False)

        # Flatten every scene's keyframes into one ordered list.
        # NB: this spans ALL mini scenes, i.e. mini_train and mini_val together.
        # That is fine for the live demo, which is this node's only purpose --
        # every reported accuracy figure comes from tools/test.py on mini_val,
        # not from this path. Do not evaluate through here without splitting.
        self.samples = [s for s in self.nusc.sample]
        if not self.samples:
            raise RuntimeError('No samples found -- check dataroot/version')
        self.get_logger().info(
            f'{len(self.samples)} keyframes across '
            f'{len(self.nusc.scene)} scenes, camera={self.camera}')

        self.bridge = CvBridge()
        self.idx = 0

        self.pub_img = self.create_publisher(Image, gp('image_topic').value, 10)
        self.pub_info = self.create_publisher(
            CameraInfo, gp('camera_info_topic').value, 10)
        self.pub_gt = self.create_publisher(MarkerArray, gp('gt_topic').value, 10)

        period = 1.0 / max(float(gp('rate_hz').value), 0.01)
        self.timer = self.create_timer(period, self.tick)
        self.get_logger().info(f'Publishing at {gp("rate_hz").value} Hz')

    # ------------------------------------------------------------------ helpers
    def _photometric(self, img: np.ndarray) -> np.ndarray:
        """Apply the controlled brightness/gamma perturbation, if any."""
        if self.gain == 1.0 and self.gamma == 1.0:
            return img
        out = img.astype(np.float32) / 255.0
        if self.gamma != 1.0:
            out = np.power(out, self.gamma)
        if self.gain != 1.0:
            out = out * self.gain
        return np.clip(out * 255.0, 0, 255).astype(np.uint8)

    # --------------------------------------------------------------------- tick
    def tick(self):
        if self.idx >= len(self.samples):
            if not self.loop:
                self.get_logger().info('Replay complete')
                self.timer.cancel()
                return
            self.idx = 0

        sample = self.samples[self.idx]
        self.idx += 1

        cam_token = sample['data'].get(self.camera)
        if cam_token is None:
            self.get_logger().warn(f'{self.camera} missing on this sample')
            return

        sd = self.nusc.get('sample_data', cam_token)
        cs = self.nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        img_path = self.nusc.get_sample_data_path(cam_token)

        img = cv2.imread(img_path)                # BGR, as FCOS3D expects
        if img is None:
            self.get_logger().warn(f'Failed to read {img_path}')
            return
        img = self._photometric(img)

        stamp = self.get_clock().now().to_msg()

        msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id

        K = np.array(cs['camera_intrinsic'], dtype=np.float64)
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = self.frame_id
        info.height, info.width = img.shape[:2]
        info.k = K.reshape(-1).tolist()
        info.p = np.hstack([K, np.zeros((3, 1))]).reshape(-1).tolist()
        info.distortion_model = 'plumb_bob'
        info.d = [0.0] * 5

        # CameraInfo first: the detector drops frames until it has intrinsics.
        self.pub_info.publish(info)
        self.pub_img.publish(msg)

        if self.publish_gt:
            self.pub_gt.publish(self._gt_markers(cam_token, stamp))

        if self.idx % 20 == 0:
            self.get_logger().info(f'published {self.idx}/{len(self.samples)}')

    # ----------------------------------------------------------------- ground truth
    def _gt_markers(self, cam_token, stamp) -> MarkerArray:
        """GT boxes transformed into the camera frame.

        get_sample_data with box_vis_level returns boxes already moved into the
        sensor frame, which is the same convention FCOS3D predicts in -- so
        these markers and the detector's markers are directly comparable in a
        viewer. If they disagree visually, the frame handling is wrong.
        """
        from nuscenes.utils.geometry_utils import BoxVisibility

        markers = MarkerArray()
        clear = Marker()
        clear.header.stamp = stamp
        clear.header.frame_id = self.frame_id
        clear.ns = 'nuscenes_gt'
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        try:
            _, boxes, _ = self.nusc.get_sample_data(
                cam_token, box_vis_level=BoxVisibility.ANY)
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f'GT lookup failed: {e}')
            return markers

        for i, box in enumerate(boxes):
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = self.frame_id
            m.ns = 'nuscenes_gt'
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = float(box.center[0])
            m.pose.position.y = float(box.center[1])
            m.pose.position.z = float(box.center[2])
            q: Quaternion = box.orientation
            m.pose.orientation.w = float(q.w)
            m.pose.orientation.x = float(q.x)
            m.pose.orientation.y = float(q.y)
            m.pose.orientation.z = float(q.z)
            # nuScenes Box.wlh unpacks as (width, length, height), but the
            # box's LOCAL axes -- the frame box.orientation rotates -- are
            # x=length, y=width, z=height (see Box.corners in the devkit:
            # x_corners uses l/2, y_corners uses w/2, z_corners uses h/2).
            # Marker scale is applied in that same local frame, so it must be
            # (l, w, h). Publishing (w, h, l) permuted every GT box.
            w, l, h = box.wlh
            m.scale = Vector3(x=float(l), y=float(w), z=float(h))
            m.color = ColorRGBA(r=0.15, g=1.0, b=0.35, a=0.25)
            m.lifetime.sec = 1
            markers.markers.append(m)

        return markers


def main(args=None):
    rclpy.init(args=args)
    node = NuScenesPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

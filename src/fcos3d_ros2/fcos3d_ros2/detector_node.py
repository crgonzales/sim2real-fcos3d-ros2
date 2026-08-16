#!/usr/bin/env python3
"""FCOS3D monocular 3D object detection as a ROS 2 node.

Subscribes to a camera stream, runs the pretrained FCOS3D detector from
MMDetection3D, and publishes 3D detections plus RViz2 markers.

Why this does not call ``mmdet3d.apis.inference_mono_3d_detector``
------------------------------------------------------------------
That helper requires an annotation *file on disk*, loads ``data_list`` from
it, and asserts the image basename matches an entry (mmdet3d 1.4.0,
inference.py:342). It is built for offline batch inference over a prepared
dataset and cannot consume an in-memory ``sensor_msgs/Image``. So we rebuild
the (short) test pipeline ourselves and inject the decoded frame directly.

For the FCOS3D config the test pipeline is only:
    LoadImageFromFileMono3D -> mmdet.Resize(scale_factor=1.0)
                            -> Pack3DDetInputs(keys=['img'])
We replace the first transform by constructing the dict it would have
produced (img, img_shape, ori_shape, cam2img) from the ROS messages.

Contributor: Carlos Gonzales
"""

import time
from collections import deque
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import (
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
    BoundingBox3D,
)
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Vector3
from std_msgs.msg import ColorRGBA
from cv_bridge import CvBridge

import torch
from mmengine.dataset import Compose, pseudo_collate
from mmdet3d.apis import init_model
from mmdet3d.structures import get_box_type


# nuScenes class order used by the FCOS3D head. Index -> name.
NUSCENES_CLASSES = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier',
]

# Stable per-class marker colors (RGBA), so the demo video reads clearly.
CLASS_COLORS = {
    'car':                  (0.20, 0.80, 1.00),
    'truck':                (0.00, 0.55, 1.00),
    'trailer':              (0.40, 0.40, 0.95),
    'bus':                  (0.60, 0.30, 1.00),
    'construction_vehicle': (0.95, 0.65, 0.20),
    'bicycle':              (0.20, 1.00, 0.55),
    'motorcycle':           (0.10, 0.90, 0.35),
    'pedestrian':           (1.00, 0.30, 0.35),
    'traffic_cone':         (1.00, 0.85, 0.10),
    'barrier':              (0.75, 0.75, 0.80),
}


class Fcos3dDetectorNode(Node):

    def __init__(self):
        super().__init__('fcos3d_detector')

        self.declare_parameter('config_file', '')
        self.declare_parameter('checkpoint_file', '')
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('fp16', False)
        self.declare_parameter('score_threshold', 0.3)
        self.declare_parameter('image_topic', '/camera/front/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/front/camera_info')
        self.declare_parameter('detections_topic', '/perception/detections_3d')
        self.declare_parameter('markers_topic', '/perception/markers')
        self.declare_parameter('marker_lifetime_sec', 0.2)
        self.declare_parameter('log_every_n', 20)

        gp = self.get_parameter
        self.config_file = gp('config_file').value
        self.checkpoint_file = gp('checkpoint_file').value
        self.device = gp('device').value
        self.fp16 = gp('fp16').value
        self.score_threshold = gp('score_threshold').value
        self.marker_lifetime = gp('marker_lifetime_sec').value
        self.log_every_n = gp('log_every_n').value

        if not self.config_file or not self.checkpoint_file:
            raise RuntimeError(
                'config_file and checkpoint_file parameters are required')

        self.bridge = CvBridge()
        self.cam2img: Optional[np.ndarray] = None
        self.frame_count = 0
        # Rolling windows for the latency report (requirement 4).
        self.lat_preprocess = deque(maxlen=500)
        self.lat_inference = deque(maxlen=500)
        self.lat_postprocess = deque(maxlen=500)
        self.lat_end_to_end = deque(maxlen=500)

        self._load_model()

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.pub_det = self.create_publisher(
            Detection3DArray, gp('detections_topic').value, 10)
        self.pub_markers = self.create_publisher(
            MarkerArray, gp('markers_topic').value, 10)

        # CameraInfo is latched-ish and low rate; keep it reliable.
        self.create_subscription(
            CameraInfo, gp('camera_info_topic').value, self.on_camera_info, 10)
        self.create_subscription(
            Image, gp('image_topic').value, self.on_image, sensor_qos)

        self.get_logger().info(
            f"FCOS3D node ready. device={self.device} fp16={self.fp16} "
            f"score_thr={self.score_threshold}")
        self.get_logger().info(
            f"Waiting for CameraInfo on {gp('camera_info_topic').value} ...")

    # ------------------------------------------------------------------ setup
    def _load_model(self):
        t0 = time.perf_counter()
        self.model = init_model(
            self.config_file, self.checkpoint_file, device=self.device)

        if self.fp16:
            # Half precision for the FP32-vs-FP16 comparison. DCNv2 in the
            # FCOS3D backbone is fp16-safe under autocast but not always under
            # a blanket .half(), so we use autocast at inference instead.
            self.get_logger().info('FP16 enabled via torch.autocast')

        cfg = self.model.cfg
        # Drop the file-loading transform; we supply the image in memory.
        pipeline_cfg = [
            t for t in cfg.test_dataloader.dataset.pipeline
            if t['type'] not in ('LoadImageFromFileMono3D', 'LoadImageFromFile')
        ]
        self.test_pipeline = Compose(pipeline_cfg)
        self.box_type_3d, self.box_mode_3d = get_box_type(
            cfg.test_dataloader.dataset.box_type_3d)

        classes = cfg.test_dataloader.dataset.get(
            'metainfo', {}).get('classes', NUSCENES_CLASSES)
        self.classes = list(classes)

        dt = time.perf_counter() - t0
        self.get_logger().info(
            f'Model loaded in {dt:.1f}s with {len(self.classes)} classes')

    # --------------------------------------------------------------- callbacks
    def on_camera_info(self, msg: CameraInfo):
        if self.cam2img is not None:
            return
        k = np.array(msg.k, dtype=np.float32).reshape(3, 3)
        if not np.any(k):
            self.get_logger().warn('CameraInfo.k is all zeros; ignoring')
            return
        self.cam2img = k
        self.get_logger().info(
            f'Got camera intrinsics: fx={k[0,0]:.1f} fy={k[1,1]:.1f} '
            f'cx={k[0,2]:.1f} cy={k[1,2]:.1f}')

    def on_image(self, msg: Image):
        if self.cam2img is None:
            self.get_logger().warn(
                'Dropping frame: no CameraInfo yet', throttle_duration_sec=5.0)
            return

        t_start = time.perf_counter()

        # FCOS3D uses caffe-style BGR normalization (bgr_to_rgb=False,
        # mean=[103.530, 116.280, 123.675]). Decode to bgr8 -- converting to
        # RGB here silently degrades accuracy.
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        data = self._preprocess(img)
        t_pre = time.perf_counter()

        result = self._infer(data)
        t_inf = time.perf_counter()

        det_array, markers = self._to_ros(result, msg.header)
        self.pub_det.publish(det_array)
        self.pub_markers.publish(markers)
        t_end = time.perf_counter()

        self.lat_preprocess.append((t_pre - t_start) * 1e3)
        self.lat_inference.append((t_inf - t_pre) * 1e3)
        self.lat_postprocess.append((t_end - t_inf) * 1e3)
        self.lat_end_to_end.append((t_end - t_start) * 1e3)
        self.frame_count += 1

        if self.log_every_n and self.frame_count % self.log_every_n == 0:
            self._log_latency(len(det_array.detections))

    # ------------------------------------------------------------- inference
    def _preprocess(self, img: np.ndarray) -> dict:
        h, w = img.shape[:2]
        # Reproduce what LoadImageFromFileMono3D would have emitted.
        data_ = dict(
            img=img,
            img_shape=(h, w),
            ori_shape=(h, w),
            cam2img=self.cam2img.copy(),
            box_type_3d=self.box_type_3d,
            box_mode_3d=self.box_mode_3d,
        )
        data_ = self.test_pipeline(data_)
        return pseudo_collate([data_])

    def _infer(self, data):
        with torch.no_grad():
            if self.fp16 and self.device.startswith('cuda'):
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    results = self.model.test_step(data)
            else:
                results = self.model.test_step(data)
        if self.device.startswith('cuda'):
            # Kernels are async; sync so the latency number is real.
            torch.cuda.synchronize()
        return results[0]

    # ---------------------------------------------------------- ROS conversion
    def _to_ros(self, result, header):
        """Convert a Det3DDataSample into Detection3DArray + MarkerArray.

        Frame convention: mmdet3d's Camera box frame (x right, y down,
        z forward) is exactly REP-103's *camera optical* frame, so we publish
        in header.frame_id unchanged and expect that to be an optical frame.
        If boxes appear rotated in RViz2, that is the frame to fix first.
        """
        det_array = Detection3DArray()
        det_array.header = header
        markers = MarkerArray()

        pred = result.pred_instances_3d
        scores = pred.scores_3d.cpu().numpy()
        keep = scores >= self.score_threshold
        if not np.any(keep):
            markers.markers.append(self._clear_marker(header))
            return det_array, markers

        bboxes = pred.bboxes_3d[keep]
        scores = scores[keep]
        labels = pred.labels_3d.cpu().numpy()[keep]

        # gravity_center is the true box center; bboxes_3d.tensor[:, :3] is the
        # *bottom* center for CameraInstance3DBoxes (origin 0.5, 1.0, 0.5).
        centers = bboxes.gravity_center.cpu().numpy()
        dims = bboxes.dims.cpu().numpy()      # (x_size, y_size, z_size)
        yaws = bboxes.yaw.cpu().numpy()       # rotation about camera y (down)

        markers.markers.append(self._clear_marker(header))

        for i in range(len(scores)):
            name = (self.classes[labels[i]]
                    if labels[i] < len(self.classes) else str(labels[i]))

            det = Detection3D()
            det.header = header
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = name
            hyp.hypothesis.score = float(scores[i])

            bbox = BoundingBox3D()
            bbox.center.position.x = float(centers[i][0])
            bbox.center.position.y = float(centers[i][1])
            bbox.center.position.z = float(centers[i][2])
            qw, qx, qy, qz = self._yaw_to_quat_about_y(float(yaws[i]))
            bbox.center.orientation.w = qw
            bbox.center.orientation.x = qx
            bbox.center.orientation.y = qy
            bbox.center.orientation.z = qz
            bbox.size = Vector3(
                x=float(dims[i][0]), y=float(dims[i][1]), z=float(dims[i][2]))

            det.bbox = bbox
            hyp.pose.pose = bbox.center
            det.results.append(hyp)
            det_array.detections.append(det)

            markers.markers.append(
                self._box_marker(header, i, name, bbox, float(scores[i])))

        return det_array, markers

    @staticmethod
    def _yaw_to_quat_about_y(yaw: float):
        """Rotation about the camera-frame y axis (which points down).

        Sign convention is the thing to verify visually in RViz2 first --
        if boxes are consistently mirrored in heading, negate `yaw` here.
        """
        half = yaw / 2.0
        return (np.cos(half), 0.0, np.sin(half), 0.0)  # (w, x, y, z)

    def _box_marker(self, header, idx, name, bbox, score) -> Marker:
        m = Marker()
        m.header = header
        m.ns = 'fcos3d'
        m.id = idx
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose = bbox.center
        m.scale = bbox.size
        r, g, b = CLASS_COLORS.get(name, (0.8, 0.8, 0.8))
        m.color = ColorRGBA(r=r, g=g, b=b, a=0.45)
        m.lifetime.sec = int(self.marker_lifetime)
        m.lifetime.nanosec = int(
            (self.marker_lifetime - int(self.marker_lifetime)) * 1e9)
        return m

    @staticmethod
    def _clear_marker(header) -> Marker:
        m = Marker()
        m.header = header
        m.ns = 'fcos3d'
        m.action = Marker.DELETEALL
        return m

    # ------------------------------------------------------------- reporting
    def _log_latency(self, n_det):
        def stats(d):
            a = np.array(d)
            return a.mean(), np.percentile(a, 95)

        pre_m, _ = stats(self.lat_preprocess)
        inf_m, inf_95 = stats(self.lat_inference)
        post_m, _ = stats(self.lat_postprocess)
        e2e_m, e2e_95 = stats(self.lat_end_to_end)
        self.get_logger().info(
            f'[{self.frame_count:5d}] dets={n_det:3d} | '
            f'pre {pre_m:6.1f} | inf {inf_m:6.1f} (p95 {inf_95:6.1f}) | '
            f'post {post_m:5.1f} | e2e {e2e_m:6.1f}ms (p95 {e2e_95:6.1f}) | '
            f'{1000.0 / max(e2e_m, 1e-6):5.1f} FPS')


def main(args=None):
    rclpy.init(args=args)
    node = Fcos3dDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

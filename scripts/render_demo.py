#!/usr/bin/env python3
"""Render an MP4 demo: FCOS3D predictions projected onto real nuScenes video.

Draws predicted 3D boxes (per-class colour) and ground-truth boxes (green)
projected into the image, with a live latency/FPS overlay. Produces a file that
can be played or embedded directly, which is more robust than screen-recording
a live viewer -- Runpod secure pods have no public IP, their HTTP proxy cannot
carry a WebSocket, and the SSH proxy does not forward TCP, so a browser-based
viewer such as Foxglove cannot reach the bridge.

Runs the same in-memory inference path as the ROS 2 node.

Usage (on the pod):
    python3 scripts/render_demo.py \
        --config /workspace/sim2real/configs/fcos3d_nus_mini.py \
        --ckpt   /workspace/checkpoints/fcos3d_r101_nus.pth \
        --dataroot /workspace/data/nuscenes \
        --frames 120 --fps 10 --out /workspace/demo.mp4

Contributor: Carlos Gonzales
"""

import argparse
import os
import time

import cv2
import numpy as np
import torch

NUSCENES_CLASSES = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier',
]

# BGR, matching the RViz2 marker palette in detector_node.py.
CLASS_COLORS = {
    'car': (255, 204, 51), 'truck': (255, 140, 0), 'trailer': (242, 102, 102),
    'bus': (255, 77, 153), 'construction_vehicle': (51, 166, 242),
    'bicycle': (140, 255, 51), 'motorcycle': (89, 230, 26),
    'pedestrian': (89, 77, 255), 'traffic_cone': (26, 217, 255),
    'barrier': (204, 191, 191),
}
GT_COLOR = (89, 255, 89)

# 12 edges of a cuboid, indexing mmdet3d's corner ordering.
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),
         (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]


def project(corners_3d, cam2img):
    """Camera-frame corners (N,3) -> pixel coords (N,2), plus in-front mask."""
    depths = corners_3d[:, 2]
    valid = depths > 0.1
    pts = corners_3d @ cam2img.T
    pts = pts[:, :2] / np.clip(pts[:, 2:3], 1e-6, None)
    return pts, valid


def draw_box(img, corners_3d, cam2img, color, thickness=2):
    pts, valid = project(corners_3d, cam2img)
    if valid.sum() < 8:          # skip boxes partly behind the camera
        return False
    h, w = img.shape[:2]
    if not np.any((pts[:, 0] > -w) & (pts[:, 0] < 2 * w) &
                  (pts[:, 1] > -h) & (pts[:, 1] < 2 * h)):
        return False
    p = pts.astype(np.int32)
    for a, b in EDGES:
        cv2.line(img, tuple(p[a]), tuple(p[b]), color, thickness, cv2.LINE_AA)
    # Mark the front face so orientation is readable.
    cv2.line(img, tuple(p[0]), tuple(p[5]), color, 1, cv2.LINE_AA)
    cv2.line(img, tuple(p[1]), tuple(p[4]), color, 1, cv2.LINE_AA)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--dataroot', default='/workspace/data/nuscenes')
    ap.add_argument('--version', default='v1.0-mini')
    ap.add_argument('--camera', default='CAM_FRONT')
    ap.add_argument('--frames', type=int, default=120)
    ap.add_argument('--fps', type=int, default=10)
    ap.add_argument('--scale', type=float, default=0.6)
    ap.add_argument('--score-thr', type=float, default=0.3)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--no-gt', action='store_true')
    # Far-away GT clutters the frame without adding information; FCOS3D's own
    # useful range is well inside this anyway (see mATE in the report).
    ap.add_argument('--gt-max-dist', type=float, default=45.0)
    ap.add_argument('--out', default='/workspace/demo.mp4')
    args = ap.parse_args()

    from mmengine.dataset import Compose, pseudo_collate
    from mmdet3d.apis import init_model
    from mmdet3d.structures import get_box_type
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.geometry_utils import BoxVisibility

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    model = init_model(args.config, args.ckpt, device=args.device)
    cfg = model.cfg
    pipeline = Compose([
        t for t in cfg.test_dataloader.dataset.pipeline
        if t['type'] not in ('LoadImageFromFileMono3D', 'LoadImageFromFile')
    ])
    box_type_3d, box_mode_3d = get_box_type(
        cfg.test_dataloader.dataset.box_type_3d)

    writer = None
    lat = []
    n = 0

    for sample in nusc.sample:
        if n >= args.frames:
            break
        tok = sample['data'].get(args.camera)
        if tok is None:
            continue
        sd = nusc.get('sample_data', tok)
        cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        cam2img = np.array(cs['camera_intrinsic'], dtype=np.float32)
        img = cv2.imread(nusc.get_sample_data_path(tok))
        if img is None:
            continue

        h, w = img.shape[:2]
        t0 = time.perf_counter()
        data = pseudo_collate([pipeline(dict(
            img=img, img_shape=(h, w), ori_shape=(h, w),
            cam2img=cam2img.copy(),
            box_type_3d=box_type_3d, box_mode_3d=box_mode_3d))])
        with torch.no_grad():
            result = model.test_step(data)[0]
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1e3
        lat.append(dt)

        canvas = img.copy()

        # ---- ground truth (already in the sensor frame) -------------------
        n_gt = 0
        if not args.no_gt:
            try:
                _, boxes, _ = nusc.get_sample_data(
                    tok, box_vis_level=BoxVisibility.ANY)
                for b in boxes:
                    if float(b.center[2]) > args.gt_max_dist:
                        continue
                    if draw_box(canvas, b.corners().T, cam2img, GT_COLOR, 1):
                        n_gt += 1
            except Exception as e:  # noqa: BLE001
                # Never swallow this silently: a failed lookup renders as
                # "ground truth 0" and produces a plausible but misleading
                # video in which the detector appears to have nothing to miss.
                print(f'  WARNING: GT lookup failed for sample_data {tok}: '
                      f'{type(e).__name__}: {e}', flush=True)

        # ---- predictions --------------------------------------------------
        pred = result.pred_instances_3d
        scores = pred.scores_3d.cpu().numpy()
        keep = scores >= args.score_thr
        n_det = 0
        if np.any(keep):
            bboxes = pred.bboxes_3d[keep]
            labels = pred.labels_3d.cpu().numpy()[keep]
            corners = bboxes.corners.cpu().numpy()
            centers = bboxes.gravity_center.cpu().numpy()
            ss = scores[keep]
            order = np.argsort(-centers[:, 2])       # far to near
            for i in order:
                name = (NUSCENES_CLASSES[labels[i]]
                        if labels[i] < len(NUSCENES_CLASSES) else '?')
                color = CLASS_COLORS.get(name, (200, 200, 200))
                if not draw_box(canvas, corners[i], cam2img, color, 2):
                    continue
                n_det += 1
                pts, _ = project(corners[i], cam2img)
                x, y = pts.min(axis=0).astype(int)
                # Keep the label on-screen: boxes clipped by the left edge
                # would otherwise have their first characters cut off.
                x = int(np.clip(x, 6, canvas.shape[1] - 190))
                label = f'{name} {ss[i]:.2f} {centers[i][2]:.0f}m'
                cv2.putText(canvas, label, (x, max(y - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4,
                            cv2.LINE_AA)
                cv2.putText(canvas, label, (x, max(y - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1,
                            cv2.LINE_AA)

        # ---- HUD ----------------------------------------------------------
        cv2.rectangle(canvas, (0, 0), (w, 92), (0, 0, 0), -1)
        cv2.addWeighted(canvas[:92], 0.55, img[:92], 0.45, 0, canvas[:92])
        hud = [
            'FCOS3D  |  ROS 2 Humble  |  nuScenes v1.0-mini  |  RTX 4090',
            f'frame {n+1:3d}   detections {n_det:2d}   ground truth {n_gt:2d}',
            f'latency {dt:6.1f} ms   mean {np.mean(lat):6.1f} ms'
            f'   {1000/np.mean(lat):4.1f} FPS',
        ]
        for i, line in enumerate(hud):
            cv2.putText(canvas, line, (14, 26 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1,
                        cv2.LINE_AA)
        cv2.putText(canvas, 'green = ground truth', (w - 260, h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, GT_COLOR, 1, cv2.LINE_AA)

        if args.scale != 1.0:
            canvas = cv2.resize(canvas, None, fx=args.scale, fy=args.scale,
                                interpolation=cv2.INTER_AREA)
        if writer is None:
            writer = cv2.VideoWriter(
                args.out, cv2.VideoWriter_fourcc(*'mp4v'), args.fps,
                (canvas.shape[1], canvas.shape[0]))
        writer.write(canvas)

        n += 1
        if n % 20 == 0:
            print(f'  {n}/{args.frames} frames', flush=True)

    if writer is not None:
        writer.release()
    print(f'\nwrote {args.out}  ({n} frames, {n/args.fps:.1f}s at {args.fps} fps)')
    print(f'size: {os.path.getsize(args.out)/1e6:.1f} MB')
    print(f'mean latency {np.mean(lat):.1f} ms  -> {1000/np.mean(lat):.1f} FPS')


if __name__ == '__main__':
    main()

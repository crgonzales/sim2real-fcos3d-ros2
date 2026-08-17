#!/usr/bin/env python3
"""Offline smoke test for the FCOS3D inference path used by the ROS 2 node.

This deliberately does NOT call mmdet3d.apis.inference_mono_3d_detector.
It reproduces the exact in-memory preprocessing that
fcos3d_ros2/detector_node.py performs, so that if this passes we know the
node's core logic is sound and anything that breaks later is ROS plumbing,
not the model path.

Usage (on the pod):
    python3 scripts/smoke_test.py \
        --config   /workspace/mmdetection3d/configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d.py \
        --ckpt     /workspace/checkpoints/fcos3d_r101_nus.pth \
        --demo-dir /workspace/mmdetection3d/demo/data/nuscenes

Contributor: Carlos Gonzales
"""

import argparse
import glob
import os
import time

import numpy as np
import torch
import mmengine
from mmengine.dataset import Compose, pseudo_collate
from mmdet3d.apis import init_model
from mmdet3d.structures import get_box_type

NUSCENES_CLASSES = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier',
]


def build_pipeline(model):
    """Same construction as Fcos3dDetectorNode._load_model."""
    cfg = model.cfg
    pipeline_cfg = [
        t for t in cfg.test_dataloader.dataset.pipeline
        if t['type'] not in ('LoadImageFromFileMono3D', 'LoadImageFromFile')
    ]
    box_type_3d, box_mode_3d = get_box_type(
        cfg.test_dataloader.dataset.box_type_3d)
    return Compose(pipeline_cfg), box_type_3d, box_mode_3d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--demo-dir', required=True)
    ap.add_argument('--cam', default='CAM_FRONT')
    ap.add_argument('--score-thr', type=float, default=0.3)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--runs', type=int, default=10)
    args = ap.parse_args()

    import mmcv

    # ---------------------------------------------------------------- inputs
    # Match the camera name EXACTLY, delimited by the '__' separators nuScenes
    # uses. A loose '*CAM_FRONT*' also matches CAM_FRONT_LEFT/RIGHT (and sorts
    # them first, since 'L' < '_'), which would silently pair one camera's
    # image with another camera's intrinsics -- the exact class of frame bug
    # that produces plausible-looking but wrong detections.
    imgs = sorted(glob.glob(os.path.join(args.demo_dir, f'*__{args.cam}__*.jpg')))
    if not imgs:
        raise SystemExit(f'No {args.cam} image found in {args.demo_dir}')
    img_path = imgs[0]
    assert f'__{args.cam}__' in os.path.basename(img_path), 'camera mismatch'

    pkls = sorted(glob.glob(os.path.join(args.demo_dir, '*.pkl')))
    if not pkls:
        raise SystemExit(f'No .pkl info file in {args.demo_dir}')
    info = mmengine.load(pkls[0])
    data_info = info['data_list'][0]
    cam2img = np.array(
        data_info['images'][args.cam]['cam2img'], dtype=np.float32)

    print(f'image   : {os.path.basename(img_path)}')
    print(f'cam2img :\n{cam2img}')

    # BGR on purpose: the FCOS3D config uses bgr_to_rgb=False with caffe means.
    img = mmcv.imread(img_path, channel_order='bgr')
    print(f'img     : {img.shape} dtype={img.dtype}')

    # ----------------------------------------------------------------- model
    t0 = time.perf_counter()
    model = init_model(args.config, args.ckpt, device=args.device)
    print(f'model loaded in {time.perf_counter() - t0:.1f}s')

    test_pipeline, box_type_3d, box_mode_3d = build_pipeline(model)
    print(f'pipeline: {[t["type"] for t in model.cfg.test_dataloader.dataset.pipeline]}')

    def preprocess():
        h, w = img.shape[:2]
        data_ = dict(
            img=img,
            img_shape=(h, w),
            ori_shape=(h, w),
            cam2img=cam2img.copy(),
            box_type_3d=box_type_3d,
            box_mode_3d=box_mode_3d,
        )
        return pseudo_collate([test_pipeline(data_)])

    # ------------------------------------------------------------- inference
    data = preprocess()
    with torch.no_grad():
        result = model.test_step(data)[0]
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()

    pred = result.pred_instances_3d
    scores = pred.scores_3d.cpu().numpy()
    labels = pred.labels_3d.cpu().numpy()
    bboxes = pred.bboxes_3d

    keep = scores >= args.score_thr
    print(f'\nraw detections      : {len(scores)}')
    print(f'above score {args.score_thr}      : {int(keep.sum())}')

    if keep.sum():
        centers = bboxes.gravity_center.cpu().numpy()[keep]
        dims = bboxes.dims.cpu().numpy()[keep]
        yaws = bboxes.yaw.cpu().numpy()[keep]
        ss = scores[keep]
        ll = labels[keep]
        order = np.argsort(-ss)
        print(f'\n{"class":22s} {"score":>6s} {"x":>7s} {"y":>7s} {"z":>7s} '
              f'{"w":>5s} {"h":>5s} {"l":>5s} {"yaw":>6s}')
        for i in order[:12]:
            name = NUSCENES_CLASSES[ll[i]] if ll[i] < len(NUSCENES_CLASSES) else str(ll[i])
            print(f'{name:22s} {ss[i]:6.3f} '
                  f'{centers[i][0]:7.2f} {centers[i][1]:7.2f} {centers[i][2]:7.2f} '
                  f'{dims[i][0]:5.2f} {dims[i][1]:5.2f} {dims[i][2]:5.2f} '
                  f'{yaws[i]:6.2f}')

        # Sanity: nuScenes cameras look down +z, so depths must be positive and
        # in a plausible range. Catches a bad cam2img or a frame-convention bug.
        z = centers[:, 2]
        print(f'\ndepth z: min {z.min():.1f}  max {z.max():.1f}  mean {z.mean():.1f}')
        if z.min() <= 0:
            print('  WARNING: non-positive depth -- check cam2img / frame convention')
    else:
        print('  WARNING: nothing above threshold -- suspicious for a demo frame')

    # ------------------------------------------------------------- profiling
    for _ in range(3):                       # warmup
        with torch.no_grad():
            model.test_step(preprocess())
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()

    lat = []
    for _ in range(args.runs):
        t = time.perf_counter()
        d = preprocess()
        with torch.no_grad():
            model.test_step(d)
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - t) * 1e3)

    lat = np.array(lat)
    print(f'\nlatency over {args.runs} runs: mean {lat.mean():.1f} ms  '
          f'p95 {np.percentile(lat, 95):.1f} ms  -> {1000/lat.mean():.1f} FPS')
    if args.device.startswith('cuda'):
        print(f'peak GPU mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB')

    print('\nSMOKE TEST PASSED')


if __name__ == '__main__':
    main()

# Sim2Real: FCOS3D in a ROS 2 Pipeline — Project Context

CMPE 249 final project. Carlos Gonzales. **Due Aug 20, 2026.**
Focused area: **Application** (Option 1 — Apply & Evaluate).

## Hard-won facts (do not re-derive)

### Version matrix — VERIFIED, do not "upgrade"
The MMDetection3D install docs are actively misleading. They say
`mim install 'mmcv>=2.0.0rc4'`, which resolves to mmcv 2.2.x and then fails
mmdet3d's own import-time assert. The real bounds live in `mmdet3d/__init__.py`:

| package  | bound (from source asserts) | we pin |
|----------|-----------------------------|--------|
| mmengine | `>=0.8.0, <1.0.0`           | 0.10.4 |
| mmcv     | `>=2.0.0rc4, <2.2.0`        | 2.1.0  |
| mmdet    | `>=3.0.0rc5, <3.4.0`        | 3.3.0  |
| mmdet3d  | —                           | 1.4.0  |
| torch    | cu118 (matches mmcv wheel)  | 2.1.0  |
| numpy    | mmcv/devkit break on numpy2 | <2.0   |

The dependency chain that forces Ubuntu 22.04:
`22.04 -> Python 3.10 -> ROS 2 Humble -> prebuilt mmcv cp310 wheel`.
Use the wheel index (`download.openmmlab.com/mmcv/dist/cu118/torch2.1.0`);
building mmcv from source costs 30–40 min.

### Model
- Config: `configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d.py`
- Weights (base): `https://download.openmmlab.com/mmdetection3d/v0.1.0_models/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_20210715_235813-4bed5239.pth`
- Weights (finetuned): `..._finetune_20210717_095645-8d806dc2.pth`

**Published reference numbers — use these to validate the pipeline is wired
correctly before trusting any of our own numbers:**

| checkpoint | mAP  | NDS  |
|------------|------|------|
| base       | 29.9 | 37.3 |
| finetune   | 32.1 | 39.3 |

If our nuScenes val reproduction is far off these, the bug is in our data/eval
plumbing, not a finding. This is the single most important sanity gate.

### Coordinate conventions — the silent-failure zone
This is where sim-to-real evaluations produce confident garbage. Write down
and verify before comparing any boxes:
- FCOS3D outputs boxes in the **camera** frame; nuScenes GT annotations are in
  the **global/ego** frame. The devkit expects global-frame submissions.
- Box format is `(x, y, z, w, l, h, yaw)` with mmdet3d's `CameraInstance3DBoxes`
  convention; yaw reference differs from nuScenes' quaternion convention.
- Any Isaac Sim ground truth we export must be converted into the *same*
  convention as the nuScenes GT, not just "into some 3D box."
- Sanity check: project predicted boxes back onto the image and eyeball them in
  RViz2 before running any metric.

## Scope decisions
- Primary (must ship): FCOS3D as a ROS 2 node, nuScenes replayed as ROS 2 bags,
  devkit mAP/NDS, latency/throughput/resource profiling FP32 vs FP16.
- Stretch (time-boxed): Isaac Sim ROS 2 bridge + day/dusk lighting sweep.
- Insurance if Isaac Sim fails: photometric-perturbation sweep on real nuScenes
  images as a rendering-gap proxy (see docs/PLAN.md).
- Dropped unless everything else is green: BEVFormer, TensorRT.

## Dataset
Use nuScenes **mini** (v1.0-mini, ~4 GB), not the full trainval (~300+ GB).
Mini has 10 scenes and is enough for every experiment in the plan.

## Environment
No Docker and no CUDA on the Mac — this repo is authored here but everything
executes on the RTX 4090 box or a RunPod instance via `docker/Dockerfile`.

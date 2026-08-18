# FCOS3D in ROS 2 — Applying and Evaluating a Pretrained Monocular 3D Detector

CMPE 249 final project — **Carlos Gonzales** — Option 1 (Apply & Evaluate).
**Focused area: Application.** No training was performed.

A pretrained [FCOS3D](https://arxiv.org/abs/2104.10956) monocular 3D object
detector wrapped as a ROS 2 Humble node, applied to nuScenes replayed onto ROS
topics, and evaluated for accuracy, latency and resource utilisation across two
GPU environments.

**Full write-up: [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md)**

---

## Results at a glance

| | |
|---|---|
| Accuracy (nuScenes `mini_val`) | mAP **0.2943**, NDS **0.3217** |
| End-to-end latency (RTX 4090, FP32) | **72.05 ms** mean, 97.75 ms p95 |
| Sustained throughput | **13.85 FPS** |
| Peak GPU memory | **597 MB** |
| RTX 4090 vs A40 | 4090 is **2.0x** faster |
| FP16 vs FP32 | **+6.8%** on A40, **−3.9%** on RTX 4090 |
| Accuracy at 40% brightness | **−33%** mAP |

Three findings worth reading the report for:

- **The pipeline is not GPU-bound** (utilisation peaks near 54%), which is why
  FP16 helps the slower card and *hurts* the faster one — and why TensorRT was
  dropped on principle rather than for lack of time.
- **The headline mAP is misleading.** Three of the ten classes have no ground
  truth at all in `mini_val`; the evaluator still averages over all ten. On the
  seven classes actually present, mAP is 0.4204.
- **Lighting degradation is nonlinear and wildly class-dependent.** At 40%
  brightness, bus AP falls 95% while traffic cone falls 12%.

## Architecture

```
                    ┌─────────────────────────┐
  nuScenes v1.0-mini│   nuscenes_publisher    │
  (404 keyframes) ─▶│  replay + ground truth  │
                    └───────────┬─────────────┘
                                │
        /camera/front/image_raw │ sensor_msgs/Image      (bgr8)
     /camera/front/camera_info  │ sensor_msgs/CameraInfo
          /nuscenes/gt_markers  │ visualization_msgs/MarkerArray
                                ▼
                    ┌─────────────────────────┐
                    │     detector_node       │
                    │  FCOS3D (MMDetection3D) │
                    └───────────┬─────────────┘
                                │
     /perception/detections_3d  │ vision_msgs/Detection3DArray
          /perception/markers   │ visualization_msgs/MarkerArray
                                ▼
                        RViz2 / Foxglove
```

| Direction | Topic | Type |
|-----------|-------|------|
| Subscribe | `/camera/front/image_raw` | `sensor_msgs/Image` |
| Subscribe | `/camera/front/camera_info` | `sensor_msgs/CameraInfo` |
| Publish | `/perception/detections_3d` | `vision_msgs/Detection3DArray` |
| Publish | `/perception/markers` | `visualization_msgs/MarkerArray` |
| Publish | `/nuscenes/gt_markers` | `visualization_msgs/MarkerArray` |

## Quick start

Tested on a Runpod pod from
`runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`. That base image is
not incidental — see *Environment* below.

```bash
git clone https://github.com/crgonzales/sim2real-fcos3d-ros2.git /workspace/sim2real
cd /workspace/sim2real

# 1. ROS 2 Humble + pinned OpenMMLab stack + FCOS3D weights (~20 min)
bash scripts/setup_pod.sh

# 2. nuScenes v1.0-mini (~4 GB). URL is account-gated: register free at
#    nuscenes.org, then Downloads -> Full dataset (v1.0) -> Mini.
bash scripts/get_nuscenes_mini.sh "<your-url>"

# 3. Evaluation info files
cd /workspace/mmdetection3d
PYTHONPATH=$PWD python3 /workspace/sim2real/scripts/prepare_nuscenes.py \
    --root-path ./data/nuscenes --version v1.0-mini

# 4. Build the ROS 2 package
cd /workspace/sim2real
source /opt/ros/humble/setup.bash
colcon build --symlink-install && source install/setup.bash
```

### Run the pipeline

```bash
# terminal 1 — replay nuScenes onto ROS topics
ros2 run fcos3d_ros2 nuscenes_publisher --ros-args \
    -p dataroot:=/workspace/data/nuscenes -p version:=v1.0-mini -p rate_hz:=2.0

# terminal 2 — detector
ros2 run fcos3d_ros2 detector_node --ros-args \
    -p config_file:=/workspace/sim2real/configs/fcos3d_nus_mini.py \
    -p checkpoint_file:=/workspace/checkpoints/fcos3d_r101_nus.pth \
    -p device:=cuda:0 -p score_threshold:=0.3
```

### Reproduce the experiments

```bash
# offline sanity check of the node's exact inference path
python3 scripts/smoke_test.py --config <cfg> --ckpt <ckpt> \
    --demo-dir /workspace/mmdetection3d/demo/data/nuscenes

# official nuScenes accuracy
cd /workspace/mmdetection3d && python3 tools/test.py \
    /workspace/sim2real/configs/fcos3d_nus_mini.py <ckpt>

# latency / throughput / CPU / GPU / memory
python3 scripts/profile_system.py --config <cfg> --ckpt <ckpt> \
    --dataroot /workspace/data/nuscenes --frames 100 \
    --precision fp32 --label "RTX 4090"

# controlled lighting sweep (8 points, full evaluation each)
cd /workspace/mmdetection3d && python3 /workspace/sim2real/scripts/lighting_sweep.py \
    --config /workspace/sim2real/configs/fcos3d_nus_mini.py --ckpt <ckpt>
```

## Environment

The version matrix is load-bearing and the official install docs are wrong.
`mim install 'mmcv>=2.0.0rc4'` resolves to mmcv 2.2.x, which fails mmdet3d's
own import-time assert. The real bounds live in `mmdet3d/__init__.py`:

| Package | Pin | Constraint |
|---------|-----|------------|
| Ubuntu | 22.04 | ROS 2 Humble; 24.04 means Jazzy |
| Python | 3.10 | prebuilt `cp310` mmcv wheel |
| torch | 2.1.0+cu118 | matches the mmcv wheel index |
| mmengine | 0.10.4 | `>=0.8.0,<1.0.0` |
| mmcv | 2.1.0 | `>=2.0.0rc4,<2.2.0` |
| mmdet | 3.3.0 | `>=3.0.0rc5,<3.4.0` |
| mmdet3d | 1.4.0 | last release |
| numpy | 1.26.4 | mmcv is built against the numpy 1.x ABI |

`scripts/setup_pod.sh` routes every `pip install` through a constraints file so
the numpy-1.x-compatible set resolves in one pass. Do not "fix" a numpy 2.x
install by downgrading afterwards — pip overwrites compiled packages without
removing the old tree, and the resulting mixed installs fail in ways that look
unrelated to numpy. See §8.1 of the report.

## Layout

| Path | Contents |
|------|----------|
| `src/fcos3d_ros2/` | ROS 2 package — `detector_node`, `nuscenes_publisher`, `fp16_compat` |
| `scripts/setup_pod.sh` | Full environment bootstrap |
| `scripts/get_nuscenes_mini.sh` | Dataset download + extraction |
| `scripts/prepare_nuscenes.py` | Info-file generation |
| `scripts/smoke_test.py` | Offline test of the node's inference path |
| `scripts/profile_system.py` | Latency / throughput / CPU / GPU / memory |
| `scripts/lighting_sweep.py` | Controlled photometric sweep |
| `scripts/pod.sh` | SSH helper for the Runpod proxy |
| `configs/fcos3d_nus_mini.py` | Evaluation config for the mini split |
| `docs/PROJECT_REPORT.md` | Full write-up |
| `eval/` | All results |
| `CONVENTIONS.md` | Version matrix and coordinate conventions |

## Notes

- The node deliberately does **not** use `inference_mono_3d_detector`: it
  requires an annotation file on disk and asserts image basenames match, so it
  cannot consume an in-memory `sensor_msgs/Image`. The node rebuilds the short
  test pipeline instead. See report §5.1.
- FCOS3D expects **BGR** input (`bgr_to_rgb=False` with caffe means). Feeding
  RGB degrades accuracy silently.
- Boxes use origin `(0.5, 1.0, 0.5)` — bottom-centre. Use `gravity_center`, not
  the raw tensor columns.

## License

MIT. Pretrained weights and nuScenes data are subject to their own licences.

## References

1. Wang et al. *FCOS3D.* ICCV Workshops 2021. https://arxiv.org/abs/2104.10956
2. MMDetection3D v1.4.0. https://github.com/open-mmlab/mmdetection3d
3. Caesar et al. *nuScenes.* CVPR 2020. https://www.nuscenes.org

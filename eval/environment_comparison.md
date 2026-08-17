# FCOS3D across computing environments

Assignment requirement 4: *"measure system-level latency and resource
utilization (CPU/GPU/Memory) under different computing environments."*

Both environments are Runpod secure-cloud pods running the identical container
image (`runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`) and the
identical software stack, built by `scripts/setup_pod.sh`. Holding the cloud
tier, OS, driver family, Python, torch and every OpenMMLab version constant
means the GPU is the only variable.

Workload: 100 real nuScenes `CAM_FRONT` keyframes at 1600x900, 10 warmup
frames discarded, measured through the same in-memory path the ROS 2 node
uses. Date: 2026-08-17.

## Results

| | RTX 4090 FP32 | RTX 4090 FP16 | A40 FP32 | A40 FP16 |
|---|---|---|---|---|
| Mean latency (ms)     | **72.05** | 74.85 | 144.57 | 134.79 |
| Median latency (ms)   | 72.54 | 76.33 | 131.65 | — |
| p95 latency (ms)      | 97.75 | 98.44 | 191.36 | 196.52 |
| Throughput (FPS)      | **13.85** | 13.33 | 6.91 | 7.41 |
| GPU utilisation (mean %) | 54.69 | 36.00 | 53.58 | 38.28 |
| GPU memory used (MB)  | 2101.7 | 1781.0 | 2051.2 | 1731.7 |
| torch peak alloc (MB) | 597.0 | 575.9 | 597.0 | 575.9 |
| Process RSS (MB)      | 1904.8 | 1998.8 | 1914.6 | 2015.8 |
| CPU (mean %, 96 cores)| 26.50 | 25.41 | 10.41 | 11.53 |
| Detections / frame    | 6.28 | 6.28 | 6.28 | 6.30 |
| GPU VRAM (GB)         | 25.3 | 25.3 | 47.7 | 47.7 |
| Cost (USD/hr, secure) | 0.74 | 0.74 | 0.44 | 0.44 |

## Findings

### 1. The RTX 4090 is 2.0x faster than the A40

144.57 / 72.05 = 2.007. Consumer Ada beats datacenter Ampere by exactly a
factor of two on this workload, despite the A40 costing 40% less per hour.
Per unit of throughput the 4090 is the better value here: $0.053/FPS versus
$0.064/FPS.

The comparison is well controlled. `torch` peak allocation is byte-identical
(597.0 MB both) and detections per frame are identical (6.28), so the two
machines are running the same computation and producing the same results --
only the speed differs.

### 2. FP16 helps the A40 and hurts the 4090

| GPU | FP32 -> FP16 |
|-----|--------------|
| RTX 4090 | 72.05 -> 74.85 ms (**3.9% slower**) |
| A40      | 144.57 -> 134.79 ms (**6.8% faster**) |

The explanation is in the utilisation column. FP16 drops GPU utilisation on
both cards (54.7 -> 36.0 and 53.6 -> 38.3), meaning less time is spent in
arithmetic. On the A40 that translates into a real speedup because the card is
slow enough for compute to matter. On the 4090 the arithmetic was never the
bottleneck, so all that is left is autocast's casting overhead, and the net
effect is negative.

The practical lesson: mixed precision is not a free win. On fast hardware
running a model that does not saturate it, FP16 can cost more than it saves.

### 3. The pipeline is not GPU-bound on either card

GPU utilisation peaks around 54% in FP32 on both. Roughly half the wall time
is spent outside GPU arithmetic -- image preprocessing, kernel launch
overhead, and CPU-side box decoding and NMS.

This is why TensorRT was dropped rather than merely deferred. TensorRT
optimises GPU kernel execution, which is not where the time is going; the
upper bound on any such gain here is well under 2x, and probably much less.

### 4. Memory is not a constraint

Peak torch allocation is 597 MB. The 4090's 24 GB and the A40's 48 GB are both
enormously oversized for this model. VRAM should not drive GPU selection for
monocular 3D detection -- a much cheaper 8-12 GB card would fit comfortably,
and the relevant question is purely throughput per dollar.

The ~2 GB gap between `torch` peak allocation (597 MB) and NVML reported usage
(~2050 MB) is the CUDA context plus cuDNN workspace, which is fixed overhead
independent of model size.

### 5. FP16 slightly perturbs results

Detections per frame is 6.28 on three configurations and 6.30 on A40 FP16 --
about two extra detections across 100 frames. Reduced precision does change
the output near the score threshold, if only marginally. Any accuracy claim
should therefore state the precision it was measured at.

## Reproducing

```bash
# on each machine
bash scripts/setup_pod.sh
bash scripts/get_nuscenes_mini.sh "<account-gated URL>"
python3 scripts/profile_system.py \
    --config configs/fcos3d_nus_mini.py \
    --ckpt   /workspace/checkpoints/fcos3d_r101_nus.pth \
    --dataroot /workspace/data/nuscenes \
    --frames 100 --precision fp32 --label "<name>"
```

Raw JSON: `eval/profile_{fp32,fp16}_{RTX_4090,A40}.json`.

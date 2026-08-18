#!/usr/bin/env python3
"""System-level profiling of FCOS3D: latency, throughput, CPU/GPU/memory.

Covers assignment requirement 4 ("measure system-level latency and resource
utilization (CPU/GPU/Memory) under different computing environments"). Run the
same command on each machine and diff the JSON.

Measures the same in-memory inference path the ROS 2 node uses, so the numbers
describe the deployed node rather than an offline benchmark harness.

Resource sampling runs on a background thread at a fixed interval, because
polling inside the inference loop would both perturb the latency measurement
and miss the peaks that occur mid-forward-pass.

Usage (on the pod):
    python3 scripts/profile_system.py \
        --config /workspace/sim2real/configs/fcos3d_nus_mini.py \
        --ckpt   /workspace/checkpoints/fcos3d_r101_nus.pth \
        --dataroot /workspace/data/nuscenes \
        --frames 100 --precision fp32 --label "RTX 4090 (Runpod secure)"

Contributor: Carlos Gonzales
"""

import argparse
import json
import os
import platform
import statistics
import threading
import time
from datetime import datetime, timezone

import numpy as np
import torch
import psutil


def _cpu_model() -> str:
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or 'unknown'


# --------------------------------------------------------------------- sampler
class ResourceSampler(threading.Thread):
    """Polls CPU / RAM / GPU on a background thread while inference runs."""

    def __init__(self, interval=0.1):
        super().__init__(daemon=True)
        self.interval = interval
        # NB: not self._stop -- that shadows Thread._stop(), which CPython's
        # threading internals call during join(), giving
        # 'TypeError: Event object is not callable'.
        self._stop_evt = threading.Event()
        self.cpu_pct = []
        self.rss_mb = []
        self.gpu_util = []
        self.gpu_mem_mb = []
        self.proc = psutil.Process(os.getpid())
        self.n_cpu = psutil.cpu_count(logical=True)

        self.nvml = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            pass

        # Prime the cpu_percent counter; the first call always returns 0.0.
        self.proc.cpu_percent(None)
        psutil.cpu_percent(None)

    def run(self):
        while not self._stop_evt.is_set():
            try:
                # System-wide CPU across all cores, plus this process's share
                # normalised to a percentage of one core.
                self.cpu_pct.append(psutil.cpu_percent(None))
                self.rss_mb.append(self.proc.memory_info().rss / 1e6)
                if self.nvml is not None:
                    u = self.nvml.nvmlDeviceGetUtilizationRates(self.handle)
                    m = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
                    self.gpu_util.append(float(u.gpu))
                    self.gpu_mem_mb.append(m.used / 1e6)
            except Exception:
                pass
            self._stop_evt.wait(self.interval)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=2.0)

    def summary(self):
        def stats(xs):
            if not xs:
                return None
            return {
                'mean': round(float(np.mean(xs)), 2),
                'max': round(float(np.max(xs)), 2),
                'samples': len(xs),
            }
        return {
            'cpu_percent_systemwide': stats(self.cpu_pct),
            'process_rss_mb': stats(self.rss_mb),
            'gpu_utilization_percent': stats(self.gpu_util),
            'gpu_memory_used_mb': stats(self.gpu_mem_mb),
            'logical_cpus': self.n_cpu,
        }


# ----------------------------------------------------------------------- setup
def collect_frames(dataroot, version, camera, limit):
    """Real nuScenes frames + their true intrinsics."""
    from nuscenes.nuscenes import NuScenes
    import cv2

    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    out = []
    for sample in nusc.sample:
        tok = sample['data'].get(camera)
        if tok is None:
            continue
        sd = nusc.get('sample_data', tok)
        cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        path = nusc.get_sample_data_path(tok)
        img = cv2.imread(path)          # BGR, matching the FCOS3D config
        if img is None:
            continue
        out.append((img, np.array(cs['camera_intrinsic'], dtype=np.float32)))
        if len(out) >= limit:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--dataroot', default='/workspace/data/nuscenes')
    ap.add_argument('--version', default='v1.0-mini')
    ap.add_argument('--camera', default='CAM_FRONT')
    ap.add_argument('--frames', type=int, default=100)
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--precision', choices=['fp32', 'fp16'], default='fp32')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--label', default='')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    from mmengine.dataset import Compose, pseudo_collate
    from mmdet3d.apis import init_model
    from mmdet3d.structures import get_box_type

    cuda = args.device.startswith('cuda')
    label = args.label or (
        torch.cuda.get_device_name(0) if cuda else platform.processor() or 'CPU')

    print(f'== {label} | {args.precision} | {args.frames} frames ==')

    frames = collect_frames(
        args.dataroot, args.version, args.camera, args.frames)
    if not frames:
        raise SystemExit('No frames collected -- check dataroot/version')
    print(f'frames: {len(frames)}  shape: {frames[0][0].shape}')

    model = init_model(args.config, args.ckpt, device=args.device)
    cfg = model.cfg
    pipeline = Compose([
        t for t in cfg.test_dataloader.dataset.pipeline
        if t['type'] not in ('LoadImageFromFileMono3D', 'LoadImageFromFile')
    ])
    box_type_3d, box_mode_3d = get_box_type(
        cfg.test_dataloader.dataset.box_type_3d)

    def preprocess(img, cam2img):
        h, w = img.shape[:2]
        return pseudo_collate([pipeline(dict(
            img=img, img_shape=(h, w), ori_shape=(h, w),
            cam2img=cam2img.copy(),
            box_type_3d=box_type_3d, box_mode_3d=box_mode_3d))])

    use_amp = args.precision == 'fp16' and cuda
    if use_amp:
        # FCOS3D's decode inverts the intrinsics matrix, and linalg.inv has no
        # half-precision kernel. Keep just that step in fp32.
        import sys as _sys
        _sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'src', 'fcos3d_ros2'))
        from fcos3d_ros2.fp16_compat import patch_points_img2cam_fp32
        patch_points_img2cam_fp32()
        print('fp16: patched points_img2cam to fp32')

    def infer(data):
        with torch.no_grad():
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    return model.test_step(data)
            return model.test_step(data)

    # ------------------------------------------------------------------ warmup
    for i in range(min(args.warmup, len(frames))):
        infer(preprocess(*frames[i]))
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # ----------------------------------------------------------------- measure
    sampler = ResourceSampler(interval=0.1)
    sampler.start()

    lat_pre, lat_inf, lat_total, n_det = [], [], [], []
    t_wall0 = time.perf_counter()
    for img, cam2img in frames:
        t0 = time.perf_counter()
        data = preprocess(img, cam2img)
        t1 = time.perf_counter()
        result = infer(data)
        if cuda:
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        lat_pre.append((t1 - t0) * 1e3)
        lat_inf.append((t2 - t1) * 1e3)
        lat_total.append((t2 - t0) * 1e3)
        n_det.append(int((result[0].pred_instances_3d.scores_3d >= 0.3).sum()))
    wall = time.perf_counter() - t_wall0

    sampler.stop()

    def pct(xs, p):
        return round(float(np.percentile(xs, p)), 2)

    report = {
        'label': label,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'precision': args.precision,
        'device': args.device,
        'frames': len(frames),
        'environment': {
            'gpu': torch.cuda.get_device_name(0) if cuda else None,
            'gpu_total_mem_gb': (
                round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
                if cuda else None),
            'torch': torch.__version__,
            'cuda': torch.version.cuda if cuda else None,
            'python': platform.python_version(),
            'platform': platform.platform(),
            # platform.processor() returns a bare 'x86_64' on Linux, which is
            # useless for controlling a cross-machine comparison. Read the real
            # model from /proc/cpuinfo -- see CR_w1_v0.1.0.md finding M1.
            'cpu_model': _cpu_model(),
        },
        'latency_ms': {
            'preprocess_mean': round(statistics.mean(lat_pre), 2),
            'inference_mean': round(statistics.mean(lat_inf), 2),
            'total_mean': round(statistics.mean(lat_total), 2),
            'total_median': pct(lat_total, 50),
            'total_p95': pct(lat_total, 95),
            'total_p99': pct(lat_total, 99),
            'total_min': round(min(lat_total), 2),
            'total_max': round(max(lat_total), 2),
            'total_stdev': round(statistics.pstdev(lat_total), 2),
        },
        'throughput_fps': round(len(frames) / wall, 2),
        'resources': sampler.summary(),
        'torch_peak_gpu_mem_mb': (
            round(torch.cuda.max_memory_allocated() / 1e6, 1) if cuda else None),
        'detections_per_frame_mean': round(statistics.mean(n_det), 2),
    }

    print(json.dumps(report, indent=2))

    out = args.out or (
        f'eval/profile_{args.precision}_'
        f'{label.replace(" ", "_").replace("/", "_")}.json')
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()

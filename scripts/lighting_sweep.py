#!/usr/bin/env python3
"""Controlled lighting sweep: measure FCOS3D accuracy vs image brightness.

Motivation
----------
The project proposal set out to quantify the sim-to-real gap by rendering the
same scene in Isaac Sim under different lighting. Isaac Sim was scoped out
(the assignment requires simulation *or* a real dataset, and four days did not
allow both). This experiment recovers the underlying scientific question --
"which classes suffer, and does it worsen with lighting?" -- by perturbing
real nuScenes images instead.

A gamma/gain change is a rendering-gap *proxy*, not a substitute for a
simulator: it alters the photometric response only, leaving geometry, texture
and sensor noise untouched. It therefore isolates one variable cleanly, which
is exactly the controlled-experiment property the proposal wanted, while
making no claim about synthetic imagery.

Each sweep point runs the full official nuScenes evaluation, so the reported
mAP/NDS are directly comparable to the unperturbed baseline.

Usage (on the pod, from the mmdetection3d repo root):
    cd /workspace/mmdetection3d
    python3 /workspace/sim2real/scripts/lighting_sweep.py \
        --config /workspace/sim2real/configs/fcos3d_nus_mini.py \
        --ckpt   /workspace/checkpoints/fcos3d_r101_nus.pth \
        --out    /workspace/sim2real/eval/lighting_sweep.json

Contributor: Carlos Gonzales
"""

import argparse
import json
import os

import numpy as np
from mmcv.transforms import BaseTransform
from mmengine.config import Config
from mmengine.registry import TRANSFORMS
from mmengine.runner import Runner


@TRANSFORMS.register_module()
class DeterministicPhotometric(BaseTransform):
    """Fixed brightness gain + gamma, applied to the decoded image.

    Deliberately deterministic, unlike mmdet's PhotoMetricDistortion which
    randomises: a sweep needs every frame at a sweep point to receive exactly
    the same perturbation, otherwise the independent variable is not controlled.

    Applied after loading and before Pack3DDetInputs, so the model's own
    normalisation (caffe BGR means) still runs afterwards exactly as in the
    baseline.

    Must subclass BaseTransform: mmengine's Compose requires a callable, and
    BaseTransform is what supplies __call__ -> transform(). A bare class with
    only a transform() method fails with
    "transform should be a callable object".
    """

    def __init__(self, gain: float = 1.0, gamma: float = 1.0):
        self.gain = float(gain)
        self.gamma = float(gamma)

    def transform(self, results: dict) -> dict:
        if self.gain == 1.0 and self.gamma == 1.0:
            return results
        img = results['img'].astype(np.float32) / 255.0
        if self.gamma != 1.0:
            img = np.power(img, self.gamma)
        if self.gain != 1.0:
            img = img * self.gain
        results['img'] = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        return results

    def __repr__(self):
        return f'{type(self).__name__}(gain={self.gain}, gamma={self.gamma})'


def run_point(base_cfg_path, ckpt, gain, gamma, workdir):
    cfg = Config.fromfile(base_cfg_path)
    cfg.load_from = ckpt
    cfg.work_dir = workdir
    cfg.log_level = 'ERROR'
    # Silence per-iteration logging; we only want the final metrics.
    if 'default_hooks' in cfg and 'logger' in cfg.default_hooks:
        cfg.default_hooks.logger.interval = 10000

    # Insert the perturbation directly before Pack3DDetInputs.
    pipeline = list(cfg.test_dataloader.dataset.pipeline)
    idx = next(i for i, t in enumerate(pipeline)
               if t['type'] == 'Pack3DDetInputs')
    pipeline.insert(idx, dict(
        type='DeterministicPhotometric', gain=gain, gamma=gamma))
    cfg.test_dataloader.dataset.pipeline = pipeline
    cfg.test_evaluator = cfg.val_evaluator

    runner = Runner.from_cfg(cfg)
    metrics = runner.test()
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', default='eval/lighting_sweep.json')
    ap.add_argument('--workdir', default='/workspace/sweep_out')
    args = ap.parse_args()

    # gain < 1 darkens (dusk); gamma > 1 darkens midtones nonlinearly.
    # 1.0/1.0 is the untouched baseline and must reproduce the headline run.
    points = [
        ('baseline',      1.00, 1.00),
        ('bright +20%',   1.20, 1.00),
        ('dim -20%',      0.80, 1.00),
        ('dim -40%',      0.60, 1.00),
        ('dusk -60%',     0.40, 1.00),
        ('night -80%',    0.20, 1.00),
        ('gamma 1.5',     1.00, 1.50),
        ('gamma 2.2',     1.00, 2.20),
    ]

    results = []
    for name, gain, gamma in points:
        print(f'\n===== {name}  gain={gain} gamma={gamma} =====', flush=True)
        try:
            m = run_point(args.config, args.ckpt, gain, gamma,
                          os.path.join(args.workdir, name.replace(' ', '_')))
            # Metric keys are prefixed by the evaluator; find them by suffix.
            mAP = next((v for k, v in m.items() if k.endswith('/mAP')), None)
            nds = next((v for k, v in m.items() if k.endswith('/NDS')), None)
            per_class = {
                k.split('/')[-1].replace('_AP_dist_2.0', ''): v
                for k, v in m.items() if k.endswith('_AP_dist_2.0')
            }
            row = dict(name=name, gain=gain, gamma=gamma,
                       mAP=mAP, NDS=nds, per_class_ap_2m=per_class)
            print(f'  mAP={mAP:.4f}  NDS={nds:.4f}', flush=True)
        except Exception as e:  # noqa: BLE001
            row = dict(name=name, gain=gain, gamma=gamma, error=str(e))
            print(f'  FAILED: {e}', flush=True)
        results.append(row)

        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)

    print('\n\n===== SUMMARY =====')
    print(f'{"setting":15s} {"gain":>5s} {"gamma":>6s} {"mAP":>8s} {"NDS":>8s}')
    base = next((r for r in results if r['name'] == 'baseline'), None)
    for r in results:
        if 'error' in r:
            print(f'{r["name"]:15s} {r["gain"]:5.2f} {r["gamma"]:6.2f}   ERROR')
            continue
        delta = ''
        if base and base.get('mAP') and r is not base:
            delta = f'  ({100*(r["mAP"]-base["mAP"])/base["mAP"]:+.1f}%)'
        print(f'{r["name"]:15s} {r["gain"]:5.2f} {r["gamma"]:6.2f} '
              f'{r["mAP"]:8.4f} {r["NDS"]:8.4f}{delta}')
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()

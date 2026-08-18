# Controlled lighting sweep — FCOS3D on nuScenes mini_val

Each point applies a deterministic brightness gain (and/or gamma) to every
image, then runs the **full official nuScenes evaluation**. Scene, geometry,
ground truth and model are identical across points; only the photometric
response changes.

The `baseline` point (gain 1.0, gamma 1.0) reproduces the unperturbed headline
run to four decimal places (mAP 0.2943, NDS 0.3217), confirming the transform
is a true no-op there and that every other row differs only because of the
perturbation.

## Overall

| Setting | gain | gamma | mAP | NDS | ΔmAP |
|---------|------|-------|-----|-----|------|
| bright +20% | 1.20 | 1.0 | 0.3022 | 0.3275 | **+2.7%** |
| **baseline** | 1.00 | 1.0 | **0.2943** | **0.3217** | — |
| dim −20% | 0.80 | 1.0 | 0.2874 | 0.3170 | −2.3% |
| dim −40% | 0.60 | 1.0 | 0.2706 | 0.3037 | −8.1% |
| dusk −60% | 0.40 | 1.0 | 0.1962 | 0.2586 | **−33.3%** |
| night −80% | 0.20 | 1.0 | 0.1202 | 0.1552 | **−59.2%** |
| gamma 1.5 | 1.00 | 1.5 | 0.2783 | 0.3073 | −5.4% |
| gamma 2.2 | 1.00 | 2.2 | 0.2141 | 0.2665 | −27.3% |

## Per-class AP (2 m matching threshold)

| Setting | car | pedestrian | truck | bus | motorcycle | bicycle | traffic_cone |
|---------|-----|------------|-------|-----|------------|---------|--------------|
| baseline | 0.680 | 0.578 | 0.522 | 0.647 | 0.452 | 0.332 | 0.740 |
| bright +20% | 0.683 | 0.573 | 0.555 | 0.637 | 0.440 | 0.274 | 0.765 |
| dim −20% | 0.680 | 0.575 | 0.464 | 0.653 | 0.466 | 0.347 | 0.716 |
| dim −40% | 0.648 | 0.550 | 0.472 | 0.665 | 0.439 | 0.351 | 0.680 |
| dusk −60% | 0.494 | 0.477 | 0.434 | **0.033** | 0.281 | 0.244 | 0.648 |
| night −80% | 0.377 | 0.358 | 0.209 | **0.007** | **0.000** | 0.144 | 0.468 |
| gamma 1.5 | 0.674 | 0.597 | 0.531 | 0.659 | 0.383 | 0.336 | 0.694 |
| gamma 2.2 | 0.613 | 0.541 | 0.424 | 0.549 | 0.145 | 0.203 | 0.601 |

## Findings

### Degradation is strongly nonlinear

Mild dimming is nearly free: −20% brightness costs 2.3% mAP, and −40% costs
8.1%. Past that the model falls off a cliff — −60% costs a third of all
accuracy and −80% costs nearly 60%. There is a usable operating band roughly
down to 0.6x brightness, and a collapse beyond it.

The practical implication for anyone validating perception in simulation: a
renderer only needs to land *within* that band to give trustworthy numbers.
Matching exposure precisely matters far less than avoiding gross
underexposure.

### Slightly brighter is slightly better

+20% brightness improves mAP by 2.7%. This suggests nuScenes' native exposure
sits marginally below this model's optimum, plausibly because the training
distribution includes a substantial share of dusk and night scenes.

### Buses fail catastrophically; cones and cars degrade gracefully

The class spread at dusk −60% is enormous:

| Class | baseline → dusk −60% | change |
|-------|---------------------|--------|
| bus | 0.647 → 0.033 | **−95%** |
| motorcycle | 0.452 → 0.281 | −38% |
| car | 0.680 → 0.494 | −27% |
| bicycle | 0.332 → 0.244 | −27% |
| truck | 0.522 → 0.434 | −17% |
| pedestrian | 0.578 → 0.477 | −17% |
| traffic_cone | 0.740 → 0.648 | **−12%** |

By −80%, motorcycle detection fails completely (AP 0.000) and bus is
effectively gone (0.007), while traffic cones still score 0.468 — higher than
*any* class's baseline except car and cone themselves.

A plausible explanation is that cones and pedestrians are recognised by
silhouette, which survives loss of contrast, whereas buses are large,
low-texture surfaces whose internal detail and boundaries wash out when
darkened. This is a hypothesis the data is consistent with, not something the
experiment establishes.

### Gamma is milder than gain at comparable darkening

gamma 2.2 (−27.3% mAP) sits between dim −40% and dusk −60%. Gamma compresses
midtones while preserving highlights, so bright, high-contrast objects survive
better than they do under a uniform gain reduction.

## Caveats

- `mini_val` is 2 scenes / 81 keyframes. Rare classes have small support:
  41 bus and 52 bicycle instances. The bus collapse is dramatic but rests on
  few samples, and bicycle's apparent *improvement* at −20%/−40%
  (0.332 → 0.347 → 0.351) is almost certainly noise rather than signal.
- A gain/gamma change is a **proxy** for a rendering gap. It alters photometric
  response only; geometry, texture, and sensor noise are untouched. These
  results describe sensitivity to exposure, and should not be read as
  measuring the sim-to-real gap itself.
- barrier, trailer and construction_vehicle are omitted throughout: they have
  no ground truth in this split (see `nuscenes_mini_fp32_rtx4090.md`).

## Reproducing

```bash
cd /workspace/mmdetection3d
python3 /workspace/sim2real/scripts/lighting_sweep.py \
    --config /workspace/sim2real/configs/fcos3d_nus_mini.py \
    --ckpt   /workspace/checkpoints/fcos3d_r101_nus.pth \
    --out    eval/lighting_sweep.json
```

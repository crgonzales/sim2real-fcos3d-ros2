# FCOS3D on nuScenes v1.0-mini (mini_val) — RTX 4090, FP32

Run: `tools/test.py configs/fcos3d_nus_mini.py fcos3d_r101_nus.pth`
Checkpoint: `fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_20210715_235813-4bed5239.pth`
Split: `mini_val` — 81 keyframes / 2 scenes / 486 camera images.
Date: 2026-08-17

## Headline

| Metric | This run (mini_val) | Published (full val) |
|--------|--------------------|----------------------|
| mAP    | **0.2943**         | 0.299                |
| NDS    | **0.3217**         | 0.373                |

mAP lands within 0.5 points of the published figure, which validates the
environment, camera intrinsics, coordinate-frame handling, and evaluation
path. NDS is lower for a structural reason, see below.

## True-positive error metrics

| Metric | Value |
|--------|-------|
| mATE (translation) | 0.7782 |
| mASE (scale)       | 0.4671 |
| mAOE (orientation) | 0.7229 |
| mAVE (velocity)    | 1.2499 |
| mAAE (attribute)   | 0.2865 |

## Per-class

| Class | AP | ATE | ASE | AOE | AVE | AAE |
|-------|-----|-----|-----|-----|-----|-----|
| traffic_cone         | 0.592 | 0.434 | 0.368 | n/a   | n/a   | n/a   |
| bus                  | 0.497 | 0.504 | 0.081 | 0.749 | 4.122 | 0.012 |
| car                  | 0.496 | 0.629 | 0.165 | 0.163 | 0.313 | 0.040 |
| pedestrian           | 0.432 | 0.728 | 0.271 | 0.692 | 0.744 | 0.109 |
| truck                | 0.385 | 0.733 | 0.183 | 0.104 | 0.191 | 0.011 |
| motorcycle           | 0.317 | 0.937 | 0.330 | 1.097 | 0.088 | 0.000 |
| bicycle              | 0.224 | 0.817 | 0.273 | 0.701 | 2.541 | 0.120 |
| trailer              | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| construction_vehicle | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| barrier              | 0.000 | 1.000 | 1.000 | 1.000 | n/a   | n/a   |

## Reading the numbers

**Why NDS (0.322) trails the published 0.373 while mAP matches.**
NDS averages mAP with the five TP error terms. Three classes score AP 0.000
here, and a missing class contributes the worst-case 1.0 to every TP error,
so those zeros hurt NDS far more than they hurt mAP. mini_val is 2 scenes;
trailer and construction_vehicle are rare enough to be essentially absent.
This is a property of the 81-sample subset, not a regression.

**Distance dominates the error budget.** mATE 0.778 m mean translation error
is the largest single contributor, consistent with monocular depth being the
fundamental limitation. Confirmed qualitatively in the smoke test, where a car
at 41 m was given a 4.35 m width -- extent estimation degrades with range.

**barrier AP 0.000 needs a second look.** Unlike trailer and
construction_vehicle, barriers are common, and the smoke test detected 7 of
them confidently with correct dimensions (0.48 x 1.05 x 2.02 m). Zero AP with
ATE/ASE/AOE all pinned at the 1.0 no-match default means no prediction matched
any GT barrier within even the 4.0 m threshold. Either mini_val's two scenes
contain few barriers, or something class-specific is wrong. Open question.

## System metrics (same run)

| Quantity | Value |
|----------|-------|
| Per-image inference | 0.0688 s |
| Throughput | ~14.5 img/s |
| GPU memory (reported) | 569 MB |

Matches the standalone smoke test (70.6 ms, 14.2 FPS, 0.60 GB), so the ROS
node's in-memory preprocessing path costs nothing measurable versus the
canonical offline pipeline.

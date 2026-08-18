# Code Review: FCOS3D ROS 2 node, evaluation harness, and results

**Review Date**: 2026-08-17
**Version**: 0.1.0
**Files Reviewed**:
- `src/fcos3d_ros2/fcos3d_ros2/detector_node.py`
- `src/fcos3d_ros2/fcos3d_ros2/nuscenes_publisher.py`
- `src/fcos3d_ros2/fcos3d_ros2/fp16_compat.py`
- `src/fcos3d_ros2/{setup.py,setup.cfg,package.xml}`
- `scripts/{setup_pod.sh,get_nuscenes_mini.sh,prepare_nuscenes.py,smoke_test.py,profile_system.py,lighting_sweep.py,render_demo.py,build_demo_video.py,pod.sh}`
- `configs/fcos3d_nus_mini.py`
- `eval/*` (results), `docs/PROJECT_REPORT.md`, `README.md`

**Plan**: no plan — TRIP was adopted mid-project, so this is the audit path
(`/TRIP-review`) applied retroactively to unplanned work. The project has no
`docs/ARCHI.md`; `CLAUDE.md` serves the equivalent role (pinned version matrix
and coordinate conventions) and was used for the architectural pass.

---

## Executive Summary

Reviews the complete CMPE 249 deliverable: a pretrained FCOS3D detector wrapped
as a ROS 2 Humble node, an evaluation and profiling harness, and the reported
results. All four assignment requirements are implemented and independently
verified. One previously-unverified correctness claim (the yaw-to-quaternion
conversion used for published poses) was tested during this review and is
correct. One overclaim was found in the report's experimental methodology.

**APPROVED with observations.**

---

## Changes Overview

Adds a ROS 2 package with two nodes (`detector_node`, `nuscenes_publisher`) and
an fp16 compatibility shim; a reproducible environment bootstrap; four analysis
scripts (smoke test, system profiler, lighting sweep, demo renderer); an
evaluation config for the nuScenes mini split; and the full result set plus
project report. Behaviour introduced: monocular RGB frames in on ROS topics,
`vision_msgs/Detection3DArray` plus RViz2 markers out, with per-frame latency
instrumentation.

---

## Findings

### Critical Issues

None.

### Major Issues

**M1 — Report overclaims that the GPU is the only variable in the
cross-environment comparison.**
`docs/PROJECT_REPORT.md` §6.2 and `eval/environment_comparison.md` both state
that holding the container, OS, driver family and library versions constant
means "the GPU is the only variable". This is not established. The measured
GPU utilisation is ~54%, so roughly 46% of wall time is host-side work
(preprocessing, kernel launch, box decoding, NMS). The RTX 4090 pod runs an
AMD EPYC 7K62 (96 logical CPUs); the A40 pod's CPU model was never recorded,
and `platform.processor()` captured only the uninformative string `x86_64`.
A slower host on the A40 pod could therefore contribute to the measured 2.0x
gap, and the report does not bound that contribution.

*Disposition*: open. The 2.0x figure is the honest measurement of
*end-to-end pod performance*, which is what the assignment asks for, but the
causal attribution to the GPU alone is stronger than the evidence. Recommend
softening the claim and recording `/proc/cpuinfo` in `profile_system.py`.

### Minor Issues

**m1 — Stale speculative comment, now falsified by verification.**
`detector_node.py:_yaw_to_quat_about_y` carries: *"Sign convention is the thing
to verify visually in RViz2 first — if boxes are consistently mirrored in
heading, negate `yaw` here."* This review verified the conversion numerically
against `CameraInstance3DBoxes.corners` at yaw ∈ {0, 0.5, π/2, 2.5, −1.0}:
max corner mismatch **0.0000 m** as written, versus up to **2.66 m** if
negated. The comment should state the verified result rather than invite an
incorrect "fix".
*Disposition*: addressed below.

**m2 — Unused import.** `nuscenes_publisher.py:21` imports `Optional`, never
used. *Disposition*: addressed below.

**m3 — Unused parameter.** `nuscenes_publisher.py:158` `_gt_markers(self,
sample, cam_token, stamp)` never reads `sample`; ground truth is fetched via
`cam_token`. *Disposition*: addressed below.

**m4 — No exception guard on the image callback.** `detector_node.on_image`
performs decode, preprocess and inference with no `try/except`. A single
malformed frame or transient CUDA error propagates out of the callback. rclpy
logs and the node survives, but the failure is noisy and the frame is lost
silently from the latency statistics.
*Disposition*: open — accepted for a coursework deliverable; would be a fix in
production.

**m5 — Publisher replays train and val together.** `nuscenes_publisher`
iterates `self.nusc.sample`, i.e. all 404 keyframes across all 10 mini scenes,
which mixes `mini_train` and `mini_val`. This is harmless for the live demo
(its purpose) and no reported metric comes from this path — all accuracy
numbers come from `tools/test.py` on `mini_val`. It would be wrong if anyone
later evaluated through the ROS path.
*Disposition*: open — documented here so the constraint is not rediscovered.

### Suggestions

**s1 — No automated test suite.** `scripts/smoke_test.py` is a manual
verification script, not a test. The yaw-quaternion check written for this
review is exactly the kind of invariant that should be a regression test, since
it guards a silent-failure mode. See coverage debt below.

**s2 — Record host CPU in profiles.** Adding `/proc/cpuinfo` model name and
`nproc` to `profile_system.py`'s `environment` block would close M1 for any
future comparison.

**s3 — `render_demo.py` and `profile_system.py` duplicate the preprocess
construction** already present in `detector_node._preprocess` (~10 lines, three
copies). Factoring it into the package would remove the drift risk.

---

## Checklist

- [x] 1. Functional Requirements — passed. All four assignment requirements
  implemented and verified end-to-end; interfaces match the documented topic
  table; yaw conversion verified numerically this review.
- [ ] 2. Code Quality — passed with caveats. m2 (unused import), m3 (unused
  parameter), s3 (duplicated preprocess block).
- [x] 3. Architectural Compliance — passed. No `ARCHI.md` exists (TRIP adopted
  mid-project); reviewed against `CLAUDE.md`'s version matrix and coordinate
  conventions, both of which the code honours — notably `gravity_center` for the
  bottom-centre origin and BGR channel order.
- [ ] 4. Error Handling — passed with caveats. m4: no guard on the image
  callback. Elsewhere handling is appropriate — `CameraInfo` absence is warned
  with throttling, GT lookup failure is caught, missing frames are skipped.
- [x] 5. Security — passed. Secret scan clean; the only `token` matches are
  nuScenes dataset sample identifiers. No credentials in the repository, per the
  submission requirement. SSH keys were never committed.
- [x] 6. Performance — passed. Profiled across two GPUs and two precisions;
  hot-path costs measured rather than assumed; the "not GPU-bound" conclusion is
  supported by utilisation data and correctly used to justify dropping TensorRT.

---

## Coverage Debt

Per the hard-to-cover policy, recorded rather than blocking:

| Item | Why untested | Risk |
|------|--------------|------|
| `detector_node` ROS callbacks | Needs a running ROS 2 graph + GPU; no CI available | Medium — verified manually live (2.016 Hz sustained, correct intrinsics ingested) |
| Yaw→quaternion conversion | Was untested; **verified numerically this review** | Now low — should be promoted to a regression test (s1) |
| `fp16_compat` patch | Requires CUDA | Low — exercised by both FP16 profile runs |
| Photometric transform | No unit test | Low — baseline point reproduces the unperturbed run to 4 decimals, which is a strong functional check |

---

## Verdict

**APPROVED with observations.**

The implementation is correct where it has been checked, and this review closed
the one open correctness question (yaw→quaternion) by verification rather than
inspection. The approval gate is not met literally — there is no unit test
suite, so "affected unit tests pass" cannot be evaluated; that is recorded as
coverage debt above rather than waived silently.

The one finding a future reader should carry forward is **M1**: the 2.0x
RTX 4090 vs A40 result is a sound measurement of end-to-end performance, but
the report attributes it to the GPU alone without having controlled for the
host CPU, and ~46% of wall time is host-side. The number is reportable; the
causal claim around it should be softened.

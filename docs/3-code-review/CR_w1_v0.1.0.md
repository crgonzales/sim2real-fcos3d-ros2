# Code Review: FCOS3D ROS 2 node, evaluation harness, and results

**Review Date**: 2026-08-19
**Version**: 0.1.0
**Reviewer**: AI review agent, run iteratively to convergence over four turns
**Files Reviewed**: entire tracked tree — `src/fcos3d_ros2/`, `scripts/`, `configs/`, `tests/`, `docs/`, `eval/`
**Plan**: no plan — TRIP adopted after implementation, reviewed as unplanned work against `docs/ARCHI.md`

Supersedes `CR_w1_v0.1.0-manual-prepass.md` (the manual pass that preceded the
AI review loop). Findings from that pass carried forward as `m4`/`m5`.

---

## Executive Summary

Full review of the CMPE 249 deliverable: a pretrained FCOS3D monocular 3D
detector wrapped as a ROS 2 Humble node, applied to nuScenes, and evaluated for
accuracy, latency and resource utilisation across two GPUs and two precisions.
Four review turns surfaced nine findings — four Major, five Minor — all of
which were addressed. **APPROVED.**

---

## Changes Overview

20 files / 2684 lines: the ROS 2 package (`detector_node`,
`nuscenes_publisher`, `fp16_compat`), a reproducible environment bootstrap,
five analysis scripts, an evaluation config, and 27 test cases. `git diff HEAD`
was empty throughout (work committed), so the reviewer read the tree directly.

---

## Findings

### Critical Issues

None, in any turn.

### Major Issues

**J1 — Default launch failed after the documented bootstrap.**
`detector.launch.py` defaulted `config_file` to `/opt/mmdetection3d` while
`setup_pod.sh` clones to `/workspace/mmdetection3d`; `checkpoint_file` had no
default at all. Anyone following the README would have `ros2 launch` fail
immediately. *Addressed* — defaults now match the bootstrap layout.

**J2 — The node's FP16 path omitted its own compatibility patch.**
`fp16_compat.py` existed and documented the `linalg.inv` failure precisely, but
only the offline profiler installed it. `detector_node` entered autocast
directly, so `fp16:=true` died on the first frame. *Addressed* — the node calls
`patch_points_img2cam_fp32()` when FP16 is enabled.

**J3 — Every ground-truth marker had permuted dimensions.**
Markers published scale `(w, h, l)`, but a nuScenes `Box`'s local axes — the
frame its `orientation` rotates — are `x=length, y=width, z=height` (confirmed
against `Box.corners` in the devkit). *Addressed* — now `Vector3(x=l, y=w, z=h)`.
Scope: visualisation only. The demo video draws GT via `Box.corners()` and was
always correct; no reported metric derives from this path.

**J4 — Reported process RSS included the benchmark corpus.**
The profiler decoded and cached all 100 frames before sampling, inflating RSS by
~432 MB relative to the depth-1 node. *Addressed* — frames are decoded one at a
time inside the timed region. **This reached published results**: re-measured
RSS is 1476.8 MB, not 1904.8 MB. See `eval/environment_comparison.md`.

**J5 — The per-frame decode fix regressed unreadable-frame handling.**
A self-inflicted regression: moving decode into the loop dropped the `None`
check `collect_frames` had, so a missing or corrupt file crashed on `.shape`
where it was previously skipped. *Addressed* — warned and skipped in both
warmup and measurement; frame count and throughput derive from frames actually
processed.

### Minor Issues

**n1 — Nonzero CUDA selections measured GPU 0.** NVML was pinned to index 0
while `--device` accepted `cuda:N`. *Addressed* over two turns — the first fix
was partial (the auto label and the measurement-loop `synchronize()` still hit
GPU 0), which the loop caught.

**n2 — Demo ground-truth failures were swallowed.** A failed lookup rendered as
"ground truth 0", producing a plausible but misleading video. *Addressed* — logs
the sample token and exception.

**n3 — `ARCHI.md` claimed no automated tests existed.** *Addressed.*

**n4 — New CUDA-ordinal logic lacked coverage.** *Addressed* — six parameterized
cases plus a static guard asserting no bare GPU-0 call survives in the profiler.

**n5 — The all-unreadable run still crashed.** Only `throughput_fps` was
guarded; empty lists still reached `statistics.mean` and `np.percentile`.
*Addressed* — exits with an actionable message before aggregation.

**n6 — "Dependency-free" tests were gated by the full stack.** `importorskip`
at module scope skipped the entire file, including the new portable tests. The
reviewer was right and the author's claim was wrong. *Addressed* — split into
`tests/test_conventions.py`.

### Carried forward from the manual pre-pass

**m4** — no `try/except` on `detector_node.on_image`; accepted for a coursework
deliverable, not reopened by the reviewer.
**m5** — `nuscenes_publisher` spans `mini_train` + `mini_val`; demo-only path,
no reported metric derives from it. Documented in-code.

### Suggestions

Unaddressed by choice: factoring the duplicated preprocess construction shared
by the node, profiler and renderer (~10 lines, three copies).

---

## Checklist

- [x] 1. Functional Requirements — pass
- [x] 2. Code Quality — pass
- [x] 3. Architectural Compliance — pass
- [x] 4. Error Handling — pass, subject to the accepted `m4` callback gap
- [x] 5. Security — pass; no credentials, no actionable vulnerability
- [x] 6. Performance — pass; CUDA device selection and frame accounting consistent

---

## Testing Gate

| Module | Cases | Needs the stack? | Result |
| --- | --- | --- | --- |
| `tests/test_conventions.py` | 7 | no | **7 passed** under pytest, laptop, no GPU/torch/mmdet3d |
| `tests/test_geometry.py` | 20 | yes | **20 passed** on the GPU host; file is byte-identical (md5 `fa521ba883f63a7a03175be17041543b`) to that run |

Not claimed: a single combined 27-case run. Both Runpod pods were reclaimed by
their hosts before the final edits, and the reviewer accepted this as a stated
environment limitation rather than an unverified assertion.

Lint and type-check are not configured for this project. `colcon build`
succeeds and both node executables register.

---

## Verdict

**APPROVED.**

Four turns, nine findings, all addressed. Two are worth carrying forward. First,
**J4 changed a published number** — the reported RSS was a property of the
benchmark harness, not the deployed node; the report and
`eval/environment_comparison.md` carry an explicit correction rather than a
silent edit, and the A40 could not be re-measured (host capacity), so the
comparison table deliberately keeps both GPUs on the original identical
methodology with the ~2 ms decode offset documented. The 2.0x headline is
unaffected either way.

Second, **J5 and the two-turn resolution of n1 both came from fixes introducing
or half-completing problems** — evidence that the iterative loop earned its
cost rather than merely ratifying the first pass.

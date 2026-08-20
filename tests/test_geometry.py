"""Regression tests for the conventions that fail silently.

Every test here guards a failure mode that actually occurred during this
project and produced plausible-looking wrong output rather than an error.
See CONVENTIONS.md and docs/ARCHI.md section 8.

Run on a machine with the full stack:
    python3 -m pytest tests/ -v
Tests requiring torch/mmdet3d skip cleanly elsewhere.
"""

import glob
import os

import numpy as np
import pytest

torch = pytest.importorskip('torch', reason='needs the inference stack')
mmdet3d = pytest.importorskip('mmdet3d', reason='needs the inference stack')

from mmdet3d.structures import CameraInstance3DBoxes  # noqa: E402


def _quat_to_R(w, x, y, z):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _yaw_to_quat_about_y(yaw):
    """Mirror of Fcos3dDetectorNode._yaw_to_quat_about_y.

    Duplicated rather than imported because importing the node pulls rclpy and
    a live ROS environment, which this test does not need.
    """
    half = yaw / 2.0
    return (np.cos(half), 0.0, np.sin(half), 0.0)


# --------------------------------------------------------------------- yaw
@pytest.mark.parametrize('yaw', [0.0, 0.5, np.pi / 2, 2.5, -1.0, 3.0, -2.2])
def test_yaw_quaternion_matches_mmdet3d_corners(yaw):
    """The published quaternion must reproduce mmdet3d's own box corners.

    Guards a mirrored-heading bug: a wrong sign here rotates every published
    box the wrong way while everything still looks superficially reasonable.
    """
    box = CameraInstance3DBoxes(
        torch.tensor([[0.0, 0.0, 20.0, 2.0, 1.6, 4.5, yaw]]))
    reference = box.corners.numpy()[0]
    center = box.gravity_center.numpy()[0]
    w, h, l = box.dims.numpy()[0]

    local = np.array([[ix * w / 2, iy * h / 2, iz * l / 2]
                      for ix in (-1, 1) for iy in (-1, 1) for iz in (-1, 1)])
    ours = (_quat_to_R(*_yaw_to_quat_about_y(yaw)) @ local.T).T + center

    # Compare as unordered point sets: corner ordering is not part of the contract.
    worst = max(np.min(np.linalg.norm(ours - p, axis=1)) for p in reference)
    assert worst < 1e-4, f'corner mismatch {worst:.4f} m at yaw={yaw}'


def test_negated_yaw_is_actually_wrong():
    """Negating the yaw must NOT also pass — otherwise the test proves nothing."""
    yaw = 0.5
    box = CameraInstance3DBoxes(
        torch.tensor([[0.0, 0.0, 20.0, 2.0, 1.6, 4.5, yaw]]))
    reference = box.corners.numpy()[0]
    center = box.gravity_center.numpy()[0]
    w, h, l = box.dims.numpy()[0]
    local = np.array([[ix * w / 2, iy * h / 2, iz * l / 2]
                      for ix in (-1, 1) for iy in (-1, 1) for iz in (-1, 1)])
    flipped = (_quat_to_R(*_yaw_to_quat_about_y(-yaw)) @ local.T).T + center
    worst = max(np.min(np.linalg.norm(flipped - p, axis=1)) for p in reference)
    assert worst > 1.0, 'a negated yaw should be clearly wrong, not equivalent'


# ------------------------------------------------------------------ origin
def test_gravity_center_is_half_a_height_above_the_raw_tensor():
    """Boxes use origin (0.5, 1.0, 0.5) -- bottom centre, not centre.

    Using tensor[:, :3] directly drops every box by half its height. y is down
    in the camera frame, so the true centre is at a smaller y.
    """
    height = 1.6
    box = CameraInstance3DBoxes(
        torch.tensor([[1.0, 2.0, 20.0, 2.0, height, 4.5, 0.3]]))
    raw = box.tensor.numpy()[0, :3]
    gravity = box.gravity_center.numpy()[0]

    assert gravity[0] == pytest.approx(raw[0]), 'x must not move'
    assert gravity[2] == pytest.approx(raw[2]), 'z must not move'
    assert gravity[1] == pytest.approx(raw[1] - height / 2, abs=1e-5), \
        'gravity centre must sit half a height above the bottom-centre origin'


def test_dims_order_is_width_height_length():
    """dims is (x_size, y_size, z_size) = (width, height, length).

    y is the down axis, so y_size is height. Swapping width and length silently
    corrupts every IoU computation.
    """
    box = CameraInstance3DBoxes(
        torch.tensor([[0.0, 0.0, 20.0, 2.0, 1.6, 4.5, 0.0]]))
    w, h, l = box.dims.numpy()[0]
    assert (w, h, l) == pytest.approx((2.0, 1.6, 4.5))


# ------------------------------------------------------------- camera match
@pytest.mark.parametrize('pattern,should_match_left', [
    ('*CAM_FRONT*', True),        # the bug: loose glob also matches _LEFT
    ('*__CAM_FRONT__*', False),   # the fix: delimited match
])
def test_camera_glob_must_be_delimited(tmp_path, pattern, should_match_left):
    """A loose camera glob pairs one camera's image with another's intrinsics.

    This actually happened: '*CAM_FRONT*' matched CAM_FRONT_LEFT, which sorts
    first because 'L' < '_'. Detections fell from 200 to 19 with no error.
    """
    for cam in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']:
        (tmp_path / f'scene__{cam}__1532402927612460.jpg').write_bytes(b'x')

    hits = sorted(glob.glob(os.path.join(str(tmp_path), f'{pattern}.jpg')))
    matched_left = any('CAM_FRONT_LEFT' in os.path.basename(h) for h in hits)
    assert matched_left is should_match_left

    if not should_match_left:
        assert len(hits) == 1
        assert '__CAM_FRONT__' in os.path.basename(hits[0])


def test_left_sorts_before_underscore():
    """The reason the loose glob picked the wrong camera, pinned as a fact."""
    assert 'CAM_FRONT_LEFT' < 'CAM_FRONT__1532'
    assert ord('L') < ord('_')


# ------------------------------------------------------------- photometric
def _photometric(img, gain, gamma):
    """Mirror of DeterministicPhotometric.transform / publisher._photometric."""
    if gain == 1.0 and gamma == 1.0:
        return img
    out = img.astype(np.float32) / 255.0
    if gamma != 1.0:
        out = np.power(out, gamma)
    if gain != 1.0:
        out = out * gain
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def test_photometric_is_a_noop_at_unity():
    """The sweep baseline must be byte-identical to the unperturbed input.

    If this drifts, the baseline point stops being a control and every
    reported delta in the lighting sweep loses its reference.
    """
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (32, 48, 3), dtype=np.uint8)
    assert np.array_equal(_photometric(img, 1.0, 1.0), img)


@pytest.mark.parametrize('gain', [0.8, 0.6, 0.4, 0.2])
def test_photometric_gain_darkens_monotonically(gain):
    rng = np.random.default_rng(1)
    img = rng.integers(1, 256, (32, 48, 3), dtype=np.uint8)
    out = _photometric(img, gain, 1.0)
    assert out.mean() < img.mean()
    assert out.max() <= 255 and out.min() >= 0


def test_photometric_gamma_above_one_darkens_midtones():
    mid = np.full((8, 8, 3), 128, dtype=np.uint8)
    assert _photometric(mid, 1.0, 2.2).mean() < mid.mean()


# ------------------------------------------------------------------ config
def test_fcos3d_config_expects_bgr():
    """bgr_to_rgb=False with caffe means -- feeding RGB degrades accuracy silently."""
    cfg_path = os.environ.get(
        'FCOS3D_CONFIG',
        '/workspace/mmdetection3d/configs/fcos3d/'
        'fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d.py')
    if not os.path.exists(cfg_path):
        pytest.skip(f'config not present at {cfg_path}')
    from mmengine.config import Config
    cfg = Config.fromfile(cfg_path)
    pre = cfg.model.data_preprocessor
    assert pre.bgr_to_rgb is False, 'model expects BGR input'
    assert pre['mean'][0] == pytest.approx(103.530), 'caffe BGR means'


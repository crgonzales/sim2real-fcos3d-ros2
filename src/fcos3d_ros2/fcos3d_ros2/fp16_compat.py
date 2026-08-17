"""FP16 compatibility patch for FCOS3D box decoding.

Problem
-------
Running FCOS3D under ``torch.autocast(float16)`` fails in post-processing:

    fcos_mono3d_head._predict_by_feat_single
      -> points_img2cam
        -> torch.inverse(pad_cam2img)
    RuntimeError: linalg.inv: Low precision dtypes not supported. Got Half

``points_img2cam`` un-projects predicted image-space points to camera space by
inverting the intrinsics. Autocast makes those tensors half, and cuSOLVER has
no half-precision matrix inverse.

Fix
---
Only the *network* benefits from fp16; the decode is a handful of tiny matrix
ops whose cost is negligible and whose numerical conditioning genuinely wants
fp32 (inverting an intrinsics matrix with ~1e3 focal lengths in half precision
would be poor even if it were supported). So we force just that function back
to fp32.

The patch targets the name as imported into the head's module namespace, which
is the reference actually called at runtime.

Contributor: Carlos Gonzales
"""

import numpy as np
import torch

_PATCHED = False


def patch_points_img2cam_fp32() -> bool:
    """Force FCOS3D's image->camera un-projection to run in fp32.

    Idempotent. Returns True if the patch is in place.
    """
    global _PATCHED
    if _PATCHED:
        return True

    import mmdet3d.models.dense_heads.fcos_mono3d_head as head_mod

    original = head_mod.points_img2cam

    def _to_fp32(x):
        # points_img2cam is wrapped in mmdet3d's @array_converter, so either
        # argument may arrive as a torch.Tensor or a numpy.ndarray.
        if isinstance(x, torch.Tensor):
            return x.float()
        if isinstance(x, np.ndarray):
            return x.astype(np.float32)
        return x

    def points_img2cam_fp32(points, cam2img):
        # enabled=False disables autocast for this region, so the inverse runs
        # in true fp32 rather than being re-cast to half by an enclosing
        # autocast context.
        with torch.autocast(device_type='cuda', enabled=False):
            return original(_to_fp32(points), _to_fp32(cam2img))

    head_mod.points_img2cam = points_img2cam_fp32
    _PATCHED = True
    return True

# FCOS3D evaluation config for the nuScenes v1.0-mini split.
#
# Why this file exists
# --------------------
# mmdet3d's NuScenesDataset.METAINFO hardcodes 'version': 'v1.0-trainval'.
# That class default wins over the 'version': 'v1.0-mini' recorded in the
# generated info .pkl, so NuScenesMetric builds its NuScenes object with the
# wrong version and dies with:
#     AssertionError: Database version not found: data/nuscenes/v1.0-trainval
#
# NuScenesMetric already knows how to map versions to eval splits --
#     {'v1.0-mini': 'mini_val', 'v1.0-trainval': 'val'}
# -- it just needs metainfo['version'] to be right, which we set here.
#
# Run from the mmdetection3d repo root (its configs use a relative data_root):
#   cd /workspace/mmdetection3d
#   python3 tools/test.py /workspace/sim2real/configs/fcos3d_nus_mini.py \
#       /workspace/checkpoints/fcos3d_r101_nus.pth \
#       --work-dir /workspace/eval_out
#
# Contributor: Carlos Gonzales

_base_ = [
    '/workspace/mmdetection3d/configs/fcos3d/'
    'fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d.py'
]

class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

# The 'version' key is the whole point of this override.
metainfo = dict(classes=class_names, version='v1.0-mini')

val_dataloader = dict(dataset=dict(metainfo=metainfo))
test_dataloader = dict(dataset=dict(metainfo=metainfo))

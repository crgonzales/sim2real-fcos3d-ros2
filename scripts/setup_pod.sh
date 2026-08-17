#!/usr/bin/env bash
# Bootstrap a Runpod pod for the FCOS3D / ROS 2 sim2real project.
#
# Run this ON THE POD, not on your Mac:
#   bash scripts/setup_pod.sh
#
# Assumes the pod was created from:
#   runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04
# which already provides Ubuntu 22.04 + Python 3.10 + CUDA 11.8 + torch 2.1.0.
# That exact combination is what makes the prebuilt mmcv wheel usable and what
# allows ROS 2 Humble to install from apt. Do not "upgrade" the base image:
# Ubuntu 24.04 means ROS 2 Jazzy and a Python version with no mmcv wheel.
#
# Contributor: Carlos Gonzales
set -euo pipefail

WS=/workspace
CKPT_DIR=$WS/checkpoints
MMDET3D_DIR=$WS/mmdetection3d

log() { echo -e "\n\033[1;36m==> $*\033[0m"; }

# --------------------------------------------------------------- sanity check
log "GPU / driver"
nvidia-smi || { echo "No GPU visible -- wrong pod type?"; exit 1; }
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"

# ------------------------------------------------------------------ ROS 2 Humble
log "Installing ROS 2 Humble"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
    curl gnupg2 lsb-release software-properties-common git wget \
    ffmpeg libsm6 libxext6 libgl1

if [ ! -f /usr/share/keyrings/ros-archive-keyring.gpg ]; then
  curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
      > /etc/apt/sources.list.d/ros2.list
  apt-get update -qq
fi

apt-get install -y -qq --no-install-recommends \
    ros-humble-ros-base \
    ros-humble-vision-msgs \
    ros-humble-visualization-msgs \
    ros-humble-cv-bridge \
    ros-humble-rosbag2 \
    ros-humble-rosbag2-storage-mcap \
    ros-humble-foxglove-bridge \
    python3-colcon-common-extensions

# ------------------------------------------------------------------ OpenMMLab
# Version bounds come from mmdet3d/__init__.py asserts, NOT the install docs.
# The docs' `mim install 'mmcv>=2.0.0rc4'` resolves to 2.2.x and fails on import.
log "Installing OpenMMLab stack (pinned)"

# CRITICAL: this constraints file is applied to EVERY pip call below, from the
# very first one. It must never be used as an after-the-fact repair.
#
# Why: mmcv 2.1.0's compiled extensions are built against the numpy 1.x ABI.
# If any transitive dependency installs numpy 2.x first, the fix is NOT to
# downgrade afterwards -- pip overwrites a compiled package's files without
# removing the old tree, leaving a mix of both versions that fails in ways that
# look like unrelated bugs:
#   numpy   -> two dist-info dirs; pip reports "already satisfied" while the
#              files on disk are the wrong version
#   scipy   -> TypeError in interpolate/_fitpack_impl.py (stale .so vs new .py)
#   mpl     -> ImportError: cannot import name 'mplDeprecation'
# Recovering from that requires deleting the package directories by hand.
# Constraining from the start avoids the entire class of problem.
cat > /tmp/constraints.txt <<'CONSTRAINTS'
numpy==1.26.4
scipy==1.13.1
numba==0.59.1
llvmlite==0.42.0
scikit-image==0.22.0
scikit-learn==1.3.2
pandas==2.1.4
matplotlib==3.5.3
CONSTRAINTS
PIP="pip install -q --no-cache-dir -c /tmp/constraints.txt"

$PIP -r /tmp/constraints.txt
$PIP mmengine==0.10.4
$PIP mmcv==2.1.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html
$PIP mmdet==3.3.0

# mmdet3d is installed WITHOUT its dependency closure, on purpose.
# Its declared deps pull open3d (~400 MB), which pulls Flask, which tries to
# uninstall Ubuntu's distutils-installed `blinker` and hard-fails with:
#   "Cannot uninstall 'blinker'. It is a distutils installed project..."
# open3d is only used for interactive 3D visualization, which we never do on a
# headless pod. mmdet3d itself is a pure-Python wheel, so --no-deps is safe;
# we then install the deps we actually use, with --ignore-installed blinker so
# anything that still wants Flask can't trip over the system package.
$PIP --no-deps mmdet3d==1.4.0

# A pip constraints file is used instead of bare pins because the numpy 2.x
# problem is transitive and cascades. What happened without it:
#   1. nuscenes-devkit/scikit-image pulled numpy 2.2.6, breaking mmcv's ABI.
#   2. Downgrading numpy alone then broke scipy, because pip had installed a
#      scipy wheel compiled against numpy 2.x
#      ("ValueError: All ufuncs must have type numpy.ufunc" via mmdet's
#       hungarian_assigner -> scipy.optimize).
# Constraints apply to the whole transitive closure, so every binary package
# resolves to a numpy-1.x-compatible build in one pass.
# nvidia-ml-py provides the pynvml module used for GPU utilisation sampling in
# scripts/profile_system.py (without it those fields come back null).
# lyft-dataset-sdk is required despite being unrelated to nuScenes:
# mmdet3d.evaluation imports lyft_eval at module scope, so the whole evaluation
# package fails to import without it.
# --ignore-installed blinker: Ubuntu ships blinker as a distutils project, so
# pip cannot uninstall it and Flask (via nuscenes-devkit) would hard-fail.
$PIP --ignore-installed blinker \
    nuscenes-devkit==1.1.11 pyquaternion psutil tabulate \
    plyfile trimesh networkx \
    nvidia-ml-py \
    lyft-dataset-sdk

log "Verifying version matrix"
python3 - <<'PY'
import mmdet3d, mmcv, mmdet, mmengine, torch, numpy
print(f"  torch    {torch.__version__}")
print(f"  numpy    {numpy.__version__}")
print(f"  mmengine {mmengine.__version__}")
print(f"  mmcv     {mmcv.__version__}")
print(f"  mmdet    {mmdet.__version__}")
print(f"  mmdet3d  {mmdet3d.__version__}")
print("  VERSION MATRIX OK")
PY

# ------------------------------------------------------------------ configs
# We install mmdet3d from pip for the library, but clone the repo for its
# configs/ directory, which pip does not ship.
log "Cloning mmdetection3d for configs"
if [ ! -d "$MMDET3D_DIR" ]; then
  git clone --depth 1 --branch v1.4.0 \
      https://github.com/open-mmlab/mmdetection3d.git "$MMDET3D_DIR"
fi

# ------------------------------------------------------------------ checkpoint
log "Downloading FCOS3D checkpoint"
mkdir -p "$CKPT_DIR"
CKPT_URL="https://download.openmmlab.com/mmdetection3d/v0.1.0_models/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_20210715_235813-4bed5239.pth"
CKPT_PATH="$CKPT_DIR/fcos3d_r101_nus.pth"
[ -f "$CKPT_PATH" ] || wget -q --show-progress -O "$CKPT_PATH" "$CKPT_URL"
ls -lh "$CKPT_PATH"

# ------------------------------------------------------------------ ROS env
log "Wiring shell environment"
grep -q 'ros/humble/setup.bash' ~/.bashrc || {
  echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
  echo "[ -f $WS/sim2real/install/setup.bash ] && source $WS/sim2real/install/setup.bash" >> ~/.bashrc
}

cat <<EOF

==============================================================
 Setup complete.

 Next:
   1. source /opt/ros/humble/setup.bash
   2. cd $WS/sim2real && colcon build --symlink-install
   3. bash scripts/get_nuscenes_mini.sh     # dataset
   4. ros2 launch fcos3d_ros2 detector.launch.py \\
        checkpoint_file:=$CKPT_PATH

 Checkpoint: $CKPT_PATH
 Configs:    $MMDET3D_DIR/configs/fcos3d/
==============================================================
EOF

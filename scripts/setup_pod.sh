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
pip install -q --no-cache-dir "numpy<2.0"
pip install -q --no-cache-dir mmengine==0.10.4
pip install -q --no-cache-dir mmcv==2.1.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html
pip install -q --no-cache-dir mmdet==3.3.0
pip install -q --no-cache-dir mmdet3d==1.4.0
pip install -q --no-cache-dir \
    nuscenes-devkit==1.1.11 pyquaternion psutil pynvml tabulate

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

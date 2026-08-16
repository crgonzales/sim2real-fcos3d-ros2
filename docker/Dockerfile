# Verified pins (2026-08-16). The version bounds below are read from
# mmdet3d/__init__.py asserts, NOT from the install docs. The official docs say
# `mim install 'mmcv>=2.0.0rc4'`, which resolves to mmcv 2.2.x and fails
# mmdet3d's own assert on import. Do not "upgrade" these pins.
#
#   mmengine >=0.8.0,<1.0.0
#   mmcv     >=2.0.0rc4,<2.2.0
#   mmdet    >=3.0.0rc5,<3.4.0
#
# Ubuntu 22.04 -> Python 3.10 -> ROS 2 Humble -> prebuilt mmcv cp310 wheel.
# This chain is why 22.04 is non-negotiable.

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# Ampere (RTX 4090 is 8.9). Set explicitly so mmcv/torch ops build for the right arch.
ENV TORCH_CUDA_ARCH_LIST="8.9"
ENV FORCE_CUDA=1

# ---------------------------------------------------------------- base system
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gnupg2 lsb-release software-properties-common \
        git wget build-essential ninja-build \
        python3-pip python3-dev \
        ffmpeg libsm6 libxext6 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------ ROS 2 Humble
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
        http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
        > /etc/apt/sources.list.d/ros2.list

RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-ros-base \
        ros-humble-vision-msgs \
        ros-humble-visualization-msgs \
        ros-humble-cv-bridge \
        ros-humble-rviz2 \
        ros-humble-rosbag2 \
        ros-humble-rosbag2-storage-mcap \
        python3-colcon-common-extensions \
        python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------ PyTorch
# cu118 to match the prebuilt mmcv wheel index below.
RUN pip3 install --no-cache-dir \
        torch==2.1.0 torchvision==0.16.0 \
        --index-url https://download.pytorch.org/whl/cu118

# ------------------------------------------------------------------ OpenMMLab
# mmcv from the prebuilt wheel index -- building from source costs ~30-40 min.
RUN pip3 install --no-cache-dir mmengine==0.10.4 && \
    pip3 install --no-cache-dir mmcv==2.1.0 \
        -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html && \
    pip3 install --no-cache-dir mmdet==3.3.0 && \
    pip3 install --no-cache-dir mmdet3d==1.4.0

# nuScenes devkit for the official mAP/NDS evaluation.
# numpy pinned <2.0: mmcv 2.1.0 and nuscenes-devkit both break on the numpy 2 ABI.
RUN pip3 install --no-cache-dir \
        "numpy<2.0" \
        nuscenes-devkit==1.1.11 \
        pyquaternion \
        psutil \
        pynvml \
        tabulate

# Fail the build early and loudly if the pin matrix is wrong, rather than
# discovering it at runtime on the GPU box.
RUN python3 -c "\
import mmdet3d, mmcv, mmdet, mmengine, torch; \
print('torch   ', torch.__version__); \
print('mmengine', mmengine.__version__); \
print('mmcv    ', mmcv.__version__); \
print('mmdet   ', mmdet.__version__); \
print('mmdet3d ', mmdet3d.__version__); \
print('VERSION MATRIX OK')"

WORKDIR /ws
RUN echo 'source /opt/ros/humble/setup.bash' >> /root/.bashrc && \
    echo '[ -f /ws/install/setup.bash ] && source /ws/install/setup.bash' >> /root/.bashrc

CMD ["/bin/bash"]

FROM ros:humble-desktop

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ENV MPLBACKEND=Agg
ENV ROS_WS=/workspace
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-colcon-common-extensions \
    python3-pytest \
    python3-vcstool \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-tf-transformations \
    ros-humble-xacro \
    ros-humble-ros2bag \
    ros-humble-ros2topic \
    gazebo \
    git \
    tmux \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install \
    casadi \
    cvxpy \
    matplotlib \
    numpy \
    pyyaml \
    pytest \
    scipy

COPY . /workspace

RUN source /opt/ros/${ROS_DISTRO}/setup.bash && \
    colcon build --symlink-install

RUN chmod +x /workspace/docker/entrypoint.sh \
    /workspace/docker/run_validation_suite.sh \
    /workspace/docker/run_gazebo_suite.sh \
    /workspace/docker/run_full_pipeline.sh

ENTRYPOINT ["/workspace/docker/entrypoint.sh"]
CMD ["bash"]

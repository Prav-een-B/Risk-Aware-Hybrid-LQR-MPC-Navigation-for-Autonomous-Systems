#!/usr/bin/env bash
set -euo pipefail

source /opt/ros/"${ROS_DISTRO}"/setup.bash

if [[ -f /workspace/src/hybrid_controller/setup.py && ! -f /workspace/install/setup.bash ]]; then
  echo "[entrypoint] Building ROS workspace"
  cd /workspace
  colcon build --symlink-install
fi

if [[ -f /workspace/install/setup.bash ]]; then
  source /workspace/install/setup.bash
fi

exec "$@"

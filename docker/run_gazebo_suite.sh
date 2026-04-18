#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/workspace}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-${ROOT_DIR}/artifacts}
RUN_LABEL=${RUN_LABEL:-gazebo_hybrid}
RUN_ID=$(date +%Y%m%d_%H%M%S)
TMP_DIR="${ARTIFACT_ROOT}/tmp_${RUN_LABEL}_${RUN_ID}"

cd "${ROOT_DIR}"
mkdir -p "${TMP_DIR}/rosbags" "${TMP_DIR}/ros_logs" "${ARTIFACT_ROOT}"
export ROS_LOG_DIR="${TMP_DIR}/ros_logs"

ros2 bag record \
  -o "${TMP_DIR}/rosbags/hybrid_topics" \
  /cmd_vel \
  /current_reference \
  /hybrid/blend_weight \
  /hybrid/mode \
  /hybrid/predicted_path \
  /hybrid/reference_pose \
  /hybrid/risk \
  /mpc_obstacles \
  /odom \
  /reference_trajectory \
  >/dev/null 2>&1 &
BAG_PID=$!

cleanup() {
  if kill -0 "${BAG_PID}" >/dev/null 2>&1; then
    kill -INT "${BAG_PID}" >/dev/null 2>&1 || true
    wait "${BAG_PID}" || true
  fi
}
trap cleanup EXIT

LAUNCH_LOG="${TMP_DIR}/hybrid_gazebo_launch.log"
ros2 launch hybrid_controller hybrid_gazebo.launch.py \
  headless:=true \
  duration:=20.0 \
  trajectory_type:=checkpoint_path \
  checkpoint_preset:=warehouse \
  >"${LAUNCH_LOG}" 2>&1

cleanup
trap - EXIT

python3 tools/collect_results.py \
  --destination "${ARTIFACT_ROOT}" \
  --label "${RUN_LABEL}" \
  --source logs \
  --source outputs \
  --source "${TMP_DIR}" \
  --metadata stage=ros2_gazebo_validation

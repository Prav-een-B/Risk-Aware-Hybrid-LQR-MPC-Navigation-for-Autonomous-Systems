#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/workspace}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-${ROOT_DIR}/artifacts}
RUN_LABEL=${RUN_LABEL:-validation_suite}

cd "${ROOT_DIR}"
mkdir -p "${ARTIFACT_ROOT}" logs outputs evaluation/results

python3 -m pytest tests -q
python3 -m py_compile \
  run_simulation.py \
  evaluation/statistical_runner.py \
  tools/collect_results.py \
  src/hybrid_controller/hybrid_controller/nodes/hybrid_node.py \
  src/hybrid_controller/hybrid_controller/nodes/kinematic_sim_node.py

python3 run_simulation.py --mode lqr --duration 8.0 --trajectory figure8
python3 run_simulation.py --mode mpc --duration 8.0 --scenario corridor --trajectory slalom
python3 run_simulation.py --mode hybrid --duration 8.0 --scenario sparse --trajectory checkpoint_path --checkpoint-preset warehouse
python3 run_simulation.py --mode adaptive --duration 8.0 --scenario default --trajectory figure8
python3 run_simulation.py --mode hybrid_adaptive --duration 8.0 --scenario default --trajectory figure8

python3 evaluation/statistical_runner.py \
  --configs 4 \
  --modes lqr mpc hybrid adaptive hybrid_adaptive \
  --scenario corridor \
  --duration 8.0 \
  --output evaluation/results/docker_validation

python3 tools/collect_results.py \
  --destination "${ARTIFACT_ROOT}" \
  --label "${RUN_LABEL}" \
  --source logs \
  --source outputs \
  --source evaluation/results \
  --metadata stage=standalone_validation

#!/usr/bin/env bash
set -euo pipefail

/workspace/docker/run_validation_suite.sh
/workspace/docker/run_gazebo_suite.sh

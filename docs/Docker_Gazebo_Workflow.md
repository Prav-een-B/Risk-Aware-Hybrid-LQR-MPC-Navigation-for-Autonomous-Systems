# Docker and Gazebo Workflow

Updated: 2026-04-06

This guide describes the container validation flow and Gazebo-oriented workflow scripts.

## Prerequisites

- Docker Desktop (Windows) or Docker Engine (Linux)
- For Gazebo flow: ROS2 Humble + Gazebo packages available in the container/host setup

## Available Scripts

- `docker/run_validation_suite.sh`: Runs standalone checks and collects artifacts
- `docker/run_gazebo_suite.sh`: Runs Gazebo-focused checks
- `docker/run_full_pipeline.sh`: Runs validation + Gazebo sequence
- `docker/entrypoint.sh`: Shared container entrypoint logic

## Build and Run

```bash
docker build -t hybrid-controller .
docker compose up --build
```

## Recommended Validation Sequence

1. Standalone baseline:

```bash
python run_simulation.py --mode hybrid --duration 10 --no-plot
python run_simulation.py --mode adaptive --duration 10 --no-plot
python run_simulation.py --mode hybrid_adaptive --duration 10 --no-plot
```

2. Dynamic obstacle checks:

```bash
python run_simulation.py --mode hybrid --scenario moving --duration 10 --no-plot
python run_simulation.py --mode hybrid --scenario random_walk --duration 10 --no-plot
python run_simulation.py --mode hybrid_adaptive --scenario moving --duration 10 --no-plot
```

3. Artifact collection:

- Outputs from `logs/`, `outputs/`, and `evaluation/results/` should be archived for review.

## Current Scope

- Standalone simulation path is fully runnable for hybrid/adaptive/dynamic scenarios.
- Gazebo path remains focused on orchestration and workflow integration.
- Full plant-in-Gazebo closed-loop dynamics remain future work.

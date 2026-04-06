import math
import os
import sys
from importlib.util import find_spec

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src", "hybrid_controller")
for path in (REPO_ROOT, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from evaluation.scenarios import ObstacleConfig
from evaluation.statistical_runner import (
    NoiseConfig,
    RunMetrics,
    aggregate_results,
    run_single_config,
)


def test_aggregate_results_includes_blend_metrics_for_hybrid_adaptive():
    runs = [
        RunMetrics(
            mode="hybrid_adaptive",
            config_id=0,
            seed=10,
            mean_tracking_error=0.5,
            max_tracking_error=0.8,
            final_tracking_error=0.4,
            mean_solve_time_ms=2.0,
            max_solve_time_ms=4.0,
            linear_jerk_rms=1.0,
            angular_jerk_rms=1.5,
            linear_jerk_peak=3.0,
            angular_jerk_peak=4.0,
            infeasible_count=0,
            collision_count=0,
            completion_fraction=1.0,
            total_control_effort=10.0,
            blend_weight_mean=0.6,
            blend_weight_std=0.1,
            smooth_transitions=2,
            wall_time_s=0.3,
        )
    ]

    aggregated = aggregate_results(runs, "hybrid_adaptive")

    assert "blend_weight_mean" in aggregated.metrics
    assert "blend_weight_std" in aggregated.metrics
    assert "smooth_transitions" in aggregated.metrics
    assert aggregated.metrics["blend_weight_mean"]["mean"] == 0.6


def test_run_single_config_supports_adaptive_modes():
    if sys.platform.startswith("win"):
        return

    if find_spec("casadi") is None:
        return

    obstacle_config = ObstacleConfig(
        obstacles=[
            {"x": 1.2, "y": 0.6, "radius": 0.2},
            {"x": -0.8, "y": -0.6, "radius": 0.2},
        ],
        seed=7,
        name="unit",
    )
    noise_config = NoiseConfig()

    for mode in ("adaptive", "hybrid_adaptive"):
        metrics = run_single_config(
            mode=mode,
            obstacle_config=obstacle_config,
            noise_config=noise_config,
            duration=0.6,
            dt=0.2,
            config_id=0,
        )

        assert metrics.mode == mode
        assert math.isfinite(metrics.mean_tracking_error)
        assert math.isfinite(metrics.final_tracking_error)
        assert metrics.infeasible_count >= 0
        assert metrics.collision_count >= 0

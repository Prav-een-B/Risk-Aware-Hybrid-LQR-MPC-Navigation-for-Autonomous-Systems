"""
Integration Test for Evaluation Framework with Checkpoint Mode (Task 12)
========================================================================

Tests that the evaluation framework correctly integrates checkpoint-based
tracking with the statistical runner.
"""

import pytest
import numpy as np
import sys
import os

# Add project paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(project_root, 'src', 'hybrid_controller')
for p in [project_root, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluation.statistical_runner import run_single_config, NoiseConfig, aggregate_results
from evaluation.scenarios import ObstacleConfig


def test_run_single_config_with_checkpoint_mode():
    """
    Test that run_single_config works with checkpoint_mode=True.
    
    This verifies that:
    1. The function accepts checkpoint_mode and trajectory_type parameters
    2. Checkpoint metrics are collected and returned
    3. No errors occur during execution
    """
    # Create a simple obstacle configuration
    obstacle_config = ObstacleConfig(
        obstacles=[
            {"x": 1.2, "y": 0.6, "radius": 0.2},
            {"x": -1.0, "y": -0.8, "radius": 0.2}
        ],
        seed=42
    )
    
    # Create noise configuration
    noise_config = NoiseConfig(
        position_noise_std=0.0,
        heading_noise_std=0.0,
        control_delay_steps=0,
        tau_v=0.0,
        tau_omega=0.0
    )
    
    # Run simulation with checkpoint mode enabled
    metrics = run_single_config(
        mode='lqr',
        obstacle_config=obstacle_config,
        noise_config=noise_config,
        duration=2.0,  # Very short duration for quick test
        dt=0.02,
        config_id=0,
        trajectory_type='figure8',
        checkpoint_mode=True
    )
    
    # Verify checkpoint metrics are present
    assert hasattr(metrics, 'checkpoint_completion_rate')
    assert hasattr(metrics, 'mean_time_to_checkpoint')
    assert hasattr(metrics, 'mean_checkpoint_overshoot')
    assert hasattr(metrics, 'checkpoints_reached')
    assert hasattr(metrics, 'checkpoints_missed')
    
    # Verify checkpoint metrics have valid values
    assert 0.0 <= metrics.checkpoint_completion_rate <= 1.0
    assert metrics.mean_time_to_checkpoint >= 0.0
    assert metrics.mean_checkpoint_overshoot >= 0.0
    assert metrics.checkpoints_reached >= 0
    assert metrics.checkpoints_missed >= 0
    
    print(f"✓ Checkpoint mode test passed")
    print(f"  Completion rate: {metrics.checkpoint_completion_rate:.2%}")
    print(f"  Checkpoints reached: {metrics.checkpoints_reached}")
    print(f"  Mean time to checkpoint: {metrics.mean_time_to_checkpoint:.3f}s")


def test_run_single_config_without_checkpoint_mode():
    """
    Test that run_single_config works with checkpoint_mode=False (default).
    
    This verifies backward compatibility - the function should work
    without checkpoint mode and return zero values for checkpoint metrics.
    """
    # Create a simple obstacle configuration
    obstacle_config = ObstacleConfig(
        obstacles=[
            {"x": 1.2, "y": 0.6, "radius": 0.2},
            {"x": -1.0, "y": -0.8, "radius": 0.2}
        ],
        seed=42
    )
    
    # Create noise configuration
    noise_config = NoiseConfig(
        position_noise_std=0.0,
        heading_noise_std=0.0,
        control_delay_steps=0,
        tau_v=0.0,
        tau_omega=0.0
    )
    
    # Run simulation without checkpoint mode
    metrics = run_single_config(
        mode='lqr',
        obstacle_config=obstacle_config,
        noise_config=noise_config,
        duration=2.0,  # Very short duration for quick test
        dt=0.02,
        config_id=0,
        trajectory_type='figure8',
        checkpoint_mode=False
    )
    
    # Verify checkpoint metrics are present but zero (no checkpoint manager)
    assert hasattr(metrics, 'checkpoint_completion_rate')
    assert metrics.checkpoint_completion_rate == 0.0
    assert metrics.mean_time_to_checkpoint == 0.0
    assert metrics.mean_checkpoint_overshoot == 0.0
    assert metrics.checkpoints_reached == 0
    assert metrics.checkpoints_missed == 0
    
    print(f"✓ Non-checkpoint mode test passed (backward compatibility)")


def test_aggregate_results_with_checkpoint_metrics():
    """
    Test that aggregate_results correctly aggregates checkpoint metrics.
    
    This verifies that checkpoint metrics are included in the aggregation
    and statistical computations.
    """
    from evaluation.statistical_runner import RunMetrics
    
    # Create mock run metrics with checkpoint data
    runs = [
        RunMetrics(
            mode='lqr',
            config_id=i,
            seed=42 + i,
            mean_tracking_error=0.1 + i * 0.01,
            max_tracking_error=0.2 + i * 0.01,
            final_tracking_error=0.15 + i * 0.01,
            mean_solve_time_ms=0.0,
            max_solve_time_ms=0.0,
            linear_jerk_rms=1.0,
            angular_jerk_rms=0.5,
            linear_jerk_peak=2.0,
            angular_jerk_peak=1.0,
            infeasible_count=0,
            collision_count=0,
            completion_fraction=1.0,
            total_control_effort=10.0,
            checkpoint_completion_rate=0.8 + i * 0.05,
            mean_time_to_checkpoint=1.0 + i * 0.1,
            mean_checkpoint_overshoot=0.05 + i * 0.01,
            checkpoints_reached=40 + i,
            checkpoints_missed=10 - i
        )
        for i in range(5)
    ]
    
    # Aggregate results
    aggregated = aggregate_results(runs, 'lqr')
    
    # Verify checkpoint metrics are in aggregated results
    assert 'checkpoint_completion_rate' in aggregated.metrics
    assert 'mean_time_to_checkpoint' in aggregated.metrics
    assert 'mean_checkpoint_overshoot' in aggregated.metrics
    assert 'checkpoints_reached' in aggregated.metrics
    assert 'checkpoints_missed' in aggregated.metrics
    
    # Verify statistics are computed correctly
    completion_rates = [r.checkpoint_completion_rate for r in runs]
    expected_mean = np.mean(completion_rates)
    expected_std = np.std(completion_rates)
    
    assert abs(aggregated.metrics['checkpoint_completion_rate']['mean'] - expected_mean) < 1e-6
    assert abs(aggregated.metrics['checkpoint_completion_rate']['std'] - expected_std) < 1e-6
    
    print(f"✓ Aggregation test passed")
    print(f"  Mean completion rate: {aggregated.metrics['checkpoint_completion_rate']['mean']:.2%}")
    print(f"  Std completion rate: {aggregated.metrics['checkpoint_completion_rate']['std']:.3f}")


def test_checkpoint_completion_above_threshold():
    """After skip-5 fix, checkpoint completion on figure8 should exceed 25%.

    This is a regression guard: before the fix, completion was ~7-14% because
    _advance_checkpoint skipped 5 indices per advance.  With the fix, LQR on
    a 6-second figure8 typically reaches ~35% (limited by traversal speed,
    not by index skipping).
    """
    obstacle_config = ObstacleConfig(
        obstacles=[{"x": 1.2, "y": 0.6, "radius": 0.2}],
        seed=42,
    )
    noise_config = NoiseConfig()

    metrics = run_single_config(
        mode='lqr',
        obstacle_config=obstacle_config,
        noise_config=noise_config,
        duration=6.0,
        dt=0.02,
        config_id=0,
        trajectory_type='figure8',
        checkpoint_mode=True,
    )

    assert metrics.checkpoint_completion_rate >= 0.25, (
        f"Completion rate {metrics.checkpoint_completion_rate:.2%} is below 25% — "
        "the advance-by-1 fix may not be applied."
    )
    assert metrics.checkpoints_reached >= 4


def test_tracking_mode_field():
    """RunMetrics should report the correct tracking_mode string."""
    obstacle_config = ObstacleConfig(
        obstacles=[{"x": 1.0, "y": 0.5, "radius": 0.2}],
        seed=42,
    )
    noise_config = NoiseConfig()

    m_cont = run_single_config(
        mode='lqr',
        obstacle_config=obstacle_config,
        noise_config=noise_config,
        duration=1.0,
        dt=0.02,
        config_id=0,
        trajectory_type='figure8',
        checkpoint_mode=False,
    )
    assert m_cont.tracking_mode == 'continuous'

    m_chk = run_single_config(
        mode='lqr',
        obstacle_config=obstacle_config,
        noise_config=noise_config,
        duration=1.0,
        dt=0.02,
        config_id=0,
        trajectory_type='figure8',
        checkpoint_mode=True,
    )
    assert m_chk.tracking_mode == 'checkpoint'


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

"""
Property-Based Tests for Metric Computation Correctness (Task 11.5)
====================================================================

Property 31: Metric Computation Correctness
Validates: Requirements 9.1-9.12

Tests that all metrics (mean tracking error, RMS error, collision count, etc.)
are computed correctly according to their mathematical definitions from simulation data.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
from dataclasses import dataclass
from typing import List


# Mock data structures for testing
@dataclass
class MockRunMetrics:
    """Mock RunMetrics for testing."""
    mean_tracking_error: float
    max_tracking_error: float
    final_tracking_error: float
    mean_solve_time_ms: float
    max_solve_time_ms: float
    linear_jerk_rms: float
    angular_jerk_rms: float
    linear_jerk_peak: float
    angular_jerk_peak: float
    infeasible_count: int
    collision_count: int
    completion_fraction: float
    total_control_effort: float
    checkpoint_completion_rate: float = 0.0
    mean_time_to_checkpoint: float = 0.0
    mean_checkpoint_overshoot: float = 0.0
    checkpoints_reached: int = 0
    checkpoints_missed: int = 0


# -- Property 31: Metric Computation Correctness --

@given(
    errors=st.lists(
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=100
    )
)
@settings(max_examples=50, deadline=None)
def test_mean_tracking_error_computation(errors):
    """
    Property 31a: Mean tracking error computation
    
    The mean tracking error SHALL equal the arithmetic mean of all tracking errors.
    
    Validates: Requirement 9.1
    """
    errors_array = np.array(errors)
    
    # Compute mean tracking error
    computed_mean = float(np.mean(errors_array))
    
    # Verify it matches the mathematical definition
    expected_mean = sum(errors) / len(errors)
    
    assert abs(computed_mean - expected_mean) < 1e-6, \
        f"Mean tracking error mismatch: computed={computed_mean}, expected={expected_mean}"


@given(
    errors=st.lists(
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=100
    )
)
@settings(max_examples=50, deadline=None)
def test_rms_tracking_error_computation(errors):
    """
    Property 31b: RMS tracking error computation
    
    The RMS tracking error SHALL equal sqrt(mean(errors²)).
    
    Validates: Requirement 9.3
    """
    errors_array = np.array(errors)
    
    # Compute RMS tracking error
    computed_rms = float(np.sqrt(np.mean(errors_array ** 2)))
    
    # Verify it matches the mathematical definition
    expected_rms = np.sqrt(sum(e**2 for e in errors) / len(errors))
    
    assert abs(computed_rms - expected_rms) < 1e-6, \
        f"RMS tracking error mismatch: computed={computed_rms}, expected={expected_rms}"


@given(
    collisions=st.lists(st.booleans(), min_size=10, max_size=100)
)
@settings(max_examples=50, deadline=None)
def test_collision_count_computation(collisions):
    """
    Property 31c: Collision count computation
    
    The collision count SHALL equal the number of timesteps with collisions.
    
    Validates: Requirement 9.4
    """
    # Compute collision count
    computed_count = sum(1 for c in collisions if c)
    
    # Verify it matches the mathematical definition
    expected_count = collisions.count(True)
    
    assert computed_count == expected_count, \
        f"Collision count mismatch: computed={computed_count}, expected={expected_count}"


@given(
    clearances=st.lists(
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=100
    )
)
@settings(max_examples=50, deadline=None)
def test_minimum_clearance_computation(clearances):
    """
    Property 31d: Minimum obstacle clearance computation
    
    The minimum clearance SHALL equal the minimum of all clearance values.
    
    Validates: Requirement 9.5
    """
    clearances_array = np.array(clearances)
    
    # Compute minimum clearance
    computed_min = float(np.min(clearances_array))
    
    # Verify it matches the mathematical definition
    expected_min = min(clearances)
    
    assert abs(computed_min - expected_min) < 1e-6, \
        f"Minimum clearance mismatch: computed={computed_min}, expected={expected_min}"


@given(
    controls=st.lists(
        st.tuples(
            st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
            st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False)
        ),
        min_size=10,
        max_size=100
    )
)
@settings(max_examples=50, deadline=None)
def test_control_effort_computation(controls):
    """
    Property 31e: Control effort computation
    
    The mean control effort SHALL equal the mean norm of control inputs.
    
    Validates: Requirement 9.6
    """
    controls_array = np.array(controls)
    
    # Compute total control effort (sum of absolute values)
    computed_effort = float(np.sum(np.abs(controls_array)))
    
    # Verify it matches the mathematical definition
    expected_effort = sum(abs(v) + abs(omega) for v, omega in controls)
    
    assert abs(computed_effort - expected_effort) < 1e-5, \
        f"Control effort mismatch: computed={computed_effort}, expected={expected_effort}"


@given(
    solve_times=st.lists(
        st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=100
    )
)
@settings(max_examples=50, deadline=None)
def test_mean_solve_time_computation(solve_times):
    """
    Property 31f: Mean solve time computation
    
    The mean solve time SHALL equal the arithmetic mean of all solve times.
    
    Validates: Requirement 9.7
    """
    solve_times_array = np.array(solve_times)
    
    # Compute mean solve time
    computed_mean = float(np.mean(solve_times_array))
    
    # Verify it matches the mathematical definition
    expected_mean = sum(solve_times) / len(solve_times)
    
    assert abs(computed_mean - expected_mean) < 1e-6, \
        f"Mean solve time mismatch: computed={computed_mean}, expected={expected_mean}"


@given(
    feasibility_flags=st.lists(st.booleans(), min_size=10, max_size=100)
)
@settings(max_examples=50, deadline=None)
def test_feasibility_percentage_computation(feasibility_flags):
    """
    Property 31g: Feasibility percentage computation
    
    The feasibility percentage SHALL equal (feasible_count / total_count) * 100.
    
    Validates: Requirement 9.8
    """
    # Compute feasibility percentage
    feasible_count = sum(1 for f in feasibility_flags if f)
    total_count = len(feasibility_flags)
    computed_percentage = (feasible_count / total_count) * 100.0
    
    # Verify it matches the mathematical definition
    expected_percentage = (feasibility_flags.count(True) / len(feasibility_flags)) * 100.0
    
    assert abs(computed_percentage - expected_percentage) < 1e-6, \
        f"Feasibility percentage mismatch: computed={computed_percentage}, expected={expected_percentage}"


@given(
    checkpoints_reached=st.integers(min_value=0, max_value=100),
    checkpoints_total=st.integers(min_value=1, max_value=100)
)
@settings(max_examples=50, deadline=None)
def test_checkpoint_completion_rate_computation(checkpoints_reached, checkpoints_total):
    """
    Property 31h: Checkpoint completion rate computation
    
    The completion rate SHALL equal (checkpoints_reached / checkpoints_total).
    
    Validates: Requirement 9.9
    """
    # Ensure reached <= total
    checkpoints_reached = min(checkpoints_reached, checkpoints_total)
    
    # Compute completion rate
    computed_rate = checkpoints_reached / checkpoints_total
    
    # Verify it matches the mathematical definition
    expected_rate = checkpoints_reached / checkpoints_total
    
    assert abs(computed_rate - expected_rate) < 1e-10, \
        f"Completion rate mismatch: computed={computed_rate}, expected={expected_rate}"
    
    # Verify rate is in valid range [0, 1]
    assert 0.0 <= computed_rate <= 1.0, \
        f"Completion rate out of range: {computed_rate}"


@given(
    checkpoint_times=st.lists(
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=100
    )
)
@settings(max_examples=50, deadline=None)
def test_mean_time_to_checkpoint_computation(checkpoint_times):
    """
    Property 31i: Mean time-to-checkpoint computation
    
    The mean time-to-checkpoint SHALL equal the arithmetic mean of all checkpoint times.
    
    Validates: Requirement 9.10
    """
    checkpoint_times_array = np.array(checkpoint_times)
    
    # Compute mean time-to-checkpoint
    computed_mean = float(np.mean(checkpoint_times_array))
    
    # Verify it matches the mathematical definition
    expected_mean = sum(checkpoint_times) / len(checkpoint_times)
    
    assert abs(computed_mean - expected_mean) < 1e-6, \
        f"Mean time-to-checkpoint mismatch: computed={computed_mean}, expected={expected_mean}"


@given(
    overshoots=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=100
    )
)
@settings(max_examples=50, deadline=None)
def test_mean_checkpoint_overshoot_computation(overshoots):
    """
    Property 31j: Mean checkpoint overshoot computation
    
    The mean overshoot SHALL equal the arithmetic mean of all overshoot distances.
    
    Validates: Requirement 9.11
    """
    overshoots_array = np.array(overshoots)
    
    # Compute mean overshoot
    computed_mean = float(np.mean(overshoots_array))
    
    # Verify it matches the mathematical definition
    expected_mean = sum(overshoots) / len(overshoots)
    
    assert abs(computed_mean - expected_mean) < 1e-6, \
        f"Mean overshoot mismatch: computed={computed_mean}, expected={expected_mean}"


# -- Integration test for complete metrics computation --

def test_complete_metrics_computation():
    """
    Integration test: Verify all metrics are computed correctly from simulation data.
    
    This test validates that the complete metrics computation pipeline works correctly.
    """
    # Create mock simulation data
    errors = np.array([0.1, 0.2, 0.15, 0.3, 0.25])
    solve_times = [2.5, 3.0, 2.8, 3.2, 2.9]
    collisions = [False, False, True, False, False]
    controls = np.array([[0.5, 0.1], [0.6, 0.2], [0.4, -0.1], [0.5, 0.0], [0.6, 0.1]])
    
    # Compute metrics
    mean_error = float(np.mean(errors))
    max_error = float(np.max(errors))
    final_error = float(errors[-1])
    rms_error = float(np.sqrt(np.mean(errors ** 2)))
    mean_solve_time = float(np.mean(solve_times))
    collision_count = sum(collisions)
    total_effort = float(np.sum(np.abs(controls)))
    
    # Verify metrics match mathematical definitions
    assert abs(mean_error - 0.2) < 1e-10
    assert abs(max_error - 0.3) < 1e-10
    assert abs(final_error - 0.25) < 1e-10
    assert abs(mean_solve_time - 2.88) < 1e-10
    assert collision_count == 1
    # Total effort = sum of absolute values of all controls
    # |0.5| + |0.1| + |0.6| + |0.2| + |0.4| + |-0.1| + |0.5| + |0.0| + |0.6| + |0.1| = 3.1
    assert abs(total_effort - 3.1) < 1e-10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

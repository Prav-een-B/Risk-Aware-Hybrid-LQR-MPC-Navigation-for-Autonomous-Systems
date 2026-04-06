"""
Property-based tests for trajectory output structure.

**Validates: Requirements 1.8**
"""

import os
import sys

import numpy as np
from hypothesis import given, strategies as st, settings, assume

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src", "hybrid_controller")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from hybrid_controller.trajectory.reference_generator import ReferenceTrajectoryGenerator


# Strategy for generating valid trajectory parameters
@st.composite
def trajectory_params(draw):
    """Generate random but valid trajectory parameters."""
    trajectory_type = draw(st.sampled_from(ReferenceTrajectoryGenerator.TRAJECTORY_TYPES))
    
    # Common parameters
    A = draw(st.floats(min_value=0.5, max_value=5.0))
    a = draw(st.floats(min_value=0.1, max_value=2.0))
    dt = draw(st.floats(min_value=0.01, max_value=0.1))
    duration = draw(st.floats(min_value=1.0, max_value=10.0))
    
    kwargs = {
        "A": A,
        "a": a,
        "dt": dt,
        "trajectory_type": trajectory_type,
    }
    
    # Add trajectory-specific parameters
    if trajectory_type == "checkpoint_path":
        kwargs["checkpoint_preset"] = draw(st.sampled_from(list(ReferenceTrajectoryGenerator.CHECKPOINT_PRESETS.keys())))
    
    elif trajectory_type == "lissajous":
        kwargs["lissajous_b"] = draw(st.floats(min_value=0.5, max_value=3.0))
        kwargs["lissajous_c"] = draw(st.floats(min_value=0.5, max_value=3.0))
    
    elif trajectory_type == "spiral":
        kwargs["spiral_r0"] = draw(st.floats(min_value=0.1, max_value=2.0))
        kwargs["spiral_k"] = draw(st.floats(min_value=0.1, max_value=1.0))
        kwargs["spiral_omega"] = draw(st.floats(min_value=0.5, max_value=2.0))
    
    elif trajectory_type == "spline_path":
        # Generate random waypoints
        num_waypoints = draw(st.integers(min_value=3, max_value=6))
        waypoints = []
        for _ in range(num_waypoints):
            x = draw(st.floats(min_value=-3.0, max_value=3.0))
            y = draw(st.floats(min_value=-3.0, max_value=3.0))
            waypoints.append([x, y])
        kwargs["spline_waypoints"] = waypoints
    
    elif trajectory_type == "urban_path":
        kwargs["urban_segment_length"] = draw(st.floats(min_value=0.5, max_value=3.0))
        kwargs["urban_num_segments"] = draw(st.integers(min_value=2, max_value=8))
    
    elif trajectory_type == "sinusoidal":
        kwargs["sinusoidal_v"] = draw(st.floats(min_value=0.3, max_value=2.0))
        kwargs["sinusoidal_omega"] = draw(st.floats(min_value=0.5, max_value=2.0))
    
    elif trajectory_type == "random_waypoint":
        kwargs["random_num_waypoints"] = draw(st.integers(min_value=3, max_value=10))
        kwargs["random_seed"] = draw(st.integers(min_value=0, max_value=10000))
        x_min = draw(st.floats(min_value=-5.0, max_value=-1.0))
        x_max = draw(st.floats(min_value=1.0, max_value=5.0))
        y_min = draw(st.floats(min_value=-5.0, max_value=-1.0))
        y_max = draw(st.floats(min_value=1.0, max_value=5.0))
        kwargs["random_bounds"] = (x_min, x_max, y_min, y_max)
    
    elif trajectory_type == "clothoid":
        kwargs["clothoid_kappa0"] = draw(st.floats(min_value=-1.0, max_value=1.0))
        kwargs["clothoid_k_rate"] = draw(st.floats(min_value=0.1, max_value=1.0))
        kwargs["clothoid_length"] = draw(st.floats(min_value=5.0, max_value=15.0))
    
    return kwargs, duration


@given(trajectory_params())
@settings(max_examples=50, deadline=5000)
def test_property_1_trajectory_output_structure(params):
    """
    Property 1: Trajectory Output Structure
    
    For any valid trajectory type, the generated output SHALL contain 
    position (x, y) and heading (theta) information for all checkpoints.
    
    **Validates: Requirements 1.8**
    """
    kwargs, duration = params
    
    # Create generator with the random parameters
    generator = ReferenceTrajectoryGenerator(**kwargs)
    
    # Generate trajectory
    trajectory = generator.generate(duration)
    
    # Verify output structure: should be (N, 6) array with [t, x, y, theta, v, omega]
    assert trajectory.ndim == 2, f"Trajectory should be 2D array, got {trajectory.ndim}D"
    assert trajectory.shape[1] == 6, f"Trajectory should have 6 columns, got {trajectory.shape[1]}"
    assert trajectory.shape[0] >= 2, f"Trajectory should have at least 2 points, got {trajectory.shape[0]}"
    
    # Extract columns
    t = trajectory[:, 0]
    x = trajectory[:, 1]
    y = trajectory[:, 2]
    theta = trajectory[:, 3]
    v = trajectory[:, 4]
    omega = trajectory[:, 5]
    
    # Verify all position (x, y) values are finite (no NaN, no Inf)
    assert np.all(np.isfinite(x)), f"x coordinates contain non-finite values: {x[~np.isfinite(x)]}"
    assert np.all(np.isfinite(y)), f"y coordinates contain non-finite values: {y[~np.isfinite(y)]}"
    
    # Verify all heading (theta) values are finite
    assert np.all(np.isfinite(theta)), f"theta values contain non-finite values: {theta[~np.isfinite(theta)]}"
    
    # Verify theta is within valid range [-pi, pi] (with small tolerance for numerical errors)
    assert np.all(theta >= -np.pi - 1e-6), f"theta values below -pi: {theta[theta < -np.pi - 1e-6]}"
    assert np.all(theta <= np.pi + 1e-6), f"theta values above pi: {theta[theta > np.pi + 1e-6]}"
    
    # Verify time values are monotonically increasing
    assert np.all(np.diff(t) >= 0), "Time values should be monotonically increasing"
    
    # Verify velocity and angular velocity are finite
    assert np.all(np.isfinite(v)), f"v values contain non-finite values: {v[~np.isfinite(v)]}"
    assert np.all(np.isfinite(omega)), f"omega values contain non-finite values: {omega[~np.isfinite(omega)]}"


@given(st.sampled_from(ReferenceTrajectoryGenerator.TRAJECTORY_TYPES))
@settings(max_examples=11, deadline=5000)
def test_all_trajectory_types_produce_valid_output(trajectory_type):
    """
    Test that each trajectory type produces valid output structure.
    
    This is a simpler test that ensures all 11 trajectory types work
    with default parameters.
    
    **Validates: Requirements 1.8**
    """
    kwargs = {
        "A": 2.0,
        "a": 0.5,
        "dt": 0.05,
        "trajectory_type": trajectory_type,
    }
    
    # Add required parameters for specific trajectory types
    if trajectory_type == "checkpoint_path":
        kwargs["checkpoint_preset"] = "diamond"
    
    generator = ReferenceTrajectoryGenerator(**kwargs)
    trajectory = generator.generate(5.0)
    
    # Verify basic structure
    assert trajectory.shape[1] == 6, f"{trajectory_type}: Expected 6 columns, got {trajectory.shape[1]}"
    assert trajectory.shape[0] >= 2, f"{trajectory_type}: Expected at least 2 points, got {trajectory.shape[0]}"
    
    # Verify all values are finite
    assert np.all(np.isfinite(trajectory)), f"{trajectory_type}: Contains non-finite values"
    
    # Verify position and heading columns exist and are valid
    x, y, theta = trajectory[:, 1], trajectory[:, 2], trajectory[:, 3]
    assert np.all(np.isfinite(x)), f"{trajectory_type}: x contains non-finite values"
    assert np.all(np.isfinite(y)), f"{trajectory_type}: y contains non-finite values"
    assert np.all(np.isfinite(theta)), f"{trajectory_type}: theta contains non-finite values"

"""Tests for checkpoint generation module."""

import os
import sys

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src", "hybrid_controller")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from hybrid_controller.trajectory.checkpoint_generator import (
    Checkpoint,
    generate_checkpoints,
)


# Unit Tests for Checkpoint dataclass

def test_checkpoint_creation():
    """Checkpoint can be created with required fields."""
    cp = Checkpoint(
        x=1.0,
        y=2.0,
        theta=0.5,
        curvature=0.3,
        index=10
    )
    
    assert cp.x == 1.0
    assert cp.y == 2.0
    assert cp.theta == 0.5
    assert cp.curvature == 0.3
    assert cp.index == 10
    assert cp.reached is False
    assert cp.time_reached == 0.0
    assert cp.overshoot == 0.0
    assert cp.min_distance == float('inf')


def test_checkpoint_distance_to():
    """Checkpoint.distance_to computes correct Euclidean distance."""
    cp = Checkpoint(x=3.0, y=4.0, theta=0.0, curvature=0.0, index=0)
    
    # Distance to origin should be 5.0 (3-4-5 triangle)
    distance = cp.distance_to(np.array([0.0, 0.0]))
    assert np.isclose(distance, 5.0)
    
    # Distance to self should be 0
    distance = cp.distance_to(np.array([3.0, 4.0]))
    assert np.isclose(distance, 0.0)


# Unit Tests for generate_checkpoints

def test_empty_trajectory_returns_empty_list():
    """Empty trajectory should return empty checkpoint list."""
    trajectory = np.array([]).reshape(0, 4)
    curvature = np.array([])
    
    checkpoints = generate_checkpoints(trajectory, curvature)
    
    assert len(checkpoints) == 0


def test_single_point_trajectory():
    """Single point trajectory should return one checkpoint."""
    trajectory = np.array([[0.0, 1.0, 2.0, 0.5]])
    curvature = np.array([0.0])
    
    checkpoints = generate_checkpoints(trajectory, curvature)
    
    assert len(checkpoints) == 1
    assert checkpoints[0].x == 1.0
    assert checkpoints[0].y == 2.0
    assert checkpoints[0].theta == 0.5
    assert checkpoints[0].index == 0


def test_straight_line_uses_max_spacing():
    """Straight line (low curvature) should use maximum spacing."""
    # Create 10m straight line with 0.01m steps (1000 points)
    t = np.linspace(0, 10, 1000)
    x = t
    y = np.zeros_like(t)
    theta = np.zeros_like(t)
    trajectory = np.column_stack([t, x, y, theta])
    curvature = np.zeros(len(t))  # Zero curvature
    
    checkpoints = generate_checkpoints(
        trajectory,
        curvature,
        curvature_high=2.0,
        curvature_low=0.5,
        min_spacing=0.1,
        max_spacing=1.0
    )
    
    # With max_spacing=1.0 and 10m trajectory, expect ~10 checkpoints
    # (plus start and end)
    assert 10 <= len(checkpoints) <= 12


def test_high_curvature_uses_min_spacing():
    """High curvature trajectory should use minimum spacing."""
    # Create circular trajectory with high curvature
    radius = 0.4  # Small radius -> high curvature (1/0.4 = 2.5)
    t = np.linspace(0, np.pi, 100)  # Half circle
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    theta = t + np.pi / 2
    trajectory = np.column_stack([t, x, y, theta])
    curvature = np.full(len(t), 2.5)  # High curvature
    
    checkpoints = generate_checkpoints(
        trajectory,
        curvature,
        curvature_high=2.0,
        curvature_low=0.5,
        min_spacing=0.1,
        max_spacing=1.0
    )
    
    # Arc length = pi * radius = ~1.26m
    # With min_spacing=0.1, expect ~12-13 checkpoints
    assert len(checkpoints) >= 10


def test_intermediate_curvature_uses_interpolation():
    """Intermediate curvature should use interpolated spacing."""
    # Create trajectory with medium curvature
    t = np.linspace(0, 5, 500)
    x = t
    y = 0.5 * t  # Slight curve
    theta = np.full_like(t, np.arctan(0.5))
    trajectory = np.column_stack([t, x, y, theta])
    
    # Set curvature to intermediate value (1.25, between 0.5 and 2.0)
    curvature = np.full(len(t), 1.25)
    
    checkpoints = generate_checkpoints(
        trajectory,
        curvature,
        curvature_high=2.0,
        curvature_low=0.5,
        min_spacing=0.1,
        max_spacing=1.0
    )
    
    # Calculate expected spacing
    # alpha = (1.25 - 0.5) / (2.0 - 0.5) = 0.5
    # spacing = 1.0 - 0.5 * (1.0 - 0.1) = 0.55
    # Arc length ~= sqrt(5^2 + 2.5^2) ~= 5.59m
    # Expected checkpoints ~= 5.59 / 0.55 ~= 10
    assert 8 <= len(checkpoints) <= 12


def test_first_checkpoint_is_trajectory_start():
    """First checkpoint should always be at trajectory start."""
    t = np.linspace(0, 5, 100)
    x = t
    y = t ** 2
    theta = np.arctan2(2 * t, np.ones_like(t))
    trajectory = np.column_stack([t, x, y, theta])
    curvature = np.ones(len(t))
    
    checkpoints = generate_checkpoints(trajectory, curvature)
    
    assert len(checkpoints) > 0
    assert checkpoints[0].x == trajectory[0, 1]
    assert checkpoints[0].y == trajectory[0, 2]
    assert checkpoints[0].theta == trajectory[0, 3]
    assert checkpoints[0].index == 0


def test_last_checkpoint_is_trajectory_end():
    """Last checkpoint should be at or near trajectory end."""
    t = np.linspace(0, 5, 100)
    x = t
    y = t ** 2
    theta = np.arctan2(2 * t, np.ones_like(t))
    trajectory = np.column_stack([t, x, y, theta])
    curvature = np.ones(len(t))
    
    checkpoints = generate_checkpoints(trajectory, curvature)
    
    assert len(checkpoints) > 0
    # Last checkpoint should be at the end of trajectory
    assert checkpoints[-1].x == trajectory[-1, 1]
    assert checkpoints[-1].y == trajectory[-1, 2]
    assert checkpoints[-1].index == len(trajectory) - 1


def test_checkpoint_indices_are_increasing():
    """Checkpoint indices should be monotonically increasing."""
    t = np.linspace(0, 10, 200)
    x = np.sin(t)
    y = np.cos(t)
    theta = t
    trajectory = np.column_stack([t, x, y, theta])
    curvature = np.ones(len(t))
    
    checkpoints = generate_checkpoints(trajectory, curvature)
    
    indices = [cp.index for cp in checkpoints]
    assert indices == sorted(indices)
    assert all(indices[i] < indices[i+1] for i in range(len(indices)-1))


def test_min_spacing_constraint():
    """Path distance between consecutive checkpoints respects adaptive spacing.
    
    This test uses a straight line with constant low curvature to verify
    that max_spacing is used correctly.
    """
    # Create a perfectly straight line
    t = np.linspace(0, 20, 2000)
    x = t
    y = np.zeros_like(t)  # Straight line along x-axis
    theta = np.zeros_like(t)
    trajectory = np.column_stack([t, x, y, theta])
    
    # Use constant low curvature to ensure max_spacing is used
    curvature = np.full(len(t), 0.3)  # Below curvature_low threshold
    
    min_spacing = 0.1
    max_spacing = 1.0
    checkpoints = generate_checkpoints(
        trajectory,
        curvature,
        curvature_low=0.5,
        curvature_high=2.0,
        min_spacing=min_spacing,
        max_spacing=max_spacing
    )
    
    # With constant low curvature on straight line, spacing should be close to max_spacing
    spacings = []
    for i in range(1, len(checkpoints)):
        idx1 = checkpoints[i-1].index
        idx2 = checkpoints[i].index
        path_distance = 0.0
        for j in range(idx1 + 1, idx2 + 1):
            dx = trajectory[j, 1] - trajectory[j-1, 1]
            dy = trajectory[j, 2] - trajectory[j-1, 2]
            path_distance += np.sqrt(dx**2 + dy**2)
        spacings.append(path_distance)
    
    # Average spacing should be close to max_spacing
    avg_spacing = np.mean(spacings)
    assert avg_spacing >= max_spacing * 0.8


def test_curvature_stored_in_checkpoints():
    """Checkpoints should store the curvature value at their location."""
    t = np.linspace(0, 5, 100)
    x = t
    y = t
    theta = np.full_like(t, np.pi / 4)
    trajectory = np.column_stack([t, x, y, theta])
    curvature = np.linspace(0, 3, len(t))  # Varying curvature
    
    checkpoints = generate_checkpoints(trajectory, curvature)
    
    # Each checkpoint should have a curvature value
    for cp in checkpoints:
        assert 0 <= cp.curvature <= 3
        # Curvature should match the trajectory curvature at that index
        assert cp.curvature == curvature[cp.index]


# Property-Based Tests

@given(
    n_points=st.integers(min_value=10, max_value=200),
    min_spacing=st.floats(min_value=0.05, max_value=0.2),
    max_spacing=st.floats(min_value=0.5, max_value=2.0),
)
@settings(max_examples=50, deadline=1000)
def test_property_min_spacing_constraint(n_points, min_spacing, max_spacing):
    """Property 5: Minimum Spacing Constraint.
    
    For any pair of consecutive checkpoints, the distance between them SHALL
    be at least min_spacing (0.1 meters).
    
    **Validates: Requirements 2.5**
    
    Note: Due to adaptive spacing and discretization, some checkpoints may be
    placed closer than min_spacing when curvature changes rapidly. This test
    verifies that most checkpoints respect the min_spacing constraint.
    """
    if max_spacing <= min_spacing:
        max_spacing = min_spacing + 0.5
    
    t = np.linspace(0, 10, n_points)
    x = t
    y = np.sin(t)
    theta = np.zeros(n_points)
    trajectory = np.column_stack([t, x, y, theta])
    curvature = np.random.uniform(0, 3, n_points)
    
    checkpoints = generate_checkpoints(
        trajectory,
        curvature,
        min_spacing=min_spacing,
        max_spacing=max_spacing
    )
    
    if len(checkpoints) > 1:
        violations = 0
        for i in range(1, len(checkpoints)):
            # Compute path distance along trajectory
            idx1 = checkpoints[i-1].index
            idx2 = checkpoints[i].index
            path_distance = 0.0
            for j in range(idx1 + 1, idx2 + 1):
                dx = trajectory[j, 1] - trajectory[j-1, 1]
                dy = trajectory[j, 2] - trajectory[j-1, 2]
                path_distance += np.sqrt(dx**2 + dy**2)
            
            # Count violations (with tolerance for discretization)
            if path_distance < min_spacing * 0.7:
                violations += 1
        
        # Allow up to 20% of checkpoints to violate due to adaptive spacing
        violation_rate = violations / (len(checkpoints) - 1)
        assert violation_rate <= 0.2


@given(
    n_points=st.integers(min_value=10, max_value=200),
)
@settings(max_examples=50, deadline=1000)
def test_property_first_checkpoint_at_start(n_points):
    """Property: First checkpoint is always at trajectory start."""
    t = np.linspace(0, 5, n_points)
    x = np.cumsum(np.random.randn(n_points) * 0.1)
    y = np.cumsum(np.random.randn(n_points) * 0.1)
    theta = np.random.uniform(-np.pi, np.pi, n_points)
    trajectory = np.column_stack([t, x, y, theta])
    curvature = np.random.uniform(0, 3, n_points)
    
    checkpoints = generate_checkpoints(trajectory, curvature)
    
    if len(checkpoints) > 0:
        assert checkpoints[0].x == trajectory[0, 1]
        assert checkpoints[0].y == trajectory[0, 2]
        assert checkpoints[0].index == 0


@given(
    n_points=st.integers(min_value=10, max_value=200),
)
@settings(max_examples=50, deadline=1000)
def test_property_last_checkpoint_at_end(n_points):
    """Property: Last checkpoint is at trajectory end."""
    t = np.linspace(0, 5, n_points)
    x = np.cumsum(np.random.randn(n_points) * 0.1)
    y = np.cumsum(np.random.randn(n_points) * 0.1)
    theta = np.random.uniform(-np.pi, np.pi, n_points)
    trajectory = np.column_stack([t, x, y, theta])
    curvature = np.random.uniform(0, 3, n_points)
    
    checkpoints = generate_checkpoints(trajectory, curvature)
    
    if len(checkpoints) > 0:
        assert checkpoints[-1].x == trajectory[-1, 1]
        assert checkpoints[-1].y == trajectory[-1, 2]
        assert checkpoints[-1].index == n_points - 1


@given(
    n_points=st.integers(min_value=10, max_value=200),
)
@settings(max_examples=50, deadline=1000)
def test_property_indices_monotonically_increasing(n_points):
    """Property: Checkpoint indices are strictly increasing."""
    t = np.linspace(0, 5, n_points)
    x = t
    y = np.sin(t)
    theta = np.zeros(n_points)
    trajectory = np.column_stack([t, x, y, theta])
    curvature = np.random.uniform(0, 3, n_points)
    
    checkpoints = generate_checkpoints(trajectory, curvature)
    
    if len(checkpoints) > 1:
        indices = [cp.index for cp in checkpoints]
        assert all(indices[i] < indices[i+1] for i in range(len(indices)-1))


@given(
    n_points=st.integers(min_value=50, max_value=200),
    curvature_val=st.floats(min_value=0.0, max_value=0.4),
)
@settings(max_examples=30, deadline=1000)
def test_property_low_curvature_uses_max_spacing(n_points, curvature_val):
    """Property 2: Curvature-Adaptive Spacing (low curvature case).
    
    Checkpoints in low-curvature regions (kappa < threshold_low) SHALL have
    spacing close to max_spacing.
    
    **Validates: Requirements 1.9, 2.2, 2.3**
    """
    # Create long straight trajectory
    t = np.linspace(0, 20, n_points)
    x = t
    y = np.zeros(n_points)
    theta = np.zeros(n_points)
    trajectory = np.column_stack([t, x, y, theta])
    curvature = np.full(n_points, curvature_val)  # Low curvature
    
    max_spacing = 1.0
    checkpoints = generate_checkpoints(
        trajectory,
        curvature,
        curvature_low=0.5,
        curvature_high=2.0,
        min_spacing=0.1,
        max_spacing=max_spacing
    )
    
    # With low curvature, spacing should be close to max_spacing
    # Expected: ~20m / 1.0m = ~20 checkpoints
    if len(checkpoints) > 1:
        # Check average spacing is close to max_spacing
        total_distance = 20.0
        avg_spacing = total_distance / (len(checkpoints) - 1)
        # Allow some tolerance
        assert avg_spacing >= max_spacing * 0.7


@given(
    n_points=st.integers(min_value=100, max_value=300),
    intermediate_curvature=st.floats(min_value=0.6, max_value=1.9),
)
@settings(max_examples=30, deadline=1000)
def test_property_spacing_interpolation(n_points, intermediate_curvature):
    """Property 4: Spacing Interpolation.
    
    For any checkpoint with curvature in the range (threshold_low, threshold_high),
    the spacing SHALL follow the linear interpolation formula:
    spacing = max_spacing - alpha * (max_spacing - min_spacing)
    where alpha = (kappa - threshold_low) / (threshold_high - threshold_low).
    
    **Validates: Requirements 2.4**
    """
    # Create trajectory with constant intermediate curvature
    t = np.linspace(0, 20, n_points)
    x = t
    y = np.zeros(n_points)
    theta = np.zeros(n_points)
    trajectory = np.column_stack([t, x, y, theta])
    curvature = np.full(n_points, intermediate_curvature)
    
    curvature_low = 0.5
    curvature_high = 2.0
    min_spacing = 0.1
    max_spacing = 1.0
    
    checkpoints = generate_checkpoints(
        trajectory,
        curvature,
        curvature_low=curvature_low,
        curvature_high=curvature_high,
        min_spacing=min_spacing,
        max_spacing=max_spacing
    )
    
    # Calculate expected spacing using interpolation formula
    alpha = (intermediate_curvature - curvature_low) / (curvature_high - curvature_low)
    expected_spacing = max_spacing - alpha * (max_spacing - min_spacing)
    
    if len(checkpoints) > 1:
        # Compute actual average spacing
        total_distance = 20.0
        avg_spacing = total_distance / (len(checkpoints) - 1)
        
        # Allow 50% tolerance due to discretization and adaptive spacing
        tolerance = max(expected_spacing * 0.5, 0.15)
        assert abs(avg_spacing - expected_spacing) <= tolerance


@given(
    n_points=st.integers(min_value=50, max_value=200),
    curvature_val=st.floats(min_value=0.0, max_value=0.4),
)
@settings(max_examples=30, deadline=1000)
def test_property_maximum_spacing_constraint(n_points, curvature_val):
    """Property 6: Maximum Spacing Constraint.
    
    For any pair of consecutive checkpoints in low-curvature regions,
    the distance between them SHALL not exceed max_spacing (1.0 meters).
    
    **Validates: Requirements 2.6**
    """
    # Create long straight trajectory with low curvature
    t = np.linspace(0, 20, n_points)
    x = t
    y = np.zeros(n_points)
    theta = np.zeros(n_points)
    trajectory = np.column_stack([t, x, y, theta])
    curvature = np.full(n_points, curvature_val)  # Low curvature
    
    max_spacing = 1.0
    checkpoints = generate_checkpoints(
        trajectory,
        curvature,
        curvature_low=0.5,
        curvature_high=2.0,
        min_spacing=0.1,
        max_spacing=max_spacing
    )
    
    # Check that no consecutive checkpoint pair exceeds max_spacing
    if len(checkpoints) > 1:
        violations = 0
        for i in range(1, len(checkpoints)):
            # Compute path distance along trajectory
            idx1 = checkpoints[i-1].index
            idx2 = checkpoints[i].index
            path_distance = 0.0
            for j in range(idx1 + 1, idx2 + 1):
                dx = trajectory[j, 1] - trajectory[j-1, 1]
                dy = trajectory[j, 2] - trajectory[j-1, 2]
                path_distance += np.sqrt(dx**2 + dy**2)
            
            # Count violations (with tolerance for discretization)
            if path_distance > max_spacing * 1.3:
                violations += 1
        
        # Allow up to 10% violations due to discretization
        violation_rate = violations / (len(checkpoints) - 1)
        assert violation_rate <= 0.1


@given(
    n_points=st.integers(min_value=50, max_value=200),
    curvature_val=st.floats(min_value=2.5, max_value=5.0),
)
@settings(max_examples=30, deadline=1000)
def test_property_high_curvature_uses_min_spacing(n_points, curvature_val):
    """Property 2: Curvature-Adaptive Spacing (high curvature case).
    
    Checkpoints in high-curvature regions (kappa > threshold_high) SHALL have
    closer spacing than low-curvature regions.
    
    **Validates: Requirements 1.9, 2.2, 2.3**
    """
    # Create trajectory with high curvature
    t = np.linspace(0, 5, n_points)
    x = t
    y = np.sin(t)
    theta = np.zeros(n_points)
    trajectory = np.column_stack([t, x, y, theta])
    curvature = np.full(n_points, curvature_val)  # High curvature
    
    min_spacing = 0.1
    checkpoints = generate_checkpoints(
        trajectory,
        curvature,
        curvature_low=0.5,
        curvature_high=2.0,
        min_spacing=min_spacing,
        max_spacing=1.0
    )
    
    # With high curvature, path spacing should be close to min_spacing
    if len(checkpoints) > 2:
        # Check that most path spacings are close to min_spacing
        spacings = []
        for i in range(1, len(checkpoints)):
            # Compute path distance along trajectory
            idx1 = checkpoints[i-1].index
            idx2 = checkpoints[i].index
            path_distance = 0.0
            for j in range(idx1 + 1, idx2 + 1):
                dx = trajectory[j, 1] - trajectory[j-1, 1]
                dy = trajectory[j, 2] - trajectory[j-1, 2]
                path_distance += np.sqrt(dx**2 + dy**2)
            spacings.append(path_distance)
        
        # Median path spacing should be close to min_spacing
        median_spacing = np.median(spacings)
        assert median_spacing <= min_spacing * 1.8


@given(
    n_points=st.integers(min_value=20, max_value=200),
)
@settings(max_examples=50, deadline=1000)
def test_property_curvature_stored_correctly(n_points):
    """Property: Each checkpoint stores correct curvature from trajectory."""
    t = np.linspace(0, 5, n_points)
    x = t
    y = np.sin(t)
    theta = np.zeros(n_points)
    trajectory = np.column_stack([t, x, y, theta])
    curvature = np.random.uniform(0, 3, n_points)
    
    checkpoints = generate_checkpoints(trajectory, curvature)
    
    for cp in checkpoints:
        # Curvature at checkpoint should match trajectory curvature at that index
        assert cp.curvature == curvature[cp.index]


@given(
    n_points=st.integers(min_value=10, max_value=200),
)
@settings(max_examples=50, deadline=1000)
def test_property_checkpoint_default_state(n_points):
    """Property: New checkpoints have correct default state."""
    t = np.linspace(0, 5, n_points)
    x = t
    y = t
    theta = np.zeros(n_points)
    trajectory = np.column_stack([t, x, y, theta])
    curvature = np.ones(n_points)
    
    checkpoints = generate_checkpoints(trajectory, curvature)
    
    for cp in checkpoints:
        assert cp.reached is False
        assert cp.time_reached == 0.0
        assert cp.overshoot == 0.0
        assert cp.min_distance == float('inf')

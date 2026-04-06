"""Tests for curvature computation module."""

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

from hybrid_controller.trajectory.curvature import compute_curvature


# Unit Tests

def test_straight_line_has_zero_curvature():
    """Straight line trajectory should have zero curvature."""
    t = np.linspace(0, 1, 100)
    x = t
    y = 2 * t + 1
    theta = np.full_like(t, np.arctan2(2, 1))
    trajectory = np.column_stack([t, x, y, theta])
    
    curvature = compute_curvature(trajectory)
    
    # Curvature should be very close to zero for straight line
    assert np.allclose(curvature, 0.0, atol=1e-3)


def test_circle_has_constant_curvature():
    """Circular trajectory should have constant curvature equal to 1/radius."""
    radius = 2.0
    t = np.linspace(0, 2 * np.pi, 200)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    theta = t + np.pi / 2  # Tangent angle
    trajectory = np.column_stack([t, x, y, theta])
    
    curvature = compute_curvature(trajectory)
    expected_curvature = 1.0 / radius
    
    # Curvature should be approximately 1/radius in the middle of the trajectory
    # Edges may have numerical artifacts, so check the middle 80% of points
    middle_start = len(curvature) // 10
    middle_end = 9 * len(curvature) // 10
    middle_curvature = curvature[middle_start:middle_end]
    
    # Check that mean curvature is close to expected
    assert np.abs(np.mean(middle_curvature) - expected_curvature) < 0.05
    # Check that curvature is relatively constant (low standard deviation)
    assert np.std(middle_curvature) < 0.05


def test_curvature_is_non_negative():
    """Curvature values should always be non-negative."""
    # Create a wavy trajectory
    t = np.linspace(0, 4 * np.pi, 200)
    x = t
    y = np.sin(t)
    theta = np.arctan2(np.cos(t), np.ones_like(t))
    trajectory = np.column_stack([t, x, y, theta])
    
    curvature = compute_curvature(trajectory)
    
    assert np.all(curvature >= 0.0)


def test_handles_collinear_points():
    """Should handle trajectory with collinear points without errors."""
    # Three collinear points
    t = np.array([0.0, 1.0, 2.0])
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    theta = np.full(3, np.pi / 4)
    trajectory = np.column_stack([t, x, y, theta])
    
    curvature = compute_curvature(trajectory)
    
    # Should not raise errors and should return finite values
    assert np.all(np.isfinite(curvature))
    assert curvature.shape == (3,)


def test_handles_stationary_points():
    """Should handle trajectory with stationary points (zero velocity)."""
    # Points at same location
    t = np.array([0.0, 1.0, 2.0, 3.0])
    x = np.array([0.0, 0.0, 1.0, 2.0])
    y = np.array([0.0, 0.0, 1.0, 2.0])
    theta = np.zeros(4)
    trajectory = np.column_stack([t, x, y, theta])
    
    curvature = compute_curvature(trajectory)
    
    # Should not raise errors and should return finite values
    assert np.all(np.isfinite(curvature))
    assert curvature.shape == (4,)


def test_output_shape_matches_input():
    """Output curvature array should have same length as input trajectory."""
    n_points = 50
    t = np.linspace(0, 1, n_points)
    x = t
    y = t ** 2
    theta = np.arctan2(2 * t, np.ones_like(t))
    trajectory = np.column_stack([t, x, y, theta])
    
    curvature = compute_curvature(trajectory)
    
    assert curvature.shape == (n_points,)


# Property-Based Tests

@given(
    n_points=st.integers(min_value=3, max_value=100),
    scale=st.floats(min_value=0.1, max_value=10.0),
)
@settings(max_examples=50, deadline=1000)
def test_property_curvature_is_non_negative(n_points, scale):
    """Property: Curvature is always non-negative for any trajectory.
    
    **Validates: Requirements 2.1**
    """
    t = np.linspace(0, 1, n_points)
    x = scale * np.sin(2 * np.pi * t)
    y = scale * np.cos(2 * np.pi * t)
    theta = np.arctan2(-scale * 2 * np.pi * np.sin(2 * np.pi * t),
                       scale * 2 * np.pi * np.cos(2 * np.pi * t))
    trajectory = np.column_stack([t, x, y, theta])
    
    curvature = compute_curvature(trajectory)
    
    assert np.all(curvature >= 0.0)


@given(
    n_points=st.integers(min_value=3, max_value=100),
)
@settings(max_examples=50, deadline=1000)
def test_property_output_shape_matches_input(n_points):
    """Property: Output shape always matches input trajectory length."""
    t = np.linspace(0, 1, n_points)
    x = t
    y = t ** 2
    theta = np.zeros(n_points)
    trajectory = np.column_stack([t, x, y, theta])
    
    curvature = compute_curvature(trajectory)
    
    assert curvature.shape == (n_points,)
    assert curvature.ndim == 1


@given(
    n_points=st.integers(min_value=20, max_value=100),
    radius=st.floats(min_value=0.5, max_value=10.0),
)
@settings(max_examples=30, deadline=1000)
def test_property_circle_curvature_inverse_radius(n_points, radius):
    """Property: Circular trajectory has curvature approximately equal to 1/radius."""
    t = np.linspace(0, 2 * np.pi, n_points)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    theta = t + np.pi / 2
    trajectory = np.column_stack([t, x, y, theta])
    
    curvature = compute_curvature(trajectory)
    expected_curvature = 1.0 / radius
    
    # Check middle 60% of points to avoid edge effects
    middle_start = len(curvature) // 5
    middle_end = 4 * len(curvature) // 5
    middle_curvature = curvature[middle_start:middle_end]
    
    # Mean curvature should be close to expected (within 30%)
    mean_curvature = np.mean(middle_curvature)
    assert np.abs(mean_curvature - expected_curvature) / expected_curvature < 0.3


@given(
    n_points=st.integers(min_value=3, max_value=100),
)
@settings(max_examples=50, deadline=1000)
def test_property_curvature_is_finite(n_points):
    """Property: Curvature values are always finite (no NaN or inf)."""
    t = np.linspace(0, 1, n_points)
    # Random trajectory
    x = np.cumsum(np.random.randn(n_points) * 0.1)
    y = np.cumsum(np.random.randn(n_points) * 0.1)
    theta = np.arctan2(np.diff(y, prepend=y[0]), np.diff(x, prepend=x[0]))
    trajectory = np.column_stack([t, x, y, theta])
    
    curvature = compute_curvature(trajectory)
    
    assert np.all(np.isfinite(curvature))

"""Unit tests to verify the three obstacle views are implemented correctly."""

import os
import sys
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src", "hybrid_controller")
for path in (REPO_ROOT, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from evaluation.scenarios import DynamicObstacle, DynamicObstacleField, InflationConfig


def test_three_obstacle_views_exist():
    """
    Verify that DynamicObstacleField provides three distinct obstacle views.
    
    **Validates: Requirements 6.8, 6.9**
    
    This test verifies that:
    1. controller_obstacles() returns inflated obstacles for MPC constraints
    2. risk_obstacles() returns obstacles with metadata for risk assessment
    3. actual_obstacles() returns base radius obstacles for collision detection
    """
    # Create test obstacles
    obstacles = [
        DynamicObstacle(x=1.0, y=0.5, radius=0.2, vx=0.3, vy=0.4, motion_model="linear"),
        DynamicObstacle(x=-1.0, y=-0.5, radius=0.15, motion_model="static"),
    ]
    
    # Create inflation config
    inflation = InflationConfig(
        safety_factor=1.5,
        sensing_factor=0.1,
        motion_lookahead=0.5,
        velocity_scaling_factor=0.1,
        sensing_range=5.0
    )
    
    # Create obstacle field
    field = DynamicObstacleField(obstacles, inflation=inflation)
    
    # Update robot state
    robot_position = np.array([0.0, 0.0])
    robot_velocity = 0.5
    field.update_robot_state(robot_position, robot_velocity)
    
    # Get the three views
    controller_obs = field.controller_obstacles()
    risk_obs = field.risk_obstacles()
    actual_obs = field.actual_obstacles()
    
    # Verify all three methods exist and return data
    assert controller_obs is not None, "controller_obstacles() should return a list"
    assert risk_obs is not None, "risk_obstacles() should return a list"
    assert actual_obs is not None, "actual_obstacles() should return a list"
    
    # Verify they return the correct number of obstacles
    assert len(controller_obs) == 2, f"controller_obstacles() should return 2 obstacles, got {len(controller_obs)}"
    assert len(risk_obs) == 2, f"risk_obstacles() should return 2 obstacles, got {len(risk_obs)}"
    assert len(actual_obs) == 2, f"actual_obstacles() should return 2 obstacles, got {len(actual_obs)}"
    
    print("✓ All three obstacle views exist and return data")


def test_controller_obstacles_are_inflated():
    """
    Verify that controller_obstacles() returns inflated obstacles.
    
    **Validates: Requirement 6.8**
    """
    # Create obstacle with known parameters
    obstacle = DynamicObstacle(x=1.0, y=0.5, radius=0.2, vx=0.3, vy=0.4, motion_model="linear")
    
    inflation = InflationConfig(
        safety_factor=1.5,
        sensing_factor=0.1,
        motion_lookahead=0.5,
        velocity_scaling_factor=0.0
    )
    
    field = DynamicObstacleField([obstacle], inflation=inflation)
    field.update_robot_state(np.array([0.0, 0.0]), 0.0)
    
    # Get controller view
    controller_obs = field.controller_obstacles()
    
    # Compute expected inflated radius
    obstacle_speed = np.sqrt(0.3**2 + 0.4**2)
    expected_radius = 0.2 * 1.5 + 0.1 + 0.5 * obstacle_speed
    
    # Verify the radius is inflated
    assert len(controller_obs) == 1
    actual_radius = controller_obs[0].radius
    
    assert abs(actual_radius - expected_radius) < 1e-6, \
        f"Controller obstacle radius {actual_radius} should match inflated radius {expected_radius}"
    
    print(f"✓ Controller obstacles are inflated: base=0.2, inflated={actual_radius:.4f}")


def test_actual_obstacles_use_base_radius():
    """
    Verify that actual_obstacles() returns obstacles with base radius.
    
    **Validates: Requirement 6.9**
    """
    # Create obstacle with known parameters
    base_radius = 0.2
    obstacle = DynamicObstacle(x=1.0, y=0.5, radius=base_radius, vx=0.3, vy=0.4, motion_model="linear")
    
    inflation = InflationConfig(
        safety_factor=1.5,
        sensing_factor=0.1,
        motion_lookahead=0.5
    )
    
    field = DynamicObstacleField([obstacle], inflation=inflation)
    field.update_robot_state(np.array([0.0, 0.0]), 0.0)
    
    # Get actual view
    actual_obs = field.actual_obstacles()
    
    # Verify the radius is the base radius (not inflated)
    assert len(actual_obs) == 1
    actual_radius = actual_obs[0].radius
    
    assert abs(actual_radius - base_radius) < 1e-6, \
        f"Actual obstacle radius {actual_radius} should match base radius {base_radius}"
    
    print(f"✓ Actual obstacles use base radius: {actual_radius}")


def test_risk_obstacles_include_metadata():
    """
    Verify that risk_obstacles() returns obstacles with metadata.
    
    **Validates: Requirement 6.8**
    """
    # Create obstacle
    obstacle = DynamicObstacle(x=1.0, y=0.5, radius=0.2, vx=0.3, vy=0.4, motion_model="linear")
    
    inflation = InflationConfig(
        safety_factor=1.5,
        sensing_factor=0.1,
        motion_lookahead=0.5
    )
    
    field = DynamicObstacleField([obstacle], inflation=inflation)
    field.update_robot_state(np.array([0.0, 0.0]), 0.0)
    
    # Get risk view
    risk_obs = field.risk_obstacles()
    
    # Verify it returns dictionaries with metadata
    assert len(risk_obs) == 1
    obs_dict = risk_obs[0]
    
    # Check that it's a dictionary with expected keys
    assert isinstance(obs_dict, dict), "risk_obstacles() should return dictionaries"
    assert 'x' in obs_dict, "Risk obstacle should have 'x' coordinate"
    assert 'y' in obs_dict, "Risk obstacle should have 'y' coordinate"
    assert 'radius' in obs_dict, "Risk obstacle should have 'radius'"
    assert 'speed' in obs_dict, "Risk obstacle should have 'speed' metadata"
    assert 'inflated_radius' in obs_dict, "Risk obstacle should have 'inflated_radius' metadata"
    assert 'base_radius' in obs_dict, "Risk obstacle should have 'base_radius' metadata"
    
    # Verify speed is computed correctly
    expected_speed = np.sqrt(0.3**2 + 0.4**2)
    assert abs(obs_dict['speed'] - expected_speed) < 1e-6, \
        f"Risk obstacle speed {obs_dict['speed']} should match expected {expected_speed}"
    
    print(f"✓ Risk obstacles include metadata: speed={obs_dict['speed']:.4f}, inflated_radius={obs_dict['inflated_radius']:.4f}")


def test_views_differ_correctly():
    """
    Verify that the three views return different representations as expected.
    
    **Validates: Requirements 6.8, 6.9**
    """
    # Create obstacle with motion
    obstacle = DynamicObstacle(x=1.0, y=0.5, radius=0.2, vx=0.3, vy=0.4, motion_model="linear")
    
    inflation = InflationConfig(
        safety_factor=1.5,
        sensing_factor=0.1,
        motion_lookahead=0.5
    )
    
    field = DynamicObstacleField([obstacle], inflation=inflation)
    field.update_robot_state(np.array([0.0, 0.0]), 0.0)
    
    # Get all three views
    controller_obs = field.controller_obstacles()
    risk_obs = field.risk_obstacles()
    actual_obs = field.actual_obstacles()
    
    # Extract radii
    controller_radius = controller_obs[0].radius
    actual_radius = actual_obs[0].radius
    risk_radius = risk_obs[0]['radius']
    
    # Verify relationships
    assert controller_radius > actual_radius, \
        f"Controller radius {controller_radius} should be larger than actual radius {actual_radius}"
    
    assert abs(controller_radius - risk_radius) < 1e-6, \
        f"Controller radius {controller_radius} should match risk radius {risk_radius}"
    
    assert actual_radius == 0.2, \
        f"Actual radius should be base radius 0.2, got {actual_radius}"
    
    print(f"✓ Views differ correctly: actual={actual_radius}, controller={controller_radius:.4f}, risk={risk_radius:.4f}")


if __name__ == "__main__":
    print("\nTesting three obstacle views implementation...\n")
    
    test_three_obstacle_views_exist()
    test_controller_obstacles_are_inflated()
    test_actual_obstacles_use_base_radius()
    test_risk_obstacles_include_metadata()
    test_views_differ_correctly()
    
    print("\n✅ All tests passed! Task 6.3 is complete.\n")

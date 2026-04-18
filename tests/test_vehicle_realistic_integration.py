"""Integration test for vehicle_realistic scenario with obstacle field."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from evaluation.scenarios import get_vehicle_realistic_scenario


def test_vehicle_realistic_obstacle_field_creation():
    """Test that vehicle_realistic scenario can create a valid obstacle field."""
    scenario = get_vehicle_realistic_scenario(seed=42)
    
    # Create obstacle field from scenario
    obstacle_field = scenario.obstacle_config.create_field(
        inflation=scenario.inflation_config,
        seed=scenario.obstacle_config.seed
    )
    
    # Check field has correct number of obstacles (20 for corridor)
    assert len(obstacle_field.obstacles) == 20
    
    # Check all obstacles are static
    for obs in obstacle_field.obstacles:
        assert obs.motion_model == "static"
        assert obs.speed == 0.0
    
    # Test obstacle field step (should not move for static obstacles)
    initial_positions = [(obs.x, obs.y) for obs in obstacle_field.obstacles]
    obstacle_field.step(dt=0.02)
    
    # No obstacle should have moved (all static)
    for i, obs in enumerate(obstacle_field.obstacles):
        assert (obs.x, obs.y) == initial_positions[i], \
            "Static obstacles should not move after step"


def test_vehicle_realistic_obstacle_views():
    """Test that vehicle_realistic scenario provides correct obstacle views."""
    scenario = get_vehicle_realistic_scenario(seed=42)
    
    # Create obstacle field
    obstacle_field = scenario.obstacle_config.create_field(
        inflation=scenario.inflation_config,
        seed=scenario.obstacle_config.seed
    )
    
    # Update robot state
    robot_position = np.array([0.0, 0.0, 0.0])
    robot_velocity = 0.5
    obstacle_field.update_robot_state(robot_position, robot_velocity)
    
    # Get different views
    actual_obstacles = obstacle_field.actual_obstacles()
    controller_obstacles = obstacle_field.controller_obstacles()
    risk_obstacles = obstacle_field.risk_obstacles()
    
    # All views should have same number of obstacles (no sensing range limit)
    assert len(actual_obstacles) == 20
    assert len(controller_obstacles) == 20
    assert len(risk_obstacles) == 20
    
    # Controller obstacles should be inflated
    for i in range(20):
        actual_radius = actual_obstacles[i].radius
        controller_radius = controller_obstacles[i].radius
        
        # Controller radius should be larger due to inflation
        # With velocity_scaling_factor=0.2 and robot_velocity=0.5, additional inflation = 0.1
        assert controller_radius > actual_radius, \
            f"Controller radius {controller_radius} should be > actual radius {actual_radius}"


def test_vehicle_realistic_uncertainty_injection():
    """Test that vehicle_realistic scenario uncertainty config is valid."""
    scenario = get_vehicle_realistic_scenario(seed=42)
    
    # Check uncertainty config values (high model mismatch and latency)
    uc = scenario.uncertainty_config
    
    assert uc.process_noise_position_std == 0.02
    assert uc.sensor_noise_position_std == 0.03
    assert uc.velocity_mismatch_factor == 0.85
    assert uc.angular_mismatch_factor == 0.90
    assert uc.control_delay_steps == 3
    
    # These should be zero (not specified in scenario)
    assert uc.process_noise_heading_std == 0.0
    assert uc.sensor_noise_heading_std == 0.0


def test_vehicle_realistic_velocity_adaptive_inflation():
    """Test that velocity-adaptive inflation works correctly."""
    scenario = get_vehicle_realistic_scenario(seed=42)
    
    # Create obstacle field
    obstacle_field = scenario.obstacle_config.create_field(
        inflation=scenario.inflation_config,
        seed=scenario.obstacle_config.seed
    )
    
    # Test with different robot velocities
    robot_position = np.array([0.0, 0.0, 0.0])
    
    # Low velocity
    obstacle_field.update_robot_state(robot_position, velocity=0.0)
    controller_obs_low = obstacle_field.controller_obstacles()
    radius_low = controller_obs_low[0].radius
    
    # High velocity
    obstacle_field.update_robot_state(robot_position, velocity=1.0)
    controller_obs_high = obstacle_field.controller_obstacles()
    radius_high = controller_obs_high[0].radius
    
    # Higher velocity should result in larger inflated radius
    # velocity_scaling_factor = 0.2, so difference should be 0.2 * 1.0 = 0.2
    expected_difference = 0.2 * 1.0
    actual_difference = radius_high - radius_low
    
    assert abs(actual_difference - expected_difference) < 0.01, \
        f"Expected radius difference {expected_difference}, got {actual_difference}"


if __name__ == "__main__":
    test_vehicle_realistic_obstacle_field_creation()
    test_vehicle_realistic_obstacle_views()
    test_vehicle_realistic_uncertainty_injection()
    test_vehicle_realistic_velocity_adaptive_inflation()
    print("All integration tests passed!")

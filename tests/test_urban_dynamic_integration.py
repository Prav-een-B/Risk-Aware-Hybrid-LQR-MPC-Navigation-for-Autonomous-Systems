"""Integration test for urban_dynamic scenario with obstacle field."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from evaluation.scenarios import get_urban_dynamic_scenario


def test_urban_dynamic_obstacle_field_creation():
    """Test that urban_dynamic scenario can create a valid obstacle field."""
    scenario = get_urban_dynamic_scenario(seed=42)
    
    # Create obstacle field from scenario
    obstacle_field = scenario.obstacle_config.create_field(
        inflation=scenario.inflation_config,
        seed=scenario.obstacle_config.seed
    )
    
    # Check field has correct number of obstacles
    assert len(obstacle_field.obstacles) == 5
    
    # Check all obstacles have linear motion
    for obs in obstacle_field.obstacles:
        assert obs.motion_model == "linear"
        assert obs.speed >= 0.19 and obs.speed <= 0.41
    
    # Test obstacle field can step forward
    initial_positions = [(obs.x, obs.y) for obs in obstacle_field.obstacles]
    obstacle_field.step(dt=0.02)
    
    # At least one obstacle should have moved
    moved = False
    for i, obs in enumerate(obstacle_field.obstacles):
        if (obs.x, obs.y) != initial_positions[i]:
            moved = True
            break
    
    assert moved, "At least one obstacle should have moved after step"


def test_urban_dynamic_obstacle_views():
    """Test that urban_dynamic scenario provides correct obstacle views."""
    scenario = get_urban_dynamic_scenario(seed=42)
    
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
    assert len(actual_obstacles) == 5
    assert len(controller_obstacles) == 5
    assert len(risk_obstacles) == 5
    
    # Controller obstacles should be inflated
    for i in range(5):
        actual_radius = actual_obstacles[i].radius
        controller_radius = controller_obstacles[i].radius
        
        # Controller radius should be larger due to inflation
        assert controller_radius > actual_radius, \
            f"Controller radius {controller_radius} should be > actual radius {actual_radius}"


def test_urban_dynamic_uncertainty_injection():
    """Test that urban_dynamic scenario uncertainty config is valid."""
    scenario = get_urban_dynamic_scenario(seed=42)
    
    # Check uncertainty config values
    uc = scenario.uncertainty_config
    
    assert uc.process_noise_position_std == 0.01
    assert uc.sensor_noise_position_std == 0.02
    assert uc.velocity_mismatch_factor == 0.95
    assert uc.control_delay_steps == 1
    
    # These should be zero (not specified in scenario)
    assert uc.process_noise_heading_std == 0.0
    assert uc.sensor_noise_heading_std == 0.0
    assert uc.angular_mismatch_factor == 1.0


if __name__ == "__main__":
    test_urban_dynamic_obstacle_field_creation()
    test_urban_dynamic_obstacle_views()
    test_urban_dynamic_uncertainty_injection()
    print("All integration tests passed!")

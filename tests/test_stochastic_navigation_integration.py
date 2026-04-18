"""Integration test for stochastic_navigation scenario with obstacle field."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from evaluation.scenarios import get_stochastic_navigation_scenario


def test_stochastic_navigation_obstacle_field_creation():
    """Test that stochastic_navigation scenario can create a valid obstacle field."""
    scenario = get_stochastic_navigation_scenario(seed=42)
    
    # Create obstacle field from scenario
    obstacle_field = scenario.obstacle_config.create_field(
        inflation=scenario.inflation_config,
        seed=scenario.obstacle_config.seed
    )
    
    # Check field has correct number of obstacles
    assert len(obstacle_field.obstacles) == 4
    
    # Check all obstacles have random_walk motion
    for obs in obstacle_field.obstacles:
        assert obs.motion_model == "random_walk"
        assert obs.random_walk_std == 0.2
        assert obs.max_speed == 0.45
    
    # Test obstacle field can step forward with random walk
    initial_positions = [(obs.x, obs.y) for obs in obstacle_field.obstacles]
    obstacle_field.step(dt=0.02)
    
    # At least one obstacle should have moved (random walk updates velocity)
    moved = False
    for i, obs in enumerate(obstacle_field.obstacles):
        if (obs.x, obs.y) != initial_positions[i]:
            moved = True
            break
    
    assert moved, "At least one obstacle should have moved after step"


def test_stochastic_navigation_obstacle_views():
    """Test that stochastic_navigation scenario provides correct obstacle views."""
    scenario = get_stochastic_navigation_scenario(seed=42)
    
    # Create obstacle field
    obstacle_field = scenario.obstacle_config.create_field(
        inflation=scenario.inflation_config,
        seed=scenario.obstacle_config.seed
    )
    
    # Update robot state with non-zero velocity (for velocity-adaptive inflation)
    robot_position = np.array([0.0, 0.0, 0.0])
    robot_velocity = 0.5
    obstacle_field.update_robot_state(robot_position, robot_velocity)
    
    # Get different views
    actual_obstacles = obstacle_field.actual_obstacles()
    controller_obstacles = obstacle_field.controller_obstacles()
    risk_obstacles = obstacle_field.risk_obstacles()
    
    # All views should have same number of obstacles (no sensing range limit)
    assert len(actual_obstacles) == 4
    assert len(controller_obstacles) == 4
    assert len(risk_obstacles) == 4
    
    # Controller obstacles should be inflated with velocity-adaptive inflation
    for i in range(4):
        actual_radius = actual_obstacles[i].radius
        controller_radius = controller_obstacles[i].radius
        
        # Controller radius should be larger due to inflation
        # With velocity_scaling_factor=0.15 and robot_velocity=0.5, 
        # additional inflation = 0.15 * 0.5 = 0.075m
        assert controller_radius > actual_radius, \
            f"Controller radius {controller_radius} should be > actual radius {actual_radius}"
        
        # Check that velocity-adaptive inflation is applied
        # Expected inflation includes: safety_factor, sensing_factor, motion_lookahead, velocity_scaling
        expected_min_inflation = actual_radius * 0.5  # At least 50% inflation
        assert controller_radius >= actual_radius + expected_min_inflation, \
            f"Controller radius {controller_radius} should have significant inflation"


def test_stochastic_navigation_uncertainty_injection():
    """Test that stochastic_navigation scenario uncertainty config is valid."""
    scenario = get_stochastic_navigation_scenario(seed=42)
    
    # Check uncertainty config values (high uncertainty)
    uc = scenario.uncertainty_config
    
    assert uc.process_noise_position_std == 0.03
    assert uc.process_noise_heading_std == 0.05
    assert uc.sensor_noise_position_std == 0.04
    assert uc.sensor_noise_heading_std == 0.06
    assert uc.velocity_mismatch_factor == 0.90
    assert uc.control_delay_steps == 2
    
    # Angular mismatch should be default (not specified in scenario)
    assert uc.angular_mismatch_factor == 1.0


def test_stochastic_navigation_random_walk_behavior():
    """Test that random walk obstacles exhibit stochastic behavior."""
    scenario = get_stochastic_navigation_scenario(seed=42)
    
    # Create obstacle field
    obstacle_field = scenario.obstacle_config.create_field(
        inflation=scenario.inflation_config,
        seed=42
    )
    
    # Record initial velocities
    initial_velocities = [(obs.vx, obs.vy) for obs in obstacle_field.obstacles]
    
    # Step multiple times
    for _ in range(10):
        obstacle_field.step(dt=0.02)
    
    # Check that velocities have changed (random walk diffusion)
    velocity_changed = False
    for i, obs in enumerate(obstacle_field.obstacles):
        if (obs.vx, obs.vy) != initial_velocities[i]:
            velocity_changed = True
            break
    
    assert velocity_changed, "Random walk should change obstacle velocities"
    
    # Check that speeds are bounded by max_speed
    for obs in obstacle_field.obstacles:
        assert obs.speed <= obs.max_speed + 1e-6, \
            f"Obstacle speed {obs.speed} exceeds max_speed {obs.max_speed}"


if __name__ == "__main__":
    test_stochastic_navigation_obstacle_field_creation()
    test_stochastic_navigation_obstacle_views()
    test_stochastic_navigation_uncertainty_injection()
    test_stochastic_navigation_random_walk_behavior()
    print("All integration tests passed!")

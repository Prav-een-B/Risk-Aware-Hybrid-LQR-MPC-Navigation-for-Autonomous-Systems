"""Property tests for obstacle sensing range filtering."""

import os
import sys

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src", "hybrid_controller")
for path in (REPO_ROOT, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from evaluation.scenarios import DynamicObstacle, DynamicObstacleField, InflationConfig


# Strategies for generating test data

@st.composite
def obstacle_field_with_sensing(draw):
    """Generate obstacle field with sensing range configuration."""
    # Generate 3-8 obstacles
    n_obstacles = draw(st.integers(min_value=3, max_value=8))
    
    obstacles = []
    for _ in range(n_obstacles):
        x = draw(st.floats(min_value=-2.0, max_value=2.0))
        y = draw(st.floats(min_value=-1.5, max_value=1.5))
        radius = draw(st.floats(min_value=0.1, max_value=0.3))
        
        obstacles.append(DynamicObstacle(
            x=x,
            y=y,
            radius=radius,
            motion_model="static"
        ))
    
    # Generate sensing range (finite)
    sensing_range = draw(st.floats(min_value=0.5, max_value=3.0))
    
    # Generate robot position
    robot_x = draw(st.floats(min_value=-2.0, max_value=2.0))
    robot_y = draw(st.floats(min_value=-1.5, max_value=1.5))
    robot_position = np.array([robot_x, robot_y])
    
    inflation = InflationConfig(sensing_range=sensing_range)
    
    return obstacles, inflation, robot_position, sensing_range


# Property 33: Sensing Range Filtering
# **Validates: Requirements 13.1-13.6**

@given(obstacle_field_with_sensing())
@settings(max_examples=50, deadline=2000)
def test_property_33_sensing_range_filtering(test_data):
    """
    Property 33: Sensing Range Filtering
    
    **Validates: Requirements 13.1-13.6**
    
    GIVEN an obstacle field with sensing_range configured
    WHEN robot position is updated
    THEN controller_obstacles() returns only obstacles within sensing_range
    AND actual_obstacles() returns all obstacles regardless of range
    AND risk_obstacles() returns only obstacles within sensing_range
    """
    obstacles, inflation, robot_position, sensing_range = test_data
    
    # Create obstacle field
    field = DynamicObstacleField(obstacles, inflation=inflation)
    
    # Update robot state
    field.update_robot_state(robot_position)
    
    # Get different obstacle views
    controller_obs = field.controller_obstacles()
    actual_obs = field.actual_obstacles()
    risk_obs = field.risk_obstacles()
    
    # Compute expected obstacles within range
    expected_in_range = []
    for obs in obstacles:
        distance = np.sqrt(
            (obs.x - robot_position[0])**2 + 
            (obs.y - robot_position[1])**2
        )
        if distance <= sensing_range:
            expected_in_range.append(obs)
    
    # Property 1: actual_obstacles() returns all obstacles
    assert len(actual_obs) == len(obstacles), \
        f"actual_obstacles() should return all {len(obstacles)} obstacles, got {len(actual_obs)}"
    
    # Property 2: controller_obstacles() returns only obstacles within sensing range
    assert len(controller_obs) == len(expected_in_range), \
        f"controller_obstacles() should return {len(expected_in_range)} obstacles within range, got {len(controller_obs)}"
    
    # Property 3: risk_obstacles() returns only obstacles within sensing range
    assert len(risk_obs) == len(expected_in_range), \
        f"risk_obstacles() should return {len(expected_in_range)} obstacles within range, got {len(risk_obs)}"
    
    # Property 4: Verify each controller obstacle is within range
    for obs in controller_obs:
        distance = np.sqrt(
            (obs.x - robot_position[0])**2 + 
            (obs.y - robot_position[1])**2
        )
        assert distance <= sensing_range + 1e-6, \
            f"Controller obstacle at distance {distance} exceeds sensing_range {sensing_range}"
    
    # Property 5: Verify each risk obstacle is within range
    for obs_dict in risk_obs:
        distance = np.sqrt(
            (obs_dict['x'] - robot_position[0])**2 + 
            (obs_dict['y'] - robot_position[1])**2
        )
        assert distance <= sensing_range + 1e-6, \
            f"Risk obstacle at distance {distance} exceeds sensing_range {sensing_range}"


@given(
    n_obstacles=st.integers(min_value=2, max_value=6),
    robot_x=st.floats(min_value=-2.0, max_value=2.0),
    robot_y=st.floats(min_value=-1.5, max_value=1.5),
)
@settings(max_examples=50, deadline=2000)
def test_property_33_infinite_sensing_range(n_obstacles, robot_x, robot_y):
    """
    Property 33: Infinite Sensing Range
    
    **Validates: Requirements 13.6**
    
    GIVEN an obstacle field with infinite sensing_range
    WHEN robot position is updated
    THEN controller_obstacles() returns all obstacles
    """
    # Generate obstacles
    rng = np.random.RandomState(42)
    obstacles = []
    for _ in range(n_obstacles):
        obstacles.append(DynamicObstacle(
            x=rng.uniform(-2.0, 2.0),
            y=rng.uniform(-1.5, 1.5),
            radius=rng.uniform(0.1, 0.3),
            motion_model="static"
        ))
    
    # Create field with infinite sensing range
    inflation = InflationConfig(sensing_range=float('inf'))
    field = DynamicObstacleField(obstacles, inflation=inflation)
    
    # Update robot state
    robot_position = np.array([robot_x, robot_y])
    field.update_robot_state(robot_position)
    
    # Get obstacle views
    controller_obs = field.controller_obstacles()
    actual_obs = field.actual_obstacles()
    
    # With infinite sensing range, controller should see all obstacles
    assert len(controller_obs) == len(obstacles), \
        f"With infinite sensing_range, controller should see all {len(obstacles)} obstacles, got {len(controller_obs)}"
    
    assert len(actual_obs) == len(obstacles), \
        f"actual_obstacles() should always return all {len(obstacles)} obstacles"


@given(
    robot_velocity=st.floats(min_value=0.0, max_value=2.0),
    obstacle_speed=st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=50, deadline=2000)
def test_property_velocity_adaptive_inflation(robot_velocity, obstacle_speed):
    """
    Property: Velocity-Adaptive Inflation
    
    **Validates: Requirements 14.1-14.5**
    
    GIVEN obstacles with motion and robot with velocity
    WHEN velocity_scaling_factor is configured
    THEN inflated radius increases with robot velocity
    AND inflated radius increases with obstacle speed
    """
    # Create obstacle with motion
    obstacle = DynamicObstacle(
        x=1.0,
        y=0.5,
        radius=0.2,
        vx=obstacle_speed * 0.6,  # Split speed into components
        vy=obstacle_speed * 0.8,
        motion_model="linear"
    )
    
    # Test with velocity scaling
    inflation_with_scaling = InflationConfig(
        safety_factor=1.0,
        sensing_factor=0.05,
        motion_lookahead=0.5,
        velocity_scaling_factor=0.1
    )
    
    inflation_without_scaling = InflationConfig(
        safety_factor=1.0,
        sensing_factor=0.05,
        motion_lookahead=0.5,
        velocity_scaling_factor=0.0
    )
    
    # Compute inflated radii
    radius_with_velocity = obstacle.inflated_radius(inflation_with_scaling, robot_velocity)
    radius_without_velocity = obstacle.inflated_radius(inflation_without_scaling, robot_velocity)
    radius_zero_velocity = obstacle.inflated_radius(inflation_with_scaling, 0.0)
    
    # Property 1: Radius with velocity scaling should be larger when robot has velocity
    if robot_velocity > 0.01:
        assert radius_with_velocity > radius_zero_velocity, \
            f"Radius with robot velocity {robot_velocity} should be larger than with zero velocity"
    
    # Property 2: Radius without velocity scaling should not depend on robot velocity
    radius_without_velocity_2 = obstacle.inflated_radius(inflation_without_scaling, robot_velocity * 2)
    assert abs(radius_without_velocity - radius_without_velocity_2) < 1e-6, \
        "Radius without velocity scaling should not change with robot velocity"
    
    # Property 3: Radius should include motion lookahead for obstacle speed
    expected_motion_margin = obstacle.speed * inflation_with_scaling.motion_lookahead
    base_inflated = obstacle.radius * inflation_with_scaling.safety_factor + inflation_with_scaling.sensing_factor
    
    # The radius should be at least base + motion margin
    assert radius_zero_velocity >= base_inflated + expected_motion_margin - 1e-6, \
        f"Radius should include motion lookahead margin for obstacle speed"


@given(
    n_obstacles=st.integers(min_value=3, max_value=8),
)
@settings(max_examples=30, deadline=2000)
def test_property_no_robot_state_shows_all_obstacles(n_obstacles):
    """
    Property: Without robot state update, all obstacles are visible
    
    GIVEN an obstacle field with sensing range
    WHEN robot state is NOT updated
    THEN controller_obstacles() returns all obstacles (default behavior)
    """
    # Generate obstacles
    rng = np.random.RandomState(42)
    obstacles = []
    for _ in range(n_obstacles):
        obstacles.append(DynamicObstacle(
            x=rng.uniform(-2.0, 2.0),
            y=rng.uniform(-1.5, 1.5),
            radius=rng.uniform(0.1, 0.3),
            motion_model="static"
        ))
    
    # Create field with finite sensing range
    inflation = InflationConfig(sensing_range=1.0)
    field = DynamicObstacleField(obstacles, inflation=inflation)
    
    # Do NOT update robot state
    # Get obstacle views
    controller_obs = field.controller_obstacles()
    
    # Without robot state, should return all obstacles (robot_position is None)
    assert len(controller_obs) == len(obstacles), \
        f"Without robot state update, controller should see all {len(obstacles)} obstacles, got {len(controller_obs)}"


# Property 26: Obstacle Inflation Formula
# **Validates: Requirements 6.6, 6.7**

@given(
    base_radius=st.floats(min_value=0.1, max_value=0.5),
    safety_factor=st.floats(min_value=1.0, max_value=2.0),
    sensing_factor=st.floats(min_value=0.0, max_value=0.2),
    motion_lookahead=st.floats(min_value=0.0, max_value=1.0),
    obstacle_speed=st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=50, deadline=2000)
def test_property_26_obstacle_inflation_formula(
    base_radius, safety_factor, sensing_factor, motion_lookahead, obstacle_speed
):
    """
    Property 26: Obstacle Inflation Formula
    
    **Validates: Requirements 6.6, 6.7**
    
    GIVEN an obstacle with base radius and speed
    AND an InflationConfig with safety_factor, sensing_factor, motion_lookahead
    WHEN computing inflated radius
    THEN the result follows the formula:
        r_inflated = base_radius * safety_factor + sensing_factor + motion_lookahead * obstacle_speed
    """
    # Create obstacle with specified speed
    vx = obstacle_speed * 0.6  # Split into components
    vy = obstacle_speed * 0.8
    
    obstacle = DynamicObstacle(
        x=1.0,
        y=0.5,
        radius=base_radius,
        vx=vx,
        vy=vy,
        motion_model="linear"
    )
    
    # Create inflation config
    inflation = InflationConfig(
        safety_factor=safety_factor,
        sensing_factor=sensing_factor,
        motion_lookahead=motion_lookahead,
        velocity_scaling_factor=0.0  # No robot velocity for this test
    )
    
    # Compute inflated radius using the method
    inflated = obstacle.inflated_radius(inflation, robot_velocity=0.0)
    
    # Compute expected value using the formula
    expected = (
        base_radius * safety_factor
        + sensing_factor
        + motion_lookahead * obstacle.speed
    )
    
    # Verify the formula is correct
    assert abs(inflated - expected) < 1e-6, \
        f"Inflated radius {inflated} does not match expected {expected} from formula"
    
    # Also test using InflationConfig.compute_inflated_radius directly
    inflated_direct = inflation.compute_inflated_radius(base_radius, obstacle.speed, 0.0)
    
    assert abs(inflated_direct - expected) < 1e-6, \
        f"InflationConfig.compute_inflated_radius {inflated_direct} does not match expected {expected}"
    
    # Verify both methods give the same result
    assert abs(inflated - inflated_direct) < 1e-6, \
        f"DynamicObstacle.inflated_radius {inflated} and InflationConfig.compute_inflated_radius {inflated_direct} should match"


@given(
    base_radius=st.floats(min_value=0.1, max_value=0.5),
    safety_factor=st.floats(min_value=1.0, max_value=2.0),
    sensing_factor=st.floats(min_value=0.0, max_value=0.2),
    motion_lookahead=st.floats(min_value=0.0, max_value=1.0),
    velocity_scaling_factor=st.floats(min_value=0.0, max_value=0.3),
    obstacle_speed=st.floats(min_value=0.0, max_value=1.0),
    robot_speed=st.floats(min_value=0.0, max_value=2.0),
)
@settings(max_examples=50, deadline=2000)
def test_property_26_full_inflation_formula(
    base_radius, safety_factor, sensing_factor, motion_lookahead, 
    velocity_scaling_factor, obstacle_speed, robot_speed
):
    """
    Property 26: Full Obstacle Inflation Formula (with robot velocity)
    
    **Validates: Requirements 6.6, 6.7, 14.1-14.5**
    
    GIVEN an obstacle with base radius and speed
    AND an InflationConfig with all parameters
    AND a robot with velocity
    WHEN computing inflated radius
    THEN the result follows the complete formula:
        r_inflated = base_radius * safety_factor 
                   + sensing_factor
                   + motion_lookahead * obstacle_speed
                   + velocity_scaling_factor * robot_speed
    """
    # Create obstacle with specified speed
    vx = obstacle_speed * 0.6
    vy = obstacle_speed * 0.8
    
    obstacle = DynamicObstacle(
        x=1.0,
        y=0.5,
        radius=base_radius,
        vx=vx,
        vy=vy,
        motion_model="linear"
    )
    
    # Create inflation config with all parameters
    inflation = InflationConfig(
        safety_factor=safety_factor,
        sensing_factor=sensing_factor,
        motion_lookahead=motion_lookahead,
        velocity_scaling_factor=velocity_scaling_factor
    )
    
    # Compute inflated radius
    inflated = obstacle.inflated_radius(inflation, robot_velocity=robot_speed)
    
    # Compute expected value using the complete formula
    expected = (
        base_radius * safety_factor
        + sensing_factor
        + motion_lookahead * obstacle.speed
        + velocity_scaling_factor * robot_speed
    )
    
    # Verify the formula is correct
    assert abs(inflated - expected) < 1e-6, \
        f"Inflated radius {inflated} does not match expected {expected} from complete formula"
    
    # Also test using InflationConfig.compute_inflated_radius directly
    inflated_direct = inflation.compute_inflated_radius(base_radius, obstacle.speed, robot_speed)
    
    assert abs(inflated_direct - expected) < 1e-6, \
        f"InflationConfig.compute_inflated_radius {inflated_direct} does not match expected {expected}"
    
    # Verify both methods give the same result
    assert abs(inflated - inflated_direct) < 1e-6, \
        f"DynamicObstacle.inflated_radius {inflated} and InflationConfig.compute_inflated_radius {inflated_direct} should match"


# Property 24: Random Walk Velocity Distribution
# **Validates: Requirements 6.4**

@given(
    random_walk_std=st.floats(min_value=0.05, max_value=0.2),
    dt=st.floats(min_value=0.01, max_value=0.03),
    seed=st.integers(min_value=0, max_value=10000),
)
@settings(max_examples=30, deadline=5000)
def test_property_24_random_walk_velocity_distribution(random_walk_std, dt, seed):
    """
    Property 24: Random Walk Velocity Distribution
    
    **Validates: Requirements 6.4**
    
    GIVEN an obstacle with random_walk motion model
    WHEN stepping the obstacle many times
    THEN velocity updates follow a Gaussian distribution with specified std
    AND the velocity changes have approximately zero mean
    
    Note: This test measures the raw velocity changes applied by the random walk.
    We reset the obstacle periodically to avoid velocity accumulation and clipping.
    """
    rng = np.random.RandomState(seed)
    
    # Collect velocity changes over many steps
    # Reset obstacle periodically to avoid accumulation
    n_resets = 20
    steps_per_reset = 10
    vx_changes = []
    vy_changes = []
    
    for _ in range(n_resets):
        # Create fresh obstacle with zero velocity
        obstacle = DynamicObstacle(
            x=0.0,
            y=0.0,
            radius=0.2,
            vx=0.0,
            vy=0.0,
            motion_model="random_walk",
            random_walk_std=random_walk_std,
            max_speed=10.0  # High limit
        )
        
        for _ in range(steps_per_reset):
            vx_before = obstacle.vx
            vy_before = obstacle.vy
            
            obstacle.step(dt, rng)
            
            vx_changes.append(obstacle.vx - vx_before)
            vy_changes.append(obstacle.vy - vy_before)
    
    vx_changes = np.array(vx_changes)
    vy_changes = np.array(vy_changes)
    
    # Expected standard deviation for velocity changes
    expected_std = random_walk_std * np.sqrt(dt)
    
    # Property 1: Mean of velocity changes should be close to zero
    mean_vx = np.mean(vx_changes)
    mean_vy = np.mean(vy_changes)
    
    # With many samples, use reasonable tolerance
    # 95% confidence interval for mean is ~2*std/sqrt(n)
    n_samples = len(vx_changes)
    tolerance = 4 * expected_std / np.sqrt(n_samples)
    
    assert abs(mean_vx) < tolerance, \
        f"Mean vx change {mean_vx} should be close to 0 (tolerance {tolerance})"
    assert abs(mean_vy) < tolerance, \
        f"Mean vy change {mean_vy} should be close to 0 (tolerance {tolerance})"
    
    # Property 2: Standard deviation should match expected value
    std_vx = np.std(vx_changes, ddof=1)
    std_vy = np.std(vy_changes, ddof=1)
    
    # Allow 50% tolerance for std estimation with finite samples
    # This is reasonable given the stochastic nature
    std_tolerance = 0.5 * expected_std
    
    assert abs(std_vx - expected_std) < std_tolerance, \
        f"Std of vx changes {std_vx} should be close to {expected_std} (tolerance {std_tolerance})"
    assert abs(std_vy - expected_std) < std_tolerance, \
        f"Std of vy changes {std_vy} should be close to {expected_std} (tolerance {std_tolerance})"


# Property 25: Boundary Reflection
# **Validates: Requirements 6.5**

@given(
    initial_vx=st.floats(min_value=-0.5, max_value=0.5),
    initial_vy=st.floats(min_value=-0.5, max_value=0.5),
    dt=st.floats(min_value=0.01, max_value=0.05),
)
@settings(max_examples=50, deadline=2000)
def test_property_25_boundary_reflection(initial_vx, initial_vy, dt):
    """
    Property 25: Boundary Reflection
    
    **Validates: Requirements 6.5**
    
    GIVEN an obstacle at an environment boundary
    WHEN the obstacle would move beyond the boundary
    THEN the velocity component perpendicular to the boundary reverses sign
    AND the obstacle position is clamped to the boundary
    """
    bounds = (-2.5, 2.5, -1.8, 1.8)
    radius = 0.2
    
    # Test x-min boundary
    obstacle = DynamicObstacle(
        x=bounds[0] + radius - 0.01,  # Just inside boundary
        y=0.0,
        radius=radius,
        vx=-0.3,  # Moving toward boundary
        vy=initial_vy,
        motion_model="linear",
        bounds=bounds
    )
    
    rng = np.random.RandomState(42)
    obstacle.step(dt, rng)
    
    # After stepping, should be at boundary with reversed vx
    assert obstacle.x >= bounds[0] + radius - 1e-6, \
        f"Obstacle x={obstacle.x} should be at or inside x_min boundary {bounds[0] + radius}"
    assert obstacle.vx >= 0, \
        f"Velocity vx={obstacle.vx} should be positive (reflected) after hitting x_min boundary"
    
    # Test x-max boundary
    obstacle = DynamicObstacle(
        x=bounds[1] - radius + 0.01,  # Just inside boundary
        y=0.0,
        radius=radius,
        vx=0.3,  # Moving toward boundary
        vy=initial_vy,
        motion_model="linear",
        bounds=bounds
    )
    
    obstacle.step(dt, rng)
    
    assert obstacle.x <= bounds[1] - radius + 1e-6, \
        f"Obstacle x={obstacle.x} should be at or inside x_max boundary {bounds[1] - radius}"
    assert obstacle.vx <= 0, \
        f"Velocity vx={obstacle.vx} should be negative (reflected) after hitting x_max boundary"
    
    # Test y-min boundary
    obstacle = DynamicObstacle(
        x=0.0,
        y=bounds[2] + radius - 0.01,  # Just inside boundary
        radius=radius,
        vx=initial_vx,
        vy=-0.3,  # Moving toward boundary
        motion_model="linear",
        bounds=bounds
    )
    
    obstacle.step(dt, rng)
    
    assert obstacle.y >= bounds[2] + radius - 1e-6, \
        f"Obstacle y={obstacle.y} should be at or inside y_min boundary {bounds[2] + radius}"
    assert obstacle.vy >= 0, \
        f"Velocity vy={obstacle.vy} should be positive (reflected) after hitting y_min boundary"
    
    # Test y-max boundary
    obstacle = DynamicObstacle(
        x=0.0,
        y=bounds[3] - radius + 0.01,  # Just inside boundary
        radius=radius,
        vx=initial_vx,
        vy=0.3,  # Moving toward boundary
        motion_model="linear",
        bounds=bounds
    )
    
    obstacle.step(dt, rng)
    
    assert obstacle.y <= bounds[3] - radius + 1e-6, \
        f"Obstacle y={obstacle.y} should be at or inside y_max boundary {bounds[3] - radius}"
    assert obstacle.vy <= 0, \
        f"Velocity vy={obstacle.vy} should be negative (reflected) after hitting y_max boundary"

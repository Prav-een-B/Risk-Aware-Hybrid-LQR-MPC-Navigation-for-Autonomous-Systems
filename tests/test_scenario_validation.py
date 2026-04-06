"""
Comprehensive validation tests for all five evaluation scenarios.

This test suite validates:
1. Each scenario configuration is valid
2. Scenarios produce expected obstacle counts
3. Uncertainty parameters are applied correctly
4. Inflation parameters are correctly configured
5. Trajectory parameters are present
6. Scenarios can create valid obstacle fields
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.scenarios import (
    get_baseline_static_scenario,
    get_urban_dynamic_scenario,
    get_stochastic_navigation_scenario,
    get_oscillatory_tracking_scenario,
    get_vehicle_realistic_scenario,
)


# ============================================================================
# Test 1: Baseline Static Scenario
# ============================================================================

def test_baseline_static_scenario_structure():
    """Test that baseline_static scenario has correct structure."""
    scenario = get_baseline_static_scenario(seed=42)
    
    # Check basic structure
    assert scenario.name == "baseline_static"
    assert scenario.trajectory_type == "lissajous"
    assert scenario.duration == 20.0
    assert scenario.dt == 0.02
    
    # Check trajectory parameters
    assert "amplitude" in scenario.trajectory_params
    assert "frequency" in scenario.trajectory_params
    assert scenario.trajectory_params["amplitude"] == 2.0
    assert scenario.trajectory_params["frequency"] == 0.5
    
    # Check obstacle configuration (3 sparse static obstacles)
    assert len(scenario.obstacle_config.obstacles) == 3
    assert scenario.obstacle_config.name == "sparse_static"
    
    # Check all obstacles are static
    for obs in scenario.obstacle_config.obstacles:
        assert obs.get("motion_model", "static") == "static"
        assert obs.get("vx", 0.0) == 0.0
        assert obs.get("vy", 0.0) == 0.0
    
    # Check inflation config (baseline parameters)
    assert scenario.inflation_config.safety_factor == 1.0
    assert scenario.inflation_config.sensing_factor == 0.05
    assert scenario.inflation_config.motion_lookahead == 0.5
    assert scenario.inflation_config.velocity_scaling_factor == 0.0
    
    # Check uncertainty config (no uncertainty)
    assert scenario.uncertainty_config.process_noise_position_std == 0.0
    assert scenario.uncertainty_config.process_noise_heading_std == 0.0
    assert scenario.uncertainty_config.sensor_noise_position_std == 0.0
    assert scenario.uncertainty_config.sensor_noise_heading_std == 0.0
    assert scenario.uncertainty_config.velocity_mismatch_factor == 1.0
    assert scenario.uncertainty_config.angular_mismatch_factor == 1.0
    assert scenario.uncertainty_config.control_delay_steps == 0


def test_baseline_static_obstacle_field_creation():
    """Test that baseline_static scenario can create valid obstacle field."""
    scenario = get_baseline_static_scenario(seed=42)
    
    # Create obstacle field
    field = scenario.obstacle_config.create_field(scenario.inflation_config)
    
    # Check field has correct number of obstacles
    assert len(field.obstacles) == 3
    
    # Check all obstacles are static
    for obs in field.obstacles:
        assert obs.motion_model == "static"
        assert obs.vx == 0.0
        assert obs.vy == 0.0


# ============================================================================
# Test 2: Urban Dynamic Scenario
# ============================================================================

def test_urban_dynamic_scenario_structure():
    """Test that urban_dynamic scenario has correct structure."""
    scenario = get_urban_dynamic_scenario(seed=42)
    
    # Check basic structure
    assert scenario.name == "urban_dynamic"
    assert scenario.trajectory_type == "urban_path"
    assert scenario.duration == 25.0
    assert scenario.dt == 0.02
    
    # Check trajectory parameters
    assert "block_size" in scenario.trajectory_params
    assert "num_blocks" in scenario.trajectory_params
    assert scenario.trajectory_params["block_size"] == 2.0
    assert scenario.trajectory_params["num_blocks"] == 4
    
    # Check obstacle configuration (5 moving obstacles)
    assert len(scenario.obstacle_config.obstacles) == 5
    assert scenario.obstacle_config.name == "urban_dynamic"
    
    # Check all obstacles have linear motion with speed in range [0.2, 0.4]
    for obs in scenario.obstacle_config.obstacles:
        assert obs["motion_model"] == "linear"
        assert "vx" in obs
        assert "vy" in obs
        assert "bounds" in obs
        
        # Check speed is in range [0.2, 0.4]
        speed = (obs["vx"]**2 + obs["vy"]**2)**0.5
        assert 0.19 <= speed <= 0.41, f"Speed {speed} out of range [0.2, 0.4]"
    
    # Check inflation config (moderate safety margins)
    assert scenario.inflation_config.safety_factor == 1.2
    assert scenario.inflation_config.sensing_factor == 0.08
    assert scenario.inflation_config.motion_lookahead == 0.8
    assert scenario.inflation_config.velocity_scaling_factor == 0.0
    
    # Check uncertainty config (moderate uncertainty)
    assert scenario.uncertainty_config.process_noise_position_std == 0.01
    assert scenario.uncertainty_config.sensor_noise_position_std == 0.02
    assert scenario.uncertainty_config.velocity_mismatch_factor == 0.95
    assert scenario.uncertainty_config.control_delay_steps == 1


def test_urban_dynamic_obstacle_field_creation():
    """Test that urban_dynamic scenario can create valid obstacle field."""
    scenario = get_urban_dynamic_scenario(seed=42)
    
    # Create obstacle field
    field = scenario.obstacle_config.create_field(scenario.inflation_config)
    
    # Check field has correct number of obstacles
    assert len(field.obstacles) == 5
    
    # Check all obstacles have linear motion
    for obs in field.obstacles:
        assert obs.motion_model == "linear"
        assert obs.speed > 0.0


# ============================================================================
# Test 3: Stochastic Navigation Scenario
# ============================================================================

def test_stochastic_navigation_scenario_structure():
    """Test that stochastic_navigation scenario has correct structure."""
    scenario = get_stochastic_navigation_scenario(seed=42)
    
    # Check basic structure
    assert scenario.name == "stochastic_navigation"
    assert scenario.trajectory_type == "sinusoidal"
    assert scenario.duration == 20.0
    assert scenario.dt == 0.02
    
    # Check trajectory parameters
    assert "amplitude" in scenario.trajectory_params
    assert "frequency" in scenario.trajectory_params
    assert scenario.trajectory_params["amplitude"] == 1.5
    assert scenario.trajectory_params["frequency"] == 0.8
    
    # Check obstacle configuration (4 random_walk obstacles)
    assert len(scenario.obstacle_config.obstacles) == 4
    assert scenario.obstacle_config.name == "stochastic_navigation"
    
    # Check all obstacles have random_walk motion
    for obs in scenario.obstacle_config.obstacles:
        assert obs["motion_model"] == "random_walk"
        assert "vx" in obs
        assert "vy" in obs
        assert "bounds" in obs
        assert obs["random_walk_std"] == 0.2
        assert obs["max_speed"] == 0.45
    
    # Check inflation config (high safety margins with velocity-adaptive inflation)
    assert scenario.inflation_config.safety_factor == 1.5
    assert scenario.inflation_config.sensing_factor == 0.1
    assert scenario.inflation_config.motion_lookahead == 1.0
    assert scenario.inflation_config.velocity_scaling_factor == 0.15
    
    # Check uncertainty config (high uncertainty)
    assert scenario.uncertainty_config.process_noise_position_std == 0.03
    assert scenario.uncertainty_config.process_noise_heading_std == 0.05
    assert scenario.uncertainty_config.sensor_noise_position_std == 0.04
    assert scenario.uncertainty_config.sensor_noise_heading_std == 0.06
    assert scenario.uncertainty_config.velocity_mismatch_factor == 0.90
    assert scenario.uncertainty_config.control_delay_steps == 2


def test_stochastic_navigation_obstacle_field_creation():
    """Test that stochastic_navigation scenario can create valid obstacle field."""
    scenario = get_stochastic_navigation_scenario(seed=42)
    
    # Create obstacle field
    field = scenario.obstacle_config.create_field(scenario.inflation_config)
    
    # Check field has correct number of obstacles
    assert len(field.obstacles) == 4
    
    # Check all obstacles have random_walk motion
    for obs in field.obstacles:
        assert obs.motion_model == "random_walk"
        assert obs.random_walk_std == 0.2
        assert obs.max_speed == 0.45


# ============================================================================
# Test 4: Oscillatory Tracking Scenario
# ============================================================================

def test_oscillatory_tracking_scenario_structure():
    """Test that oscillatory_tracking scenario has correct structure."""
    scenario = get_oscillatory_tracking_scenario(seed=42)
    
    # Check basic structure
    assert scenario.name == "oscillatory_tracking"
    assert scenario.trajectory_type == "lissajous"
    assert scenario.duration == 20.0
    assert scenario.dt == 0.02
    
    # Check trajectory parameters (high-frequency Lissajous)
    assert "amplitude" in scenario.trajectory_params
    assert "a" in scenario.trajectory_params
    assert "b" in scenario.trajectory_params
    assert "c" in scenario.trajectory_params
    assert scenario.trajectory_params["amplitude"] == 1.5
    assert scenario.trajectory_params["a"] == 2.0
    assert scenario.trajectory_params["b"] == 3.0
    assert scenario.trajectory_params["c"] == 2.5
    
    # Check obstacle configuration (8 dense static obstacles)
    assert len(scenario.obstacle_config.obstacles) == 8
    assert scenario.obstacle_config.name == "dense_static"
    
    # Check all obstacles are static
    for obs in scenario.obstacle_config.obstacles:
        assert obs.get("motion_model", "static") == "static"
        assert obs.get("vx", 0.0) == 0.0
        assert obs.get("vy", 0.0) == 0.0
        
        # Check obstacles avoid center region (radius > 0.75m from origin)
        distance_from_origin = (obs["x"]**2 + obs["y"]**2)**0.5
        assert distance_from_origin > 0.75, f"Obstacle at ({obs['x']}, {obs['y']}) too close to origin"
    
    # Check inflation config (baseline parameters)
    assert scenario.inflation_config.safety_factor == 1.0
    assert scenario.inflation_config.sensing_factor == 0.05
    assert scenario.inflation_config.motion_lookahead == 0.5
    assert scenario.inflation_config.velocity_scaling_factor == 0.0
    
    # Check uncertainty config (low uncertainty)
    assert scenario.uncertainty_config.process_noise_position_std == 0.005
    assert scenario.uncertainty_config.sensor_noise_position_std == 0.01
    assert scenario.uncertainty_config.velocity_mismatch_factor == 1.0
    assert scenario.uncertainty_config.control_delay_steps == 0


def test_oscillatory_tracking_obstacle_field_creation():
    """Test that oscillatory_tracking scenario can create valid obstacle field."""
    scenario = get_oscillatory_tracking_scenario(seed=42)
    
    # Create obstacle field
    field = scenario.obstacle_config.create_field(scenario.inflation_config)
    
    # Check field has correct number of obstacles
    assert len(field.obstacles) == 8
    
    # Check all obstacles are static
    for obs in field.obstacles:
        assert obs.motion_model == "static"
        assert obs.vx == 0.0
        assert obs.vy == 0.0


# ============================================================================
# Test 5: Vehicle Realistic Scenario
# ============================================================================

def test_vehicle_realistic_scenario_structure():
    """Test that vehicle_realistic scenario has correct structure."""
    scenario = get_vehicle_realistic_scenario(seed=42)
    
    # Check basic structure
    assert scenario.name == "vehicle_realistic"
    assert scenario.trajectory_type == "clothoid"
    assert scenario.duration == 25.0
    assert scenario.dt == 0.02
    
    # Check trajectory parameters (clothoid)
    assert "kappa0" in scenario.trajectory_params
    assert "k_rate" in scenario.trajectory_params
    assert scenario.trajectory_params["kappa0"] == 0.0
    assert scenario.trajectory_params["k_rate"] == 0.5
    
    # Check obstacle configuration (corridor: 20 obstacles forming two walls)
    assert len(scenario.obstacle_config.obstacles) == 20
    assert scenario.obstacle_config.name == "corridor"
    
    # Check all obstacles are static and form corridor
    for obs in scenario.obstacle_config.obstacles:
        assert obs.get("motion_model", "static") == "static"
        assert obs.get("vx", 0.0) == 0.0
        assert obs.get("vy", 0.0) == 0.0
        assert obs["radius"] == 0.15
        
        # Check obstacles are on corridor walls (y ≈ ±0.6)
        assert abs(abs(obs["y"]) - 0.6) < 0.01, f"Obstacle y={obs['y']} not on corridor wall"
    
    # Check inflation config (higher safety margins with velocity-adaptive scaling)
    assert scenario.inflation_config.safety_factor == 1.3
    assert scenario.inflation_config.sensing_factor == 0.1
    assert scenario.inflation_config.motion_lookahead == 0.6
    assert scenario.inflation_config.velocity_scaling_factor == 0.2
    
    # Check uncertainty config (high model mismatch and latency)
    assert scenario.uncertainty_config.process_noise_position_std == 0.02
    assert scenario.uncertainty_config.sensor_noise_position_std == 0.03
    assert scenario.uncertainty_config.velocity_mismatch_factor == 0.85
    assert scenario.uncertainty_config.angular_mismatch_factor == 0.90
    assert scenario.uncertainty_config.control_delay_steps == 3


def test_vehicle_realistic_obstacle_field_creation():
    """Test that vehicle_realistic scenario can create valid obstacle field."""
    scenario = get_vehicle_realistic_scenario(seed=42)
    
    # Create obstacle field
    field = scenario.obstacle_config.create_field(scenario.inflation_config)
    
    # Check field has correct number of obstacles
    assert len(field.obstacles) == 20
    
    # Check all obstacles are static
    for obs in field.obstacles:
        assert obs.motion_model == "static"
        assert obs.vx == 0.0
        assert obs.vy == 0.0


# ============================================================================
# Cross-Scenario Tests
# ============================================================================

def test_all_scenarios_reproducibility():
    """Test that all scenarios are reproducible with same seed."""
    scenarios = [
        get_baseline_static_scenario,
        get_urban_dynamic_scenario,
        get_stochastic_navigation_scenario,
        get_oscillatory_tracking_scenario,
        get_vehicle_realistic_scenario,
    ]
    
    for scenario_func in scenarios:
        scenario1 = scenario_func(seed=123)
        scenario2 = scenario_func(seed=123)
        
        # Check obstacles are identical
        assert len(scenario1.obstacle_config.obstacles) == len(scenario2.obstacle_config.obstacles)
        
        for obs1, obs2 in zip(scenario1.obstacle_config.obstacles, scenario2.obstacle_config.obstacles):
            assert obs1["x"] == obs2["x"]
            assert obs1["y"] == obs2["y"]
            assert obs1["radius"] == obs2["radius"]


def test_all_scenarios_different_seeds():
    """Test that scenarios with randomized obstacles produce different configurations with different seeds."""
    # Only test scenarios that use randomization
    # baseline_static and vehicle_realistic have deterministic obstacles
    scenarios_with_randomization = [
        get_urban_dynamic_scenario,
        get_stochastic_navigation_scenario,
        get_oscillatory_tracking_scenario,
    ]
    
    for scenario_func in scenarios_with_randomization:
        scenario1 = scenario_func(seed=42)
        scenario2 = scenario_func(seed=999)
        
        # At least one obstacle should be different
        different = False
        for obs1, obs2 in zip(scenario1.obstacle_config.obstacles, scenario2.obstacle_config.obstacles):
            if obs1["x"] != obs2["x"] or obs1["y"] != obs2["y"]:
                different = True
                break
        
        assert different, f"{scenario_func.__name__}: Different seeds should produce different configurations"


def test_all_scenarios_have_valid_configs():
    """Test that all scenarios have valid configuration objects."""
    scenarios = [
        get_baseline_static_scenario(seed=42),
        get_urban_dynamic_scenario(seed=42),
        get_stochastic_navigation_scenario(seed=42),
        get_oscillatory_tracking_scenario(seed=42),
        get_vehicle_realistic_scenario(seed=42),
    ]
    
    for scenario in scenarios:
        # Check all required attributes exist
        assert hasattr(scenario, 'name')
        assert hasattr(scenario, 'trajectory_type')
        assert hasattr(scenario, 'trajectory_params')
        assert hasattr(scenario, 'obstacle_config')
        assert hasattr(scenario, 'inflation_config')
        assert hasattr(scenario, 'uncertainty_config')
        assert hasattr(scenario, 'duration')
        assert hasattr(scenario, 'dt')
        
        # Check types
        assert isinstance(scenario.name, str)
        assert isinstance(scenario.trajectory_type, str)
        assert isinstance(scenario.trajectory_params, dict)
        assert isinstance(scenario.duration, (int, float))
        assert isinstance(scenario.dt, (int, float))
        
        # Check positive values
        assert scenario.duration > 0
        assert scenario.dt > 0


def test_all_scenarios_inflation_parameters():
    """Test that all scenarios have correctly configured inflation parameters."""
    scenarios = [
        get_baseline_static_scenario(seed=42),
        get_urban_dynamic_scenario(seed=42),
        get_stochastic_navigation_scenario(seed=42),
        get_oscillatory_tracking_scenario(seed=42),
        get_vehicle_realistic_scenario(seed=42),
    ]
    
    for scenario in scenarios:
        inflation = scenario.inflation_config
        
        # Check all required attributes exist
        assert hasattr(inflation, 'safety_factor')
        assert hasattr(inflation, 'sensing_factor')
        assert hasattr(inflation, 'motion_lookahead')
        assert hasattr(inflation, 'velocity_scaling_factor')
        assert hasattr(inflation, 'sensing_range')
        
        # Check valid ranges
        assert inflation.safety_factor >= 1.0, f"{scenario.name}: safety_factor must be >= 1.0"
        assert inflation.sensing_factor >= 0.0, f"{scenario.name}: sensing_factor must be >= 0.0"
        assert inflation.motion_lookahead >= 0.0, f"{scenario.name}: motion_lookahead must be >= 0.0"
        assert inflation.velocity_scaling_factor >= 0.0, f"{scenario.name}: velocity_scaling_factor must be >= 0.0"


def test_all_scenarios_uncertainty_parameters():
    """Test that all scenarios have correctly configured uncertainty parameters."""
    scenarios = [
        get_baseline_static_scenario(seed=42),
        get_urban_dynamic_scenario(seed=42),
        get_stochastic_navigation_scenario(seed=42),
        get_oscillatory_tracking_scenario(seed=42),
        get_vehicle_realistic_scenario(seed=42),
    ]
    
    for scenario in scenarios:
        uncertainty = scenario.uncertainty_config
        
        # Check all required attributes exist
        assert hasattr(uncertainty, 'process_noise_position_std')
        assert hasattr(uncertainty, 'process_noise_heading_std')
        assert hasattr(uncertainty, 'sensor_noise_position_std')
        assert hasattr(uncertainty, 'sensor_noise_heading_std')
        assert hasattr(uncertainty, 'velocity_mismatch_factor')
        assert hasattr(uncertainty, 'angular_mismatch_factor')
        assert hasattr(uncertainty, 'control_delay_steps')
        
        # Check valid ranges
        assert uncertainty.process_noise_position_std >= 0.0
        assert uncertainty.process_noise_heading_std >= 0.0
        assert uncertainty.sensor_noise_position_std >= 0.0
        assert uncertainty.sensor_noise_heading_std >= 0.0
        assert 0.0 < uncertainty.velocity_mismatch_factor <= 1.5
        assert 0.0 < uncertainty.angular_mismatch_factor <= 1.5
        assert uncertainty.control_delay_steps >= 0


def test_all_scenarios_can_create_obstacle_fields():
    """Test that all scenarios can successfully create obstacle fields."""
    scenarios = [
        get_baseline_static_scenario(seed=42),
        get_urban_dynamic_scenario(seed=42),
        get_stochastic_navigation_scenario(seed=42),
        get_oscillatory_tracking_scenario(seed=42),
        get_vehicle_realistic_scenario(seed=42),
    ]
    
    for scenario in scenarios:
        # Create obstacle field
        field = scenario.obstacle_config.create_field(scenario.inflation_config)
        
        # Check field is valid
        assert field is not None
        assert len(field.obstacles) > 0
        
        # Check field methods work
        actual_obstacles = field.actual_obstacles()
        controller_obstacles = field.controller_obstacles()
        risk_obstacles = field.risk_obstacles()
        
        assert len(actual_obstacles) == len(field.obstacles)
        assert len(controller_obstacles) == len(field.obstacles)
        assert len(risk_obstacles) == len(field.obstacles)


if __name__ == "__main__":
    # Run all tests
    print("Testing baseline_static scenario...")
    test_baseline_static_scenario_structure()
    test_baseline_static_obstacle_field_creation()
    
    print("Testing urban_dynamic scenario...")
    test_urban_dynamic_scenario_structure()
    test_urban_dynamic_obstacle_field_creation()
    
    print("Testing stochastic_navigation scenario...")
    test_stochastic_navigation_scenario_structure()
    test_stochastic_navigation_obstacle_field_creation()
    
    print("Testing oscillatory_tracking scenario...")
    test_oscillatory_tracking_scenario_structure()
    test_oscillatory_tracking_obstacle_field_creation()
    
    print("Testing vehicle_realistic scenario...")
    test_vehicle_realistic_scenario_structure()
    test_vehicle_realistic_obstacle_field_creation()
    
    print("Testing cross-scenario properties...")
    test_all_scenarios_reproducibility()
    test_all_scenarios_different_seeds()
    test_all_scenarios_have_valid_configs()
    test_all_scenarios_inflation_parameters()
    test_all_scenarios_uncertainty_parameters()
    test_all_scenarios_can_create_obstacle_fields()
    
    print("\n✓ All scenario validation tests passed!")

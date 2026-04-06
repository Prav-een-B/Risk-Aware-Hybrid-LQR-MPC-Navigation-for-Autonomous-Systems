"""Test vehicle_realistic scenario configuration."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.scenarios import get_vehicle_realistic_scenario


def test_vehicle_realistic_scenario_structure():
    """Test that vehicle_realistic scenario has correct structure."""
    scenario = get_vehicle_realistic_scenario(seed=42)
    
    # Check basic structure
    assert scenario.name == "vehicle_realistic"
    assert scenario.trajectory_type == "clothoid"
    assert scenario.duration == 25.0
    assert scenario.dt == 0.02
    
    # Check trajectory parameters
    assert "kappa0" in scenario.trajectory_params
    assert "k_rate" in scenario.trajectory_params
    assert scenario.trajectory_params["kappa0"] == 0.0
    assert scenario.trajectory_params["k_rate"] == 0.5
    
    # Check obstacle configuration (corridor with 20 obstacles: 10 per side)
    assert len(scenario.obstacle_config.obstacles) == 20
    assert scenario.obstacle_config.name == "corridor"
    
    # Check all obstacles are static (no motion_model or motion_model is static)
    for obs in scenario.obstacle_config.obstacles:
        motion_model = obs.get("motion_model", "static")
        assert motion_model == "static"
        
        # Check obstacles form corridor (y positions should be ±0.6)
        assert abs(abs(obs["y"]) - 0.6) < 0.01, \
            f"Obstacle y={obs['y']} should be near ±0.6 for corridor width 1.2m"
    
    # Check inflation config
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


def test_vehicle_realistic_scenario_reproducibility():
    """Test that same seed produces same scenario."""
    scenario1 = get_vehicle_realistic_scenario(seed=123)
    scenario2 = get_vehicle_realistic_scenario(seed=123)
    
    # Check obstacles are identical
    assert len(scenario1.obstacle_config.obstacles) == len(scenario2.obstacle_config.obstacles)
    
    for obs1, obs2 in zip(scenario1.obstacle_config.obstacles, scenario2.obstacle_config.obstacles):
        assert obs1["x"] == obs2["x"]
        assert obs1["y"] == obs2["y"]
        assert obs1["radius"] == obs2["radius"]


def test_vehicle_realistic_scenario_different_seeds():
    """Test that different seeds produce same scenario (corridor is deterministic)."""
    scenario1 = get_vehicle_realistic_scenario(seed=42)
    scenario2 = get_vehicle_realistic_scenario(seed=999)
    
    # Corridor obstacles should be identical (deterministic placement)
    # Only the seed in obstacle_config should differ
    assert len(scenario1.obstacle_config.obstacles) == len(scenario2.obstacle_config.obstacles)
    
    for obs1, obs2 in zip(scenario1.obstacle_config.obstacles, scenario2.obstacle_config.obstacles):
        assert obs1["x"] == obs2["x"]
        assert obs1["y"] == obs2["y"]
        assert obs1["radius"] == obs2["radius"]


if __name__ == "__main__":
    test_vehicle_realistic_scenario_structure()
    test_vehicle_realistic_scenario_reproducibility()
    test_vehicle_realistic_scenario_different_seeds()
    print("All tests passed!")

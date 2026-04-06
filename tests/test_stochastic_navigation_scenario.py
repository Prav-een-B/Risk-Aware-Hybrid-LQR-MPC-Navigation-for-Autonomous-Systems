"""Test stochastic_navigation scenario configuration."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.scenarios import get_stochastic_navigation_scenario


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
    
    # Check obstacle configuration
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


def test_stochastic_navigation_scenario_reproducibility():
    """Test that same seed produces same scenario."""
    scenario1 = get_stochastic_navigation_scenario(seed=123)
    scenario2 = get_stochastic_navigation_scenario(seed=123)
    
    # Check obstacles are identical
    assert len(scenario1.obstacle_config.obstacles) == len(scenario2.obstacle_config.obstacles)
    
    for obs1, obs2 in zip(scenario1.obstacle_config.obstacles, scenario2.obstacle_config.obstacles):
        assert obs1["x"] == obs2["x"]
        assert obs1["y"] == obs2["y"]
        assert obs1["radius"] == obs2["radius"]
        assert obs1["vx"] == obs2["vx"]
        assert obs1["vy"] == obs2["vy"]


def test_stochastic_navigation_scenario_different_seeds():
    """Test that different seeds produce different scenarios."""
    scenario1 = get_stochastic_navigation_scenario(seed=42)
    scenario2 = get_stochastic_navigation_scenario(seed=999)
    
    # At least one obstacle should be different
    different = False
    for obs1, obs2 in zip(scenario1.obstacle_config.obstacles, scenario2.obstacle_config.obstacles):
        if obs1["x"] != obs2["x"] or obs1["y"] != obs2["y"]:
            different = True
            break
    
    assert different, "Different seeds should produce different obstacle configurations"


if __name__ == "__main__":
    test_stochastic_navigation_scenario_structure()
    test_stochastic_navigation_scenario_reproducibility()
    test_stochastic_navigation_scenario_different_seeds()
    print("All tests passed!")

"""Test urban_dynamic scenario configuration."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.scenarios import get_urban_dynamic_scenario


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
    
    # Check obstacle configuration
    assert len(scenario.obstacle_config.obstacles) == 5
    assert scenario.obstacle_config.name == "urban_dynamic"
    
    # Check all obstacles have linear motion
    for obs in scenario.obstacle_config.obstacles:
        assert obs["motion_model"] == "linear"
        assert "vx" in obs
        assert "vy" in obs
        assert "bounds" in obs
        
        # Check speed is in range [0.2, 0.4]
        speed = (obs["vx"]**2 + obs["vy"]**2)**0.5
        assert 0.19 <= speed <= 0.41, f"Speed {speed} out of range [0.2, 0.4]"
    
    # Check inflation config
    assert scenario.inflation_config.safety_factor == 1.2
    assert scenario.inflation_config.sensing_factor == 0.08
    assert scenario.inflation_config.motion_lookahead == 0.8
    
    # Check uncertainty config (moderate uncertainty)
    assert scenario.uncertainty_config.process_noise_position_std == 0.01
    assert scenario.uncertainty_config.sensor_noise_position_std == 0.02
    assert scenario.uncertainty_config.velocity_mismatch_factor == 0.95
    assert scenario.uncertainty_config.control_delay_steps == 1


def test_urban_dynamic_scenario_reproducibility():
    """Test that same seed produces same scenario."""
    scenario1 = get_urban_dynamic_scenario(seed=123)
    scenario2 = get_urban_dynamic_scenario(seed=123)
    
    # Check obstacles are identical
    assert len(scenario1.obstacle_config.obstacles) == len(scenario2.obstacle_config.obstacles)
    
    for obs1, obs2 in zip(scenario1.obstacle_config.obstacles, scenario2.obstacle_config.obstacles):
        assert obs1["x"] == obs2["x"]
        assert obs1["y"] == obs2["y"]
        assert obs1["radius"] == obs2["radius"]
        assert obs1["vx"] == obs2["vx"]
        assert obs1["vy"] == obs2["vy"]


def test_urban_dynamic_scenario_different_seeds():
    """Test that different seeds produce different scenarios."""
    scenario1 = get_urban_dynamic_scenario(seed=42)
    scenario2 = get_urban_dynamic_scenario(seed=999)
    
    # At least one obstacle should be different
    different = False
    for obs1, obs2 in zip(scenario1.obstacle_config.obstacles, scenario2.obstacle_config.obstacles):
        if obs1["x"] != obs2["x"] or obs1["y"] != obs2["y"]:
            different = True
            break
    
    assert different, "Different seeds should produce different obstacle configurations"


if __name__ == "__main__":
    test_urban_dynamic_scenario_structure()
    test_urban_dynamic_scenario_reproducibility()
    test_urban_dynamic_scenario_different_seeds()
    print("All tests passed!")

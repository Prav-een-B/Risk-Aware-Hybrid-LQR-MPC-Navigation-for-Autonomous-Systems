"""Test that urban_dynamic scenario matches design specification exactly."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.scenarios import get_urban_dynamic_scenario


def test_urban_dynamic_matches_design_spec():
    """
    Test that urban_dynamic scenario matches the design specification exactly.
    
    Design spec from design.md:
    ```yaml
    name: urban_dynamic
    trajectory_type: urban_path
    trajectory_params:
      block_size: 2.0
      num_blocks: 4
    obstacle_config:
      type: moving
      count: 5
      motion: linear
      speed_range: [0.2, 0.4]
    inflation_config:
      safety_factor: 1.2
      sensing_factor: 0.08
      motion_lookahead: 0.8
    uncertainty_config:
      process_noise_position_std: 0.01
      sensor_noise_position_std: 0.02
      velocity_mismatch_factor: 0.95
      control_delay_steps: 1
    duration: 25.0
    ```
    
    Expected Behavior:
    - Moderate checkpoint completion rate (>80%)
    - Moderate tracking error (<0.2m mean)
    - Rare collisions (<2)
    """
    scenario = get_urban_dynamic_scenario(seed=42)
    
    # Test name
    assert scenario.name == "urban_dynamic", \
        f"Expected name 'urban_dynamic', got '{scenario.name}'"
    
    # Test trajectory_type
    assert scenario.trajectory_type == "urban_path", \
        f"Expected trajectory_type 'urban_path', got '{scenario.trajectory_type}'"
    
    # Test trajectory_params
    assert scenario.trajectory_params["block_size"] == 2.0, \
        f"Expected block_size 2.0, got {scenario.trajectory_params['block_size']}"
    assert scenario.trajectory_params["num_blocks"] == 4, \
        f"Expected num_blocks 4, got {scenario.trajectory_params['num_blocks']}"
    
    # Test obstacle_config
    assert len(scenario.obstacle_config.obstacles) == 5, \
        f"Expected 5 obstacles, got {len(scenario.obstacle_config.obstacles)}"
    
    # Test all obstacles have linear motion and speed in range [0.2, 0.4]
    for i, obs in enumerate(scenario.obstacle_config.obstacles):
        assert obs["motion_model"] == "linear", \
            f"Obstacle {i}: Expected motion_model 'linear', got '{obs['motion_model']}'"
        
        speed = (obs["vx"]**2 + obs["vy"]**2)**0.5
        assert 0.2 <= speed <= 0.4, \
            f"Obstacle {i}: Speed {speed:.3f} not in range [0.2, 0.4]"
    
    # Test inflation_config
    assert scenario.inflation_config.safety_factor == 1.2, \
        f"Expected safety_factor 1.2, got {scenario.inflation_config.safety_factor}"
    assert scenario.inflation_config.sensing_factor == 0.08, \
        f"Expected sensing_factor 0.08, got {scenario.inflation_config.sensing_factor}"
    assert scenario.inflation_config.motion_lookahead == 0.8, \
        f"Expected motion_lookahead 0.8, got {scenario.inflation_config.motion_lookahead}"
    
    # Test uncertainty_config
    assert scenario.uncertainty_config.process_noise_position_std == 0.01, \
        f"Expected process_noise_position_std 0.01, got {scenario.uncertainty_config.process_noise_position_std}"
    assert scenario.uncertainty_config.sensor_noise_position_std == 0.02, \
        f"Expected sensor_noise_position_std 0.02, got {scenario.uncertainty_config.sensor_noise_position_std}"
    assert scenario.uncertainty_config.velocity_mismatch_factor == 0.95, \
        f"Expected velocity_mismatch_factor 0.95, got {scenario.uncertainty_config.velocity_mismatch_factor}"
    assert scenario.uncertainty_config.control_delay_steps == 1, \
        f"Expected control_delay_steps 1, got {scenario.uncertainty_config.control_delay_steps}"
    
    # Test duration
    assert scenario.duration == 25.0, \
        f"Expected duration 25.0, got {scenario.duration}"
    
    # Test dt (default)
    assert scenario.dt == 0.02, \
        f"Expected dt 0.02, got {scenario.dt}"
    
    print("✓ All design specification requirements met!")


if __name__ == "__main__":
    test_urban_dynamic_matches_design_spec()
    print("\nDesign specification compliance test passed!")

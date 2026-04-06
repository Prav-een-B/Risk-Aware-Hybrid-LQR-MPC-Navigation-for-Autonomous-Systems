"""Test that stochastic_navigation scenario matches design specification exactly."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.scenarios import get_stochastic_navigation_scenario


def test_stochastic_navigation_matches_design_spec():
    """
    Test that stochastic_navigation scenario matches the design specification exactly.
    
    Design spec from design.md:
    ```yaml
    name: stochastic_navigation
    trajectory_type: sinusoidal
    trajectory_params:
      amplitude: 1.5
      frequency: 0.8
    obstacle_config:
      type: random_walk
      count: 4
      motion: random_walk
      random_walk_std: 0.2
    inflation_config:
      safety_factor: 1.5
      sensing_factor: 0.1
      motion_lookahead: 1.0
      velocity_scaling_factor: 0.15
    uncertainty_config:
      process_noise_position_std: 0.03
      sensor_noise_position_std: 0.04
      process_noise_heading_std: 0.05
      sensor_noise_heading_std: 0.06
      velocity_mismatch_factor: 0.90
      control_delay_steps: 2
    duration: 20.0
    ```
    
    Expected Behavior:
    - Lower checkpoint completion rate (>70%)
    - Higher tracking error (<0.3m mean)
    - Occasional collisions (<5)
    - Hybrid blending adapts to high uncertainty
    """
    scenario = get_stochastic_navigation_scenario(seed=42)
    
    # Test name
    assert scenario.name == "stochastic_navigation", \
        f"Expected name 'stochastic_navigation', got '{scenario.name}'"
    
    # Test trajectory_type
    assert scenario.trajectory_type == "sinusoidal", \
        f"Expected trajectory_type 'sinusoidal', got '{scenario.trajectory_type}'"
    
    # Test trajectory_params
    assert scenario.trajectory_params["amplitude"] == 1.5, \
        f"Expected amplitude 1.5, got {scenario.trajectory_params['amplitude']}"
    assert scenario.trajectory_params["frequency"] == 0.8, \
        f"Expected frequency 0.8, got {scenario.trajectory_params['frequency']}"
    
    # Test obstacle_config
    assert len(scenario.obstacle_config.obstacles) == 4, \
        f"Expected 4 obstacles, got {len(scenario.obstacle_config.obstacles)}"
    
    # Test all obstacles have random_walk motion with correct parameters
    for i, obs in enumerate(scenario.obstacle_config.obstacles):
        assert obs["motion_model"] == "random_walk", \
            f"Obstacle {i}: Expected motion_model 'random_walk', got '{obs['motion_model']}'"
        assert obs["random_walk_std"] == 0.2, \
            f"Obstacle {i}: Expected random_walk_std 0.2, got {obs['random_walk_std']}"
        assert obs["max_speed"] == 0.45, \
            f"Obstacle {i}: Expected max_speed 0.45, got {obs['max_speed']}"
    
    # Test inflation_config
    assert scenario.inflation_config.safety_factor == 1.5, \
        f"Expected safety_factor 1.5, got {scenario.inflation_config.safety_factor}"
    assert scenario.inflation_config.sensing_factor == 0.1, \
        f"Expected sensing_factor 0.1, got {scenario.inflation_config.sensing_factor}"
    assert scenario.inflation_config.motion_lookahead == 1.0, \
        f"Expected motion_lookahead 1.0, got {scenario.inflation_config.motion_lookahead}"
    assert scenario.inflation_config.velocity_scaling_factor == 0.15, \
        f"Expected velocity_scaling_factor 0.15, got {scenario.inflation_config.velocity_scaling_factor}"
    
    # Test uncertainty_config (high uncertainty)
    assert scenario.uncertainty_config.process_noise_position_std == 0.03, \
        f"Expected process_noise_position_std 0.03, got {scenario.uncertainty_config.process_noise_position_std}"
    assert scenario.uncertainty_config.process_noise_heading_std == 0.05, \
        f"Expected process_noise_heading_std 0.05, got {scenario.uncertainty_config.process_noise_heading_std}"
    assert scenario.uncertainty_config.sensor_noise_position_std == 0.04, \
        f"Expected sensor_noise_position_std 0.04, got {scenario.uncertainty_config.sensor_noise_position_std}"
    assert scenario.uncertainty_config.sensor_noise_heading_std == 0.06, \
        f"Expected sensor_noise_heading_std 0.06, got {scenario.uncertainty_config.sensor_noise_heading_std}"
    assert scenario.uncertainty_config.velocity_mismatch_factor == 0.90, \
        f"Expected velocity_mismatch_factor 0.90, got {scenario.uncertainty_config.velocity_mismatch_factor}"
    assert scenario.uncertainty_config.control_delay_steps == 2, \
        f"Expected control_delay_steps 2, got {scenario.uncertainty_config.control_delay_steps}"
    
    # Test duration
    assert scenario.duration == 20.0, \
        f"Expected duration 20.0, got {scenario.duration}"
    
    # Test dt (default)
    assert scenario.dt == 0.02, \
        f"Expected dt 0.02, got {scenario.dt}"
    
    print("✓ All design specification requirements met!")


if __name__ == "__main__":
    test_stochastic_navigation_matches_design_spec()
    print("\nDesign specification compliance test passed!")

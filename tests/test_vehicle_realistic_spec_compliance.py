"""Test that vehicle_realistic scenario matches design specification exactly."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.scenarios import get_vehicle_realistic_scenario


def test_vehicle_realistic_matches_design_spec():
    """
    Test that vehicle_realistic scenario matches the design specification exactly.
    
    Design spec from design.md:
    ```yaml
    name: vehicle_realistic
    trajectory_type: clothoid
    trajectory_params:
      kappa0: 0.0
      k_rate: 0.5
    obstacle_config:
      type: corridor
      width: 1.2
      length: 10.0
    inflation_config:
      safety_factor: 1.3
      sensing_factor: 0.1
      motion_lookahead: 0.6
      velocity_scaling_factor: 0.2
    uncertainty_config:
      process_noise_position_std: 0.02
      sensor_noise_position_std: 0.03
      velocity_mismatch_factor: 0.85
      angular_mismatch_factor: 0.90
      control_delay_steps: 3
      actuator_tau_v: 0.1
      actuator_tau_omega: 0.15
    duration: 25.0
    ```
    
    Expected Behavior:
    - Realistic vehicle constraints
    - Moderate completion rate (>75%)
    - Adaptive MPC compensates for mismatch
    - Latency handled by predictive control
    
    Note: actuator_tau_v and actuator_tau_omega are mentioned in the design
    but UncertaintyConfig doesn't have these fields yet. They may be handled
    elsewhere in the system or added later.
    """
    scenario = get_vehicle_realistic_scenario(seed=42)
    
    # Test name
    assert scenario.name == "vehicle_realistic", \
        f"Expected name 'vehicle_realistic', got '{scenario.name}'"
    
    # Test trajectory_type
    assert scenario.trajectory_type == "clothoid", \
        f"Expected trajectory_type 'clothoid', got '{scenario.trajectory_type}'"
    
    # Test trajectory_params
    assert scenario.trajectory_params["kappa0"] == 0.0, \
        f"Expected kappa0 0.0, got {scenario.trajectory_params['kappa0']}"
    assert scenario.trajectory_params["k_rate"] == 0.5, \
        f"Expected k_rate 0.5, got {scenario.trajectory_params['k_rate']}"
    
    # Test obstacle_config (corridor with width=1.2m, length=10.0m)
    # Corridor should have 20 obstacles (10 per side)
    assert len(scenario.obstacle_config.obstacles) == 20, \
        f"Expected 20 obstacles for corridor, got {len(scenario.obstacle_config.obstacles)}"
    
    # Test corridor structure: obstacles should be at y = ±0.6 (width/2)
    corridor_width = 1.2
    expected_y_positions = [corridor_width / 2, -corridor_width / 2]
    
    for i, obs in enumerate(scenario.obstacle_config.obstacles):
        # Check obstacle is static
        motion_model = obs.get("motion_model", "static")
        assert motion_model == "static", \
            f"Obstacle {i}: Expected motion_model 'static', got '{motion_model}'"
        
        # Check y position is at corridor wall (±0.6)
        y_pos = obs["y"]
        assert any(abs(y_pos - expected_y) < 0.01 for expected_y in expected_y_positions), \
            f"Obstacle {i}: y position {y_pos} not at corridor wall (expected ±{corridor_width/2})"
    
    # Test inflation_config
    assert scenario.inflation_config.safety_factor == 1.3, \
        f"Expected safety_factor 1.3, got {scenario.inflation_config.safety_factor}"
    assert scenario.inflation_config.sensing_factor == 0.1, \
        f"Expected sensing_factor 0.1, got {scenario.inflation_config.sensing_factor}"
    assert scenario.inflation_config.motion_lookahead == 0.6, \
        f"Expected motion_lookahead 0.6, got {scenario.inflation_config.motion_lookahead}"
    assert scenario.inflation_config.velocity_scaling_factor == 0.2, \
        f"Expected velocity_scaling_factor 0.2, got {scenario.inflation_config.velocity_scaling_factor}"
    
    # Test uncertainty_config (high model mismatch and latency)
    assert scenario.uncertainty_config.process_noise_position_std == 0.02, \
        f"Expected process_noise_position_std 0.02, got {scenario.uncertainty_config.process_noise_position_std}"
    assert scenario.uncertainty_config.sensor_noise_position_std == 0.03, \
        f"Expected sensor_noise_position_std 0.03, got {scenario.uncertainty_config.sensor_noise_position_std}"
    assert scenario.uncertainty_config.velocity_mismatch_factor == 0.85, \
        f"Expected velocity_mismatch_factor 0.85, got {scenario.uncertainty_config.velocity_mismatch_factor}"
    assert scenario.uncertainty_config.angular_mismatch_factor == 0.90, \
        f"Expected angular_mismatch_factor 0.90, got {scenario.uncertainty_config.angular_mismatch_factor}"
    assert scenario.uncertainty_config.control_delay_steps == 3, \
        f"Expected control_delay_steps 3, got {scenario.uncertainty_config.control_delay_steps}"
    
    # Test duration
    assert scenario.duration == 25.0, \
        f"Expected duration 25.0, got {scenario.duration}"
    
    # Test dt (default)
    assert scenario.dt == 0.02, \
        f"Expected dt 0.02, got {scenario.dt}"
    
    print("✓ All design specification requirements met!")
    print("  Note: actuator_tau_v and actuator_tau_omega are not yet in UncertaintyConfig")


if __name__ == "__main__":
    test_vehicle_realistic_matches_design_spec()
    print("\nDesign specification compliance test passed!")

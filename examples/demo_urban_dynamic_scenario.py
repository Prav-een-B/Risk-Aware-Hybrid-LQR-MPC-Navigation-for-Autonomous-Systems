#!/usr/bin/env python3
"""
Demonstration of the urban_dynamic scenario.

This script shows how to:
1. Load the urban_dynamic scenario configuration
2. Create an obstacle field from the scenario
3. Visualize the scenario parameters
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.scenarios import get_urban_dynamic_scenario


def main():
    """Demonstrate urban_dynamic scenario usage."""
    print("=" * 70)
    print("Urban Dynamic Scenario Demonstration")
    print("=" * 70)
    print()
    
    # Load the scenario
    scenario = get_urban_dynamic_scenario(seed=42)
    
    # Display scenario configuration
    print(f"Scenario Name: {scenario.name}")
    print(f"Trajectory Type: {scenario.trajectory_type}")
    print(f"Duration: {scenario.duration}s")
    print(f"Timestep: {scenario.dt}s")
    print()
    
    # Display trajectory parameters
    print("Trajectory Parameters:")
    for key, value in scenario.trajectory_params.items():
        print(f"  {key}: {value}")
    print()
    
    # Display obstacle configuration
    print(f"Obstacles: {len(scenario.obstacle_config.obstacles)} moving obstacles")
    print("Obstacle Details:")
    for i, obs in enumerate(scenario.obstacle_config.obstacles):
        speed = (obs["vx"]**2 + obs["vy"]**2)**0.5
        print(f"  Obstacle {i+1}:")
        print(f"    Position: ({obs['x']:.2f}, {obs['y']:.2f})")
        print(f"    Radius: {obs['radius']:.2f}m")
        print(f"    Velocity: ({obs['vx']:.2f}, {obs['vy']:.2f}) m/s")
        print(f"    Speed: {speed:.2f} m/s")
        print(f"    Motion: {obs['motion_model']}")
    print()
    
    # Display inflation configuration
    print("Inflation Configuration:")
    print(f"  Safety Factor: {scenario.inflation_config.safety_factor}")
    print(f"  Sensing Factor: {scenario.inflation_config.sensing_factor}m")
    print(f"  Motion Lookahead: {scenario.inflation_config.motion_lookahead}s")
    print(f"  Velocity Scaling: {scenario.inflation_config.velocity_scaling_factor}")
    print(f"  Sensing Range: {scenario.inflation_config.sensing_range}m")
    print()
    
    # Display uncertainty configuration
    print("Uncertainty Configuration:")
    print(f"  Process Noise (position): {scenario.uncertainty_config.process_noise_position_std}m")
    print(f"  Sensor Noise (position): {scenario.uncertainty_config.sensor_noise_position_std}m")
    print(f"  Velocity Mismatch: {scenario.uncertainty_config.velocity_mismatch_factor}")
    print(f"  Control Delay: {scenario.uncertainty_config.control_delay_steps} steps")
    print()
    
    # Display expected behavior
    print("Expected Behavior:")
    print("  - Checkpoint completion rate: >80%")
    print("  - Mean tracking error: <0.2m")
    print("  - Collisions: <2")
    print()
    
    # Create obstacle field
    print("Creating obstacle field...")
    obstacle_field = scenario.obstacle_config.create_field(
        inflation=scenario.inflation_config,
        seed=scenario.obstacle_config.seed
    )
    print(f"✓ Obstacle field created with {len(obstacle_field.obstacles)} obstacles")
    print()
    
    # Simulate a few steps
    print("Simulating obstacle motion (5 steps)...")
    for step in range(5):
        obstacle_field.step(dt=scenario.dt)
        print(f"  Step {step+1}: Obstacles moved")
    print()
    
    print("=" * 70)
    print("Demonstration complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

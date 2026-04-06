#!/usr/bin/env python3
"""
Demonstration of the stochastic_navigation scenario.

This script shows how to:
1. Load the stochastic_navigation scenario configuration
2. Create an obstacle field from the scenario
3. Visualize the scenario parameters
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.scenarios import get_stochastic_navigation_scenario


def main():
    """Demonstrate stochastic_navigation scenario usage."""
    print("=" * 70)
    print("Stochastic Navigation Scenario Demonstration")
    print("=" * 70)
    print()
    
    # Load the scenario
    scenario = get_stochastic_navigation_scenario(seed=42)
    
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
    print(f"Obstacles: {len(scenario.obstacle_config.obstacles)} random-walk obstacles")
    print("Obstacle Details:")
    for i, obs in enumerate(scenario.obstacle_config.obstacles):
        speed = (obs["vx"]**2 + obs["vy"]**2)**0.5
        print(f"  Obstacle {i+1}:")
        print(f"    Position: ({obs['x']:.2f}, {obs['y']:.2f})")
        print(f"    Radius: {obs['radius']:.2f}m")
        print(f"    Initial Velocity: ({obs['vx']:.2f}, {obs['vy']:.2f}) m/s")
        print(f"    Initial Speed: {speed:.2f} m/s")
        print(f"    Motion: {obs['motion_model']}")
        print(f"    Random Walk Std: {obs['random_walk_std']}")
        print(f"    Max Speed: {obs['max_speed']} m/s")
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
    print("Uncertainty Configuration (HIGH UNCERTAINTY):")
    print(f"  Process Noise (position): {scenario.uncertainty_config.process_noise_position_std}m")
    print(f"  Process Noise (heading): {scenario.uncertainty_config.process_noise_heading_std}rad")
    print(f"  Sensor Noise (position): {scenario.uncertainty_config.sensor_noise_position_std}m")
    print(f"  Sensor Noise (heading): {scenario.uncertainty_config.sensor_noise_heading_std}rad")
    print(f"  Velocity Mismatch: {scenario.uncertainty_config.velocity_mismatch_factor}")
    print(f"  Control Delay: {scenario.uncertainty_config.control_delay_steps} steps")
    print()
    
    # Display expected behavior
    print("Expected Behavior:")
    print("  - Checkpoint completion rate: >70%")
    print("  - Mean tracking error: <0.3m")
    print("  - Collisions: <5")
    print("  - Hybrid blending adapts to high uncertainty")
    print()
    
    # Create obstacle field
    print("Creating obstacle field...")
    obstacle_field = scenario.obstacle_config.create_field(
        inflation=scenario.inflation_config,
        seed=scenario.obstacle_config.seed
    )
    print(f"✓ Obstacle field created with {len(obstacle_field.obstacles)} obstacles")
    print()
    
    # Simulate a few steps to show random walk behavior
    print("Simulating obstacle motion with random walk (5 steps)...")
    for step in range(5):
        # Record velocities before step
        velocities_before = [(obs.vx, obs.vy) for obs in obstacle_field.obstacles]
        
        obstacle_field.step(dt=scenario.dt)
        
        # Check velocity changes (random walk diffusion)
        velocity_changes = []
        for i, obs in enumerate(obstacle_field.obstacles):
            vx_before, vy_before = velocities_before[i]
            dvx = obs.vx - vx_before
            dvy = obs.vy - vy_before
            velocity_changes.append((dvx, dvy))
        
        print(f"  Step {step+1}: Obstacles moved with random walk velocity updates")
        for i, (dvx, dvy) in enumerate(velocity_changes):
            if abs(dvx) > 0.001 or abs(dvy) > 0.001:
                print(f"    Obstacle {i+1} velocity changed by ({dvx:.4f}, {dvy:.4f}) m/s")
    print()
    
    print("=" * 70)
    print("Demonstration complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

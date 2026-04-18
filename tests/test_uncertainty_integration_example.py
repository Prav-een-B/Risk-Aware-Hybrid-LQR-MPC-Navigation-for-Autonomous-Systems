"""
Integration example demonstrating how uncertainty mechanisms are used in simulation.

This is not a formal test but a demonstration of the API usage.
"""

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src", "hybrid_controller")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from evaluation.scenarios import UncertaintyConfig, UncertaintyInjector


def simulate_with_uncertainty():
    """
    Example simulation loop showing how uncertainty injection works.
    
    This demonstrates the typical usage pattern:
    1. Create UncertaintyConfig with desired noise/mismatch parameters
    2. Create UncertaintyInjector with config and seed
    3. In simulation loop:
       - Inject process noise after dynamics update
       - Inject sensor noise before passing to controller
       - Apply model mismatch to commanded control
       - Buffer control for latency
    """
    # Configure uncertainty
    config = UncertaintyConfig(
        process_noise_position_std=0.02,
        process_noise_heading_std=0.03,
        sensor_noise_position_std=0.015,
        sensor_noise_heading_std=0.025,
        velocity_mismatch_factor=0.95,
        angular_mismatch_factor=0.90,
        control_delay_steps=2,
    )
    
    injector = UncertaintyInjector(config, seed=42)
    
    # Initial state
    true_state = np.array([0.0, 0.0, 0.0])  # [x, y, theta]
    dt = 0.02
    
    print("Uncertainty Injection Simulation Example")
    print("=" * 50)
    print(f"Config: {config}")
    print()
    
    # Simulate a few timesteps
    for step in range(5):
        print(f"Step {step}:")
        
        # 1. Get sensor measurement (with sensor noise)
        measured_state = injector.inject_sensor_noise(true_state)
        print(f"  True state:     {true_state}")
        print(f"  Measured state: {measured_state}")
        
        # 2. Controller computes control based on noisy measurement
        # (In real simulation, this would be LQR/MPC controller)
        commanded_control = np.array([0.5, 0.1])  # [v, omega]
        print(f"  Commanded control: {commanded_control}")
        
        # 3. Apply model mismatch
        mismatched_control = injector.apply_model_mismatch(commanded_control)
        print(f"  Mismatched control: {mismatched_control}")
        
        # 4. Apply control latency
        delayed_control = injector.buffer_control(mismatched_control)
        print(f"  Delayed control: {delayed_control}")
        
        # 5. Apply control to dynamics (simplified unicycle model)
        v, omega = delayed_control
        true_state[0] += v * np.cos(true_state[2]) * dt
        true_state[1] += v * np.sin(true_state[2]) * dt
        true_state[2] += omega * dt
        
        # 6. Inject process noise
        true_state = injector.inject_process_noise(true_state)
        
        print()
    
    print("Simulation complete!")
    print(f"Final true state: {true_state}")


def demonstrate_noise_statistics():
    """
    Demonstrate that noise has correct statistical properties.
    """
    config = UncertaintyConfig(
        process_noise_position_std=0.05,
        sensor_noise_position_std=0.03,
    )
    
    # Generate many samples
    n_samples = 1000
    state = np.array([1.0, 2.0, 0.5])
    
    process_noise_samples = []
    sensor_noise_samples = []
    
    for seed in range(n_samples):
        injector = UncertaintyInjector(config, seed=seed)
        
        # Process noise
        noisy_state = injector.inject_process_noise(state)
        process_noise_samples.append(noisy_state[0] - state[0])
        
        # Sensor noise
        noisy_measurement = injector.inject_sensor_noise(state)
        sensor_noise_samples.append(noisy_measurement[0] - state[0])
    
    process_noise_samples = np.array(process_noise_samples)
    sensor_noise_samples = np.array(sensor_noise_samples)
    
    print("\nNoise Statistics Demonstration")
    print("=" * 50)
    print(f"Process noise (std={config.process_noise_position_std}):")
    print(f"  Mean: {np.mean(process_noise_samples):.6f} (should be ~0)")
    print(f"  Std:  {np.std(process_noise_samples):.6f} (should be ~{config.process_noise_position_std})")
    print()
    print(f"Sensor noise (std={config.sensor_noise_position_std}):")
    print(f"  Mean: {np.mean(sensor_noise_samples):.6f} (should be ~0)")
    print(f"  Std:  {np.std(sensor_noise_samples):.6f} (should be ~{config.sensor_noise_position_std})")


if __name__ == "__main__":
    simulate_with_uncertainty()
    print("\n" + "=" * 50 + "\n")
    demonstrate_noise_statistics()

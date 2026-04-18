"""
Property-based tests for uncertainty injection mechanisms.

Tests validate:
- Property 27: Process Noise Distribution (Requirements 7.5)
- Property 28: Sensor Noise Distribution (Requirements 7.6)
- Property 29: Model Mismatch Scaling (Requirements 7.7)
- Property 30: Control Latency Delay (Requirements 7.8)
"""

import os
import sys

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from scipy import stats

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src", "hybrid_controller")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from evaluation.scenarios import UncertaintyConfig, UncertaintyInjector


# Strategy for generating valid uncertainty configurations
@st.composite
def uncertainty_config_strategy(draw):
    """Generate valid UncertaintyConfig instances."""
    return UncertaintyConfig(
        process_noise_position_std=draw(st.floats(min_value=0.0, max_value=0.1)),
        process_noise_heading_std=draw(st.floats(min_value=0.0, max_value=0.1)),
        sensor_noise_position_std=draw(st.floats(min_value=0.0, max_value=0.1)),
        sensor_noise_heading_std=draw(st.floats(min_value=0.0, max_value=0.1)),
        velocity_mismatch_factor=draw(st.floats(min_value=0.5, max_value=1.5)),
        angular_mismatch_factor=draw(st.floats(min_value=0.5, max_value=1.5)),
        control_delay_steps=draw(st.integers(min_value=0, max_value=10)),
    )


@st.composite
def state_strategy(draw):
    """Generate valid robot states."""
    return np.array([
        draw(st.floats(min_value=-5.0, max_value=5.0)),  # x
        draw(st.floats(min_value=-5.0, max_value=5.0)),  # y
        draw(st.floats(min_value=-np.pi, max_value=np.pi)),  # theta
    ])


@st.composite
def control_strategy(draw):
    """Generate valid control inputs."""
    return np.array([
        draw(st.floats(min_value=-1.0, max_value=1.0)),  # v
        draw(st.floats(min_value=-1.0, max_value=1.0)),  # omega
    ])


class TestProcessNoise:
    """Test process noise injection (Property 27)."""
    
    @given(state=state_strategy())
    @settings(max_examples=50)
    def test_process_noise_zero_when_disabled(self, state):
        """**Property 27a: Process Noise Zero When Disabled**
        
        **Validates: Requirements 7.5**
        
        When process noise is disabled (std=0), the state should remain unchanged.
        """
        config = UncertaintyConfig(
            process_noise_position_std=0.0,
            process_noise_heading_std=0.0,
        )
        injector = UncertaintyInjector(config, seed=42)
        
        noisy_state = injector.inject_process_noise(state)
        
        np.testing.assert_array_equal(noisy_state, state)
    
    def test_process_noise_distribution(self):
        """**Property 27b: Process Noise Distribution**
        
        **Validates: Requirements 7.5**
        
        Process noise samples should have zero mean and follow Gaussian distribution
        with specified standard deviation.
        """
        position_std = 0.05
        heading_std = 0.03
        config = UncertaintyConfig(
            process_noise_position_std=position_std,
            process_noise_heading_std=heading_std,
        )
        
        # Generate many samples
        n_samples = 1000
        state = np.array([0.0, 0.0, 0.0])
        
        position_noise_x = []
        position_noise_y = []
        heading_noise = []
        
        for seed in range(n_samples):
            injector = UncertaintyInjector(config, seed=seed)
            noisy_state = injector.inject_process_noise(state)
            
            position_noise_x.append(noisy_state[0] - state[0])
            position_noise_y.append(noisy_state[1] - state[1])
            heading_noise.append(noisy_state[2] - state[2])
        
        position_noise_x = np.array(position_noise_x)
        position_noise_y = np.array(position_noise_y)
        heading_noise = np.array(heading_noise)
        
        # Test zero mean (within tolerance)
        assert abs(np.mean(position_noise_x)) < 0.01, "Position noise X should have zero mean"
        assert abs(np.mean(position_noise_y)) < 0.01, "Position noise Y should have zero mean"
        assert abs(np.mean(heading_noise)) < 0.01, "Heading noise should have zero mean"
        
        # Test standard deviation (within 10% tolerance)
        assert abs(np.std(position_noise_x) - position_std) < 0.01, \
            f"Position noise X std should be {position_std}"
        assert abs(np.std(position_noise_y) - position_std) < 0.01, \
            f"Position noise Y std should be {position_std}"
        assert abs(np.std(heading_noise) - heading_std) < 0.01, \
            f"Heading noise std should be {heading_std}"
        
        # Test normality using Shapiro-Wilk test (p > 0.05 indicates normal distribution)
        _, p_x = stats.shapiro(position_noise_x)
        _, p_y = stats.shapiro(position_noise_y)
        _, p_theta = stats.shapiro(heading_noise)
        
        assert p_x > 0.01, "Position noise X should follow normal distribution"
        assert p_y > 0.01, "Position noise Y should follow normal distribution"
        assert p_theta > 0.01, "Heading noise should follow normal distribution"


class TestSensorNoise:
    """Test sensor noise injection (Property 28)."""
    
    @given(measurement=state_strategy())
    @settings(max_examples=50)
    def test_sensor_noise_zero_when_disabled(self, measurement):
        """**Property 28a: Sensor Noise Zero When Disabled**
        
        **Validates: Requirements 7.6**
        
        When sensor noise is disabled (std=0), the measurement should remain unchanged.
        """
        config = UncertaintyConfig(
            sensor_noise_position_std=0.0,
            sensor_noise_heading_std=0.0,
        )
        injector = UncertaintyInjector(config, seed=42)
        
        noisy_measurement = injector.inject_sensor_noise(measurement)
        
        np.testing.assert_array_equal(noisy_measurement, measurement)
    
    def test_sensor_noise_distribution(self):
        """**Property 28b: Sensor Noise Distribution**
        
        **Validates: Requirements 7.6**
        
        Sensor noise samples should have zero mean and follow Gaussian distribution
        with specified standard deviation.
        """
        position_std = 0.04
        heading_std = 0.02
        config = UncertaintyConfig(
            sensor_noise_position_std=position_std,
            sensor_noise_heading_std=heading_std,
        )
        
        # Generate many samples
        n_samples = 1000
        measurement = np.array([1.0, 2.0, 0.5])
        
        position_noise_x = []
        position_noise_y = []
        heading_noise = []
        
        for seed in range(n_samples):
            injector = UncertaintyInjector(config, seed=seed)
            noisy_measurement = injector.inject_sensor_noise(measurement)
            
            position_noise_x.append(noisy_measurement[0] - measurement[0])
            position_noise_y.append(noisy_measurement[1] - measurement[1])
            heading_noise.append(noisy_measurement[2] - measurement[2])
        
        position_noise_x = np.array(position_noise_x)
        position_noise_y = np.array(position_noise_y)
        heading_noise = np.array(heading_noise)
        
        # Test zero mean (within tolerance)
        assert abs(np.mean(position_noise_x)) < 0.01, "Sensor noise X should have zero mean"
        assert abs(np.mean(position_noise_y)) < 0.01, "Sensor noise Y should have zero mean"
        assert abs(np.mean(heading_noise)) < 0.01, "Heading noise should have zero mean"
        
        # Test standard deviation (within tolerance)
        assert abs(np.std(position_noise_x) - position_std) < 0.01, \
            f"Sensor noise X std should be {position_std}"
        assert abs(np.std(position_noise_y) - position_std) < 0.01, \
            f"Sensor noise Y std should be {position_std}"
        assert abs(np.std(heading_noise) - heading_std) < 0.01, \
            f"Heading noise std should be {heading_std}"
        
        # Test normality using Shapiro-Wilk test
        _, p_x = stats.shapiro(position_noise_x)
        _, p_y = stats.shapiro(position_noise_y)
        _, p_theta = stats.shapiro(heading_noise)
        
        assert p_x > 0.01, "Sensor noise X should follow normal distribution"
        assert p_y > 0.01, "Sensor noise Y should follow normal distribution"
        assert p_theta > 0.01, "Heading noise should follow normal distribution"


class TestModelMismatch:
    """Test model mismatch scaling (Property 29)."""
    
    @given(control=control_strategy())
    @settings(max_examples=50)
    def test_model_mismatch_identity_when_disabled(self, control):
        """**Property 29a: Model Mismatch Identity When Disabled**
        
        **Validates: Requirements 7.7**
        
        When model mismatch is disabled (factors=1.0), control should remain unchanged.
        """
        config = UncertaintyConfig(
            velocity_mismatch_factor=1.0,
            angular_mismatch_factor=1.0,
        )
        injector = UncertaintyInjector(config, seed=42)
        
        mismatched_control = injector.apply_model_mismatch(control)
        
        np.testing.assert_array_almost_equal(mismatched_control, control)
    
    @given(
        control=control_strategy(),
        velocity_factor=st.floats(min_value=0.5, max_value=1.5),
        angular_factor=st.floats(min_value=0.5, max_value=1.5),
    )
    @settings(max_examples=100)
    def test_model_mismatch_scaling(self, control, velocity_factor, angular_factor):
        """**Property 29b: Model Mismatch Scaling**
        
        **Validates: Requirements 7.7**
        
        Actual applied velocity should equal commanded velocity multiplied by mismatch factor.
        """
        config = UncertaintyConfig(
            velocity_mismatch_factor=velocity_factor,
            angular_mismatch_factor=angular_factor,
        )
        injector = UncertaintyInjector(config, seed=42)
        
        mismatched_control = injector.apply_model_mismatch(control)
        
        # Check linear velocity scaling
        expected_v = control[0] * velocity_factor
        assert abs(mismatched_control[0] - expected_v) < 1e-10, \
            f"Linear velocity should be scaled by {velocity_factor}"
        
        # Check angular velocity scaling
        expected_omega = control[1] * angular_factor
        assert abs(mismatched_control[1] - expected_omega) < 1e-10, \
            f"Angular velocity should be scaled by {angular_factor}"


class TestControlLatency:
    """Test control latency delay (Property 30)."""
    
    @given(control=control_strategy())
    @settings(max_examples=50)
    def test_control_latency_zero_when_disabled(self, control):
        """**Property 30a: Control Latency Zero When Disabled**
        
        **Validates: Requirements 7.8**
        
        When control latency is disabled (delay=0), control should be returned immediately.
        """
        config = UncertaintyConfig(control_delay_steps=0)
        injector = UncertaintyInjector(config, seed=42)
        
        delayed_control = injector.buffer_control(control)
        
        np.testing.assert_array_equal(delayed_control, control)
    
    @given(delay_steps=st.integers(min_value=1, max_value=5))
    @settings(max_examples=20)
    def test_control_latency_delay(self, delay_steps):
        """**Property 30b: Control Latency Delay**
        
        **Validates: Requirements 7.8**
        
        Control should be applied exactly delay_steps timesteps after it is computed.
        """
        config = UncertaintyConfig(control_delay_steps=delay_steps)
        injector = UncertaintyInjector(config, seed=42)
        
        # Create sequence of distinct controls
        controls = [np.array([float(i), float(i * 2)]) for i in range(delay_steps + 3)]
        
        # Buffer controls and track outputs
        outputs = []
        for control in controls:
            delayed = injector.buffer_control(control)
            outputs.append(delayed)
        
        # First delay_steps outputs should be zero (buffer filling)
        for i in range(delay_steps):
            np.testing.assert_array_equal(
                outputs[i], 
                np.zeros(2),
                err_msg=f"Output {i} should be zero while buffer fills"
            )
        
        # After delay_steps, outputs should match inputs with delay
        for i in range(delay_steps, len(controls)):
            expected_control = controls[i - delay_steps]
            np.testing.assert_array_equal(
                outputs[i],
                expected_control,
                err_msg=f"Output {i} should match input {i - delay_steps}"
            )
    
    def test_control_latency_buffer_reset(self):
        """**Property 30c: Control Latency Buffer Reset**
        
        **Validates: Requirements 7.8**
        
        Buffer reset should clear all buffered controls.
        """
        config = UncertaintyConfig(control_delay_steps=3)
        injector = UncertaintyInjector(config, seed=42)
        
        # Buffer some controls
        for i in range(5):
            injector.buffer_control(np.array([float(i), float(i)]))
        
        # Reset buffer
        injector.reset_buffer()
        
        # Next control should return zero (buffer empty)
        control = np.array([10.0, 20.0])
        delayed = injector.buffer_control(control)
        np.testing.assert_array_equal(delayed, np.zeros(2))


class TestUncertaintyIntegration:
    """Integration tests for combined uncertainty mechanisms."""
    
    def test_all_mechanisms_can_be_enabled_simultaneously(self):
        """Test that all uncertainty mechanisms can work together."""
        config = UncertaintyConfig(
            process_noise_position_std=0.01,
            process_noise_heading_std=0.02,
            sensor_noise_position_std=0.015,
            sensor_noise_heading_std=0.025,
            velocity_mismatch_factor=0.95,
            angular_mismatch_factor=0.90,
            control_delay_steps=2,
        )
        injector = UncertaintyInjector(config, seed=42)
        
        # Test process noise
        state = np.array([1.0, 2.0, 0.5])
        noisy_state = injector.inject_process_noise(state)
        assert not np.array_equal(noisy_state, state)
        
        # Test sensor noise
        measurement = np.array([1.0, 2.0, 0.5])
        noisy_measurement = injector.inject_sensor_noise(measurement)
        assert not np.array_equal(noisy_measurement, measurement)
        
        # Test model mismatch
        control = np.array([0.5, 0.3])
        mismatched = injector.apply_model_mismatch(control)
        assert abs(mismatched[0] - 0.5 * 0.95) < 1e-10
        assert abs(mismatched[1] - 0.3 * 0.90) < 1e-10
        
        # Test control latency
        delayed = injector.buffer_control(control)
        np.testing.assert_array_equal(delayed, np.zeros(2))  # Buffer filling
    
    def test_uncertainty_reproducibility_with_seed(self):
        """Test that uncertainty injection is reproducible with same seed."""
        config = UncertaintyConfig(
            process_noise_position_std=0.05,
            sensor_noise_position_std=0.03,
        )
        
        state = np.array([1.0, 2.0, 0.5])
        
        # Generate with same seed
        injector1 = UncertaintyInjector(config, seed=42)
        result1 = injector1.inject_process_noise(state)
        
        injector2 = UncertaintyInjector(config, seed=42)
        result2 = injector2.inject_process_noise(state)
        
        np.testing.assert_array_equal(result1, result2)
        
        # Generate with different seed
        injector3 = UncertaintyInjector(config, seed=123)
        result3 = injector3.inject_process_noise(state)
        
        assert not np.array_equal(result1, result3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Unit tests for CheckpointManager class."""

import os
import sys
import pytest
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src", "hybrid_controller")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from hybrid_controller.navigation.checkpoint_manager import CheckpointManager, Checkpoint


class TestCheckpoint:
    """Test Checkpoint dataclass."""
    
    def test_checkpoint_creation(self):
        """Test checkpoint initialization with required fields."""
        cp = Checkpoint(
            x=1.0, y=2.0, theta=0.5,
            curvature=0.3, index=0
        )
        assert cp.x == 1.0
        assert cp.y == 2.0
        assert cp.theta == 0.5
        assert cp.curvature == 0.3
        assert cp.index == 0
        assert cp.reached is False
        assert cp.time_reached == 0.0
        assert cp.overshoot == 0.0
        assert cp.min_distance == float('inf')
    
    def test_distance_to(self):
        """Test distance computation to a position."""
        cp = Checkpoint(x=0.0, y=0.0, theta=0.0, curvature=0.0, index=0)
        
        # Test distance to origin
        position = np.array([0.0, 0.0])
        assert cp.distance_to(position) == 0.0
        
        # Test distance to (3, 4) - should be 5
        position = np.array([3.0, 4.0])
        assert np.isclose(cp.distance_to(position), 5.0)
        
        # Test distance to (-1, 0)
        position = np.array([-1.0, 0.0])
        assert np.isclose(cp.distance_to(position), 1.0)


class TestCheckpointManagerInit:
    """Test CheckpointManager initialization."""
    
    def test_default_initialization(self):
        """Test initialization with default parameters."""
        manager = CheckpointManager()
        
        assert manager.base_radius == 0.3
        assert manager.curvature_scaling == 0.2
        assert manager.hysteresis_margin == 0.1
        assert manager.forward_progress_timeout == 1.0
        assert manager.dt == 0.02
        
        assert manager.checkpoints == []
        assert manager.current_idx == 0
        assert manager.switching_radius == 0.3
        assert manager.hysteresis_active is False
        assert manager.last_distance == float('inf')
        assert manager.no_progress_time == 0.0
        
        assert manager.checkpoints_reached == 0
        assert manager.checkpoints_missed == 0
        assert manager.total_overshoot == 0.0
        assert manager.checkpoint_times == []
    
    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        manager = CheckpointManager(
            base_switching_radius=0.5,
            curvature_scaling=0.3,
            hysteresis_margin=0.15,
            forward_progress_timeout=2.0,
            dt=0.01
        )
        
        assert manager.base_radius == 0.5
        assert manager.curvature_scaling == 0.3
        assert manager.hysteresis_margin == 0.15
        assert manager.forward_progress_timeout == 2.0
        assert manager.dt == 0.01


class TestCheckpointManagerMethods:
    """Test CheckpointManager core methods."""
    
    def test_set_checkpoints(self):
        """Test setting checkpoint queue."""
        manager = CheckpointManager()
        checkpoints = [
            Checkpoint(x=0.0, y=0.0, theta=0.0, curvature=0.0, index=0),
            Checkpoint(x=1.0, y=0.0, theta=0.0, curvature=0.5, index=1),
            Checkpoint(x=2.0, y=0.0, theta=0.0, curvature=0.0, index=2),
        ]
        
        manager.set_checkpoints(checkpoints)
        
        assert len(manager.checkpoints) == 3
        assert manager.current_idx == 0
        assert manager.checkpoints[0].x == 0.0
        assert manager.checkpoints[1].x == 1.0
        assert manager.checkpoints[2].x == 2.0
    
    def test_reset(self):
        """Test reset functionality."""
        manager = CheckpointManager()
        checkpoints = [
            Checkpoint(x=0.0, y=0.0, theta=0.0, curvature=0.0, index=0),
            Checkpoint(x=1.0, y=0.0, theta=0.0, curvature=0.5, index=1),
        ]
        manager.set_checkpoints(checkpoints)
        
        # Modify state
        manager.current_idx = 1
        manager.switching_radius = 0.5
        manager.hysteresis_active = True
        manager.last_distance = 0.5
        manager.no_progress_time = 0.5
        manager.checkpoints_reached = 1
        manager.checkpoints_missed = 1
        manager.total_overshoot = 0.1
        manager.checkpoint_times = [0.5]
        manager.checkpoints[0].reached = True
        
        # Reset
        manager.reset()
        
        assert manager.current_idx == 0
        assert manager.switching_radius == manager.base_radius
        assert manager.hysteresis_active is False
        assert manager.last_distance == float('inf')
        assert manager.no_progress_time == 0.0
        assert manager.checkpoints_reached == 0
        assert manager.checkpoints_missed == 0
        assert manager.total_overshoot == 0.0
        assert manager.checkpoint_times == []
        assert manager.checkpoints[0].reached is False
    
    def test_get_current_checkpoint(self):
        """Test getting current checkpoint."""
        manager = CheckpointManager()
        checkpoints = [
            Checkpoint(x=0.0, y=0.0, theta=0.0, curvature=0.0, index=0),
            Checkpoint(x=1.0, y=0.0, theta=0.0, curvature=0.5, index=1),
        ]
        manager.set_checkpoints(checkpoints)
        
        # Get first checkpoint
        cp = manager.get_current_checkpoint()
        assert cp is not None
        assert cp.x == 0.0
        
        # Advance to second checkpoint
        manager.current_idx = 1
        cp = manager.get_current_checkpoint()
        assert cp is not None
        assert cp.x == 1.0
        
        # Beyond last checkpoint
        manager.current_idx = 2
        cp = manager.get_current_checkpoint()
        assert cp is None


class TestCheckpointSwitching:
    """Test checkpoint switching logic."""
    
    def test_basic_switching(self):
        """Test basic checkpoint switching when within radius."""
        manager = CheckpointManager(base_switching_radius=0.5)
        checkpoints = [
            Checkpoint(x=0.0, y=0.0, theta=0.0, curvature=0.0, index=0),
            Checkpoint(x=2.0, y=0.0, theta=0.0, curvature=0.0, index=1),
        ]
        manager.set_checkpoints(checkpoints)
        
        # Robot far from checkpoint - no switch
        robot_pos = np.array([1.0, 0.0])
        switched = manager.update(robot_pos, 0.0)
        assert switched is False
        assert manager.current_idx == 0
        
        # Robot within switching radius - should switch
        robot_pos = np.array([0.0, 0.0])
        switched = manager.update(robot_pos, 0.1)
        assert switched is True
        assert manager.current_idx == 1
        assert manager.checkpoints_reached == 1
    
    def test_curvature_adaptive_radius(self):
        """Test switching radius adapts to curvature."""
        manager = CheckpointManager(
            base_switching_radius=0.5,
            curvature_scaling=0.2
        )
        
        # High curvature checkpoint
        checkpoints = [
            Checkpoint(x=0.0, y=0.0, theta=0.0, curvature=2.0, index=0),
        ]
        manager.set_checkpoints(checkpoints)
        
        robot_pos = np.array([0.5, 0.0])
        manager.update(robot_pos, 0.0)
        
        # Radius should be reduced: 0.5 - 0.2*2.0 = 0.1
        assert np.isclose(manager.switching_radius, 0.1)
    
    def test_minimum_radius_constraint(self):
        """Test switching radius has minimum of 0.1m."""
        manager = CheckpointManager(
            base_switching_radius=0.3,
            curvature_scaling=0.5
        )
        
        # Very high curvature
        checkpoints = [
            Checkpoint(x=0.0, y=0.0, theta=0.0, curvature=10.0, index=0),
        ]
        manager.set_checkpoints(checkpoints)
        
        robot_pos = np.array([1.0, 0.0])
        manager.update(robot_pos, 0.0)
        
        # Radius should be clamped to minimum
        assert manager.switching_radius >= 0.1
    
    def test_hysteresis_activation(self):
        """Test hysteresis activates when moving away."""
        manager = CheckpointManager(
            base_switching_radius=0.5,
            hysteresis_margin=0.1
        )
        checkpoints = [
            Checkpoint(x=0.0, y=0.0, theta=0.0, curvature=0.0, index=0),
        ]
        manager.set_checkpoints(checkpoints)
        
        # First update - approaching
        robot_pos = np.array([1.0, 0.0])
        manager.update(robot_pos, 0.0)
        assert manager.hysteresis_active is False
        
        # Second update - moving away (distance increases)
        robot_pos = np.array([1.5, 0.0])
        manager.update(robot_pos, 0.02)
        assert manager.hysteresis_active is True
        # Radius should increase by hysteresis margin
        assert manager.switching_radius == 0.5 + 0.1
    
    def test_forward_progress_timeout(self):
        """Test checkpoint advances on forward progress timeout."""
        manager = CheckpointManager(
            base_switching_radius=0.5,
            forward_progress_timeout=0.1,
            dt=0.02
        )
        checkpoints = [
            Checkpoint(x=0.0, y=0.0, theta=0.0, curvature=0.0, index=0),
            Checkpoint(x=2.0, y=0.0, theta=0.0, curvature=0.0, index=1),
        ]
        manager.set_checkpoints(checkpoints)
        
        # Robot moving away from checkpoint repeatedly
        robot_pos = np.array([1.0, 0.0])
        for i in range(10):
            robot_pos[0] += 0.1  # Keep moving away
            switched = manager.update(robot_pos, i * 0.02)
            if switched:
                break
        
        # Should have switched due to timeout
        assert manager.current_idx == 1
        assert manager.checkpoints_missed == 1


class TestMetrics:
    """Test checkpoint tracking metrics."""
    
    def test_get_metrics_empty(self):
        """Test metrics with no checkpoints."""
        manager = CheckpointManager()
        metrics = manager.get_metrics()
        
        assert metrics['checkpoints_total'] == 0
        assert metrics['checkpoints_reached'] == 0
        assert metrics['checkpoints_missed'] == 0
        assert metrics['completion_rate'] == 0.0
        assert metrics['mean_time_to_checkpoint'] == 0.0
    
    def test_get_metrics_with_checkpoints(self):
        """Test metrics after reaching checkpoints."""
        manager = CheckpointManager(base_switching_radius=0.5)
        checkpoints = [
            Checkpoint(x=0.0, y=0.0, theta=0.0, curvature=0.0, index=0),
            Checkpoint(x=1.0, y=0.0, theta=0.0, curvature=0.0, index=1),
            Checkpoint(x=2.0, y=0.0, theta=0.0, curvature=0.0, index=2),
        ]
        manager.set_checkpoints(checkpoints)
        
        # Reach first checkpoint
        manager.update(np.array([0.0, 0.0]), 1.0)
        
        # Reach second checkpoint
        manager.update(np.array([1.0, 0.0]), 2.0)
        
        metrics = manager.get_metrics()
        
        assert metrics['checkpoints_total'] == 3
        assert metrics['checkpoints_reached'] == 2
        assert metrics['completion_rate'] == 2.0 / 3.0
        assert len(manager.checkpoint_times) == 1
        assert manager.checkpoint_times[0] == 1.0  # Time between checkpoints


class TestLocalTrajectorySegment:
    """Test local trajectory segment extraction for MPC."""
    
    def test_basic_segment_extraction(self):
        """Test extracting reference horizon."""
        manager = CheckpointManager(dt=0.1)
        checkpoints = [
            Checkpoint(x=0.0, y=0.0, theta=0.0, curvature=0.0, index=0),
            Checkpoint(x=1.0, y=0.0, theta=0.0, curvature=0.0, index=1),
            Checkpoint(x=2.0, y=0.0, theta=0.0, curvature=0.0, index=2),
        ]
        manager.set_checkpoints(checkpoints)
        
        robot_state = np.array([0.0, 0.0, 0.0])
        horizon = 3
        
        x_refs, u_refs = manager.get_local_trajectory_segment(robot_state, horizon)
        
        # Check shapes
        assert x_refs.shape == (3, 3)
        assert u_refs.shape == (2, 2)
        
        # Check reference states
        assert np.allclose(x_refs[0], [0.0, 0.0, 0.0])
        assert np.allclose(x_refs[1], [1.0, 0.0, 0.0])
        assert np.allclose(x_refs[2], [2.0, 0.0, 0.0])
    
    def test_segment_beyond_checkpoints(self):
        """Test segment extraction when horizon exceeds checkpoint count."""
        manager = CheckpointManager(dt=0.1)
        checkpoints = [
            Checkpoint(x=0.0, y=0.0, theta=0.0, curvature=0.0, index=0),
            Checkpoint(x=1.0, y=0.0, theta=0.0, curvature=0.0, index=1),
        ]
        manager.set_checkpoints(checkpoints)
        
        robot_state = np.array([0.0, 0.0, 0.0])
        horizon = 5  # More than available checkpoints
        
        x_refs, u_refs = manager.get_local_trajectory_segment(robot_state, horizon)
        
        # Should clamp to last checkpoint
        assert x_refs.shape == (5, 3)
        assert u_refs.shape == (4, 2)
        
        # Last references should repeat final checkpoint
        assert np.allclose(x_refs[-1], [1.0, 0.0, 0.0])
    
    def test_velocity_computation(self):
        """Test reference velocity computation from checkpoint spacing."""
        manager = CheckpointManager(dt=0.1)
        checkpoints = [
            Checkpoint(x=0.0, y=0.0, theta=0.0, curvature=0.0, index=0),
            Checkpoint(x=0.5, y=0.0, theta=0.0, curvature=0.0, index=1),
        ]
        manager.set_checkpoints(checkpoints)
        
        robot_state = np.array([0.0, 0.0, 0.0])
        horizon = 2
        
        x_refs, u_refs = manager.get_local_trajectory_segment(robot_state, horizon)
        
        # Distance is 0.5m, dt is 0.1s, so v_ref should be 5.0 m/s
        # But capped at 1.0 m/s
        assert u_refs[0, 0] == 1.0


# ============================================================================
# Property-Based Tests using Hypothesis
# ============================================================================

from hypothesis import given, strategies as st, assume, settings
from hypothesis.strategies import composite


@composite
def checkpoint_with_curvature(draw):
    """Generate a checkpoint with valid curvature."""
    x = draw(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False))
    y = draw(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False))
    theta = draw(st.floats(min_value=-np.pi, max_value=np.pi, allow_nan=False, allow_infinity=False))
    curvature = draw(st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False))
    index = draw(st.integers(min_value=0, max_value=100))
    
    return Checkpoint(x=x, y=y, theta=theta, curvature=curvature, index=index)


@composite
def robot_position_near_checkpoint(draw, checkpoint):
    """Generate a robot position near a checkpoint."""
    # Generate position within 2 meters of checkpoint
    offset_x = draw(st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False))
    offset_y = draw(st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False))
    
    return np.array([checkpoint.x + offset_x, checkpoint.y + offset_y])


class TestPropertyCheckpointSwitching:
    """Property 7: Checkpoint Switching - Validates: Requirements 3.2"""
    
    @given(
        base_radius=st.floats(min_value=0.2, max_value=1.0, allow_nan=False, allow_infinity=False),
        curvature=st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        distance_factor=st.floats(min_value=0.0, max_value=0.99, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=None)
    def test_checkpoint_switches_within_radius(self, base_radius, curvature, distance_factor):
        """
        **Validates: Requirements 3.2**
        
        For any checkpoint and robot position, when the robot is within the 
        switching radius, the Checkpoint_Manager SHALL advance to the next checkpoint.
        """
        manager = CheckpointManager(base_switching_radius=base_radius, curvature_scaling=0.2)
        
        # Create two checkpoints
        checkpoints = [
            Checkpoint(x=0.0, y=0.0, theta=0.0, curvature=curvature, index=0),
            Checkpoint(x=5.0, y=0.0, theta=0.0, curvature=0.0, index=1),
        ]
        manager.set_checkpoints(checkpoints)
        
        # Compute expected switching radius
        expected_radius = max(base_radius - 0.2 * curvature, 0.1)
        
        # Place robot within switching radius
        distance = expected_radius * distance_factor
        robot_pos = np.array([distance, 0.0])
        
        # Update should trigger switch
        switched = manager.update(robot_pos, 0.1)
        
        assert switched is True, f"Failed to switch when distance {distance} <= radius {expected_radius}"
        assert manager.current_idx == 1, "Should have advanced to next checkpoint"
        assert manager.checkpoints_reached == 1, "Should have recorded checkpoint as reached"


class TestPropertyCurvatureDependentRadius:
    """Property 8: Curvature-Dependent Radius - Validates: Requirements 3.3"""
    
    @given(
        base_radius=st.floats(min_value=0.3, max_value=1.0, allow_nan=False, allow_infinity=False),
        curvature_scaling=st.floats(min_value=0.1, max_value=0.5, allow_nan=False, allow_infinity=False),
        curvature=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=None)
    def test_radius_computed_from_curvature(self, base_radius, curvature_scaling, curvature):
        """
        **Validates: Requirements 3.3**
        
        For any curvature value, the switching radius SHALL be computed as 
        radius = base_radius - curvature_scaling * curvature, clamped to a minimum value.
        """
        manager = CheckpointManager(
            base_switching_radius=base_radius,
            curvature_scaling=curvature_scaling
        )
        
        checkpoints = [
            Checkpoint(x=0.0, y=0.0, theta=0.0, curvature=curvature, index=0),
        ]
        manager.set_checkpoints(checkpoints)
        
        # Trigger radius computation by updating
        robot_pos = np.array([10.0, 0.0])  # Far away to avoid switching
        manager.update(robot_pos, 0.0)
        
        # Expected radius with minimum constraint
        expected_radius = max(base_radius - curvature_scaling * curvature, 0.1)
        
        assert np.isclose(manager.switching_radius, expected_radius, atol=1e-6), \
            f"Radius {manager.switching_radius} != expected {expected_radius}"


class TestPropertyRadiusMonotonicity:
    """Property 9: Radius Monotonicity with Curvature - Validates: Requirements 3.4"""
    
    @given(
        base_radius=st.floats(min_value=0.3, max_value=1.0, allow_nan=False, allow_infinity=False),
        curvature_scaling=st.floats(min_value=0.1, max_value=0.5, allow_nan=False, allow_infinity=False),
        curvature1=st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        curvature2=st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=None)
    def test_radius_monotonic_with_curvature(self, base_radius, curvature_scaling, curvature1, curvature2):
        """
        **Validates: Requirements 3.4**
        
        For any two checkpoints with curvatures kappa1 > kappa2, the switching radius 
        for checkpoint 1 SHALL be less than or equal to the switching radius for checkpoint 2.
        """
        assume(curvature1 > curvature2)  # Only test when curvature1 > curvature2
        
        manager = CheckpointManager(
            base_switching_radius=base_radius,
            curvature_scaling=curvature_scaling
        )
        
        # Test checkpoint 1 with higher curvature
        checkpoints1 = [Checkpoint(x=0.0, y=0.0, theta=0.0, curvature=curvature1, index=0)]
        manager.set_checkpoints(checkpoints1)
        robot_pos = np.array([10.0, 0.0])
        manager.update(robot_pos, 0.0)
        radius1 = manager.switching_radius
        
        # Test checkpoint 2 with lower curvature
        checkpoints2 = [Checkpoint(x=0.0, y=0.0, theta=0.0, curvature=curvature2, index=0)]
        manager.set_checkpoints(checkpoints2)
        manager.update(robot_pos, 0.0)
        radius2 = manager.switching_radius
        
        assert radius1 <= radius2, \
            f"Higher curvature {curvature1} should have smaller/equal radius {radius1} vs {radius2}"


class TestPropertyHysteresisBehavior:
    """Property 10: Hysteresis Behavior - Validates: Requirements 3.5, 3.6"""
    
    @given(
        base_radius=st.floats(min_value=0.3, max_value=1.0, allow_nan=False, allow_infinity=False),
        hysteresis_margin=st.floats(min_value=0.05, max_value=0.15, allow_nan=False, allow_infinity=False),
        initial_distance=st.floats(min_value=1.5, max_value=3.0, allow_nan=False, allow_infinity=False),
        distance_increase=st.floats(min_value=0.1, max_value=0.5, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=None)
    def test_hysteresis_increases_radius_when_moving_away(self, base_radius, hysteresis_margin, 
                                                          initial_distance, distance_increase):
        """
        **Validates: Requirements 3.5, 3.6**
        
        For any checkpoint, when the robot moves away from it (distance increasing), 
        the switching radius SHALL increase by the hysteresis margin; when moving 
        toward it (distance decreasing), the radius SHALL decrease by the hysteresis margin.
        """
        manager = CheckpointManager(
            base_switching_radius=base_radius,
            hysteresis_margin=hysteresis_margin,
            curvature_scaling=0.0,  # Disable curvature effect for clearer test
            dt=0.02
        )
        
        checkpoints = [
            Checkpoint(x=0.0, y=0.0, theta=0.0, curvature=0.0, index=0),
        ]
        manager.set_checkpoints(checkpoints)
        
        # First update - establish baseline (approaching from far away)
        robot_pos = np.array([initial_distance, 0.0])
        manager.update(robot_pos, 0.0)
        assert manager.hysteresis_active is False, "Hysteresis should not be active initially"
        radius_before_hysteresis = manager.switching_radius
        
        # Second update - move away (increase distance)
        robot_pos = np.array([initial_distance + distance_increase, 0.0])
        manager.update(robot_pos, 0.02)
        
        # Hysteresis should activate and radius should increase by margin
        assert manager.hysteresis_active is True, "Hysteresis should activate when moving away"
        expected_radius_with_hysteresis = radius_before_hysteresis + hysteresis_margin
        assert np.isclose(manager.switching_radius, expected_radius_with_hysteresis, atol=1e-6), \
            f"Radius should increase by hysteresis margin: {manager.switching_radius} vs {expected_radius_with_hysteresis}"
        
        radius_with_hysteresis = manager.switching_radius
        
        # Third update - move toward (decrease distance significantly)
        robot_pos = np.array([initial_distance - distance_increase, 0.0])
        manager.update(robot_pos, 0.04)
        
        # Hysteresis should deactivate and radius should decrease by margin
        assert manager.hysteresis_active is False, "Hysteresis should deactivate when moving toward"
        expected_radius_after_deactivation = radius_with_hysteresis - hysteresis_margin
        assert np.isclose(manager.switching_radius, expected_radius_after_deactivation, atol=1e-6), \
            f"Radius should decrease by hysteresis margin: {manager.switching_radius} vs {expected_radius_after_deactivation}"
        
        # Verify the radius returned to the pre-hysteresis value
        assert np.isclose(manager.switching_radius, radius_before_hysteresis, atol=1e-6), \
            f"Radius should return to pre-hysteresis value: {manager.switching_radius} vs {radius_before_hysteresis}"


class TestPropertyForwardProgress:
    """Property 11: Forward Progress Requirement - Validates: Requirements 3.7"""
    
    @given(
        timeout=st.floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False),
        dt=st.floats(min_value=0.01, max_value=0.05, allow_nan=False, allow_infinity=False),
        distance_increment=st.floats(min_value=0.05, max_value=0.2, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=50, deadline=None)
    def test_forward_progress_timeout_triggers_switch(self, timeout, dt, distance_increment):
        """
        **Validates: Requirements 3.7**
        
        For any checkpoint switching event, either forward progress SHALL have been made 
        (distance decreasing) OR the forward progress timeout SHALL have been exceeded.
        """
        manager = CheckpointManager(
            base_switching_radius=0.5,
            forward_progress_timeout=timeout,
            dt=dt
        )
        
        checkpoints = [
            Checkpoint(x=0.0, y=0.0, theta=0.0, curvature=0.0, index=0),
            Checkpoint(x=5.0, y=0.0, theta=0.0, curvature=0.0, index=1),
        ]
        manager.set_checkpoints(checkpoints)
        
        # Calculate number of steps needed to exceed timeout
        steps_needed = int(np.ceil(timeout / dt)) + 2
        
        # Move away from checkpoint repeatedly (no forward progress)
        initial_distance = 1.0
        switched = False
        for i in range(steps_needed):
            robot_pos = np.array([initial_distance + i * distance_increment, 0.0])
            current_time = i * dt
            switched = manager.update(robot_pos, current_time)
            if switched:
                break
        
        # Should have switched due to timeout
        assert switched is True, f"Should switch after {steps_needed} steps exceeding timeout {timeout}"
        assert manager.current_idx == 1, "Should have advanced to next checkpoint"
        assert manager.checkpoints_missed == 1, "Should record checkpoint as missed"
    
    @given(
        base_radius=st.floats(min_value=0.3, max_value=1.0, allow_nan=False, allow_infinity=False),
        approach_distance=st.floats(min_value=0.01, max_value=0.15, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=None)
    def test_forward_progress_allows_switch_within_radius(self, base_radius, approach_distance):
        """
        **Validates: Requirements 3.7**
        
        When forward progress is made (distance decreasing) and robot is within 
        switching radius, checkpoint SHALL switch without timeout.
        """
        manager = CheckpointManager(
            base_switching_radius=base_radius,
            forward_progress_timeout=1.0,
            curvature_scaling=0.0,  # Disable curvature effect
            dt=0.02
        )
        
        checkpoints = [
            Checkpoint(x=0.0, y=0.0, theta=0.0, curvature=0.0, index=0),
            Checkpoint(x=5.0, y=0.0, theta=0.0, curvature=0.0, index=1),
        ]
        manager.set_checkpoints(checkpoints)
        
        # Approach checkpoint (forward progress) - start from outside radius
        robot_pos = np.array([base_radius * 1.5, 0.0])
        manager.update(robot_pos, 0.0)
        
        # Move closer (within radius) - ensure we're definitely within
        robot_pos = np.array([approach_distance, 0.0])
        switched = manager.update(robot_pos, 0.02)
        
        # Should switch due to being within radius with forward progress
        assert switched is True, f"Should switch when distance {approach_distance} < radius {base_radius}"
        assert manager.current_idx == 1, "Should have advanced to next checkpoint"
        assert manager.checkpoints_missed == 0, "Should not count as missed"
        assert manager.no_progress_time == 0.0, "No progress time should reset after switch"

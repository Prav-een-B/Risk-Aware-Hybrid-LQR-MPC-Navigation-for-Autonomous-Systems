"""Integration tests for Task 9: Checkpoint-based tracking with controllers."""

import numpy as np
import pytest

from src.hybrid_controller.hybrid_controller.trajectory.reference_generator import (
    ReferenceTrajectoryGenerator,
)
from src.hybrid_controller.hybrid_controller.controllers.lqr_controller import LQRController
from src.hybrid_controller.hybrid_controller.controllers.mpc_controller import MPCController
from src.hybrid_controller.hybrid_controller.controllers.adaptive_mpc_controller import (
    AdaptiveMPCController,
)
from src.hybrid_controller.hybrid_controller.models.differential_drive import DifferentialDriveRobot


class TestCheckpointModeIntegration:
    """Test checkpoint mode integration with trajectory generator."""
    
    def test_checkpoint_mode_initialization(self):
        """Test that checkpoint mode initializes CheckpointManager."""
        traj_gen = ReferenceTrajectoryGenerator(
            trajectory_type="lissajous",
            checkpoint_mode=True,
            dt=0.02
        )
        
        # Verify checkpoint manager is initialized
        assert traj_gen.checkpoint_mode is True
        assert traj_gen.checkpoint_manager is not None
    
    def test_checkpoint_generation_with_curvature(self):
        """Test that checkpoints are generated with curvature information."""
        traj_gen = ReferenceTrajectoryGenerator(
            trajectory_type="lissajous",
            checkpoint_mode=True,
            dt=0.02
        )
        
        # Generate trajectory
        trajectory = traj_gen.generate(duration=10.0)
        
        # Verify trajectory was generated
        assert trajectory is not None
        assert len(trajectory) > 0
        
        # Verify checkpoints were generated
        assert traj_gen.checkpoint_manager is not None
        assert len(traj_gen.checkpoint_manager.checkpoints) > 0
        
        # Verify checkpoints have curvature information
        for cp in traj_gen.checkpoint_manager.checkpoints:
            assert hasattr(cp, 'curvature')
            assert cp.curvature >= 0.0  # Curvature should be non-negative
    
    def test_checkpoint_mode_reference_extraction(self):
        """Test reference extraction in checkpoint mode."""
        traj_gen = ReferenceTrajectoryGenerator(
            trajectory_type="lissajous",
            checkpoint_mode=True,
            dt=0.02
        )
        
        # Generate trajectory
        traj_gen.generate(duration=10.0)
        
        # Test reference point extraction
        state = np.array([0.0, 0.0, 0.0])
        x_refs, u_refs = traj_gen.get_local_trajectory_segment(state, horizon=1)
        
        assert x_refs.shape == (1, 3)
        assert u_refs.shape == (0, 2)  # horizon-1 controls
    
    def test_checkpoint_mode_horizon_extraction(self):
        """Test horizon extraction for MPC in checkpoint mode."""
        traj_gen = ReferenceTrajectoryGenerator(
            trajectory_type="lissajous",
            checkpoint_mode=True,
            dt=0.02
        )
        
        # Generate trajectory
        traj_gen.generate(duration=10.0)
        
        # Test horizon extraction
        state = np.array([0.0, 0.0, 0.0])
        horizon = 10
        x_refs, u_refs = traj_gen.get_local_trajectory_segment(state, horizon)
        
        assert x_refs.shape == (horizon, 3)
        assert u_refs.shape == (horizon - 1, 2)
    
    def test_checkpoint_manager_update(self):
        """Test checkpoint manager updates during simulation."""
        traj_gen = ReferenceTrajectoryGenerator(
            trajectory_type="lissajous",
            checkpoint_mode=True,
            dt=0.02
        )
        
        # Generate trajectory
        traj_gen.generate(duration=10.0)
        
        # Simulate checkpoint updates
        robot_position = np.array([0.0, 0.0])
        current_time = 0.0
        
        # Update checkpoint manager
        switched = traj_gen.update_checkpoint_manager(robot_position, current_time)
        
        # Should not switch immediately (robot at first checkpoint)
        assert isinstance(switched, bool)
    
    def test_checkpoint_metrics_export(self):
        """Test checkpoint metrics export."""
        traj_gen = ReferenceTrajectoryGenerator(
            trajectory_type="lissajous",
            checkpoint_mode=True,
            dt=0.02
        )
        
        # Generate trajectory
        traj_gen.generate(duration=10.0)
        
        # Get metrics
        metrics = traj_gen.get_checkpoint_metrics()
        
        # Verify metrics structure
        assert isinstance(metrics, dict)
        assert 'checkpoints_total' in metrics
        assert 'checkpoints_reached' in metrics
        assert 'checkpoints_missed' in metrics
        assert 'completion_rate' in metrics
        assert 'mean_time_to_checkpoint' in metrics
        assert 'mean_overshoot' in metrics


class TestLQRCheckpointIntegration:
    """Test LQR controller with checkpoint mode."""
    
    def test_lqr_with_checkpoint_mode(self):
        """Test LQR controller receives checkpoint references."""
        # Create trajectory generator with checkpoint mode
        traj_gen = ReferenceTrajectoryGenerator(
            trajectory_type="lissajous",
            checkpoint_mode=True,
            dt=0.02
        )
        traj_gen.generate(duration=5.0)
        
        # Create LQR controller
        lqr = LQRController(Q_diag=[10.0, 10.0, 1.0], R_diag=[1.0, 1.0])
        
        # Create robot
        robot = DifferentialDriveRobot()
        
        # Get reference from checkpoint mode
        state = np.array([0.0, 0.0, 0.0])
        x_refs, u_refs = traj_gen.get_local_trajectory_segment(state, 1)
        x_ref = x_refs[0]
        u_ref = u_refs[0] if len(u_refs) > 0 else np.array([0.0, 0.0])
        
        # Compute control
        u, error = lqr.compute_control_at_operating_point(state, x_ref, u_ref)
        
        # Verify control output
        assert u.shape == (2,)
        assert error.shape == (3,)


class TestMPCCheckpointIntegration:
    """Test MPC controller with checkpoint mode."""
    
    def test_mpc_with_checkpoint_mode(self):
        """Test MPC controller receives checkpoint horizon."""
        # Create trajectory generator with checkpoint mode
        traj_gen = ReferenceTrajectoryGenerator(
            trajectory_type="lissajous",
            checkpoint_mode=True,
            dt=0.02
        )
        traj_gen.generate(duration=5.0)
        
        # Create MPC controller
        mpc = MPCController(horizon=10, dt=0.02)
        
        # Get reference horizon from checkpoint mode
        state = np.array([0.0, 0.0, 0.0])
        x_refs, u_refs = traj_gen.get_local_trajectory_segment(state, mpc.N + 1)
        
        # Verify horizon shapes
        assert x_refs.shape == (mpc.N + 1, 3)
        assert u_refs.shape == (mpc.N, 2)
        
        # Solve MPC (without obstacles for simplicity)
        solution = mpc.solve_with_ltv(state, x_refs, u_refs, obstacles=[])
        
        # Verify solution
        assert solution.optimal_control.shape == (2,)


class TestAdaptiveMPCCheckpointIntegration:
    """Test Adaptive MPC controller with checkpoint mode."""
    
    def test_adaptive_mpc_with_checkpoint_mode(self):
        """Test Adaptive MPC controller receives extended checkpoint horizon."""
        # Create trajectory generator with checkpoint mode
        traj_gen = ReferenceTrajectoryGenerator(
            trajectory_type="lissajous",
            checkpoint_mode=True,
            dt=0.02
        )
        traj_gen.generate(duration=5.0)
        
        # Create Adaptive MPC controller
        adaptive_mpc = AdaptiveMPCController(prediction_horizon=10, terminal_horizon=5, dt=0.02)
        
        # Get extended reference horizon from checkpoint mode
        state = np.array([0.0, 0.0, 0.0])
        x_refs, u_refs = traj_gen.get_local_trajectory_segment(state, adaptive_mpc.N_ext + 1)
        
        # Verify horizon shapes
        assert x_refs.shape == (adaptive_mpc.N_ext + 1, 3)
        assert u_refs.shape == (adaptive_mpc.N_ext, 2)
        
        # Solve Adaptive MPC (without obstacles for simplicity)
        solution = adaptive_mpc.solve_tracking(
            state, x_refs, u_refs[:adaptive_mpc.N_ext], obstacles=[]
        )
        
        # Verify solution
        assert solution.optimal_control.shape == (2,)


class TestMultipleTrajectoryTypesCheckpointMode:
    """Test checkpoint mode with different trajectory types."""
    
    @pytest.mark.parametrize("trajectory_type", [
        "lissajous",
        "spiral",
        "sinusoidal",
        "clothoid",
    ])
    def test_trajectory_types_with_checkpoint_mode(self, trajectory_type):
        """Test checkpoint mode works with various trajectory types."""
        traj_gen = ReferenceTrajectoryGenerator(
            trajectory_type=trajectory_type,
            checkpoint_mode=True,
            dt=0.02
        )
        
        # Generate trajectory
        trajectory = traj_gen.generate(duration=5.0)
        
        # Verify trajectory was generated
        assert trajectory is not None
        assert len(trajectory) > 0
        
        # Verify checkpoints were generated
        assert traj_gen.checkpoint_manager is not None
        assert len(traj_gen.checkpoint_manager.checkpoints) > 0
        
        # Test reference extraction
        state = np.array([0.0, 0.0, 0.0])
        x_refs, u_refs = traj_gen.get_local_trajectory_segment(state, 5)
        
        assert x_refs.shape == (5, 3)
        assert u_refs.shape == (4, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

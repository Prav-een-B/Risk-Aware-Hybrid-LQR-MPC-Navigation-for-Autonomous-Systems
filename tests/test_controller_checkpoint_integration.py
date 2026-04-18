"""
Integration tests for controller compatibility with checkpoint-based tracking.

Tests verify that LQR, MPC, Adaptive MPC, and hybrid blending controllers
work correctly with checkpoint-based reference extraction.
"""

import numpy as np
import pytest

from hybrid_controller.controllers.lqr_controller import LQRController
from hybrid_controller.controllers.mpc_controller import MPCController, Obstacle
from hybrid_controller.controllers.risk_metrics import RiskMetrics
from hybrid_controller.controllers.hybrid_blender import BlendingSupervisor
from hybrid_controller.models.differential_drive import DifferentialDriveRobot
from hybrid_controller.trajectory.reference_generator import ReferenceTrajectoryGenerator
from hybrid_controller.navigation.checkpoint_manager import (
    CheckpointManager,
    Checkpoint as CheckpointData,
)


def create_test_checkpoints(num_checkpoints: int = 5) -> list:
    """Create a simple test checkpoint sequence."""
    checkpoints = []
    for i in range(num_checkpoints):
        x = float(i) * 0.5
        y = 0.5 * np.sin(float(i) * 0.5)
        theta = np.arctan2(0.5 * np.cos(float(i) * 0.5) * 0.5, 0.5)
        checkpoints.append(
            CheckpointData(
                x=x,
                y=y,
                theta=theta,
                curvature=0.5,
                index=i,
            )
        )
    return checkpoints


class TestLQRCheckpointIntegration:
    """Test LQR controller with checkpoint-based references."""

    def test_lqr_with_checkpoint_reference(self):
        """Test that LQR can track a single checkpoint reference."""
        # Setup
        lqr = LQRController(
            Q_diag=[10.0, 10.0, 5.0],
            R_diag=[0.1, 0.1],
            dt=0.02,
            v_max=2.0,
            omega_max=3.0,
        )
        robot = DifferentialDriveRobot(v_max=2.0, omega_max=3.0)

        # Create checkpoint manager
        checkpoints = create_test_checkpoints(5)
        manager = CheckpointManager(dt=0.02)
        manager.set_checkpoints(checkpoints)

        # Initial state
        x = np.array([0.0, 0.0, 0.0])

        # Get reference from checkpoint
        current_cp = manager.get_current_checkpoint()
        assert current_cp is not None

        x_ref = np.array([current_cp.x, current_cp.y, current_cp.theta])
        u_ref = np.array([0.5, 0.0])  # Nominal control

        # Compute LQR control
        u, error = lqr.compute_control_at_operating_point(x, x_ref, u_ref)

        # Verify control output
        assert u.shape == (2,)
        assert error.shape == (3,)
        assert np.isfinite(u).all()
        assert np.isfinite(error).all()

        # Verify control is within limits
        assert abs(u[0]) <= 2.0
        assert abs(u[1]) <= 3.0

    def test_lqr_checkpoint_tracking_loop(self):
        """Test LQR tracking through multiple checkpoints."""
        # Setup
        lqr = LQRController(
            Q_diag=[15.0, 15.0, 8.0],
            R_diag=[0.1, 0.1],
            dt=0.02,
            v_max=2.0,
            omega_max=3.0,
        )
        robot = DifferentialDriveRobot(v_max=2.0, omega_max=3.0)

        # Create checkpoint manager
        checkpoints = create_test_checkpoints(10)
        manager = CheckpointManager(
            base_switching_radius=0.3,
            dt=0.02,
        )
        manager.set_checkpoints(checkpoints)

        # Initial state at first checkpoint
        x = np.array([checkpoints[0].x, checkpoints[0].y, checkpoints[0].theta])

        # Simulate for 100 steps
        for k in range(100):
            # Update checkpoint manager
            manager.update(x[:2], k * 0.02)

            # Get current checkpoint reference
            current_cp = manager.get_current_checkpoint()
            if current_cp is None:
                break

            x_ref = np.array([current_cp.x, current_cp.y, current_cp.theta])
            u_ref = np.array([0.5, 0.0])

            # Compute control
            u, error = lqr.compute_control_at_operating_point(x, x_ref, u_ref)

            # Simulate
            x = robot.simulate_step(x, u, 0.02)

        # Verify at least some checkpoints were reached
        assert manager.checkpoints_reached > 0
        assert manager.checkpoints_reached <= len(checkpoints)


class TestMPCCheckpointIntegration:
    """Test MPC controller with checkpoint-based reference horizon."""

    def test_mpc_with_checkpoint_horizon(self):
        """Test that MPC can use checkpoint-based reference horizon."""
        # Setup
        mpc = MPCController(
            horizon=5,
            Q_diag=[80.0, 80.0, 120.0],
            R_diag=[0.1, 0.1],
            P_diag=[20.0, 20.0, 40.0],
            d_safe=0.3,
            slack_penalty=5000.0,
            dt=0.02,
            v_max=2.0,
            omega_max=3.0,
            solver="OSQP",
        )

        # Create checkpoint manager
        checkpoints = create_test_checkpoints(10)
        manager = CheckpointManager(dt=0.02)
        manager.set_checkpoints(checkpoints)

        # Initial state
        x = np.array([0.0, 0.0, 0.0])

        # Get reference horizon from checkpoint manager
        x_refs, u_refs = manager.get_local_trajectory_segment(x, mpc.N + 1)

        # Verify horizon shapes
        assert x_refs.shape == (mpc.N + 1, 3)
        assert u_refs.shape == (mpc.N, 2)

        # Solve MPC
        obstacles = []
        solution = mpc.solve_with_ltv(x, x_refs, u_refs, obstacles)

        # Verify solution
        assert solution is not None
        assert solution.optimal_control.shape == (2,)
        assert np.isfinite(solution.optimal_control).all()

    def test_mpc_checkpoint_tracking_with_obstacles(self):
        """Test MPC tracking checkpoints while avoiding obstacles."""
        # Setup
        mpc = MPCController(
            horizon=5,
            Q_diag=[80.0, 80.0, 120.0],
            R_diag=[0.1, 0.1],
            P_diag=[20.0, 20.0, 40.0],
            d_safe=0.3,
            slack_penalty=5000.0,
            dt=0.02,
            v_max=2.0,
            omega_max=3.0,
            solver="OSQP",
        )
        robot = DifferentialDriveRobot(v_max=2.0, omega_max=3.0)

        # Create checkpoint manager
        checkpoints = create_test_checkpoints(10)
        manager = CheckpointManager(dt=0.02)
        manager.set_checkpoints(checkpoints)

        # Add obstacle near path
        obstacles = [Obstacle(x=1.0, y=0.3, radius=0.2)]

        # Initial state
        x = np.array([0.0, 0.0, 0.0])

        # Simulate for 50 steps
        for k in range(50):
            # Update checkpoint manager
            manager.update(x[:2], k * 0.02)

            # Get reference horizon
            x_refs, u_refs = manager.get_local_trajectory_segment(x, mpc.N + 1)

            # Solve MPC
            solution = mpc.solve_with_ltv(x, x_refs, u_refs, obstacles)

            # Apply control
            u = solution.optimal_control
            x = robot.simulate_step(x, u, 0.02)

        # Verify progress was made
        assert manager.checkpoints_reached > 0


class TestAdaptiveMPCCheckpointIntegration:
    """Test Adaptive MPC controller with checkpoint-based references."""

    @pytest.mark.skipif(
        not pytest.importorskip("casadi", reason="CasADi not available"),
        reason="Adaptive MPC requires CasADi",
    )
    def test_adaptive_mpc_with_checkpoint_horizon(self):
        """Test that Adaptive MPC can use checkpoint-based reference horizon."""
        try:
            from hybrid_controller.controllers.adaptive_mpc_controller import (
                AdaptiveMPCController,
            )
        except ImportError:
            pytest.skip("AdaptiveMPCController not available")

        # Setup
        adaptive_mpc = AdaptiveMPCController(
            prediction_horizon=6,
            terminal_horizon=4,
            Q_diag=[80.0, 80.0, 120.0],
            R_diag=[0.1, 0.1],
            P_diag=[20.0, 20.0, 40.0],
            d_safe=0.3,
            slack_penalty=5000.0,
            dt=0.02,
            v_max=2.0,
            omega_max=3.0,
        )

        # Create checkpoint manager
        checkpoints = create_test_checkpoints(15)
        manager = CheckpointManager(dt=0.02)
        manager.set_checkpoints(checkpoints)

        # Initial state
        x = np.array([0.0, 0.0, 0.0])

        # Get extended reference horizon
        x_refs, u_refs = manager.get_local_trajectory_segment(x, adaptive_mpc.N_ext + 1)

        # Verify horizon shapes
        assert x_refs.shape == (adaptive_mpc.N_ext + 1, 3)
        assert u_refs.shape == (adaptive_mpc.N_ext, 2)

        # Solve Adaptive MPC
        obstacles = []
        solution = adaptive_mpc.solve_tracking(
            x, x_refs, u_refs[: adaptive_mpc.N_ext], obstacles
        )

        # Verify solution
        assert solution is not None
        assert solution.optimal_control.shape == (2,)
        assert np.isfinite(solution.optimal_control).all()


class TestHybridBlendingCheckpointIntegration:
    """Test hybrid blending with checkpoint-based references."""

    def test_hybrid_blending_with_checkpoints(self):
        """Test that hybrid blending works with checkpoint references."""
        # Setup controllers
        lqr = LQRController(
            Q_diag=[15.0, 15.0, 8.0],
            R_diag=[0.1, 0.1],
            dt=0.02,
            v_max=2.0,
            omega_max=3.0,
        )
        mpc = MPCController(
            horizon=5,
            Q_diag=[80.0, 80.0, 120.0],
            R_diag=[0.1, 0.1],
            P_diag=[20.0, 20.0, 40.0],
            d_safe=0.3,
            slack_penalty=5000.0,
            dt=0.02,
            v_max=2.0,
            omega_max=3.0,
            solver="OSQP",
        )
        risk_metrics = RiskMetrics(
            d_safe=0.3,
            d_trigger=1.0,
            alpha=0.6,
            beta=0.4,
            threshold_low=0.2,
            threshold_medium=0.5,
        )
        blender = BlendingSupervisor(
            k_sigmoid=10.0,
            risk_threshold=0.3,
            dw_max=2.0,
            hysteresis_band=0.05,
            solver_time_limit=5.0,
            feasibility_decay=0.8,
            dt=0.02,
        )
        robot = DifferentialDriveRobot(v_max=2.0, omega_max=3.0)

        # Create checkpoint manager
        checkpoints = create_test_checkpoints(10)
        manager = CheckpointManager(dt=0.02)
        manager.set_checkpoints(checkpoints)

        # Add obstacle to trigger risk
        obstacles = [{"x": 1.0, "y": 0.3, "radius": 0.2, "speed": 0.0}]
        obstacle_objs = [Obstacle(x=1.0, y=0.3, radius=0.2)]

        # Initial state
        x = np.array([0.0, 0.0, 0.0])

        # Simulate for 50 steps
        blend_weights = []
        for k in range(50):
            # Update checkpoint manager
            manager.update(x[:2], k * 0.02)

            # Get current checkpoint reference
            current_cp = manager.get_current_checkpoint()
            if current_cp is None:
                break

            x_ref = np.array([current_cp.x, current_cp.y, current_cp.theta])
            u_ref = np.array([0.5, 0.0])

            # Get reference horizon for MPC
            x_refs, u_refs = manager.get_local_trajectory_segment(x, mpc.N + 1)

            # Compute LQR control
            u_lqr, _ = lqr.compute_control_at_operating_point(x, x_ref, u_ref)

            # Compute MPC control
            mpc_solution = mpc.solve_with_ltv(x, x_refs, u_refs, obstacle_objs)
            u_mpc = mpc_solution.optimal_control

            # Assess risk
            assessment = risk_metrics.assess_risk(x, obstacles)

            # Blend controls
            u_blend, blend_info = blender.blend(
                u_lqr=u_lqr,
                u_mpc=u_mpc,
                risk=assessment.combined_risk,
                solver_status=mpc_solution.status,
                solver_time_ms=mpc_solution.solve_time_ms,
                feasibility_margin=mpc_solution.feasibility_margin,
            )

            blend_weights.append(blend_info.weight)

            # Apply blended control
            x = robot.simulate_step(x, u_blend, 0.02)

        # Verify blending occurred
        assert len(blend_weights) > 0
        assert all(0.0 <= w <= 1.0 for w in blend_weights)

        # Verify some checkpoints were reached
        assert manager.checkpoints_reached > 0


class TestReferenceTrajectoryGeneratorCheckpointMode:
    """Test ReferenceTrajectoryGenerator with checkpoint mode."""

    def test_checkpoint_path_trajectory_type(self):
        """Test that checkpoint_path trajectory type works."""
        # Create generator with checkpoint_path type
        traj_gen = ReferenceTrajectoryGenerator(
            dt=0.02,
            trajectory_type="checkpoint_path",
            checkpoint_preset="diamond",
        )

        # Generate trajectory
        trajectory = traj_gen.generate(duration=10.0)

        # Verify trajectory was generated
        assert trajectory is not None
        assert len(trajectory) > 0
        assert trajectory.shape[1] == 6  # [t, px, py, theta, v, omega]

    def test_get_local_trajectory_segment_checkpoint_mode(self):
        """Test get_local_trajectory_segment with checkpoint_path."""
        # Create generator
        traj_gen = ReferenceTrajectoryGenerator(
            dt=0.02,
            trajectory_type="checkpoint_path",
            checkpoint_preset="diamond",
        )

        # Generate trajectory
        traj_gen.generate(duration=10.0)

        # Get local segment
        state = np.array([0.0, 0.0, 0.0])
        horizon = 5
        x_refs, u_refs = traj_gen.get_local_trajectory_segment(state, horizon)

        # Verify shapes
        assert x_refs.shape == (horizon, 3)
        assert u_refs.shape == (horizon - 1, 2)

    def test_multiple_trajectory_types_with_checkpoints(self):
        """Test that various trajectory types work with checkpoint extraction."""
        trajectory_types = [
            "checkpoint_path",
            "spline_path",
            "urban_path",
            "random_waypoint",
        ]

        for traj_type in trajectory_types:
            # Create generator
            traj_gen = ReferenceTrajectoryGenerator(
                dt=0.02,
                trajectory_type=traj_type,
                random_seed=42 if traj_type == "random_waypoint" else None,
            )

            # Generate trajectory
            trajectory = traj_gen.generate(duration=10.0)

            # Verify trajectory
            assert trajectory is not None
            assert len(trajectory) > 0

            # Test local segment extraction
            state = np.array([0.0, 0.0, 0.0])
            x_refs, u_refs = traj_gen.get_local_trajectory_segment(state, 5)

            assert x_refs.shape == (5, 3)
            assert u_refs.shape == (4, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

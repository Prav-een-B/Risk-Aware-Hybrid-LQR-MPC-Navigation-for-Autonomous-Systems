#!/usr/bin/env python3
"""
Standalone Simulation
=====================

Run the LQR and MPC simulations without ROS2 dependencies.
Useful for testing, validation, and generating plots.

Usage:
    python run_simulation.py --mode lqr
    python run_simulation.py --mode mpc
    python run_simulation.py --mode compare
    python run_simulation.py --mode hybrid
    python run_simulation.py --mode adaptive
    python run_simulation.py --mode hybrid_adaptive
"""

import sys
import os
import argparse
import numpy as np
from typing import Any, Dict, List

# Add the package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'hybrid_controller'))

from hybrid_controller.models.differential_drive import DifferentialDriveRobot
from hybrid_controller.trajectory.reference_generator import ReferenceTrajectoryGenerator
from hybrid_controller.controllers.lqr_controller import LQRController
from hybrid_controller.controllers.mpc_controller import MPCController, Obstacle
from hybrid_controller.controllers.risk_metrics import RiskMetrics
from hybrid_controller.controllers.hybrid_blender import BlendingSupervisor
from hybrid_controller.models.actuator_dynamics import ActuatorDynamics, ActuatorParams
from hybrid_controller.logging.simulation_logger import SimulationLogger
from hybrid_controller.utils.visualization import Visualizer
from evaluation.scenarios import DynamicObstacleField, InflationConfig, build_demo_config

TRAJECTORY_CHOICES = list(ReferenceTrajectoryGenerator.TRAJECTORY_TYPES)
CHECKPOINT_PRESET_CHOICES = sorted(ReferenceTrajectoryGenerator.CHECKPOINT_PRESETS.keys())
SCENARIO_CHOICES = ["default", "sparse", "dense", "corridor", "moving", "random_walk"]
DYNAMIC_SCENARIOS = {"moving", "random_walk"}
DEFAULT_INFLATION = InflationConfig(
    safety_factor=1.0,
    sensing_factor=0.05,
    motion_lookahead=0.5,
)


def build_trajectory_generator(
    dt: float,
    trajectory_type: str = "figure8",
    checkpoint_preset: str = "diamond",
    checkpoint_mode: bool = False,
    amplitude: float = 2.0,
    frequency: float = 0.5,
) -> ReferenceTrajectoryGenerator:
    """Create a trajectory generator with consistent defaults."""
    return ReferenceTrajectoryGenerator(
        A=amplitude,
        a=frequency,
        dt=dt,
        T_blend=0.5,
        trajectory_type=trajectory_type,
        checkpoint_preset=checkpoint_preset,
        checkpoint_mode=checkpoint_mode,
    )


def describe_trajectory(trajectory_type: str, checkpoint_preset: str) -> str:
    """Human-readable trajectory description for logs."""
    if trajectory_type == "checkpoint_path":
        return f"{trajectory_type} ({checkpoint_preset})"
    return trajectory_type


def build_obstacle_field(
    scenario: str,
    seed: int = 42,
    inflation: InflationConfig = DEFAULT_INFLATION,
) -> DynamicObstacleField:
    """Create a mutable obstacle field for the selected scenario."""
    config = build_demo_config(name=scenario, seed=seed)
    return config.create_field(inflation=inflation, seed=seed)


def get_plot_obstacles(field: DynamicObstacleField) -> List[Dict[str, float]]:
    """Convert a field snapshot to static dicts used by plotting utilities."""
    obstacle_dicts: List[Dict[str, float]] = []
    for item in field.snapshot(use_base_radius=True):
        payload = dict(item)
        x_raw: Any = payload.get("x", 0.0)
        y_raw: Any = payload.get("y", 0.0)
        x = float(x_raw)
        y = float(y_raw)
        radius_raw: Any = payload.get("base_radius", payload.get("radius", 0.0))
        obstacle_dicts.append({"x": x, "y": y, "radius": float(radius_raw)})
    return obstacle_dicts


def has_collision(state: np.ndarray, obstacles: List[Obstacle], d_safe: float) -> bool:
    """Collision helper for both static and dynamic obstacle sets."""
    px, py = float(state[0]), float(state[1])
    return any(obs.is_collision(px, py, d_safe) for obs in obstacles)


def get_adaptive_mpc_class():
    """Lazy import AdaptiveMPCController to keep CasADi optional."""
    try:
        from hybrid_controller.controllers.adaptive_mpc_controller import AdaptiveMPCController
    except Exception as exc:
        raise RuntimeError(
            "Adaptive modes require CasADi. Install optional dependency with: pip install casadi"
        ) from exc
    return AdaptiveMPCController


def get_reference_point(
    traj_gen: ReferenceTrajectoryGenerator,
    trajectory_type: str,
    state: np.ndarray,
    index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the current reference point for control.

    For checkpoint paths or checkpoint mode, use a local state-anchored reference 
    point so control does not depend solely on a global time index.
    """
    if trajectory_type == "checkpoint_path" or traj_gen.checkpoint_mode:
        x_refs, u_refs = traj_gen.get_local_trajectory_segment(state, 1)
        # CheckpointManager returns u_refs with shape (horizon-1, 2), so for
        # horizon=1 there is no control row. Fall back to cached/global
        # reference control to keep single-step controllers stable at the tail.
        if u_refs.shape[0] == 0:
            _, u_ref = traj_gen.get_reference_at_index(index)
            return x_refs[0], u_ref
        return x_refs[0], u_refs[0]
    return traj_gen.get_reference_at_index(index)


def get_reference_segment(
    traj_gen: ReferenceTrajectoryGenerator,
    trajectory_type: str,
    state: np.ndarray,
    index: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a horizon of references for predictive controllers.

    For checkpoint paths or checkpoint mode, use local horizon extraction around 
    current state.
    """
    if trajectory_type == "checkpoint_path" or traj_gen.checkpoint_mode:
        return traj_gen.get_local_trajectory_segment(state, horizon)
    return traj_gen.get_trajectory_segment(index, horizon)


def run_lqr_simulation(duration: float = 20.0, dt: float = 0.02,
                       visualize: bool = True,
                       trajectory_type: str = "figure8",
                       checkpoint_preset: str = "diamond",
                       checkpoint_mode: bool = False) -> dict:
    """
    Run LQR trajectory tracking simulation.
    
    Args:
        duration: Simulation duration (seconds)
        dt: Time step (seconds)
        visualize: Generate plots
        
    Returns:
        Dictionary with simulation results
    """
    print("=" * 60)
    print("LQR Trajectory Tracking Simulation")
    print("=" * 60)
    
    # Initialize components
    robot = DifferentialDriveRobot(v_max=2.0, omega_max=3.0)  # Increased limits to prevent saturation
    traj_gen = build_trajectory_generator(
        dt=dt,
        trajectory_type=trajectory_type,
        checkpoint_preset=checkpoint_preset,
        checkpoint_mode=checkpoint_mode,
    )
    lqr = LQRController(Q_diag=[15.0, 15.0, 8.0], R_diag=[0.1, 0.1], dt=dt, v_max=2.0, omega_max=3.0)
    logger = SimulationLogger(log_dir='logs', log_level='INFO', node_name='lqr_sim')
    
    # Generate reference trajectory
    trajectory = traj_gen.generate(duration)
    N = len(trajectory)
    
    print(
        f"Generated {N} trajectory points over {duration}s "
        f"using {describe_trajectory(trajectory_type, checkpoint_preset)}"
    )
    
    # Initialize state at the reference trajectory's starting point
    x_ref_init, _ = traj_gen.get_reference_at_index(0)
    x = x_ref_init.copy()  # Start at reference position and heading
    
    # Storage
    states = np.zeros((N, 3))
    controls = np.zeros((N - 1, 2))
    errors = np.zeros((N - 1, 3))
    
    states[0] = x
    
    # Simulation loop
    for k in range(N - 1):
        # Update checkpoint manager if in checkpoint mode
        if traj_gen.checkpoint_mode:
            t = k * dt
            traj_gen.update_checkpoint_manager(x[:2], t)
        
        # Get reference
        x_ref, u_ref = get_reference_point(
            traj_gen,
            trajectory_type,
            x,
            k,
        )
        
        # Compute LQR control
        u, error = lqr.compute_control_at_operating_point(x, x_ref, u_ref)
        
        # Log
        logger.log_state(k, x, x_ref, error)
        logger.log_control(k, u, 'LQR')
        
        # Simulate robot
        x = robot.simulate_step(x, u, dt)
        
        # Store
        states[k + 1] = x
        controls[k] = u
        errors[k] = error
        
        if k % 100 == 0:
            error_norm = np.linalg.norm(error[:2])
            print(f"  k={k:4d}: error_norm={error_norm:.4f}, u=[{u[0]:.3f}, {u[1]:.3f}]")
    
    # Final error
    final_error = np.linalg.norm(errors[-1, :2])
    mean_error = np.mean(np.linalg.norm(errors[:, :2], axis=1))
    
    print(f"\nResults:")
    print(f"  Mean tracking error: {mean_error:.4f} m")
    print(f"  Final tracking error: {final_error:.4f} m")
    
    # Export logs
    logger.finalize()
    
    # Visualization
    if visualize:
        viz = Visualizer(output_dir='outputs')
        ref_states = trajectory[:, 1:4]  # [px, py, theta]
        
        viz.plot_trajectory(states, ref_states, 
                           title="LQR Trajectory Tracking",
                           save_path="outputs/lqr_tracking.png")
        
        viz.plot_tracking_error(errors, dt,
                               title="LQR Tracking Error",
                               save_path="outputs/lqr_error.png")
        
        viz.plot_control_inputs(controls, dt,
                               v_max=2.0, omega_max=3.0,
                               title="LQR Control Inputs",
                               save_path="outputs/lqr_control.png")
        
        print("\nPlots saved to outputs/")
    
    return {
        'states': states,
        'controls': controls,
        'errors': errors,
        'reference': trajectory[:, 1:4],
        'mean_error': mean_error,
        'final_error': final_error
    }


def run_mpc_simulation(duration: float = 20.0, dt: float = 0.02,
                       with_obstacles: bool = True,
                       visualize: bool = True,
                       scenario: str = "default",
                       trajectory_type: str = "figure8",
                       checkpoint_preset: str = "diamond",
                       checkpoint_mode: bool = False) -> dict:
    """
    Run MPC obstacle avoidance simulation.
    
    Args:
        duration: Simulation duration (seconds)
        dt: Time step (seconds)
        with_obstacles: Include obstacles
        visualize: Generate plots
        
    Returns:
        Dictionary with simulation results
    """
    print("=" * 60)
    print("MPC Obstacle Avoidance Simulation")
    print("=" * 60)
    
    # Initialize components
    robot = DifferentialDriveRobot(v_max=2.0, omega_max=3.0)
    traj_gen = build_trajectory_generator(
        dt=dt,
        trajectory_type=trajectory_type,
        checkpoint_preset=checkpoint_preset,
        checkpoint_mode=checkpoint_mode,
    )
    # OPTIMIZED MPC PARAMETERS for industry-standard tolerances
    # Target: heading ≤5°, latency ≤50ms, slack ≤5
    mpc = MPCController(
        horizon=6,                        # Reduced: 10→6 for faster solves
        Q_diag=[15.0, 15.0, 50.0],       # Aggressive heading: 20→50
        R_diag=[0.1, 0.1],
        P_diag=[30.0, 30.0, 40.0],       # Strong terminal heading: 15→40
        d_safe=0.3,
        slack_penalty=5000.0,             # Stricter: 1000→5000
        dt=dt,
        v_max=2.0,
        omega_max=3.0,
        solver='OSQP',                    # Faster: ECOS→OSQP
        block_size=2,                     # Move-blocking: halves decision vars
        w_max=0.05                        # Tube MPC: disturbance bound (5cm)
    )
    
    logger = SimulationLogger(log_dir='logs', log_level='INFO', node_name='mpc_sim')
    
    obstacle_field = None
    obstacle_dicts = []
    if with_obstacles:
        obstacle_field = build_obstacle_field(scenario=scenario)
        obstacle_dicts = get_plot_obstacles(obstacle_field)
        scenario_label = f"{scenario} (dynamic)" if scenario in DYNAMIC_SCENARIOS else scenario
        print(f"Scenario: {scenario_label} | Added {len(obstacle_dicts)} obstacles")
    else:
        print("Scenario: none | Obstacles disabled")
    
    obs_positions = np.array(
        [[o['x'], o['y']] for o in obstacle_dicts], dtype=float
    ).reshape(-1, 2) if obstacle_dicts else None

    # Generate reference trajectory
    trajectory = traj_gen.generate(duration, obstacle_positions=obs_positions)
    N = len(trajectory)
    
    print(
        f"Generated {N} trajectory points over {duration}s "
        f"using {describe_trajectory(trajectory_type, checkpoint_preset)}"
    )
    
    # Initialize state at the reference trajectory's starting point
    x_ref_init, _ = traj_gen.get_reference_at_index(0)
    x = x_ref_init.copy()  # Start at reference position and heading
    
    # Storage
    states = np.zeros((N, 3))
    controls = np.zeros((N - 1, 2))
    errors = np.zeros((N - 1, 3))
    solve_times = []
    
    states[0] = x
    
    # Simulation loop (MPC runs at lower rate)
    mpc_rate = 5  # Run MPC every N steps
    solution = None
    collision_count = 0
    
    for k in range(N - 1):
        # Update checkpoint manager if in checkpoint mode
        if traj_gen.checkpoint_mode:
            t = k * dt
            traj_gen.update_checkpoint_manager(x[:2], t)
        
        # Get reference segment
        x_refs, u_refs = get_reference_segment(
            traj_gen,
            trajectory_type,
            x,
            k,
            mpc.N + 1,
        )
        controller_obstacles = obstacle_field.controller_obstacles() if obstacle_field else []
        
        # MPC control at lower rate
        if k % mpc_rate == 0 or solution is None:
            solution = mpc.solve_with_ltv(x, x_refs, u_refs, controller_obstacles)
            solve_times.append(solution.solve_time_ms)
            
            if solution.slack_used:
                logger.log_constraint_event(k, 'slack_activated', 
                                           {'reason': 'feasibility'})
        
        u = solution.optimal_control
        controller_name = 'MPC'
        solve_time = solution.solve_time_ms
        
        error = x - x_refs[0]
        error[2] = robot.normalize_angle(error[2])
        
        # Log
        logger.log_state(k, x, x_refs[0], error)
        logger.log_control(k, u, controller_name, solve_time)
        
        # Simulate robot
        x = robot.simulate_step(x, u, dt)

        if obstacle_field:
            if has_collision(x, obstacle_field.actual_obstacles(), mpc.d_safe):
                collision_count += 1
            obstacle_field.step(dt)
        
        # Store
        states[k + 1] = x
        controls[k] = u
        errors[k] = error
        
        if k % 100 == 0:
            error_norm = np.linalg.norm(error[:2])
            ctrl_info = f"solve={solve_time:.2f}ms" if controller_name == 'MPC' else "bootstrap"
            print(f"  k={k:4d}: error_norm={error_norm:.4f}, {ctrl_info}")
    
    # Results
    mean_error = np.mean(np.linalg.norm(errors[:, :2], axis=1))
    final_error = np.linalg.norm(errors[-1, :2])
    mean_solve_time = np.mean(solve_times)
    
    print(f"\nResults:")
    print(f"  Mean tracking error: {mean_error:.4f} m")
    print(f"  Final tracking error: {final_error:.4f} m")
    print(f"  Mean MPC solve time: {mean_solve_time:.2f} ms")
    
    if with_obstacles:
        print(f"  Collision events: {collision_count}")
    
    logger.finalize()
    
    # Visualization
    if visualize:
        viz = Visualizer(output_dir='outputs')
        ref_states = trajectory[:, 1:4]
        
        viz.plot_with_obstacles(states, ref_states, obstacle_dicts, mpc.d_safe,
                               title="MPC Obstacle Avoidance",
                               save_path="outputs/mpc_obstacle_avoidance.png")
        
        viz.plot_tracking_error(errors, dt,
                               title="MPC Tracking Error",
                               save_path="outputs/mpc_error.png")
        
        viz.plot_control_inputs(controls, dt,
                               v_max=2.0, omega_max=3.0,
                               title="MPC Control Inputs",
                               save_path="outputs/mpc_control.png")
        
        print("\nPlots saved to outputs/")
    
    return {
        'states': states,
        'controls': controls,
        'errors': errors,
        'reference': trajectory[:, 1:4],
        'mean_error': mean_error,
        'collision_count': collision_count,
        'mean_solve_time': mean_solve_time
    }


def run_comparison(duration: float = 20.0, dt: float = 0.02,
                   scenario: str = "default",
                   trajectory_type: str = "figure8",
                   checkpoint_preset: str = "diamond",
                   checkpoint_mode: bool = False) -> None:
    """
    Run comparison between LQR and MPC with obstacles.
    """
    print("=" * 60)
    print("LQR vs MPC Comparison Simulation")
    print("=" * 60)
    
    obstacle_field_lqr = build_obstacle_field(scenario=scenario)
    obstacle_field_mpc = build_obstacle_field(scenario=scenario)
    obstacle_dicts = get_plot_obstacles(obstacle_field_lqr)
    scenario_label = f"{scenario} (dynamic)" if scenario in DYNAMIC_SCENARIOS else scenario
    print(f"Scenario: {scenario_label} | Added {len(obstacle_dicts)} obstacles")
    
    # Run LQR (without visibility of obstacles)
    print("\n--- Running LQR (obstacle-unaware) ---")
    robot = DifferentialDriveRobot(v_max=2.0, omega_max=3.0)
    traj_gen = build_trajectory_generator(
        dt=dt,
        trajectory_type=trajectory_type,
        checkpoint_preset=checkpoint_preset,
        checkpoint_mode=checkpoint_mode,
    )
    lqr = LQRController(Q_diag=[15.0, 15.0, 8.0], R_diag=[0.1, 0.1], dt=dt, v_max=2.0, omega_max=3.0)
    trajectory = traj_gen.generate(duration)
    N = len(trajectory)
    
    x_ref_init, _ = traj_gen.get_reference_at_index(0)
    x_lqr = x_ref_init.copy()
    lqr_states = np.zeros((N, 3))
    lqr_states[0] = x_lqr
    lqr_collisions = 0
    
    for k in range(N - 1):
        x_ref, u_ref = get_reference_point(
            traj_gen,
            trajectory_type,
            x_lqr,
            k,
        )
        u, _ = lqr.compute_control_at_operating_point(x_lqr, x_ref, u_ref)
        x_lqr = robot.simulate_step(x_lqr, u, dt)
        if has_collision(x_lqr, obstacle_field_lqr.actual_obstacles(), 0.3):
            lqr_collisions += 1
        obstacle_field_lqr.step(dt)
        lqr_states[k + 1] = x_lqr

    print(f"LQR collision events: {lqr_collisions}")
    
    # Run MPC (obstacle-aware) with optimized weights
    print("\n--- Running MPC (obstacle-aware) ---")
    traj_gen.generate(duration)  # Reset
    mpc = MPCController(horizon=6, 
                       Q_diag=[15.0, 15.0, 50.0],   # Optimized heading
                       R_diag=[0.1, 0.1], 
                       P_diag=[30.0, 30.0, 40.0],   # Strong terminal heading
                       d_safe=0.3, slack_penalty=5000.0, dt=dt, 
                       v_max=2.0, omega_max=3.0, solver='OSQP')
    
    x_mpc = x_ref_init.copy()  # Use same initial state as LQR
    mpc_states = np.zeros((N, 3))
    mpc_states[0] = x_mpc
    mpc_collisions = 0
    
    for k in range(N - 1):
        x_refs, u_refs = get_reference_segment(
            traj_gen,
            trajectory_type,
            x_mpc,
            k,
            mpc.N + 1,
        )
        controller_obstacles = obstacle_field_mpc.controller_obstacles()
        solution = mpc.solve_with_ltv(x_mpc, x_refs, u_refs, controller_obstacles)
        x_mpc = robot.simulate_step(x_mpc, solution.optimal_control, dt)
        if has_collision(x_mpc, obstacle_field_mpc.actual_obstacles(), 0.3):
            mpc_collisions += 1
        obstacle_field_mpc.step(dt)
        mpc_states[k + 1] = x_mpc

    print(f"MPC collision events: {mpc_collisions}")
    
    # Comparison plot
    viz = Visualizer(output_dir='outputs')
    viz.plot_comparison(lqr_states, mpc_states, trajectory[:, 1:4],
                       obstacle_dicts, d_safe=0.3,
                       title="LQR vs MPC: Obstacle Avoidance Comparison",
                       save_path="outputs/comparison.png")
    
    print("\nComparison plot saved to outputs/comparison.png")


def run_hybrid_simulation(duration: float = 20.0, dt: float = 0.02,
                          visualize: bool = True,
                          scenario: str = "default",
                          actuator_params: ActuatorParams = None,
                          trajectory_type: str = "figure8",
                          checkpoint_preset: str = "diamond",
                          checkpoint_mode: bool = False) -> dict:
    """
    Run risk-aware hybrid LQR-MPC simulation with continuous blending.
    
    Uses BlendingSupervisor for smooth control arbitration:
        u = w(t) * u_mpc + (1 - w(t)) * u_lqr
    
    where w(t) is computed from risk metrics with anti-chatter guarantees.
    
    Args:
        duration: Simulation duration (seconds)
        dt: Time step (seconds)
        visualize: Generate plots
        visualize: Generate plots
        scenario: Obstacle scenario
        actuator_params: Optional parameters for hardware realism (lag, delay, noise)
        
    Returns:
        Dictionary with simulation results including blend weights and jerk metrics
    """
    print("=" * 60)
    print("Hybrid LQR-MPC Smooth Blending Simulation")
    print("=" * 60)
    
    # Initialize components
    robot = DifferentialDriveRobot(v_max=2.0, omega_max=3.0)
    traj_gen = build_trajectory_generator(
        dt=dt,
        trajectory_type=trajectory_type,
        checkpoint_preset=checkpoint_preset,
        checkpoint_mode=checkpoint_mode,
    )
    
    # LQR for low-risk regions (efficient, stable)
    lqr = LQRController(Q_diag=[15.0, 15.0, 8.0], R_diag=[0.1, 0.1], 
                        dt=dt, v_max=2.0, omega_max=3.0)
    
    # MPC for high-risk regions (obstacle-aware, optimal)
    mpc = MPCController(
        horizon=5,
        Q_diag=[80.0, 80.0, 120.0],
        R_diag=[0.1, 0.1],
        P_diag=[20.0, 20.0, 40.0],
        S_diag=[0.1, 0.5],
        J_diag=[0.05, 0.3],     # Second-order jerk penalty (Phase 5A)
        d_safe=0.3,
        slack_penalty=5000.0,
        dt=dt,
        v_max=2.0,
        omega_max=3.0,
        solver='OSQP',
        block_size=2,
        w_max=0.05
    )
    
    # Risk metrics
    risk_metrics = RiskMetrics(
        d_safe=0.3,
        d_trigger=1.0,
        alpha=0.6,
        beta=0.4,
        threshold_low=0.2,
        threshold_medium=0.5
    )
    
    # Blending supervisor (core novelty)
    blender = BlendingSupervisor(
        k_sigmoid=10.0,
        risk_threshold=0.3,
        dw_max=2.0,
        hysteresis_band=0.05,
        solver_time_limit=5.0,
        feasibility_decay=0.8,
        dt=dt
    )
    
    logger = SimulationLogger(log_dir='logs', log_level='INFO', node_name='hybrid_sim')
    
    # Initialize Actuator Dynamics if params are provided
    actuator = None
    if actuator_params:
        actuator = ActuatorDynamics(actuator_params, dt)
    
    obstacle_field = build_obstacle_field(scenario=scenario)
    obstacle_dicts = get_plot_obstacles(obstacle_field)
    scenario_label = f"{scenario} (dynamic)" if scenario in DYNAMIC_SCENARIOS else scenario
    print(f"Scenario: {scenario_label} | Added {len(obstacle_dicts)} obstacles")
    
    # Extract obstacle positions for obstacle-aware checkpoint generation
    obs_positions = np.array(
        [[o['x'], o['y']] for o in obstacle_dicts], dtype=float
    ).reshape(-1, 2) if obstacle_dicts else None

    # Generate trajectory
    trajectory = traj_gen.generate(duration, obstacle_positions=obs_positions)
    N = len(trajectory)
    print(f"Trajectory: {describe_trajectory(trajectory_type, checkpoint_preset)}")
    
    # Initialize state
    x_ref_init, _ = traj_gen.get_reference_at_index(0)
    x = x_ref_init.copy()
    
    # Storage
    states = np.zeros((N, 3))
    controls = np.zeros((N - 1, 2))
    errors = np.zeros((N - 1, 3))
    risk_history = np.zeros((N - 1,))
    blend_weights = np.zeros((N - 1,))
    solve_times = []
    collision_count = 0
    
    states[0] = x
    
    # MPC rate control
    mpc_rate = 5
    mpc_solution = None
    
    print(f"\nStarting smooth blending simulation ({N} steps)...")
    
    for k in range(N - 1):
        # Update checkpoint manager if in checkpoint mode
        if traj_gen.checkpoint_mode:
            t = k * dt
            traj_gen.update_checkpoint_manager(x[:2], t)
        
        # 1. Measurement (add sensor noise if actuator params provided)
        x_measured = x.copy()
        if actuator_params:
            if actuator_params.noise_v_std > 0: # Reuse noise param for state noise proxy
                 # Ideally we'd have separate sensor noise params, but for now reuse
                 x_measured[0] += np.random.normal(0, actuator_params.noise_v_std * dt)
                 x_measured[1] += np.random.normal(0, actuator_params.noise_v_std * dt)
                 x_measured[2] += np.random.normal(0, actuator_params.noise_omega_std * dt)
        
        x_ref, u_ref = get_reference_point(
            traj_gen,
            trajectory_type,
            x_measured,
            k,
        )
        
        # --- Compute risk ---
        risk_obstacles = obstacle_field.risk_obstacles()
        assessment = risk_metrics.assess_risk(x_measured, risk_obstacles)
        risk_history[k] = assessment.combined_risk
        
        # --- Compute LQR control (always available, cheap) ---
        u_lqr, _ = lqr.compute_control_at_operating_point(x_measured, x_ref, u_ref)
        
        # --- Compute MPC control (at lower rate) ---
        solver_status = 'optimal'
        solver_time_ms = 0.0
        
        if k % mpc_rate == 0 or mpc_solution is None:
            x_refs, u_refs = get_reference_segment(
                traj_gen,
                trajectory_type,
                x_measured,
                k,
                mpc.N + 1,
            )
            controller_obstacles = obstacle_field.controller_obstacles()
            mpc_solution = mpc.solve_with_ltv(x_measured, x_refs, u_refs, controller_obstacles)
            solver_status = mpc_solution.status
            solver_time_ms = mpc_solution.solve_time_ms
            solve_times.append(solver_time_ms)
        
        # Use latest MPC solution (or LQR if MPC hasn't run yet)
        if mpc_solution is not None:
            u_mpc = mpc_solution.optimal_control
        else:
            u_mpc = u_lqr  # Fallback before first MPC solve
        
        # --- Blend controls ---
        feas_margin = mpc_solution.feasibility_margin if mpc_solution is not None else 0.0
        u, blend_info = blender.blend(
            u_lqr=u_lqr,
            u_mpc=u_mpc,
            risk=assessment.combined_risk,
            solver_status=solver_status,
            solver_time_ms=solver_time_ms,
            feasibility_margin=feas_margin
        )
        
        blend_weights[k] = blend_info.weight
        
        # --- Compute jerk (from previous controls) ---
        linear_jerk = 0.0
        angular_jerk = 0.0
        if k >= 2:
            dv_curr = (controls[k-1, 0] - controls[k-2, 0]) / dt
            dv_prev = (controls[k-2, 0] - controls[max(0, k-3), 0]) / dt if k >= 3 else 0.0
            linear_jerk = (dv_curr - dv_prev) / dt
            
            domega_curr = (controls[k-1, 1] - controls[k-2, 1]) / dt
            domega_prev = (controls[k-2, 1] - controls[max(0, k-3), 1]) / dt if k >= 3 else 0.0
            angular_jerk = (domega_curr - domega_prev) / dt
        
        # --- Log ---
        error = x - x_ref
        error[2] = robot.normalize_angle(error[2])
        
        logger.log_state(k, x, x_ref, error)
        logger.log_control(k, u, f"HYBRID(w={blend_info.weight:.2f})", solver_time_ms)
        logger.log_hybrid_step(k, blend_info.weight, assessment.combined_risk,
                               blend_info.mode, linear_jerk, angular_jerk)
        
        # --- Simulate ---
        u_applied = u.copy()
        if actuator:
            u_applied[0], u_applied[1] = actuator.update(u[0], u[1])
            
        x = robot.simulate_step(x, u_applied, dt)

        if has_collision(x, obstacle_field.actual_obstacles(), mpc.d_safe):
            collision_count += 1
        obstacle_field.step(dt)
        
        # --- Store ---
        states[k + 1] = x
        controls[k] = u
        errors[k] = error
        
        if k % 100 == 0:
            error_norm = np.linalg.norm(error[:2])
            print(f"  k={k:4d}: risk={assessment.combined_risk:.2f} "
                  f"w={blend_info.weight:.3f} [{blend_info.mode:14s}] "
                  f"err={error_norm:.4f}")
    
    # --- Results ---
    mean_error = np.mean(np.linalg.norm(errors[:, :2], axis=1))
    final_error = np.linalg.norm(errors[-1, :2])
    mean_solve_time = np.mean(solve_times) if solve_times else 0.0
    
    # Jerk metrics
    jerk_metrics = SimulationLogger.compute_jerk_metrics(controls, dt)
    
    # Blending statistics
    blend_stats = blender.get_statistics()
    
    print(f"\nResults:")
    print(f"  Mean tracking error: {mean_error:.4f} m")
    print(f"  Final tracking error: {final_error:.4f} m")
    print(f"  Mean MPC solve time: {mean_solve_time:.2f} ms")
    print(f"  Collision events: {collision_count}")
    print(f"\nBlending Statistics:")
    print(f"  Weight mean: {blend_stats['weight_mean']:.3f}")
    print(f"  LQR-dominant: {100*blend_stats['lqr_dominant_fraction']:.1f}%")
    print(f"  Blended: {100*blend_stats['blended_fraction']:.1f}%")
    print(f"  MPC-dominant: {100*blend_stats['mpc_dominant_fraction']:.1f}%")
    print(f"  Weight transitions (w crosses 0.5): {blend_stats['total_switches']}")
    print(f"\nJerk Metrics:")
    print(f"  Linear jerk  - peak: {jerk_metrics['linear_jerk_peak']:.2f}, "
          f"RMS: {jerk_metrics['linear_jerk_rms']:.2f}, "
          f"p95: {jerk_metrics['linear_jerk_p95']:.2f}")
    print(f"  Angular jerk - peak: {jerk_metrics['angular_jerk_peak']:.2f}, "
          f"RMS: {jerk_metrics['angular_jerk_rms']:.2f}, "
          f"p95: {jerk_metrics['angular_jerk_p95']:.2f}")
    
    logger.finalize()
    
    # --- Visualization ---
    if visualize:
        viz = Visualizer(output_dir='outputs')
        ref_states = trajectory[:, 1:4]
        
        viz.plot_with_obstacles(states, ref_states, obstacle_dicts, mpc.d_safe,
                               title="Hybrid Smooth-Blend Trajectory",
                               save_path="outputs/hybrid_trajectory.png")
        
        viz.plot_tracking_error(errors, dt,
                               title="Hybrid Tracking Error",
                               save_path="outputs/hybrid_error.png")
        
        viz.plot_control_inputs(controls, dt,
                               v_max=2.0, omega_max=3.0,
                               title="Hybrid Control Inputs",
                               save_path="outputs/hybrid_control.png")
        
        # Plot blending weight and risk
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        t = np.arange(N - 1) * dt
        
        # Panel 1: Risk and blend weight
        ax1 = axes[0]
        ax1.plot(t, risk_history, 'b-', linewidth=1.0, alpha=0.7, label='Risk')
        ax1.plot(t, blend_weights, 'r-', linewidth=2.0, label='Blend Weight w(t)')
        ax1.axhline(blender.risk_threshold, color='g', linestyle='--', 
                    alpha=0.4, label=f'Threshold ({blender.risk_threshold})')
        ax1.fill_between(t, 0, blend_weights, alpha=0.15, color='red')
        ax1.set_ylabel('Value')
        ax1.set_title('Continuous Blending: Risk-Based Control Arbitration')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.05, 1.05)
        
        # Panel 2: Control inputs with blend regions
        ax2 = axes[1]
        ax2.plot(t, controls[:, 0], 'b-', linewidth=1.0, label='v (m/s)')
        ax2.plot(t, controls[:, 1], 'r-', linewidth=1.0, label='omega (rad/s)')
        ax2.set_ylabel('Control')
        ax2.set_title('Blended Control Inputs')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Jerk
        ax3 = axes[2]
        if len(controls) >= 3:
            dv = np.diff(controls[:, 0]) / dt
            domega = np.diff(controls[:, 1]) / dt
            lj = np.diff(dv) / dt
            aj = np.diff(domega) / dt
            t_jerk = np.arange(len(lj)) * dt
            ax3.plot(t_jerk, lj, 'b-', linewidth=0.8, alpha=0.7, label='Linear jerk')
            ax3.plot(t_jerk, aj, 'r-', linewidth=0.8, alpha=0.7, label='Angular jerk')
        ax3.set_ylabel('Jerk')
        ax3.set_xlabel('Time (s)')
        ax3.set_title('Control Jerk (d^2u/dt^2)')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig('outputs/hybrid_blending.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print("\nPlots saved to outputs/")
    
    return {
        'states': states,
        'controls': controls,
        'errors': errors,
        'risk_history': risk_history,
        'blend_weights': blend_weights,
        'blend_stats': blend_stats,
        'jerk_metrics': jerk_metrics,
        'mean_error': mean_error,
        'final_error': final_error,
        'collision_count': collision_count,
        'mean_solve_time': mean_solve_time,
    }


def run_adaptive_simulation(duration: float = 20.0, dt: float = 0.02,
                            with_obstacles: bool = True,
                            visualize: bool = True,
                            scenario: str = "default",
                            trajectory_type: str = "figure8",
                            checkpoint_preset: str = "diamond",
                            checkpoint_mode: bool = False) -> dict:
    """
    Run adaptive MPC obstacle avoidance simulation.

    Mirrors run_mpc_simulation but uses AdaptiveMPCController and online
    parameter adaptation after each simulation step.
    """
    print("=" * 60)
    print("Adaptive MPC Obstacle Avoidance Simulation")
    print("=" * 60)

    AdaptiveMPCController = get_adaptive_mpc_class()

    robot = DifferentialDriveRobot(v_max=2.0, omega_max=3.0)
    traj_gen = build_trajectory_generator(
        dt=dt,
        trajectory_type=trajectory_type,
        checkpoint_preset=checkpoint_preset,
        checkpoint_mode=checkpoint_mode,
    )

    adaptive_mpc = AdaptiveMPCController(
        prediction_horizon=6,
        terminal_horizon=4,
        Q_diag=[20.0, 20.0, 40.0],
        R_diag=[0.1, 0.1],
        d_safe=0.3,
        q_xi=1000.0,
        omega_term=8.0,
        dt=dt,
        v_max=2.0,
        omega_max=3.0,
        adaptation_gamma=0.01,
        theta_init=np.array([0.9, 0.9]),
        max_obstacles=10,
    )

    logger = SimulationLogger(log_dir='logs', log_level='INFO', node_name='adaptive_sim')

    obstacle_field = None
    obstacle_dicts = []
    if with_obstacles:
        obstacle_field = build_obstacle_field(scenario=scenario)
        obstacle_dicts = get_plot_obstacles(obstacle_field)
        scenario_label = f"{scenario} (dynamic)" if scenario in DYNAMIC_SCENARIOS else scenario
        print(f"Scenario: {scenario_label} | Added {len(obstacle_dicts)} obstacles")
    else:
        print("Scenario: none | Obstacles disabled")

    obs_positions = np.array(
        [[o['x'], o['y']] for o in obstacle_dicts], dtype=float
    ).reshape(-1, 2) if obstacle_dicts else None

    trajectory = traj_gen.generate(duration, obstacle_positions=obs_positions)
    N = len(trajectory)

    print(
        f"Generated {N} trajectory points over {duration}s "
        f"using {describe_trajectory(trajectory_type, checkpoint_preset)}"
    )

    x_ref_init, _ = traj_gen.get_reference_at_index(0)
    x = x_ref_init.copy()

    states = np.zeros((N, 3))
    controls = np.zeros((N - 1, 2))
    errors = np.zeros((N - 1, 3))
    param_history = np.zeros((N, 2))
    solve_times = []
    collision_count = 0

    states[0] = x
    param_history[0] = adaptive_mpc.param_estimates

    mpc_rate = 5
    solution = None

    for k in range(N - 1):
        # Update checkpoint manager if in checkpoint mode
        if traj_gen.checkpoint_mode:
            t = k * dt
            traj_gen.update_checkpoint_manager(x[:2], t)
        
        x_refs, u_refs = get_reference_segment(
            traj_gen,
            trajectory_type,
            x,
            k,
            adaptive_mpc.N_ext + 1,
        )
        controller_obstacles = obstacle_field.controller_obstacles() if obstacle_field else []

        solve_time = 0.0
        if k % mpc_rate == 0 or solution is None:
            solution = adaptive_mpc.solve_tracking(
                x,
                x_refs,
                u_refs[:adaptive_mpc.N_ext],
                controller_obstacles,
            )
            solve_time = solution.solve_time_ms
            solve_times.append(solve_time)
            if solution.slack_used:
                logger.log_constraint_event(k, 'slack_activated', {'reason': 'feasibility'})

        u = solution.optimal_control
        error = x - x_refs[0]
        error[2] = robot.normalize_angle(error[2])

        logger.log_state(k, x, x_refs[0], error)
        logger.log_control(k, u, 'AdaptiveMPC', solve_time if solve_time > 0.0 else None)

        x_prev = x.copy()
        theta_prev = adaptive_mpc.param_estimates
        x = robot.simulate_step(x, u, dt)
        adaptive_mpc.adapt_parameters(x_measured=x, x_prev=x_prev, u_prev=u)
        theta_now = adaptive_mpc.param_estimates
        param_history[k + 1] = theta_now

        if np.linalg.norm(theta_now - theta_prev) > 1e-8 and solve_time > 0.0:
            logger.log_parameter_change('theta_hat', theta_prev, theta_now, source='adaptive_mpc')

        if obstacle_field:
            if has_collision(x, obstacle_field.actual_obstacles(), adaptive_mpc.d_safe):
                collision_count += 1
            obstacle_field.step(dt)

        states[k + 1] = x
        controls[k] = u
        errors[k] = error

        if k % 100 == 0:
            error_norm = np.linalg.norm(error[:2])
            print(
                f"  k={k:4d}: error_norm={error_norm:.4f}, "
                f"theta=[{theta_now[0]:.3f}, {theta_now[1]:.3f}]"
            )

    mean_error = np.mean(np.linalg.norm(errors[:, :2], axis=1))
    final_error = np.linalg.norm(errors[-1, :2])
    mean_solve_time = np.mean(solve_times) if solve_times else 0.0

    print(f"\nResults:")
    print(f"  Mean tracking error: {mean_error:.4f} m")
    print(f"  Final tracking error: {final_error:.4f} m")
    print(f"  Mean Adaptive MPC solve time: {mean_solve_time:.2f} ms")
    print(f"  Final parameter estimate: [{param_history[-1, 0]:.4f}, {param_history[-1, 1]:.4f}]")
    if with_obstacles:
        print(f"  Collision events: {collision_count}")

    logger.finalize()

    if visualize:
        viz = Visualizer(output_dir='outputs')
        ref_states = trajectory[:, 1:4]

        if with_obstacles:
            viz.plot_with_obstacles(
                states,
                ref_states,
                obstacle_dicts,
                adaptive_mpc.d_safe,
                title="Adaptive MPC Obstacle Avoidance",
                save_path="outputs/adaptive_obstacle_avoidance.png",
            )
        else:
            viz.plot_trajectory(
                states,
                ref_states,
                title="Adaptive MPC Trajectory Tracking",
                save_path="outputs/adaptive_tracking.png",
            )

        viz.plot_tracking_error(
            errors,
            dt,
            title="Adaptive MPC Tracking Error",
            save_path="outputs/adaptive_error.png",
        )

        viz.plot_control_inputs(
            controls,
            dt,
            v_max=2.0,
            omega_max=3.0,
            title="Adaptive MPC Control Inputs",
            save_path="outputs/adaptive_control.png",
        )

        import matplotlib.pyplot as plt

        t_params = np.arange(N) * dt
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(t_params, param_history[:, 0], 'b-', linewidth=1.8, label='v_scale')
        axes[0].axhline(1.0, color='k', linestyle='--', linewidth=1.0, alpha=0.6)
        axes[0].set_ylabel('Theta[0]')
        axes[0].set_title('Adaptive MPC Parameter Convergence')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='best')

        axes[1].plot(t_params, param_history[:, 1], 'r-', linewidth=1.8, label='omega_scale')
        axes[1].axhline(1.0, color='k', linestyle='--', linewidth=1.0, alpha=0.6)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Theta[1]')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc='best')

        plt.tight_layout()
        fig.savefig('outputs/adaptive_parameter_convergence.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        print("\nPlots saved to outputs/")

    return {
        'states': states,
        'controls': controls,
        'errors': errors,
        'reference': trajectory[:, 1:4],
        'param_history': param_history,
        'mean_error': mean_error,
        'final_error': final_error,
        'collision_count': collision_count,
        'mean_solve_time': mean_solve_time,
    }


def run_hybrid_adaptive_simulation(duration: float = 20.0, dt: float = 0.02,
                                   visualize: bool = True,
                                   scenario: str = "default",
                                   actuator_params: ActuatorParams = None,
                                   trajectory_type: str = "figure8",
                                   checkpoint_preset: str = "diamond",
                                   checkpoint_mode: bool = False) -> dict:
    """
    Run risk-aware hybrid LQR + Adaptive MPC simulation with smooth blending.
    """
    print("=" * 60)
    print("Hybrid LQR + Adaptive MPC Smooth Blending Simulation")
    print("=" * 60)

    AdaptiveMPCController = get_adaptive_mpc_class()

    robot = DifferentialDriveRobot(v_max=2.0, omega_max=3.0)
    traj_gen = build_trajectory_generator(
        dt=dt,
        trajectory_type=trajectory_type,
        checkpoint_preset=checkpoint_preset,
        checkpoint_mode=checkpoint_mode,
    )

    lqr = LQRController(
        Q_diag=[15.0, 15.0, 8.0],
        R_diag=[0.1, 0.1],
        dt=dt,
        v_max=2.0,
        omega_max=3.0,
    )

    adaptive_mpc = AdaptiveMPCController(
        prediction_horizon=6,
        terminal_horizon=4,
        Q_diag=[40.0, 40.0, 80.0],
        R_diag=[0.1, 0.1],
        d_safe=0.3,
        q_xi=1000.0,
        omega_term=8.0,
        dt=dt,
        v_max=2.0,
        omega_max=3.0,
        adaptation_gamma=0.01,
        theta_init=np.array([0.9, 0.9]),
        max_obstacles=10,
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
        solver_time_limit=8.0,
        feasibility_decay=0.8,
        dt=dt,
    )

    logger = SimulationLogger(log_dir='logs', log_level='INFO', node_name='hybrid_adaptive_sim')

    actuator = None
    if actuator_params:
        actuator = ActuatorDynamics(actuator_params, dt)

    obstacle_field = build_obstacle_field(scenario=scenario)
    obstacle_dicts = get_plot_obstacles(obstacle_field)
    scenario_label = f"{scenario} (dynamic)" if scenario in DYNAMIC_SCENARIOS else scenario
    print(f"Scenario: {scenario_label} | Added {len(obstacle_dicts)} obstacles")

    obs_positions = np.array(
        [[o['x'], o['y']] for o in obstacle_dicts], dtype=float
    ).reshape(-1, 2) if obstacle_dicts else None

    trajectory = traj_gen.generate(duration, obstacle_positions=obs_positions)
    N = len(trajectory)
    print(f"Trajectory: {describe_trajectory(trajectory_type, checkpoint_preset)}")

    x_ref_init, _ = traj_gen.get_reference_at_index(0)
    x = x_ref_init.copy()

    states = np.zeros((N, 3))
    controls = np.zeros((N - 1, 2))
    errors = np.zeros((N - 1, 3))
    risk_history = np.zeros((N - 1,))
    blend_weights = np.zeros((N - 1,))
    param_history = np.zeros((N, 2))
    solve_times = []
    collision_count = 0

    states[0] = x
    param_history[0] = adaptive_mpc.param_estimates

    mpc_rate = 5
    adaptive_solution = None

    print(f"\nStarting hybrid-adaptive blending simulation ({N} steps)...")

    for k in range(N - 1):
        # Update checkpoint manager if in checkpoint mode
        if traj_gen.checkpoint_mode:
            t = k * dt
            traj_gen.update_checkpoint_manager(x[:2], t)
        
        x_measured = x.copy()
        if actuator_params and actuator_params.noise_v_std > 0:
            x_measured[0] += np.random.normal(0, actuator_params.noise_v_std * dt)
            x_measured[1] += np.random.normal(0, actuator_params.noise_v_std * dt)
            x_measured[2] += np.random.normal(0, actuator_params.noise_omega_std * dt)

        x_ref, u_ref = get_reference_point(
            traj_gen,
            trajectory_type,
            x_measured,
            k,
        )

        risk_obstacles = obstacle_field.risk_obstacles()
        assessment = risk_metrics.assess_risk(x_measured, risk_obstacles)
        risk_history[k] = assessment.combined_risk

        u_lqr, _ = lqr.compute_control_at_operating_point(x_measured, x_ref, u_ref)

        solver_status = 'optimal'
        solver_time_ms = 0.0
        if k % mpc_rate == 0 or adaptive_solution is None:
            x_refs, u_refs = get_reference_segment(
                traj_gen,
                trajectory_type,
                x_measured,
                k,
                adaptive_mpc.N_ext + 1,
            )
            controller_obstacles = obstacle_field.controller_obstacles()
            adaptive_solution = adaptive_mpc.solve_tracking(
                x_measured,
                x_refs,
                u_refs[:adaptive_mpc.N_ext],
                controller_obstacles,
            )
            solver_status = adaptive_solution.status
            solver_time_ms = adaptive_solution.solve_time_ms
            solve_times.append(solver_time_ms)
        elif adaptive_solution is not None:
            solver_status = adaptive_solution.status

        u_adaptive = adaptive_solution.optimal_control if adaptive_solution is not None else u_lqr
        feas_margin = adaptive_solution.feasibility_margin if adaptive_solution is not None else 0.0

        u, blend_info = blender.blend(
            u_lqr=u_lqr,
            u_mpc=u_adaptive,
            risk=assessment.combined_risk,
            solver_status=solver_status,
            solver_time_ms=solver_time_ms,
            feasibility_margin=feas_margin,
        )
        blend_weights[k] = blend_info.weight

        linear_jerk = 0.0
        angular_jerk = 0.0
        if k >= 2:
            dv_curr = (controls[k - 1, 0] - controls[k - 2, 0]) / dt
            dv_prev = (controls[k - 2, 0] - controls[max(0, k - 3), 0]) / dt if k >= 3 else 0.0
            linear_jerk = (dv_curr - dv_prev) / dt

            domega_curr = (controls[k - 1, 1] - controls[k - 2, 1]) / dt
            domega_prev = (controls[k - 2, 1] - controls[max(0, k - 3), 1]) / dt if k >= 3 else 0.0
            angular_jerk = (domega_curr - domega_prev) / dt

        error = x - x_ref
        error[2] = robot.normalize_angle(error[2])
        logger.log_state(k, x, x_ref, error)
        logger.log_control(k, u, f"HYBRID_ADAPTIVE(w={blend_info.weight:.2f})", solver_time_ms)
        logger.log_hybrid_step(
            k,
            blend_info.weight,
            assessment.combined_risk,
            blend_info.mode,
            linear_jerk,
            angular_jerk,
        )

        x_prev = x.copy()
        theta_prev = adaptive_mpc.param_estimates
        u_applied = u.copy()
        if actuator:
            u_applied[0], u_applied[1] = actuator.update(u[0], u[1])
        x = robot.simulate_step(x, u_applied, dt)

        adaptive_mpc.adapt_parameters(x_measured=x, x_prev=x_prev, u_prev=u_applied)
        theta_now = adaptive_mpc.param_estimates
        param_history[k + 1] = theta_now

        if np.linalg.norm(theta_now - theta_prev) > 1e-8 and solver_time_ms > 0.0:
            logger.log_parameter_change('theta_hat', theta_prev, theta_now, source='hybrid_adaptive')

        if has_collision(x, obstacle_field.actual_obstacles(), adaptive_mpc.d_safe):
            collision_count += 1
        obstacle_field.step(dt)

        states[k + 1] = x
        controls[k] = u
        errors[k] = error

        if k % 100 == 0:
            error_norm = np.linalg.norm(error[:2])
            print(
                f"  k={k:4d}: risk={assessment.combined_risk:.2f} "
                f"w={blend_info.weight:.3f} [{blend_info.mode:14s}] "
                f"err={error_norm:.4f} "
                f"theta=[{theta_now[0]:.3f}, {theta_now[1]:.3f}]"
            )

    mean_error = np.mean(np.linalg.norm(errors[:, :2], axis=1))
    final_error = np.linalg.norm(errors[-1, :2])
    mean_solve_time = np.mean(solve_times) if solve_times else 0.0
    jerk_metrics = SimulationLogger.compute_jerk_metrics(controls, dt)
    blend_stats = blender.get_statistics()

    print(f"\nResults:")
    print(f"  Mean tracking error: {mean_error:.4f} m")
    print(f"  Final tracking error: {final_error:.4f} m")
    print(f"  Mean Adaptive MPC solve time: {mean_solve_time:.2f} ms")
    print(f"  Final parameter estimate: [{param_history[-1, 0]:.4f}, {param_history[-1, 1]:.4f}]")
    print(f"  Collision events: {collision_count}")
    print(f"\nBlending Statistics:")
    print(f"  Weight mean: {blend_stats['weight_mean']:.3f}")
    print(f"  LQR-dominant: {100 * blend_stats['lqr_dominant_fraction']:.1f}%")
    print(f"  Blended: {100 * blend_stats['blended_fraction']:.1f}%")
    print(f"  MPC-dominant: {100 * blend_stats['mpc_dominant_fraction']:.1f}%")
    print(f"  Weight transitions (w crosses 0.5): {blend_stats['total_switches']}")
    print(f"\nJerk Metrics:")
    print(
        f"  Linear jerk  - peak: {jerk_metrics['linear_jerk_peak']:.2f}, "
        f"RMS: {jerk_metrics['linear_jerk_rms']:.2f}, "
        f"p95: {jerk_metrics['linear_jerk_p95']:.2f}"
    )
    print(
        f"  Angular jerk - peak: {jerk_metrics['angular_jerk_peak']:.2f}, "
        f"RMS: {jerk_metrics['angular_jerk_rms']:.2f}, "
        f"p95: {jerk_metrics['angular_jerk_p95']:.2f}"
    )

    logger.finalize()

    if visualize:
        viz = Visualizer(output_dir='outputs')
        ref_states = trajectory[:, 1:4]

        viz.plot_with_obstacles(
            states,
            ref_states,
            obstacle_dicts,
            adaptive_mpc.d_safe,
            title="Hybrid LQR + Adaptive MPC Trajectory",
            save_path="outputs/hybrid_adaptive_trajectory.png",
        )

        viz.plot_tracking_error(
            errors,
            dt,
            title="Hybrid Adaptive Tracking Error",
            save_path="outputs/hybrid_adaptive_error.png",
        )

        viz.plot_control_inputs(
            controls,
            dt,
            v_max=2.0,
            omega_max=3.0,
            title="Hybrid Adaptive Control Inputs",
            save_path="outputs/hybrid_adaptive_control.png",
        )

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        t = np.arange(N - 1) * dt

        ax1 = axes[0]
        ax1.plot(t, risk_history, 'b-', linewidth=1.0, alpha=0.7, label='Risk')
        ax1.plot(t, blend_weights, 'r-', linewidth=2.0, label='Blend Weight w(t)')
        ax1.axhline(
            blender.risk_threshold,
            color='g',
            linestyle='--',
            alpha=0.4,
            label=f'Threshold ({blender.risk_threshold})',
        )
        ax1.fill_between(t, 0, blend_weights, alpha=0.15, color='red')
        ax1.set_ylabel('Value')
        ax1.set_title('Hybrid Adaptive: Risk-Based Control Arbitration')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.05, 1.05)

        ax2 = axes[1]
        ax2.plot(t, controls[:, 0], 'b-', linewidth=1.0, label='v (m/s)')
        ax2.plot(t, controls[:, 1], 'r-', linewidth=1.0, label='omega (rad/s)')
        ax2.set_ylabel('Control')
        ax2.set_title('Blended Control Inputs')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        ax3 = axes[2]
        if len(controls) >= 3:
            dv = np.diff(controls[:, 0]) / dt
            domega = np.diff(controls[:, 1]) / dt
            lj = np.diff(dv) / dt
            aj = np.diff(domega) / dt
            t_jerk = np.arange(len(lj)) * dt
            ax3.plot(t_jerk, lj, 'b-', linewidth=0.8, alpha=0.7, label='Linear jerk')
            ax3.plot(t_jerk, aj, 'r-', linewidth=0.8, alpha=0.7, label='Angular jerk')
        ax3.set_ylabel('Jerk')
        ax3.set_xlabel('Time (s)')
        ax3.set_title('Control Jerk (d^2u/dt^2)')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig('outputs/hybrid_adaptive_blending.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        t_params = np.arange(N) * dt
        fig_params, ax_params = plt.subplots(1, 1, figsize=(10, 4.5))
        ax_params.plot(t_params, param_history[:, 0], 'b-', linewidth=1.8, label='v_scale')
        ax_params.plot(t_params, param_history[:, 1], 'r-', linewidth=1.8, label='omega_scale')
        ax_params.axhline(1.0, color='k', linestyle='--', linewidth=1.0, alpha=0.6)
        ax_params.set_xlabel('Time (s)')
        ax_params.set_ylabel('Estimated parameter')
        ax_params.set_title('Hybrid Adaptive Parameter Convergence')
        ax_params.grid(True, alpha=0.3)
        ax_params.legend(loc='best')
        plt.tight_layout()
        fig_params.savefig('outputs/hybrid_adaptive_parameter_convergence.png', dpi=150, bbox_inches='tight')
        plt.close(fig_params)

        print("\nPlots saved to outputs/")

    return {
        'states': states,
        'controls': controls,
        'errors': errors,
        'risk_history': risk_history,
        'blend_weights': blend_weights,
        'param_history': param_history,
        'blend_stats': blend_stats,
        'jerk_metrics': jerk_metrics,
        'mean_error': mean_error,
        'final_error': final_error,
        'collision_count': collision_count,
        'mean_solve_time': mean_solve_time,
    }


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run hybrid LQR-MPC simulation')
    parser.add_argument('--mode', type=str,
                        choices=['lqr', 'mpc', 'compare', 'hybrid', 'adaptive', 'hybrid_adaptive'],
                        default='lqr', help='Simulation mode')
    parser.add_argument('--duration', type=float, default=20.0, help='Duration in seconds')
    parser.add_argument('--dt', type=float, default=0.02, help='Time step')
    parser.add_argument('--scenario', type=str, choices=SCENARIO_CHOICES,
                        default='default', help='Obstacle scenario')
    parser.add_argument('--trajectory', type=str, choices=TRAJECTORY_CHOICES,
                        default='figure8',
                        help='Reference trajectory family')
    parser.add_argument('--checkpoint-preset', type=str,
                        choices=CHECKPOINT_PRESET_CHOICES,
                        default='diamond',
                        help='Checkpoint preset when --trajectory checkpoint_path is used')
    parser.add_argument('--checkpoint-mode', action='store_true',
                        help='Enable checkpoint-based tracking (adaptive switching)')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    parser.add_argument('--realistic', action='store_true', help='Enable actuator dynamics (lag, delay, noise)')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Configure realism
    act_params = None
    if args.realistic:
        act_params = ActuatorParams(
            tau_v=0.1, tau_omega=0.1,
            delay_steps=1,
            noise_v_std=0.02, noise_omega_std=0.05
        )
    
    if args.mode == 'lqr':
        run_lqr_simulation(
            duration=args.duration,
            dt=args.dt,
            visualize=not args.no_plot,
            trajectory_type=args.trajectory,
            checkpoint_preset=args.checkpoint_preset,
            checkpoint_mode=args.checkpoint_mode,
        )
    elif args.mode == 'mpc':
        run_mpc_simulation(
            duration=args.duration,
            dt=args.dt,
            visualize=not args.no_plot,
            scenario=args.scenario,
            trajectory_type=args.trajectory,
            checkpoint_preset=args.checkpoint_preset,
            checkpoint_mode=args.checkpoint_mode,
        )
    elif args.mode == 'compare':
        run_comparison(
            duration=args.duration,
            dt=args.dt,
            scenario=args.scenario,
            trajectory_type=args.trajectory,
            checkpoint_preset=args.checkpoint_preset,
            checkpoint_mode=args.checkpoint_mode,
        )
    elif args.mode == 'hybrid':
        run_hybrid_simulation(
            duration=args.duration,
            dt=args.dt,
            visualize=not args.no_plot,
            scenario=args.scenario,
            actuator_params=act_params,
            trajectory_type=args.trajectory,
            checkpoint_preset=args.checkpoint_preset,
            checkpoint_mode=args.checkpoint_mode,
        )
    elif args.mode == 'adaptive':
        run_adaptive_simulation(
            duration=args.duration,
            dt=args.dt,
            visualize=not args.no_plot,
            scenario=args.scenario,
            trajectory_type=args.trajectory,
            checkpoint_preset=args.checkpoint_preset,
            checkpoint_mode=args.checkpoint_mode,
        )
    elif args.mode == 'hybrid_adaptive':
        run_hybrid_adaptive_simulation(
            duration=args.duration,
            dt=args.dt,
            visualize=not args.no_plot,
            scenario=args.scenario,
            actuator_params=act_params,
            trajectory_type=args.trajectory,
            checkpoint_preset=args.checkpoint_preset,
            checkpoint_mode=args.checkpoint_mode,
        )
    
    print("\nSimulation complete!")

    # CasADi may crash during interpreter teardown on some Windows builds.
    # Force a clean process exit after simulations are fully finalized.
    if os.name == 'nt' and args.mode in {'adaptive', 'hybrid_adaptive'}:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == '__main__':
    main()

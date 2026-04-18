"""
Statistical Validation Framework
=================================

Monte Carlo runner for comparing controller modes under randomized
obstacle configurations, sensor noise, and control latency.

Usage:
    python evaluation/statistical_runner.py --configs 50 --modes hybrid lqr mpc
    python evaluation/statistical_runner.py --configs 10 --modes adaptive hybrid_adaptive checkpoint
    python evaluation/statistical_runner.py --configs 100 --noise 0.01 --delay 20

Output:
    - Console table: mean ± std for each metric × controller
    - CSV: evaluation/results/statistical_results.csv
    - JSON: evaluation/results/statistical_results.json
"""

import sys
import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple

# Default testing modes - automatically include advanced controllers
DEFAULT_MODES = ['hybrid_adaptive', 'adaptive', 'checkpoint', 'hybrid', 'lqr', 'mpc']

# Add project root and source directories to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(project_root, 'src', 'hybrid_controller')
for p in [project_root, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from hybrid_controller.models.differential_drive import DifferentialDriveRobot
from hybrid_controller.controllers.lqr_controller import LQRController
from hybrid_controller.models.actuator_dynamics import ActuatorDynamics, ActuatorParams
from hybrid_controller.controllers.mpc_controller import Obstacle
from hybrid_controller.controllers.cvxpygen_solver import CVXPYgenWrapper
from hybrid_controller.models.linearization import Linearizer
from hybrid_controller.controllers.hybrid_blender import BlendingSupervisor
from hybrid_controller.controllers.risk_metrics import RiskMetrics
from hybrid_controller.trajectory.reference_generator import ReferenceTrajectoryGenerator
from hybrid_controller.logging.simulation_logger import SimulationLogger
from evaluation.scenarios import (
    ObstacleConfig, InflationConfig, get_generator,
    get_baseline_static_scenario, get_urban_dynamic_scenario,
    get_stochastic_navigation_scenario, get_oscillatory_tracking_scenario,
    get_vehicle_realistic_scenario
)


BLENDED_MODES = {'hybrid', 'hybrid_adaptive'}
ADAPTIVE_MODES = {'adaptive', 'hybrid_adaptive'}
DEFAULT_INFLATION = InflationConfig(
    safety_factor=1.0,
    sensing_factor=0.05,
    motion_lookahead=0.5,
)
_ADAPTIVE_MPC_CLASS = None


def get_adaptive_mpc_class():
    """Lazy import so adaptive modes remain optional."""
    global _ADAPTIVE_MPC_CLASS
    if _ADAPTIVE_MPC_CLASS is not None:
        return _ADAPTIVE_MPC_CLASS

    try:
        from hybrid_controller.controllers.adaptive_mpc_controller import AdaptiveMPCController
    except Exception as exc:
        raise RuntimeError(
            "Adaptive modes require CasADi. Install optional dependency with: pip install casadi"
        ) from exc

    _ADAPTIVE_MPC_CLASS = AdaptiveMPCController
    return _ADAPTIVE_MPC_CLASS


# -- Data classes ----------------------------------------------------

@dataclass 
class NoiseConfig:
    """Sensor noise, actuator lag, and control latency parameters."""
    position_noise_std: float = 0.0    # Sensor noise (meters)
    heading_noise_std: float = 0.0     # Sensor noise (radians)
    control_delay_steps: int = 0       # Pipeline delay
    tau_v: float = 0.0                 # Actuator lag (time constant)
    tau_omega: float = 0.0             # Actuator lag
    actuator_noise_std: float = 0.0    # Execution noise (v/omega)
    
    @property
    def delay_ms(self) -> float:
        return self.control_delay_steps * 20.0  # assumes dt=0.02
    
    def to_actuator_params(self) -> ActuatorParams:
        return ActuatorParams(
            tau_v=self.tau_v, 
            tau_omega=self.tau_omega,
            noise_v_std=self.actuator_noise_std,
            noise_omega_std=self.actuator_noise_std,
            delay_steps=self.control_delay_steps
        )


@dataclass
class RunMetrics:
    """Collected metrics from a single simulation run."""
    mode: str
    config_id: int
    seed: int
    mean_tracking_error: float
    max_tracking_error: float  
    final_tracking_error: float
    mean_solve_time_ms: float
    max_solve_time_ms: float
    linear_jerk_rms: float
    angular_jerk_rms: float
    linear_jerk_peak: float
    angular_jerk_peak: float
    infeasible_count: int
    collision_count: int
    completion_fraction: float  # How much of trajectory was completed
    total_control_effort: float
    blend_weight_mean: float = 0.0
    blend_weight_std: float = 0.0
    smooth_transitions: int = 0
    wall_time_s: float = 0.0
    # Checkpoint metrics
    checkpoint_completion_rate: float = 0.0
    mean_time_to_checkpoint: float = 0.0
    mean_checkpoint_overshoot: float = 0.0
    checkpoints_reached: int = 0
    checkpoints_missed: int = 0
    tracking_mode: str = 'continuous'
    

@dataclass
class AggregatedResults:
    """Statistics across multiple runs for one controller mode."""
    mode: str
    n_runs: int
    metrics: Dict[str, Dict[str, float]]  # metric_name -> {mean, std, min, max, p5, p95}





# -- Single simulation runner ----------------------------------------

def run_single_config(mode: str, 
                       obstacle_config: ObstacleConfig,
                       noise_config: NoiseConfig,
                       duration: float = 20.0,
                       dt: float = 0.02,
                       config_id: int = 0,
                       trajectory_type: str = 'figure8',
                       checkpoint_mode: bool = False,
                       tuning: Optional[Dict[str, Any]] = None) -> RunMetrics:
    """
    Run a single simulation with given controller mode and obstacle config.
    
    Modes:
        - 'lqr':              Pure LQR (ignores obstacles)
        - 'mpc':              Pure MPC
        - 'hard_switch':      Hard LQR/MPC switching (threshold-based)
        - 'hybrid':           Smooth blending (BlendingSupervisor)
        - 'adaptive':         Adaptive MPC with online parameter estimation
        - 'hybrid_adaptive':  Smooth blending between LQR and Adaptive MPC
        
    Args:
        mode: Controller mode string
        obstacle_config: Randomized obstacle layout
        noise_config: Noise and latency parameters
        duration: Simulation duration (seconds)
        dt: Time step (seconds)
        config_id: Config index for identification
        trajectory_type: Trajectory type (Task 11.1)
        checkpoint_mode: Enable checkpoint-based tracking (Task 11.1)
        
    Returns:
        RunMetrics with all collected performance data
    """
    start_wall = time.perf_counter()
    
    rng = np.random.RandomState(obstacle_config.seed)

    # -- Initialize components --
    tuning = tuning or {}

    mpc_horizon = int(tuning.get('mpc_horizon', 5))
    mpc_q_diag = tuning.get('mpc_q_diag', [80.0, 80.0, 120.0])
    mpc_r_diag = tuning.get('mpc_r_diag', [0.1, 0.1])
    mpc_p_diag = tuning.get('mpc_p_diag', [20.0, 20.0, 40.0])
    mpc_s_diag = tuning.get('mpc_s_diag', [0.1, 0.5])
    mpc_j_diag = tuning.get('mpc_j_diag', [0.05, 0.3])

    risk_d_safe = float(tuning.get('risk_d_safe', 0.3))
    risk_d_trigger = float(tuning.get('risk_d_trigger', 1.0))
    risk_threshold_low = float(tuning.get('risk_threshold_low', 0.2))
    risk_threshold_medium = float(tuning.get('risk_threshold_medium', 0.5))

    blend_k_sigmoid = float(tuning.get('blend_k_sigmoid', 10.0))
    blend_risk_threshold = float(tuning.get('blend_risk_threshold', 0.3))
    blend_dw_max = float(tuning.get('blend_dw_max', 2.0))
    blend_hysteresis_band = float(tuning.get('blend_hysteresis_band', 0.05))
    robot = DifferentialDriveRobot(v_max=2.0, omega_max=3.0)
    # Task 11.1: Initialize trajectory generator with checkpoint mode support
    traj_gen = ReferenceTrajectoryGenerator(
        A=2.0, a=0.5, dt=dt, T_blend=0.5,
        trajectory_type=trajectory_type,
        checkpoint_mode=checkpoint_mode
    )
    
    lqr = LQRController(Q_diag=[15.0, 15.0, 8.0], R_diag=[0.1, 0.1],
                         dt=dt, v_max=2.0, omega_max=3.0)
    
    mpc = None
    linearizer = None
    if mode in {'mpc', 'hard_switch', 'hybrid'}:
        mpc = CVXPYgenWrapper(
            horizon=mpc_horizon, Q_diag=mpc_q_diag, R_diag=mpc_r_diag,
            P_diag=mpc_p_diag, S_diag=mpc_s_diag, J_diag=mpc_j_diag,
            v_max=2.0, omega_max=3.0, solver_name='OSQP'
        )
        linearizer = Linearizer(dt=dt)

    adaptive_mpc = None
    if mode in ADAPTIVE_MODES:
        AdaptiveMPCController = get_adaptive_mpc_class()
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
    
    risk_metrics = RiskMetrics(
        d_safe=risk_d_safe, d_trigger=risk_d_trigger, alpha=0.6, beta=0.4,
        threshold_low=risk_threshold_low, threshold_medium=risk_threshold_medium
    )
    
    blender = BlendingSupervisor(
        k_sigmoid=blend_k_sigmoid, risk_threshold=blend_risk_threshold, dw_max=blend_dw_max,
        hysteresis_band=blend_hysteresis_band, solver_time_limit=5.0,
        feasibility_decay=0.8, dt=dt
    )
    
    obstacle_field = obstacle_config.create_field(
        inflation=DEFAULT_INFLATION,
        seed=obstacle_config.seed,
    )
    
    # Extract obstacle positions for obstacle-aware checkpoint generation
    obs_positions = np.array(
        [[obs.x, obs.y] for obs in obstacle_field.actual_obstacles()],
        dtype=float,
    ).reshape(-1, 2) if obstacle_field.actual_obstacles() else np.empty((0, 2))

    # -- Generate trajectory --
    trajectory = traj_gen.generate(duration, obstacle_positions=obs_positions)
    N = len(trajectory)
    
    checkpoint_manager = None
    if checkpoint_mode and hasattr(traj_gen, 'checkpoint_manager'):
        checkpoint_manager = traj_gen.checkpoint_manager
    
    x_ref_init, _ = traj_gen.get_reference_at_index(0)
    x = np.array([x_ref_init[0], x_ref_init[1], x_ref_init[2]])
    
    # Storage
    states = np.zeros((N, 3))
    controls = np.zeros((N - 1, 2))
    errors = np.zeros(N)
    solve_times = []
    infeasible_count = 0
    collision_count = 0
    blend_weights = np.zeros(N - 1) if mode in BLENDED_MODES else None
    
    # Control delay buffer managed by ActuatorDynamics now
    actuator = ActuatorDynamics(noise_config.to_actuator_params(), dt)
    
    # MPC rate control
    mpc_rate = 5  # Run MPC every 5 steps
    mpc_solution = None
    adaptive_solution = None
    solver_status = 'optimal'
    solver_time_ms = 0.0
    
    # -- Simulation loop --
    for k in range(N):
        controller_obstacles = obstacle_field.controller_obstacles()
        risk_obstacles = obstacle_field.risk_obstacles()
        collision_obstacles = obstacle_field.actual_obstacles()

        # Apply noise to state measurement
        x_measured = x.copy()
        if noise_config.position_noise_std > 0:
            x_measured[0] += rng.normal(0, noise_config.position_noise_std)
            x_measured[1] += rng.normal(0, noise_config.position_noise_std)
        if noise_config.heading_noise_std > 0:
            x_measured[2] += rng.normal(0, noise_config.heading_noise_std)
        
        states[k] = x
        
        if checkpoint_manager is not None:
            current_time = k * dt
            num_obs = len(controller_obstacles)
            checkpoint_manager.update(x[:2], current_time, num_obs)
        
        # Reference — use state-anchored local segment in checkpoint mode
        if checkpoint_manager is not None:
            _xr, _ur = traj_gen.get_local_trajectory_segment(x_measured, 1)
            x_ref = _xr[0]
            x_ref_dot = _ur[0] if _ur.shape[0] > 0 else traj_gen.get_reference_at_index(k)[1]
        else:
            x_ref, x_ref_dot = traj_gen.get_reference_at_index(k)
        error = robot.compute_tracking_error(x_measured, x_ref)
        errors[k] = np.linalg.norm(error[:2])
        
        if k >= N - 1:
            break
        
        # -- Helper: get trajectory segment (state-anchored when checkpoint mode) --
        def _get_segment(horizon):
            if checkpoint_manager is not None:
                return traj_gen.get_local_trajectory_segment(x_measured, horizon)
            return traj_gen.get_trajectory_segment(k, horizon)

        # -- Compute controls based on mode --
        if mode == 'lqr':
            u = lqr.compute_control(x_measured, x_ref, x_ref_dot)
        
        elif mode == 'mpc':
            x_refs, u_refs = _get_segment(mpc.N + 1)
            
            v_ref = u_refs[0, 0] if abs(u_refs[0, 0]) > 0.01 else 0.1
            A_d, B_d = linearizer.get_discrete_model_explicit(v_ref, x_refs[0, 2])
            
            sol = mpc.solve_fast(x_measured, x_refs, A_d, B_d)
            u = sol.optimal_control
            solve_times.append(sol.solve_time_ms)
            if sol.status not in ('optimal', 'optimal_inaccurate'):
                infeasible_count += 1
        
        elif mode == 'hard_switch':
            assessment = risk_metrics.assess_risk(x_measured, risk_obstacles)
            
            if assessment.combined_risk > 0.3:
                x_refs, u_refs = _get_segment(mpc.N + 1)
                
                v_ref = u_refs[0, 0] if abs(u_refs[0, 0]) > 0.01 else 0.1
                A_d, B_d = linearizer.get_discrete_model_explicit(v_ref, x_refs[0, 2])
                
                sol = mpc.solve_fast(x_measured, x_refs, A_d, B_d)
                u = sol.optimal_control
                solve_times.append(sol.solve_time_ms)
                if sol.status not in ('optimal', 'optimal_inaccurate'):
                    infeasible_count += 1
            else:
                u = lqr.compute_control(x_measured, x_ref, x_ref_dot)
        
        elif mode == 'hybrid':
            u_lqr = lqr.compute_control(x_measured, x_ref, x_ref_dot)
            assessment = risk_metrics.assess_risk(x_measured, risk_obstacles)
            
            if k % mpc_rate == 0 or mpc_solution is None:
                x_refs, u_refs = _get_segment(mpc.N + 1)
                
                v_ref = u_refs[0, 0] if abs(u_refs[0, 0]) > 0.01 else 0.1
                A_d, B_d = linearizer.get_discrete_model_explicit(v_ref, x_refs[0, 2])
                
                mpc_solution = mpc.solve_fast(x_measured, x_refs, A_d, B_d)
                solver_status = mpc_solution.status
                solver_time_ms = mpc_solution.solve_time_ms
                solve_times.append(solver_time_ms)
                
                if solver_status not in ('optimal', 'optimal_inaccurate'):
                    infeasible_count += 1
            
            u_mpc = mpc_solution.optimal_control if mpc_solution is not None else u_lqr
            feas_margin = mpc_solution.feasibility_margin if mpc_solution is not None else 0.0
            
            u, blend_info = blender.blend(
                u_lqr=u_lqr, u_mpc=u_mpc,
                risk=assessment.combined_risk,
                solver_status=solver_status,
                solver_time_ms=solver_time_ms,
                feasibility_margin=feas_margin
            )
            blend_weights[k] = blend_info.weight

        elif mode == 'adaptive':
            x_refs, u_refs = _get_segment(adaptive_mpc.N_ext + 1)

            if k % mpc_rate == 0 or adaptive_solution is None:
                adaptive_solution = adaptive_mpc.solve_tracking(
                    x_measured,
                    x_refs,
                    u_refs[:adaptive_mpc.N_ext],
                    controller_obstacles,
                )
                solve_times.append(adaptive_solution.solve_time_ms)
                if adaptive_solution.status not in ('optimal', 'optimal_inaccurate'):
                    infeasible_count += 1

            if adaptive_solution is not None:
                u = adaptive_solution.optimal_control
            else:
                u = lqr.compute_control(x_measured, x_ref, x_ref_dot)

        elif mode == 'hybrid_adaptive':
            u_lqr = lqr.compute_control(x_measured, x_ref, x_ref_dot)
            assessment = risk_metrics.assess_risk(x_measured, risk_obstacles)

            if k % mpc_rate == 0 or adaptive_solution is None:
                x_refs, u_refs = _get_segment(adaptive_mpc.N_ext + 1)
                adaptive_solution = adaptive_mpc.solve_tracking(
                    x_measured,
                    x_refs,
                    u_refs[:adaptive_mpc.N_ext],
                    controller_obstacles,
                )
                solver_status = adaptive_solution.status
                solver_time_ms = adaptive_solution.solve_time_ms
                solve_times.append(solver_time_ms)
                if solver_status not in ('optimal', 'optimal_inaccurate'):
                    infeasible_count += 1

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
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # -- Apply Actuator Dynamics (delay + lag) --
        # actuator.update() handles delay buffer and first-order lag
        x_prev = x.copy()
        v_applied, omega_applied = actuator.update(u[0], u[1])
        u_applied = np.array([v_applied, omega_applied])
        
        controls[k] = u  # Log COMMANDED control
        
        # -- Simulate step --
        x = robot.simulate_step(x, u_applied, dt.real if hasattr(dt, 'real') else dt)

        if mode in ADAPTIVE_MODES:
            adaptive_mpc.adapt_parameters(x_measured=x, x_prev=x_prev, u_prev=u_applied)
        
        # -- Check collisions --
        for obs in collision_obstacles:
            if obs.is_collision(x[0], x[1], 0.0):
                collision_count += 1
                break

        obstacle_field.step(dt)
    
    # -- Compute metrics --
    wall_time = time.perf_counter() - start_wall
    
    # Jerk metrics
    jerk_metrics = SimulationLogger.compute_jerk_metrics(controls, dt)
    
    # Control effort
    total_effort = float(np.sum(np.abs(controls)))
    
    # Blend stats
    bw_mean, bw_std, n_transitions = 0.0, 0.0, 0
    if mode in BLENDED_MODES:
        stats = blender.get_statistics()
        bw_mean = stats.get('weight_mean', 0.0)
        bw_std = stats.get('weight_std', 0.0)
        n_transitions = stats.get('total_switches', 0)
    
    # Task 11.3: Collect checkpoint metrics
    checkpoint_completion_rate = 0.0
    mean_time_to_checkpoint = 0.0
    mean_checkpoint_overshoot = 0.0
    checkpoints_reached = 0
    checkpoints_missed = 0
    
    if checkpoint_manager is not None:
        checkpoint_metrics = checkpoint_manager.get_metrics()
        checkpoint_completion_rate = checkpoint_metrics.get('completion_rate', 0.0)
        mean_time_to_checkpoint = checkpoint_metrics.get('mean_time_to_checkpoint', 0.0)
        mean_checkpoint_overshoot = checkpoint_metrics.get('mean_overshoot', 0.0)
        checkpoints_reached = checkpoint_metrics.get('checkpoints_reached', 0)
        checkpoints_missed = checkpoint_metrics.get('checkpoints_missed', 0)
    
    return RunMetrics(
        mode=mode,
        config_id=config_id,
        seed=obstacle_config.seed,
        mean_tracking_error=float(np.mean(errors)),
        max_tracking_error=float(np.max(errors)),
        final_tracking_error=float(errors[-1]),
        mean_solve_time_ms=float(np.mean(solve_times)) if solve_times else 0.0,
        max_solve_time_ms=float(np.max(solve_times)) if solve_times else 0.0,
        linear_jerk_rms=jerk_metrics.get('linear_jerk_rms', 0.0),
        angular_jerk_rms=jerk_metrics.get('angular_jerk_rms', 0.0),
        linear_jerk_peak=jerk_metrics.get('linear_jerk_peak', 0.0),
        angular_jerk_peak=jerk_metrics.get('angular_jerk_peak', 0.0),
        infeasible_count=infeasible_count,
        collision_count=collision_count,
        completion_fraction=1.0,
        total_control_effort=total_effort,
        blend_weight_mean=bw_mean,
        blend_weight_std=bw_std,
        smooth_transitions=n_transitions,
        wall_time_s=wall_time,
        checkpoint_completion_rate=checkpoint_completion_rate,
        mean_time_to_checkpoint=mean_time_to_checkpoint,
        mean_checkpoint_overshoot=mean_checkpoint_overshoot,
        checkpoints_reached=checkpoints_reached,
        checkpoints_missed=checkpoints_missed,
        tracking_mode='checkpoint' if checkpoint_mode else 'continuous',
    )


# -- Aggregation -----------------------------------------------------

def aggregate_results(runs: List[RunMetrics], mode: str) -> AggregatedResults:
    """Compute mean, spread, and percentiles for all metrics across runs."""
    metric_names = [
        'mean_tracking_error', 'max_tracking_error', 'final_tracking_error',
        'mean_solve_time_ms', 'max_solve_time_ms',
        'linear_jerk_rms', 'angular_jerk_rms',
        'linear_jerk_peak', 'angular_jerk_peak',
        'infeasible_count', 'collision_count',
        'total_control_effort', 'wall_time_s',
        'checkpoint_completion_rate', 'mean_time_to_checkpoint',
        'mean_checkpoint_overshoot', 'checkpoints_reached', 'checkpoints_missed',
    ]

    if mode in BLENDED_MODES:
        metric_names += ['blend_weight_mean', 'blend_weight_std', 'smooth_transitions']

    metrics = {}
    for name in metric_names:
        values = np.array([getattr(r, name) for r in runs])
        metrics[name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'p5': float(np.percentile(values, 5)),
            'p95': float(np.percentile(values, 95)),
        }

    return AggregatedResults(mode=mode, n_runs=len(runs), metrics=metrics)


# -- Table formatting ------------------------------------------------

def format_comparison_table(results: Dict[str, AggregatedResults]) -> str:
    """
    Format a comparison table of all controller modes.
    
    Returns:
        Formatted table string for console output.
    """
    # Key metrics for comparison
    display_metrics = [
        ('mean_tracking_error', 'Mean Track Err (m)', '.4f'),
        ('final_tracking_error', 'Final Track Err (m)', '.4f'),
        ('linear_jerk_rms', 'Lin Jerk RMS', '.1f'),
        ('angular_jerk_rms', 'Ang Jerk RMS', '.1f'),
        ('mean_solve_time_ms', 'Solve Time (ms)', '.2f'),
        ('infeasible_count', 'Infeasible', '.1f'),
        ('collision_count', 'Collisions', '.1f'),
        ('total_control_effort', 'Ctrl Effort', '.1f'),
        # Task 11.4: Add checkpoint metrics to comparison table
        ('checkpoint_completion_rate', 'CP Completion %', '.1f'),
        ('mean_time_to_checkpoint', 'Mean CP Time (s)', '.3f'),
        ('mean_checkpoint_overshoot', 'Mean CP Overshoot (m)', '.4f'),
        ('checkpoints_reached', 'CPs Reached', '.1f'),
        ('checkpoints_missed', 'CPs Missed', '.1f'),
    ]
    
    modes = list(results.keys())
    
    # Header
    header = f"{'Metric':<24}"
    for mode in modes:
        header += f" | {mode:^22}"
    
    separator = "-" * len(header)
    
    lines = [
        "",
        "=" * len(header),
        "Statistical Comparison Results",
        "=" * len(header),
        f"Runs per mode: {results[modes[0]].n_runs}",
        "",
        header,
        separator,
    ]
    
    for metric_key, metric_name, fmt in display_metrics:
        row = f"{metric_name:<24}"
        for mode in modes:
            if metric_key in results[mode].metrics:
                m = results[mode].metrics[metric_key]
                row += f" | {m['mean']:{fmt}} ± {m['std']:{fmt}}"
                row += " " * max(0, 22 - len(f"{m['mean']:{fmt}} ± {m['std']:{fmt}}"))
            else:
                row += f" | {'N/A':^22}"
        lines.append(row)
    
    lines.append(separator)
    lines.append("")
    
    return "\n".join(lines)


# -- Main runner -----------------------------------------------------

def run_statistical_validation(
    n_configs: int = 50,
    modes: List[str] = None,
    duration: float = 20.0,
    dt: float = 0.02,
    trajectory_type: str = 'figure8',
    checkpoint_mode: bool = False,
    dual_mode: bool = False,
    noise_std: float = 0.0,
    heading_noise_std: float = 0.0,
    delay_steps: int = 0,
    tau: float = 0.0,
    actuator_noise: float = 0.0,
    scenario_type: str = 'random',
    base_seed: int = 42,
    verbose: bool = True,
    output_dir: str = 'evaluation/results',
    tuning: Optional[Dict[str, Any]] = None,
) -> Dict[str, AggregatedResults]:
    """
    Run full Monte Carlo validation across all controller modes.
    
    Args:
        n_configs: Number of randomized obstacle configs
        modes: List of controller modes to compare
        duration: Simulation duration per run
        dt: Time step
        trajectory_type: Trajectory type for reference path
        checkpoint_mode: Run in checkpoint-based tracking mode
        dual_mode: Run every mode twice (continuous + checkpoint) for comparison
        noise_std: Position noise standard deviation (meters)
        heading_noise_std: Heading noise std (radians)
        delay_steps: Control pipeline delay (timesteps)
        tau: Actuator time constant (seconds)
        actuator_noise: Actuator execution noise std
        base_seed: Base random seed
        scenario_type: Type of obstacle scenario
        verbose: Print progress
        output_dir: Directory for output files
        tuning: Optional dict of tuning overrides
        
    Returns:
        Dictionary mapping mode -> AggregatedResults
    """
    if modes is None:
        modes = ['hybrid_adaptive', 'adaptive', 'hybrid', 'mpc', 'lqr']

    if any(mode in ADAPTIVE_MODES for mode in modes):
        get_adaptive_mpc_class()
    
    noise_config = NoiseConfig(
        position_noise_std=noise_std,
        heading_noise_std=heading_noise_std,
        control_delay_steps=delay_steps,
        tau_v=tau, tau_omega=tau,
        actuator_noise_std=actuator_noise
    )
    
    predefined_scenarios = ['baseline_static', 'urban_dynamic', 'stochastic_navigation',
                           'oscillatory_tracking', 'vehicle_realistic']
    
    if scenario_type in predefined_scenarios:
        if verbose:
            print(f"Warning: Predefined scenario '{scenario_type}' not yet fully supported.")
            print(f"Falling back to 'random' obstacle generator.")
        scenario_type = 'random'
    
    if verbose:
        print(f"Generating {n_configs} obstacle configurations (type={scenario_type}, seed={base_seed})...")
    
    generator = get_generator(scenario_type)
    configs = generator.generate(n_configs, base_seed)

    # Build the list of (result_key, checkpoint_flag) pairs to run.
    # In dual mode every controller runs twice: continuous then checkpoint.
    tracking_passes: List[Tuple[str, bool]] = []
    if dual_mode:
        for mode in modes:
            tracking_passes.append((f'{mode}_continuous', False))
            tracking_passes.append((f'{mode}_checkpoint', True))
    else:
        for mode in modes:
            tracking_passes.append((mode, checkpoint_mode))

    all_results: Dict[str, List[RunMetrics]] = {key: [] for key, _ in tracking_passes}
    total_runs = n_configs * len(tracking_passes)
    completed = 0
    
    for result_key, cp_flag in tracking_passes:
        ctrl_mode = result_key.rsplit('_continuous', 1)[0].rsplit('_checkpoint', 1)[0] if dual_mode else result_key
        label = f"{ctrl_mode.upper()} ({'checkpoint' if cp_flag else 'continuous'})"
        if verbose:
            print(f"\n{'-'*50}")
            print(f"Running: {label} ({n_configs} configs)")
            print(f"{'-'*50}")
        
        for i, config in enumerate(configs):
            try:
                metrics = run_single_config(
                    mode=ctrl_mode,
                    obstacle_config=config,
                    noise_config=noise_config,
                    duration=duration,
                    dt=dt,
                    config_id=i,
                    trajectory_type=trajectory_type,
                    checkpoint_mode=cp_flag,
                    tuning=tuning,
                )
                all_results[result_key].append(metrics)
            except Exception as e:
                if verbose:
                    print(f"  Config {i} FAILED: {e}")
                all_results[result_key].append(RunMetrics(
                    mode=ctrl_mode, config_id=i, seed=config.seed,
                    mean_tracking_error=float('inf'),
                    max_tracking_error=float('inf'),
                    final_tracking_error=float('inf'),
                    mean_solve_time_ms=0.0, max_solve_time_ms=0.0,
                    linear_jerk_rms=0.0, angular_jerk_rms=0.0,
                    linear_jerk_peak=0.0, angular_jerk_peak=0.0,
                    infeasible_count=999, collision_count=999,
                    completion_fraction=0.0, total_control_effort=0.0,
                    tracking_mode='checkpoint' if cp_flag else 'continuous',
                ))
            
            completed += 1
            if verbose and (i + 1) % max(1, n_configs // 10) == 0:
                pct = 100 * completed / total_runs
                print(f"  Progress: {i+1}/{n_configs} ({pct:.0f}% overall)")
    
    # Aggregate results
    aggregated = {}
    for result_key, _ in tracking_passes:
        valid_runs = [r for r in all_results[result_key] 
                      if r.mean_tracking_error < float('inf')]
        if valid_runs:
            aggregated[result_key] = aggregate_results(valid_runs, result_key)
        else:
            if verbose:
                print(f"WARNING: No valid runs for '{result_key}'")
    
    if verbose:
        print(format_comparison_table(aggregated))
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    json_data = {
        'metadata': {
            'n_configs': n_configs,
            'duration': duration,
            'dt': dt,
            'noise_config': asdict(noise_config),
            'base_seed': base_seed,
            'dual_mode': dual_mode,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        },
        'results': {}
    }
    for key, agg in aggregated.items():
        json_data['results'][key] = {
            'n_runs': agg.n_runs,
            'metrics': agg.metrics
        }
    
    json_path = os.path.join(output_dir, 'statistical_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    if verbose:
        print(f"Results saved to {json_path}")
    
    csv_path = os.path.join(output_dir, 'statistical_results.csv')
    with open(csv_path, 'w') as f:
        metric_keys = list(next(iter(aggregated.values())).metrics.keys())
        header_parts = ['mode', 'n_runs']
        for mk in metric_keys:
            header_parts.extend([f'{mk}_mean', f'{mk}_std'])
        f.write(','.join(header_parts) + '\n')
        
        for key, agg in aggregated.items():
            row_parts = [key, str(agg.n_runs)]
            for mk in metric_keys:
                if mk in agg.metrics:
                    row_parts.extend([
                        f"{agg.metrics[mk]['mean']:.6f}",
                        f"{agg.metrics[mk]['std']:.6f}"
                    ])
                else:
                    row_parts.extend(['0', '0'])
            f.write(','.join(row_parts) + '\n')
    if verbose:
        print(f"CSV saved to {csv_path}")
    
    perrun_path = os.path.join(output_dir, 'per_run_results.csv')
    with open(perrun_path, 'w') as f:
        fields = [
            'mode', 'tracking_mode', 'config_id', 'seed',
            'mean_tracking_error', 'max_tracking_error', 'final_tracking_error',
            'mean_solve_time_ms', 'max_solve_time_ms',
            'linear_jerk_rms', 'angular_jerk_rms',
            'linear_jerk_peak', 'angular_jerk_peak',
            'infeasible_count', 'collision_count',
            'total_control_effort', 'blend_weight_mean', 'smooth_transitions',
            'checkpoint_completion_rate', 'mean_time_to_checkpoint',
            'mean_checkpoint_overshoot', 'checkpoints_reached', 'checkpoints_missed',
            'wall_time_s'
        ]
        f.write(','.join(fields) + '\n')
        
        for result_key, _ in tracking_passes:
            for r in all_results[result_key]:
                if r.mean_tracking_error < float('inf'):
                    values = [str(getattr(r, field)) for field in fields]
                    f.write(','.join(values) + '\n')
    if verbose:
        print(f"Per-run data saved to {perrun_path}")
    
    return aggregated


# -- CLI -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Monte Carlo statistical validation of controller modes',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--configs', type=int, default=50,
                        help='Number of randomized obstacle configs')
    parser.add_argument('--modes', nargs='+', 
                        default=DEFAULT_MODES,
                        choices=['lqr', 'mpc', 'hard_switch', 'hybrid', 'adaptive', 'hybrid_adaptive', 'checkpoint'],
                        help='Controller modes to compare')
    parser.add_argument('--duration', type=float, default=20.0,
                        help='Simulation duration (seconds)')
    parser.add_argument('--trajectory', type=str, default='clothoid',
                        help='Reference trajectory type')
    parser.add_argument('--checkpoint-mode', action='store_true',
                        help='Enable checkpoint-based reference tracking')
    parser.add_argument('--dual-mode', action='store_true',
                        help='Run each mode twice (continuous + checkpoint) for comparison')
    parser.add_argument('--noise', type=float, default=0.0,
                        help='Position noise std (meters)')
    parser.add_argument('--heading-noise', type=float, default=0.0,
                        help='Heading noise std (radians)')
    parser.add_argument('--delay', type=int, default=0,
                        help='Control pipeline delay (timesteps)')
    parser.add_argument('--tau', type=float, default=0.0,
                        help='Actuator time constant (seconds)')
    parser.add_argument('--actuator-noise', type=float, default=0.0,
                        help='Actuator execution noise std')
    parser.add_argument('--scenario', type=str, default='stochastic_navigation',
                        choices=['random', 'corridor', 'bugtrap', 'dense', 'moving', 'random_walk',
                                 'baseline_static', 'urban_dynamic', 'stochastic_navigation',
                                 'oscillatory_tracking', 'vehicle_realistic'],
                        help='Scenario type (random/corridor/bugtrap/dense/moving/random_walk for obstacle configs, or predefined scenarios)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed')
    parser.add_argument('--output', type=str, default='evaluation/results',
                        help='Output directory')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')
    
    args = parser.parse_args()
    
    run_statistical_validation(
        n_configs=args.configs,
        modes=args.modes,
        duration=args.duration,
        trajectory_type=args.trajectory,
        checkpoint_mode=args.checkpoint_mode,
        dual_mode=args.dual_mode,
        noise_std=args.noise,
        heading_noise_std=args.heading_noise,
        delay_steps=args.delay,
        tau=args.tau,
        actuator_noise=args.actuator_noise,
        scenario_type=args.scenario,
        base_seed=args.seed,
        verbose=not args.quiet,
        output_dir=args.output
    )

    # CasADi teardown can crash some Windows Python builds after completion.
    # Force a clean process exit once results are fully persisted.
    if os.name == 'nt' and any(mode in ADAPTIVE_MODES for mode in args.modes):
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == '__main__':
    main()

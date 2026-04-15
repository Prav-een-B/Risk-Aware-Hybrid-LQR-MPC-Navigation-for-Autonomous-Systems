# Code Review: Risk-Aware Hybrid LQR-MPC Navigation

## Complete Technical Documentation

> In-depth explanation of every module, class, and function in the codebase.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Models Module](#2-models-module)
   - [differential_drive.py](#21-differential_drivepy)
   - [linearization.py](#22-linearizationpy)
   - [actuator_dynamics.py](#23-actuator_dynamicspy)
3. [Controllers Module](#3-controllers-module)
   - [lqr_controller.py](#31-lqr_controllerpy)
   - [mpc_controller.py](#32-mpc_controllerpy)
   - [yaw_stabilizer.py](#33-yaw_stabilizerpy)
   - [cvxpygen_solver.py](#34-cvxpygen_solverpy)
   - [hybrid_blender.py](#35-hybrid_blenderpy)
   - [risk_metrics.py](#36-risk_metricspy)
   - [adaptive_mpc_controller.py](#37-adaptive_mpc_controllerpy)
4. [Trajectory Module](#4-trajectory-module)
   - [reference_generator.py](#41-reference_generatorpy)
5. [Logging Module](#5-logging-module)
   - [simulation_logger.py](#51-simulation_loggerpy)
6. [ROS2 Nodes](#6-ros2-nodes)
7. [Standalone Simulation](#7-standalone-simulation)
8. [Advanced Scenarios](#8-advanced-scenarios)
9. [Docker and Gazebo Harness](#9-docker-and-gazebo-harness)
10. [Integration Status and Next Steps](#10-integration-status-and-next-steps)

---

## 1. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                         Application Layer                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ trajectory_node │  │    lqr_node     │  │    mpc_node     │  │
│  │                 │  │                 │  │                 │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
│           │                    │                     │           │
│  ┌────────┘  ┌─────────────────┘  ┌──────────────────┘           │
│  │  ┌────────▼────────┐  ┌───────▼──────────┐  ┌────────────┐  │
│  │  │ hybrid_node     │  │ kinematic_sim    │  │ Adaptive   │  │
│  │  │ (ROS2 blending) │  │ (odom bridge)    │  │ MPC node   │  │
│  │  └────────┬────────┘  └──────────────────┘  │ (standalone wired,
│  │           │                                  │  ROS planned) │  │
│  │           │                                  └────────────┘  │
├──┼───────────┼──────────────────────────────────────────────────┤
│  │           │            Core Libraries                        │
│  │  ┌────────▼────────────────────────────────────────────────┐ │
│  │  │                    Controllers Module                    │ │
│  │  │  ┌──────────────┐  ┌───────────────┐  ┌──────────────┐ │ │
│  │  │  │ LQRController│  │ MPCController │  │ Adaptive MPC │ │ │
│  │  │  │   (DARE)     │  │  (CVXPY)      │  │ (CasADi)     │ │ │
│  │  │  └──────┬───────┘  └──────┬────────┘  └──────┬───────┘ │ │
│  │  │         │                 │                   │         │ │
│  │  │  ┌──────▼─────────────────▼───────────────────▼──────┐  │ │
│  │  │  │         BlendingSupervisor + RiskMetrics           │  │ │
│  │  │  └───────────────────────────────────────────────────┘  │ │
│  │  └─────────────────────────────────────────────────────────┘ │
│  │  ┌──────────────────────────────────────────────────────┐    │
│  │  │                   Models Module                       │    │
│  │  │  ┌──────────────────┐  ┌──────────────────┐          │    │
│  │  │  │ DifferentialDrive│  │   Linearizer     │          │    │
│  │  │  │  (Kinematics)    │◄─┤ (Jacobians, ZOH) │          │    │
│  │  │  └──────────────────┘  └──────────────────┘          │    │
│  │  │  ┌──────────────────┐                                │    │
│  │  │  │ ActuatorDynamics │                                │    │
│  │  │  │ (delay, lag)     │                                │    │
│  │  │  └──────────────────┘                                │    │
│  │  └──────────────────────────────────────────────────────┘    │
│  │  ┌─────────────────────────┐  ┌────────────────────────┐    │
│  │  │ ReferenceTrajectory     │  │   SimulationLogger     │    │
│  │  │ Generator               │  │   (CSV/JSON Export)    │    │
│  │  └─────────────────────────┘  └────────────────────────┘    │
└──┴──────────────────────────────────────────────────────────────┘
```

---

## 2. Models Module

### 2.1 differential_drive.py

**Location:** `src/hybrid_controller/hybrid_controller/models/differential_drive.py`

This module implements the nonlinear kinematic model for a two-wheeled differential drive robot.

#### Data Classes

##### `RobotState`
```python
@dataclass
class RobotState:
    px: float    # x-position (meters)
    py: float    # y-position (meters)
    theta: float # orientation (radians)
```

##### `ControlInput`
```python
@dataclass
class ControlInput:
    v: float      # linear velocity (m/s)
    omega: float  # angular velocity (rad/s)
```

#### Class: `DifferentialDriveRobot`

**Constants:** `STATE_DIM = 3`, `CONTROL_DIM = 2`

##### `__init__(self, v_max, omega_max, wheel_base)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `v_max` | 1.0 | Maximum linear velocity (m/s) |
| `omega_max` | 1.5 | Maximum angular velocity (rad/s) |
| `wheel_base` | 0.3 | Distance between wheels (m) |

##### `continuous_dynamics(self, state, control) → np.ndarray`

$$\dot{x} = f(x, u) = \begin{bmatrix} v \cos\theta \\ v \sin\theta \\ \omega \end{bmatrix}$$

##### `simulate_step(self, state, control, dt, method) → np.ndarray`

Integration methods:
- **Euler:** $x_{k+1} = x_k + \Delta t \cdot f(x_k, u_k)$
- **RK4:** Fourth-order Runge-Kutta

##### `simulate_trajectory(self, x0, controls, dt, method) → np.ndarray`

Returns array of shape `(N+1, 3)`.

##### `get_wheel_velocities(self, v, omega) → Tuple[float, float]`

$$v_L = v - \frac{L}{2}\omega, \quad v_R = v + \frac{L}{2}\omega$$

##### `from_wheel_velocities(self, v_left, v_right) → Tuple[float, float]`

$$v = \frac{v_R + v_L}{2}, \quad \omega = \frac{v_R - v_L}{L}$$

---

### 2.2 linearization.py

**Location:** `src/hybrid_controller/hybrid_controller/models/linearization.py`

#### Class: `Linearizer`

##### `get_jacobians(self, v_r, theta_r) → Tuple[np.ndarray, np.ndarray]`

$$A = \begin{bmatrix} 0 & 0 & -v_r \sin\theta_r \\ 0 & 0 & v_r \cos\theta_r \\ 0 & 0 & 0 \end{bmatrix}, \quad
B = \begin{bmatrix} \cos\theta_r & 0 \\ \sin\theta_r & 0 \\ 0 & 1 \end{bmatrix}$$

##### `discretize_euler(self, A, B) → Tuple`

$A_d \approx I + A \cdot T_s$, $B_d \approx B \cdot T_s$

##### `discretize_exact(self, A, B) → Tuple`

Uses matrix exponential: $A_d = e^{A \cdot T_s}$

##### `get_discrete_model_explicit(self, v_r, theta_r) → Tuple`

Direct ZOH computation (most efficient).

##### `build_prediction_matrices(A_d, B_d, N) → Tuple` (static)

Constructs batch prediction: $X = \Phi \cdot x_0 + \Gamma \cdot U$

---

### 2.3 actuator_dynamics.py

**Location:** `src/hybrid_controller/hybrid_controller/models/actuator_dynamics.py`

Models physical hardware limitations for sim-to-real fidelity.

#### Class: `ActuatorDynamics`

| Component | Model |
|-----------|-------|
| Control latency | Circular buffer delay: $u_{\text{delayed}}[k] = u_{\text{cmd}}[k-d]$ |
| Actuator lag | First-order: $\tau\dot{v} + v = v_{\text{cmd}}$, Euler-discretized |
| Execution noise | Additive Gaussian: $u_{\text{applied}} = u_{\text{lagged}} + \mathcal{N}(0,\sigma^2)$ |

**Key Parameters:** `tau_v`, `tau_omega` (time constants), `delay_steps` (e.g. 2 = 40ms), `noise_std`

---

## 3. Controllers Module

### 3.1 lqr_controller.py

**Location:** `src/hybrid_controller/hybrid_controller/controllers/lqr_controller.py`

#### Class: `LQRController`

##### `__init__(self, Q_diag, R_diag, dt, v_max, omega_max)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Q_diag` | [10, 10, 1] | State error weights |
| `R_diag` | [0.1, 0.1] | Control effort weights |

##### `compute_gain(self, v_r, theta_r) → np.ndarray`

Solves the DARE:
$$P = A_d^T P A_d - A_d^T P B_d (R + B_d^T P B_d)^{-1} B_d^T P A_d + Q$$
$$K = (R + B_d^T P B_d)^{-1} B_d^T P A_d$$

Uses operating-point caching; falls back to proportional gain if DARE fails.

##### `compute_control(self, x, x_ref, u_ref, K) → np.ndarray`

$$u_k = u_{r,k} - K \cdot (x_k - x_{r,k})$$

---

### 3.2 mpc_controller.py

**Location:** `src/hybrid_controller/hybrid_controller/controllers/mpc_controller.py`

#### Data Classes

```python
@dataclass
class Obstacle:
    x: float; y: float; radius: float

@dataclass
class MPCSolution:
    status: str; optimal_control: np.ndarray; control_sequence: np.ndarray
    predicted_states: np.ndarray; cost: float; solve_time_ms: float
    slack_used: bool; iterations: int
```

#### Class: `MPCController`

##### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `horizon` | 10 | Prediction horizon N |
| `P_diag` | [20, 20, 40] | Terminal cost |
| `S_diag` | [0.1, 0.5] | 1st-order Δu penalty |
| `J_diag` | None | 2nd-order jerk penalty (recommended: [0.05, 0.3]) |
| `d_safe` | 0.3 | Safety distance (m) |
| `slack_penalty` | 5000 | Soft constraint weight ρ |
| `w_max` | 0.05 | Tube MPC disturbance bound |

##### `solve(self, x0, x_refs, u_refs, obstacles) → MPCSolution`

**Cost:**
$$\min \sum_{k=0}^{N-1} \left( \|x_k - x_k^{\text{ref}}\|_Q^2 + \|u_k\|_R^2 + \|u_k - u_{k-1}\|_S^2 + \|u_k - 2u_{k-1} + u_{k-2}\|_J^2 \right) + \|x_N - x_N^{\text{ref}}\|_P^2$$

**Obstacle constraints:** Linearized half-plane (DCP-compliant):
$$n_x(x_k - x_{\text{obs}}) + n_y(y_k - y_{\text{obs}}) \geq d_{\text{safe}} + r_{\text{obs}} - \epsilon_k$$

##### `solve_with_ltv` — Linear Time-Varying variant with per-step linearization

##### `get_warm_start` — Shifts previous solution forward

---

### 3.3 yaw_stabilizer.py

**Location:** `src/hybrid_controller/hybrid_controller/controllers/yaw_stabilizer.py`

PID heading controller with derivative smoothing, anti-windup, and angle wrapping.

$$\omega_{\text{cmd}} = K_p e_\theta + K_i \int e_\theta + K_d \dot{e}_\theta + \omega_{ff}$$

---

### 3.4 cvxpygen_solver.py

**Location:** `src/hybrid_controller/hybrid_controller/controllers/cvxpygen_solver.py`

Parametrized MPC solver achieving **38× speedup** (4.7ms vs 180ms) by avoiding CVXPY re-canonicalization. Uses `cp.Parameter` for time-varying data; canonicalization happens once.

| Benchmark (N=5, OSQP) | Value |
|------------------------|-------|
| Mean solve time | 4.7ms |
| Median | 3.4ms |
| Min | 1.9ms |
| Speedup vs original | **38×** |

---

### 3.5 hybrid_blender.py

**Location:** `src/hybrid_controller/hybrid_controller/controllers/hybrid_blender.py`

#### Class: `BlendingSupervisor`

##### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k_sigmoid` | 10.0 | Sigmoid steepness |
| `risk_threshold` | 0.3 | Sigmoid midpoint |
| `dw_max` | 2.0 | Max weight rate (s⁻¹) |
| `hysteresis_band` | 0.05 | Deadband half-width |

##### Blending Pipeline

$$\text{risk} \xrightarrow{\text{sigmoid}} w_{\text{raw}} \xrightarrow{\text{hysteresis}} w_{\text{hyst}} \xrightarrow{\text{rate limit}} w_{\text{lim}} \xrightarrow{\text{feasibility}} w(t)$$

1. **Sigmoid:** $w_{\text{raw}} = \sigma(k(r - r_{\text{th}}))$
2. **Hysteresis:** hold if $r \in [r_{\text{th}} - h, r_{\text{th}} + h]$
3. **Rate limit:** $w = \text{clip}(w_{\text{hyst}},\; w_{\text{prev}} \pm \dot{w}_{\max}\Delta t)$
4. **Feasibility fallback:** exponential ramp-down on consecutive MPC failures

**Output:** $u_{\text{blend}} = w \cdot u_{\text{MPC}} + (1-w) \cdot u_{\text{LQR}}$

---

### 3.6 risk_metrics.py

**Location:** `src/hybrid_controller/hybrid_controller/controllers/risk_metrics.py`

Computes geometric risk from current obstacle positions and optional predicted robot states. Current risk model is static-obstacle-centric; uncertainty-aware prediction is planned.

---

### 3.7 adaptive_mpc_controller.py

**Location:** `src/hybrid_controller/hybrid_controller/controllers/adaptive_mpc_controller.py`

Implements a nonlinear MPC with online parameter adaptation. Uses CasADi + IPOPT
instead of CVXPY/OSQP.

#### Class: `LMSAdaptation`

Online parameter estimation using projected gradient descent on one-step prediction error:

$$\hat{x}_{k+1} = x_k + dt \begin{bmatrix} \hat{\theta}_v v_k \cos\theta_k \\ \hat{\theta}_v v_k \sin\theta_k \\ \hat{\theta}_\omega \omega_k \end{bmatrix}$$

$$\Phi_k = dt \begin{bmatrix} v_k\cos\theta_k & 0 \\ v_k\sin\theta_k & 0 \\ 0 & \omega_k \end{bmatrix}$$

$$\hat{\theta}_{k+1} = \text{clip}\!\left(\hat{\theta}_k + \Gamma \Phi_k^\top (x_{\text{meas}} - \hat{x}_{k+1}),\; [0.5, 0.5],\; [2.0, 2.0]\right)$$

Adaptation gain: $\Gamma = 0.005 \cdot I$

**Key property:** cumulative tracking error scales linearly with noise energy (Koehler, 2025).

#### Class: `AdaptiveMPCController`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prediction_horizon` | 10 | Free control steps N |
| `terminal_horizon` | 5 | LQR rollout steps M |
| `Q_diag` | [30, 30, 5] | State cost |
| `R_diag` | [0.1, 0.1] | Control cost |
| `omega_term` | 10.0 | Terminal weight multiplier |
| `q_xi` | 1000.0 | Slack penalty |
| `adaptation_gamma` | 0.005 | LMS learning rate |

**Key differences from `MPCController`:**

| Feature | MPC (CVXPY) | Adaptive MPC (CasADi) |
|---------|-------------|----------------------|
| Solver | OSQP (QP) | IPOPT (NLP) |
| Obstacle constraints | Linearised half-plane | Exact Euclidean norm |
| Horizon | N=5 (0.1s) | N+M=15 (0.3s) |
| Model parameters | Fixed | Online LMS estimation |
| Terminal controller | Cost matrix P | LQR rollout equality constraints |
| Typical solve time | 2.5ms | 20-80ms |
| Slack variables | Shared | Per-state $\xi_k$ |

**Cost function:**
$$J = \sum_{k=0}^{N-1}\!\left(\|x_k - x_k^{\text{ref}}\|_Q^2 + \|u_k - u_k^{\text{ref}}\|_R^2 + q_\xi\|\xi_k\|^2\right) + \omega\sum_{k=N}^{N+M}\!\left(\cdots\right) + \omega\|x_{N+M} - x_{N+M}^{\text{ref}}\|_Q^2$$

**Solver setup:** CasADi symbolic NLP with IPOPT backend, warm-start support,
and per-state obstacle avoidance constraints using exact distance norms.

**Integration status:** The controller is fully implemented and now wired into
standalone simulation through `--mode adaptive` and
`--mode hybrid_adaptive` in `run_simulation.py`. Statistical evaluation runner
integration remains open.

---

## 4. Trajectory Module

### 4.1 reference_generator.py

**Location:** `src/hybrid_controller/hybrid_controller/trajectory/reference_generator.py`

#### Class: `ReferenceTrajectoryGenerator`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `A` | 2.0 | Spatial amplitude (m) |
| `a` | 0.5 | Angular frequency (rad/s) |
| `dt` | 0.02 | Sampling time (s) |

**Figure-8 parametric equations:**
$$p_x(t) = A\sin(at), \quad p_y(t) = \frac{A}{2}\sin(2at)$$

**Supported trajectory families:**

| Family | Description |
|--------|-------------|
| `figure8` | Default Lissajous figure-8 |
| `circle` | Simple circular path |
| `clover` | Three-lobed clover pattern |
| `slalom` | S-curve slalom |
| `checkpoint_path` | Waypoint-based path with presets (e.g. `warehouse`) |
| `lissajous` | General Lissajous curve with tunable harmonics |
| `spiral` | Polar spiral with linearly increasing radius |
| `spline_path` | Cubic spline interpolation through waypoints |
| `urban_path` | Segment path with orthogonal turns |
| `sinusoidal` | Linear progress with sinusoidal lateral offset |
| `random_waypoint` | Piecewise-linear path through random waypoints |
| `clothoid` | Euler spiral with linearly varying curvature |

##### `generate(self, duration) → np.ndarray`

Output shape `(N, 6)`: `[time, px, py, theta, v, omega]`

##### `get_trajectory_segment(self, start_idx, horizon) → Tuple`

Extracts `(x_refs, u_refs)` for MPC prediction horizon.

### 4.2 Checkpoint-Based Navigation Stack

The trajectory stack now supports full checkpoint-mode tracking with
curvature-aware generation and adaptive switching logic.

#### CurvatureComputer

- Computes finite-difference curvature from sampled trajectories.
- Uses epsilon clamping for numerical stability.
- Feeds curvature values to adaptive checkpoint spacing.

#### CheckpointGenerator

- Converts trajectories into checkpoint sequences with adaptive spacing.
- Uses interpolation between min/max spacing based on local curvature.
- Stores checkpoint state including heading, curvature, and tracking metadata.

#### CheckpointManager

- Maintains current checkpoint index and progress metrics.
- Applies adaptive switching radius based on curvature.
- Uses hysteresis and forward-progress timeout for robust transitions.
- Extracts local reference horizons for MPC/Adaptive MPC and single-step
    references for LQR.

#### Enhanced Dynamic Obstacle Field

- Supports controller/risk/actual obstacle views.
- Includes velocity-aware and sensing-aware radius inflation.
- Supports bounded random-walk motion with reflection/wrapping.

---

## 5. Logging Module

### 5.1 simulation_logger.py

**Location:** `src/hybrid_controller/hybrid_controller/logging/simulation_logger.py`

| Method | Purpose |
|--------|---------|
| `log_state()` | Records state, reference, error |
| `log_control()` | Records control with type and solve time |
| `log_hybrid_step()` | Blend weight, risk, mode, jerk |
| `compute_jerk_metrics()` | Peak/RMS/p95 jerk from controls |
| `export_to_csv()` | State history |
| `export_controls_to_csv()` | Control history |
| `export_to_json()` | All entries with metadata |

---

## 6. ROS2 Nodes

| Node | Purpose | Key Topics |
|------|---------|------------|
| `trajectory_node.py` | Publishes reference trajectory | `/reference_trajectory`, `/current_reference` |
| `lqr_node.py` | LQR tracking | `/odom` → `/cmd_vel` |
| `mpc_node.py` | MPC obstacle avoidance | `/mpc_obstacles` → `/cmd_vel`, `/mpc_predicted_path` |
| `hybrid_node.py` | Smooth hybrid blending (ROS2) | `/odom`, `/obstacles` → `/cmd_vel` |
| `kinematic_sim_node.py` | Lightweight odom bridge | `/cmd_vel` → `/odom` |

---

## 7. Standalone Simulation

**File:** `run_simulation.py`

```bash
python run_simulation.py --mode lqr
python run_simulation.py --mode mpc
python run_simulation.py --mode compare
python run_simulation.py --mode hybrid
python run_simulation.py --mode hybrid --trajectory slalom --scenario dense
python run_simulation.py --mode lqr --trajectory checkpoint_path --checkpoint-preset warehouse
python run_simulation.py --mode hybrid --trajectory lissajous --checkpoint-mode
```

| Flag | Description |
|------|-------------|
| `--mode` | `lqr`, `mpc`, `compare`, `hybrid`, `adaptive`, `hybrid_adaptive` |
| `--trajectory` | `figure8`, `circle`, `clover`, `slalom`, `checkpoint_path`, `lissajous`, `spiral`, `spline_path`, `urban_path`, `sinusoidal`, `random_waypoint`, `clothoid` |
| `--checkpoint-mode` | Enables checkpoint-based adaptive switching and local horizon references |
| `--checkpoint-preset` | Preset route for `checkpoint_path` trajectories |
| `--scenario` | `default`, `sparse`, `dense`, `corridor`, `moving`, `random_walk` |
| `--duration` | Simulation duration (seconds) |
| `--no-plot` | Disable visualization |

---

## 8. Advanced Scenarios

**Location:** `evaluation/scenarios.py`

| Scenario | Description | Purpose |
|----------|-------------|---------|
| `CorridorScenario` | Narrow passage (two obstacle walls) | Simultaneous constraint handling |
| `BugTrapScenario` | U-shaped obstacle configuration | Local minima handling |
| `DenseClutterScenario` | High-density random field (8-15 obstacles) | Solver speed and switching stress test |
| `MovingObstacleScenario` | Bounded linear obstacle motion | Dynamic avoidance validation |
| `RandomWalkScenario` | Bounded random-walk obstacle motion | Stochastic obstacle stress testing |

Standalone `run_simulation.py` now uses dynamic obstacle fields for
`moving` and `random_walk` scenarios with per-step obstacle updates.

---

## 9. Docker and Gazebo Harness

### Infrastructure Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Container image definition |
| `docker-compose.yml` | Service orchestration |
| `docker/run_validation_suite.sh` | Standalone benchmarks inside container |
| `docker/run_gazebo_suite.sh` | ROS2/Gazebo demo with rosbag recording |
| `docker/run_full_pipeline.sh` | Both suites |

### Runtime Topology

```
Validation Suite                      Gazebo Suite
─────────────────                     ────────────
run_simulation.py → logs/, outputs/   hybrid_gazebo.launch.py
statistical_runner.py → results/        ├── Gazebo World
                                        ├── kinematic_sim_node → /odom
                                        ├── hybrid_node → /cmd_vel
                                        ├── trajectory_node
                                        └── ros2 bag record
```

### Important Limitation

The Gazebo path uses `kinematic_sim_node.py` for `/odom`, not a fully coupled
Gazebo robot model with wheel plugins. This is controller-in-the-loop validation,
not full plant-in-Gazebo integration.

---

## 10. Integration Status and Next Steps

### Current Module Status

| Module | Status |
|--------|--------|
| LQR controller | ✅ Integrated and benchmarked |
| MPC controller (CVXPY) | ✅ Integrated and benchmarked |
| Parametrised solver (38× speedup) | ✅ Integrated |
| Smooth hybrid supervisor | ✅ Integrated and benchmarked |
| Risk metrics | ✅ Integrated (dynamic obstacle view wired in standalone) |
| Adaptive MPC (CasADi) | ✅ Integrated in standalone CLI (`adaptive`, `hybrid_adaptive`) |
| Trajectory families | ✅ 12 families including lissajous, spiral, spline_path, urban_path, sinusoidal, random_waypoint, clothoid |
| Checkpoint-horizon (online) | ✅ Implemented with adaptive switching and reference extraction |
| Docker validation | ✅ Syntax-checked, not end-to-end tested |
| Gazebo harness | ⚠️ Controller-in-loop only |

### Priority Roadmap

1. Extend `evaluation/statistical_runner.py` to include adaptive and hybrid-adaptive modes
2. Expand checkpoint benchmark coverage across all scenarios and controller modes
3. Add richer uncertainty propagation to risk/inflation beyond geometric inflation
4. Replace the lightweight ROS odom shim with a full Gazebo robot model
5. Benchmark all five modes at scale: LQR, MPC, Hybrid, Adaptive MPC, Hybrid+Adaptive

---

## Authors

| Name | GitHub | Email |
|------|--------|-------|
| Kshitiz Kumar Sinha | [@Erebuzzz](https://github.com/Erebuzzz) | kshitiz23@iiserb.ac.in |
| Agolika BM | [@Agolika413](https://github.com/Agolika413) | agolika23@iiserb.ac.in |

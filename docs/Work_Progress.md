# Work Progress

Updated: 2026-04-18

## 1. Current Repository State

The repository is no longer in the proposal-only stage. The baseline hybrid
stack is implemented and the previously pending dynamic/adaptive standalone
integration work is now wired into the runnable workflow.

### Implemented now

| Area | Status | Notes |
|---|---|---|
| Differential-drive model | Implemented | `differential_drive.py` provides the kinematic model and simulation utilities |
| LQR tracking controller | Implemented | `lqr_controller.py` is wired into `run_simulation.py` and the evaluation scripts |
| Linearized MPC controller | Implemented | `mpc_controller.py` includes obstacle constraints, move blocking, tube-style tightening, and jerk-related penalties |
| Smooth hybrid supervisor | Implemented | `hybrid_blender.py` performs risk-based blending between LQR and MPC |
| Static-obstacle risk metric | Implemented | `risk_metrics.py` handles current geometric risk and optional predictive risk |
| Standalone simulation entrypoint | Implemented | `run_simulation.py` supports `lqr`, `mpc`, `compare`, `hybrid`, `adaptive`, and `hybrid_adaptive` modes |
| Static scenario evaluation | Implemented | `evaluation/statistical_runner.py` and `evaluation/scenarios.py` provide Monte Carlo-style experiments |
| Adaptive MPC runnable mode | Implemented | `--mode adaptive` now executes `AdaptiveMPCController` with online parameter adaptation |
| Hybrid LQR + Adaptive MPC mode | Implemented | `--mode hybrid_adaptive` now blends LQR and adaptive MPC in the runnable path |
| Dynamic obstacle scenarios | Implemented | `moving` and `random_walk` scenarios now run through `DynamicObstacleField` and per-step state updates |
| Safety/sensing inflation wiring | Implemented | `InflationConfig` now feeds controller obstacle inflation and risk-obstacle views |
| Adaptive MPC in evaluation runner | Implemented | Wired correctly into `evaluation/statistical_runner.py` metrics arrays |
| Checkpoint-based reference tracking | Implemented | Obstacle-density exponential checkpoint generation, adaptive switching (`CheckpointManager`), and finite-bounded local horizon extraction are fully integrated |

### Open items

| Area | Status | Notes |
|---|---|---|
| Uncertainty-aware risk model | Partial | Obstacle inflation is now wired, but risk remains primarily geometric and not yet covariance-driven end-to-end |
| Complex trajectory suite | Partial | Figure-8, circle, clover, slalom, and checkpoint presets now exist in the standalone path, but the evaluation runner defaults could utilize them more consistently |
| Docker validation workflow | Implemented | `Dockerfile`, validation scripts, artifact collection, and pytest smoke tests are now present |
| ROS2 hybrid Gazebo harness | Partial | `hybrid_node.py`, `kinematic_sim_node.py`, and `hybrid_gazebo.launch.py` now exist, but odometry still comes from the lightweight ROS simulator rather than a Gazebo robot plugin |

## 2. Checkpoint-Based Tracking Timeline and Key Decisions

### Implementation timeline

| Milestone | Status | Summary |
|---|---|---|
| Trajectory family expansion | Completed | Added seven extended trajectories: lissajous, spiral, spline_path, urban_path, sinusoidal, random_waypoint, clothoid |
| Obstacle-aware checkpoint generation | Completed | Replaced curvature metric with an exponential obstacle-density spacing function to ensure kinematically feasible segments |
| Adaptive checkpoint switching | Completed | Added spacing-scaled switching radius, hysteresis logic, and forward-progress timeout |
| Controller integration | Completed | Unified checkpoint-mode reference extraction across LQR, MPC, Adaptive MPC, Hybrid, Hybrid-Adaptive, and bounding local velocities via `v_max` |
| Scenario and metrics integration | Completed | Added five checkpoint-aware scenarios and checkpoint completion/time/overshoot reporting |
| Property and integration validation | Completed | Added properties for formulas, spacing, switching, horizon extraction, dynamics, uncertainty, and metrics |

### Key design decisions

1. Keep backward compatibility by preserving legacy trajectories while adding checkpoint-mode support behind explicit CLI flags.
2. Use exponential obstacle-density (`S_{min} + (S_{max} - S_{min}) * e^{-\gamma * N_{obs}}`) as the primary geometric signal for checkpoint spacing.
3. Use hysteresis plus forward-progress timeout to avoid oscillatory switching and deadlock near difficult checkpoints.
4. Keep obstacle representations separated by purpose (`controller_obstacles`, `risk_obstacles`, `actual_obstacles`) to avoid mixing safety inflation with collision truth.
5. Keep uncertainty injection modular (process noise, sensor noise, mismatch, delay) so stress tests can be composed per scenario.

## 4. Evidence From Existing Evaluation

The repository already contains a stored statistical evaluation at
`evaluation/results/statistical_results.json`.

Important observations from that file:

- Pure MPC currently has much lower mean tracking error than the stored hybrid
  results in the static benchmark set.
- Hybrid solve times are lower than pure MPC on average, but hybrid tracking
  and final-error metrics still need retuning.
- MPC and hybrid solve-time distributions contain large outliers, so real-time
  claims must be made carefully.

This means the current hybrid architecture is implemented, but the next phase
must focus on controller design quality and benchmarking discipline, not just
on adding more modules.

## 5. Validation Status of This Documentation Pass

This pass included feature integration and smoke validation of dynamic obstacle
and adaptive execution paths.

Validation completed on 2026-04-06:

- End-to-end standalone smoke tests (10s, no-plot):
  - Hybrid with moving obstacles
  - Hybrid with random-walk obstacles
  - Adaptive MPC mode
  - Hybrid LQR + Adaptive MPC mode
  - Hybrid LQR + Adaptive MPC with moving obstacles
- Validation details observed during runs:
  - Dynamic obstacles move over time in both dynamic scenarios.
  - Adaptive parameter estimates update continuously during adaptive modes.
  - Hybrid-adaptive mode produces risk-driven blended control outputs.
- Platform note:
  - Adaptive modes on some Windows Python builds can hit CasADi teardown
    failures. The runner now uses a Windows adaptive-mode clean-exit safeguard,
    and validation was executed with Python 3.11.

Implication:

- The repository state described here now includes both stored evidence and a
  fresh smoke validation of the new trajectory features.

## 6. Literature-Driven Gap Analysis

### What the literature already supports well

1. Adaptive MPC for uncertain nonlinear or wheeled systems is mature enough to
   be used as a principled comparison path.
2. Differential-drive safety-critical LMPC plus CBF is already well aligned
   with the current repository design.
3. Moving-obstacle handling is commonly done with predicted safe envelopes,
   often ellipsoidal, and not with static circles alone.
4. Uncertainty-aware navigation is increasingly framed in distributionally
   robust or risk-aware terms rather than with fixed margins only.
5. Adaptive horizon and waypoint guidance are credible control problems, not
   merely user-interface conveniences.

### What is still missing in this repository

1. The adaptive MPC branch is integrated in standalone execution, but not yet
   in the statistical evaluation runner.
2. Checkpoint-mode is implemented, but broader evaluation defaults still need
  to exercise checkpoint-specific metrics across all controller/situation
  combinations.
3. Risk and safety logic do not yet use full obstacle-motion uncertainty
   propagation and covariance-based sensing models.
4. Evaluation and ROS paths do not yet expose the same adaptive-mode breadth as
   standalone mode.
5. The benchmark set is not yet realistic enough for the intended comparative
   study.

### Project contribution path

The most defensible next contribution is not "adaptive MPC alone". It is:

`baseline hybrid LQR-MPC` versus `adaptive-MPC-enhanced navigation`, both under
checkpoint tracking and moving-obstacle uncertainty.

That contribution is stronger because it compares:

- a fixed-model hybrid controller
- an adaptive predictive controller
- and eventually a hybrid controller that uses adaptive MPC in the high-risk
  branch

## 7. GO / NO-GO Decisions

### 7.1 Integrate Adaptive MPC as a Comparative Controller

Research support:

- Kohler (2026), arXiv:2603.17843
- Prakash et al. (2022), arXiv:2201.00863

Feasibility:

- The repository already contains `adaptive_mpc_controller.py`
- The main missing work is integration, tuning, and evaluation wiring

Decision:

- GO

Primary files:

- `src/hybrid_controller/hybrid_controller/controllers/adaptive_mpc_controller.py`
- `run_simulation.py`
- `evaluation/statistical_runner.py`

### 7.2 Sustain and Benchmark Checkpoint-Based References

Research support:

- Li et al. (2021), adaptive horizon EMPC
- Himanshu et al. (2025), waypoint guidance

Feasibility:

- The trajectory module is already isolated in `reference_generator.py`
- The standalone workflow already supports checkpoint paths and richer analytic
  families
- The remaining work is to advance from precomputed checkpoint curves to local
  online horizon construction tied to the active checkpoint queue

Decision:

- GO

Primary files:

- `src/hybrid_controller/hybrid_controller/trajectory/reference_generator.py`
- `run_simulation.py`

### 7.3 Add Moving Obstacles in a Bounded Environment

Research support:

- Jian et al. (2023), dynamic CBF-MPC
- Rosenfelder et al. (2025), ellipsoidal obstacle avoidance

Feasibility:

- The evaluation framework already owns scenario generation
- The risk and MPC modules already accept obstacle data structures

Decision:

- GO

Primary files:

- `evaluation/scenarios.py`
- `src/hybrid_controller/hybrid_controller/controllers/risk_metrics.py`
- `src/hybrid_controller/hybrid_controller/controllers/mpc_controller.py`

### 7.4 Add Random-Walk Obstacles With Safety and Sensing Factors

Research support:

- Jian et al. (2023), predicted dynamic-obstacle envelopes
- Ryu and Mehr (2024), predictive motion uncertainty plus robust risk-aware control
- Wu and Ning (2025), distributionally robust MPC

Feasibility:

- Requires a new obstacle-motion predictor and envelope inflation rule
- Fits the current architecture without breaking existing modes

Decision:

- GO

Primary files:

- `evaluation/scenarios.py`
- `src/hybrid_controller/hybrid_controller/controllers/risk_metrics.py`
- `src/hybrid_controller/hybrid_controller/controllers/mpc_controller.py`

## 8. Recommended Mathematical Upgrade for Safety Margins

For moving obstacles, use a predicted obstacle envelope instead of a fixed
radius:

`r_eff(k) = r_obs + d_safe + d_sensor(k) + d_model(k)`

Recommended meaning:

- `r_obs` : nominal obstacle radius
- `d_safe` : fixed collision buffer
- `d_sensor(k)` : uncertainty inflation from the predicted covariance
- `d_model(k)` : extra margin for motion-model mismatch or solver conservatism

If the obstacle follows a random walk:

`o_{k+1} = o_k + w_k`

then the covariance grows over time, and `d_sensor(k)` should grow with the
prediction horizon. This gives a principled way to build the additional safety
and sensing factor requested for the next implementation phase.

## 9. Proposed Next Implementation Order

### Step 1

Advance the new checkpoint-capable trajectory library into a local online
checkpoint-horizon manager. Keep figure-8 as one benchmark mode.

### Step 2

Expose adaptive MPC as a new simulation and evaluation mode without replacing
the current hybrid baseline.

### Step 3

Add bounded moving-obstacle generators to `evaluation/scenarios.py`.

### Step 4

Extend `risk_metrics.py` to reason about predicted obstacle states and their
uncertainty, not only current obstacle positions.

### Step 5

Update the high-risk control branch to use predicted safety envelopes, starting
with inflated circles or ellipses, then advancing to stronger moving-obstacle
constraints if needed.

### Step 6

Run the full comparative study:

1. LQR
2. MPC
3. Hybrid LQR-MPC
4. Adaptive MPC
5. Hybrid LQR-Adaptive-MPC

## 10. Immediate Engineering Targets

| Target | Why First | Expected Outcome |
|---|---|---|
| Local checkpoint manager | Finishes the transition started by the new checkpoint-path support | Controllers stop depending on a fully precomputed checkpoint reference |
| Adaptive MPC integration | Uses code that already exists in the repo | Fair comparative baseline becomes possible |
| Moving-obstacle scenarios | Makes the evaluation more realistic | Static-only benchmarks are no longer the sole evidence base |
| Uncertainty-aware safety inflation | Answers the sensing-factor requirement directly | Stronger, more defensible applicative safety story |
| Full robot-in-Gazebo coupling | Replaces the current lightweight odometry shim | The ROS2 path becomes a truer plant-in-the-loop benchmark |

## 11. Documentation Map

For the current phase, the most relevant documents are:

- `README.md` for the high-level architecture and roadmap
- `code_review.md` for the current audit findings
- `docs/Code_Review.md` for the detailed technical module audit
- `docs/Docker_Gazebo_Workflow.md` for the container, artifact, and Gazebo path
- `REFERENCES.md` for the selected bibliography
- `Resources/Papers_and_Journals/2026-04-05_adaptive_mpc_research.md` for the
  full research synthesis

## Appendix A. Retained Original Mathematical Foundation Notes

The earlier work-progress document contained the mathematical foundation notes
below. They are retained here instead of being replaced.

### A.1 Differential Drive Robot Modeling

The robot state vector is

$$
x = \begin{bmatrix} p_x \\ p_y \\ \theta \end{bmatrix}
$$

with control input

$$
u = \begin{bmatrix} v \\ \omega \end{bmatrix}.
$$

The continuous-time nonlinear kinematic model is

$$
\dot{p}_x = v\cos\theta, \qquad
\dot{p}_y = v\sin\theta, \qquad
\dot{\theta} = \omega.
$$

### A.2 Linearization Around a Reference Trajectory

Let

$$
x_r(t) = \begin{bmatrix} p_{x,r}(t) \\ p_{y,r}(t) \\ \theta_r(t) \end{bmatrix},
\qquad
u_r(t) = \begin{bmatrix} v_r(t) \\ \omega_r(t) \end{bmatrix}.
$$

Define the tracking error

$$
\tilde{x} = x - x_r, \qquad \tilde{u} = u - u_r.
$$

The Jacobians are

$$
A = \left.\frac{\partial f}{\partial x}\right|_{(x_r,u_r)}, \qquad
B = \left.\frac{\partial f}{\partial u}\right|_{(x_r,u_r)}
$$

which evaluate to

$$
A = \begin{bmatrix}
0 & 0 & -v_r\sin\theta_r \\
0 & 0 & v_r\cos\theta_r \\
0 & 0 & 0
\end{bmatrix},
\qquad
B = \begin{bmatrix}
\cos\theta_r & 0 \\
\sin\theta_r & 0 \\
0 & 1
\end{bmatrix}.
$$

Hence

$$
\dot{\tilde{x}} = A\tilde{x} + B\tilde{u}.
$$

### A.3 Discretization for Digital Control

With sampling time $T_s$, the discrete-time model is

$$
\tilde{x}_{k+1} = A_d \tilde{x}_k + B_d \tilde{u}_k
$$

with

$$
A_d = e^{AT_s} \approx I + A T_s, \qquad B_d \approx B T_s.
$$

Explicitly,

$$
A_d = \begin{bmatrix}
1 & 0 & -v_r \sin\theta_r T_s \\
0 & 1 & v_r \cos\theta_r T_s \\
0 & 0 & 1
\end{bmatrix},
\qquad
B_d = \begin{bmatrix}
\cos\theta_r T_s & 0 \\
\sin\theta_r T_s & 0 \\
0 & T_s
\end{bmatrix}.
$$

### A.4 LQR Controller Formulation

The infinite-horizon cost is

$$
J = \sum_{k=0}^{\infty} \left(
\tilde{x}_k^{\top} Q \tilde{x}_k + \tilde{u}_k^{\top} R \tilde{u}_k
\right)
$$

with

$$
Q = \text{diag}(q_x, q_y, q_\theta), \qquad
R = \text{diag}(r_v, r_\omega).
$$

The DARE is

$$
P = A_d^{\top} P A_d - A_d^{\top} P B_d
(R + B_d^{\top} P B_d)^{-1} B_d^{\top} P A_d + Q
$$

and the optimal feedback gain is

$$
K = (R + B_d^{\top} P B_d)^{-1} B_d^{\top} P A_d.
$$

The final control law is

$$
\tilde{u}_k = -K\tilde{x}_k,
\qquad
u_k = u_{r,k} - K(x_k - x_{r,k}).
$$

Reference implementation retained from the earlier document:

```python
import numpy as np
from scipy.linalg import solve_discrete_are

def get_lqr_gain(v_r, theta_r, dt):
    Ad = np.array([[1, 0, -v_r * np.sin(theta_r) * dt],
                   [0, 1,  v_r * np.cos(theta_r) * dt],
                   [0, 0,  1]])

    Bd = np.array([[np.cos(theta_r) * dt, 0],
                   [np.sin(theta_r) * dt, 0],
                   [0, dt]])

    Q = np.diag([10.0, 10.0, 1.0])
    R = np.diag([0.1, 0.1])

    P = solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.inv(R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)
    return K
```

### A.5 Reference Trajectory Generation

The earlier benchmark trajectory used a figure-8:

$$
p_{x,r}(t) = A \sin(at), \qquad
p_{y,r}(t) = A \sin(at)\cos(at)
$$

with

$$
\theta_r(t) = \arctan2(\dot{p}_{y,r}(t), \dot{p}_{x,r}(t)),
\qquad
v_r(t) = \sqrt{\dot{p}_{x,r}(t)^2 + \dot{p}_{y,r}(t)^2},
\qquad
\omega_r(t_k) = \frac{\theta_r(t_{k+1}) - \theta_r(t_k)}{T_s}.
$$

Reference implementation retained from the earlier document:

```python
import numpy as np

def generate_figure_8(t_range, dt, A=2.0, a=0.5):
    t = np.arange(0, t_range, dt)
    xr = A * np.sin(a * t)
    yr = A * np.sin(a * t) * np.cos(a * t)
    dxr = a * A * np.cos(a * t)
    dyr = a * A * (np.cos(a * t)**2 - np.sin(a * t)**2)
    thetar = np.arctan2(dyr, dxr)
    vr = np.sqrt(dxr**2 + dyr**2)
    wr = np.diff(thetar, append=thetar[-1]) / dt
    return np.stack([xr, yr, thetar, vr, wr], axis=1)
```

### A.6 Safety-Critical Control Layer

The earlier document summarized the MPC layer as

$$
x_{k+1} = A_k x_k + B_k u_k
$$

with finite-horizon cost

$$
\min_{u_0,\dots,u_{N-1}}
\left(
\sum_{k=0}^{N-1}
\left(
\|x_k - x_{\text{ref},k}\|_Q^2 + \|u_k\|_R^2
\right)
+
\|x_N - x_{\text{ref},N}\|_P^2
\right)
$$

subject to actuator and obstacle constraints.

For obstacle avoidance, the earlier notes used the nonlinear safe-set template

$$
(x_k - x_{\text{obs},i})^2 + (y_k - y_{\text{obs},i})^2
\ge d_{\text{safe}}^2
$$

and soft constraints of the form

$$
(x_k - x_{\text{obs},i})^2 + (y_k - y_{\text{obs},i})^2
\ge d_{\text{safe}}^2 - \epsilon_k
$$

with a quadratic slack penalty.

### A.7 Standard QP Form

The retained QP summary is

$$
\min_{z} \frac{1}{2} z^\top P z + q^\top z,
\qquad
\text{s.t. } l \leq Az \leq u
$$

with decision vector

$$
z = \begin{bmatrix}
x_1 \\ x_2 \\ \vdots \\ x_N \\ u_0 \\ u_1 \\ \vdots \\ u_{N-1}
\end{bmatrix}.
$$

### A.8 Earlier ROS and MPC Node Notes

The retained node notes recorded:

- `/odom` as the state-estimate source
- `/cmd_vel` as the command output
- `/mpc_obstacles` as the obstacle input
- CVXPY and OSQP as the baseline linearized MPC stack

These older implementation notes remain historically relevant, while the main
body of this document now records the newer adaptive-MPC and dynamic-obstacle
roadmap.

# Risk-Aware Hybrid LQR–MPC Navigation for Autonomous Systems

## Work Progress Document

> Mathematical foundations and implementation details for the hybrid control architecture.

---

## Table of Contents

1. [Differential Drive Robot Modeling](#1-differential-drive-robot-modeling)
2. [Linearization Around Reference Trajectory](#2-linearization-around-a-reference-trajectory)
3. [Discretization for Digital Control](#3-discretization-for-digital-control)
4. [LQR Controller Formulation](#4-lqr-controller-formulation)
5. [Reference Trajectory Generation](#5-reference-trajectory-generation)
6. [Safety-Critical Control Layer (MPC)](#6-safety-critical-control-layer-mpc)
7. [QP Formulation for MPC](#7-qp-formulation-for-mpc)
8. [MPC Node Implementation](#8-mpc-node-implementation)

---

## 1. Differential Drive Robot Modeling

### 1.1 State and Input Definitions

The robot state vector:

$$
x = \begin{bmatrix} p_x \\ p_y \\ \theta \end{bmatrix}
$$

where $p_x, p_y$ represent planar position and $\theta$ is orientation.

Control inputs:

$$
u = \begin{bmatrix} v \\ \omega \end{bmatrix}
$$

where $v$ is linear velocity and $\omega$ is angular velocity.

### 1.2 Continuous-Time Nonlinear Kinematic Model

The motion of a differential drive robot is governed by:

$$
\dot{p}_x = v\cos\theta, \qquad \dot{p}_y = v\sin\theta, \qquad \dot{\theta} = \omega
$$

These equations define how the robot's pose evolves over time based on commanded velocities.

---

## 2. Linearization Around a Reference Trajectory

### 2.1 Reference Trajectory and Error State

Let the nominal planned trajectory be:

$$
x_r(t) = \begin{bmatrix} p_{x,r}(t) \\ p_{y,r}(t) \\ \theta_r(t) \end{bmatrix}, \qquad
u_r(t) = \begin{bmatrix} v_r(t) \\ \omega_r(t) \end{bmatrix}
$$

The tracking error is defined as:

$$
\tilde{x} = x - x_r, \qquad \tilde{u} = u - u_r
$$

### 2.2 Jacobian-Based Linearization

The linearized model is obtained by computing Jacobians:

$$
A = \left.\frac{\partial f}{\partial x}\right|_{(x_r,u_r)}, \qquad
B = \left.\frac{\partial f}{\partial u}\right|_{(x_r,u_r)}
$$

Evaluating the derivatives gives:

$$
A = \begin{bmatrix}
0 & 0 & -v_r\sin\theta_r \\
0 & 0 & v_r\cos\theta_r \\
0 & 0 & 0
\end{bmatrix}, \qquad
B = \begin{bmatrix}
\cos\theta_r & 0 \\
\sin\theta_r & 0 \\
0 & 1
\end{bmatrix}
$$

The linearized error dynamics become:

$$
\dot{\tilde{x}} = A\tilde{x} + B\tilde{u}
$$

---

## 3. Discretization for Digital Control

### 3.1 Zero-Order Hold Discretization

With sampling time $T_s$, the discrete-time model is:

$$
\tilde{x}_{k+1} = A_d \tilde{x}_k + B_d \tilde{u}_k
$$

where:

$$
A_d = e^{AT_s} \approx I + A T_s, \qquad B_d \approx B T_s
$$

Explicitly:

$$
A_d = \begin{bmatrix}
1 & 0 & -v_r \sin\theta_r T_s \\
0 & 1 & v_r \cos\theta_r T_s \\
0 & 0 & 1
\end{bmatrix}, \qquad
B_d = \begin{bmatrix}
\cos\theta_r T_s & 0 \\
\sin\theta_r T_s & 0 \\
0 & T_s
\end{bmatrix}
$$

---

## 4. LQR Controller Formulation

### 4.1 Quadratic Cost Function

The LQR minimizes the infinite-horizon cost:

$$
J = \sum_{k=0}^{\infty} \left( \tilde{x}_k^{\top} Q\, \tilde{x}_k + \tilde{u}_k^{\top} R\, \tilde{u}_k \right)
$$

where:
- $Q = \text{diag}(q_x, q_y, q_\theta)$ penalizes tracking errors
- $R = \text{diag}(r_v, r_\omega)$ penalizes control effort

### 4.2 Discrete Algebraic Riccati Equation (DARE)

$$
P = A_d^{\top} P A_d - A_d^{\top} P B_d (R + B_d^{\top} P B_d)^{-1} B_d^{\top} P A_d + Q
$$

### 4.3 Optimal LQR Feedback Gain

$$
K = (R + B_d^{\top} P B_d)^{-1} B_d^{\top} P A_d
$$

### 4.4 Final Control Law

$$
\tilde{u}_k = -K\tilde{x}_k, \qquad u_k = u_{r,k} - K(x_k - x_{r,k})
$$

### 4.5 Python Implementation

```python
import numpy as np
from scipy.linalg import solve_discrete_are

def get_lqr_gain(v_r, theta_r, dt):
    # 1. Define discrete linearized matrices
    Ad = np.array([[1, 0, -v_r * np.sin(theta_r) * dt],
                   [0, 1,  v_r * np.cos(theta_r) * dt],
                   [0, 0,  1]])
    
    Bd = np.array([[np.cos(theta_r) * dt, 0],
                   [np.sin(theta_r) * dt, 0],
                   [0, dt]])

    # 2. Cost weighting matrices
    Q = np.diag([10.0, 10.0, 1.0])  # State error penalty
    R = np.diag([0.1, 0.1])         # Control effort penalty

    # 3. Solve DARE
    P = solve_discrete_are(Ad, Bd, Q, R)

    # 4. Compute optimal LQR gain
    K = np.linalg.inv(R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)
    
    return K
```

---

## 5. Reference Trajectory Generation

### 5.1 Figure-8 (Lemniscate) Trajectory

The reference position is defined as:

$$
p_{x,r}(t) = A \sin(at), \qquad p_{y,r}(t) = A \sin(at)\cos(at)
$$

where:
- $A$ controls the spatial size of the trajectory
- $a$ controls the temporal speed of traversal

### 5.2 Heading and Velocities

**Heading angle:**
$$
\theta_r(t) = \arctan2(\dot{p}_{y,r}(t), \dot{p}_{x,r}(t))
$$

**Linear velocity:**
$$
v_r(t) = \sqrt{\dot{p}_{x,r}(t)^2 + \dot{p}_{y,r}(t)^2}
$$

**Angular velocity:**
$$
\omega_r(t_k) = \frac{\theta_r(t_{k+1}) - \theta_r(t_k)}{T_s}
$$

### 5.3 Python Implementation

```python
import numpy as np

def generate_figure_8(t_range, dt, A=2.0, a=0.5):
    t = np.arange(0, t_range, dt)
    
    # Reference position
    xr = A * np.sin(a * t)
    yr = A * np.sin(a * t) * np.cos(a * t)
    
    # Derivatives
    dxr = a * A * np.cos(a * t)
    dyr = a * A * (np.cos(a * t)**2 - np.sin(a * t)**2)
    
    # Heading
    thetar = np.arctan2(dyr, dxr)
    
    # Linear velocity
    vr = np.sqrt(dxr**2 + dyr**2)
    
    # Angular velocity (numerical derivative)
    wr = np.diff(thetar, append=thetar[-1]) / dt
    
    return np.stack([xr, yr, thetar, vr, wr], axis=1)
```

---

## 6. Safety-Critical Control Layer (MPC)

The MPC augments the nominal LQR controller with constraint-aware optimization.

### 6.1 Linear Time-Varying MPC Dynamics

At every control cycle, the dynamics are locally linearized:

$$
x_{k+1} = A_k x_k + B_k u_k
$$

### 6.2 Finite-Horizon Optimal Control Problem

$$
\min_{u_0,\dots,u_{N-1}} \left( \sum_{k=0}^{N-1} \left( \|x_k - x_{\text{ref},k}\|_Q^2 + \|u_k\|_R^2 \right) + \|x_N - x_{\text{ref},N}\|_P^2 \right)
$$

where:
- $x_k = [x_k, y_k, \theta_k]^\top$ is the predicted state
- $u_k = [v_k, \omega_k]^\top$ is the control input
- $Q \succeq 0$ penalizes state tracking error
- $R \succ 0$ penalizes control effort
- $P$ is the terminal cost ensuring stability

### 6.3 Obstacle Avoidance Constraints

For each obstacle at $(x_{\text{obs},i}, y_{\text{obs},i})$:

$$
(x_k - x_{\text{obs},i})^2 + (y_k - y_{\text{obs},i})^2 \geq d_{\text{safe}}^2
$$

### 6.4 Actuator Constraints

$$
|v_k| \leq v_{\max}, \qquad |\omega_k| \leq \omega_{\max}
$$

### 6.5 Soft-Constraint Mechanism

Using slack variables $\epsilon_k \geq 0$:

$$
(x_k - x_{\text{obs},i})^2 + (y_k - y_{\text{obs},i})^2 \geq d_{\text{safe}}^2 - \epsilon_k
$$

with added penalty:

$$
J_{\text{slack}} = \sum_{k=0}^{N-1} \rho \, \epsilon_k^2
$$

### 6.6 Warm-Start Strategy

The MPC employs warm-start by shifting the previous solution:

$$
u_0^{\text{warm}} = u_1^{\text{prev}}, \quad \dots, \quad u_{N-1}^{\text{warm}} = u_{N-2}^{\text{prev}}
$$

---

## 7. QP Formulation for MPC

### 7.1 Standard QP Structure

$$
\min_{z} \frac{1}{2} z^\top P z + q^\top z, \qquad \text{s.t. } l \leq Az \leq u
$$

### 7.2 Decision Variable Vector

$$
z = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_N \\ u_0 \\ u_1 \\ \vdots \\ u_{N-1} \end{bmatrix}
$$

Dimension: $\dim(z) = 3N + 2N = 5N$

### 7.3 Constraint Structure

$$
A = \begin{bmatrix} A_{\text{dyn}} \\ A_{\text{act}} \\ A_{\text{obs}} \end{bmatrix}, \qquad
l = \begin{bmatrix} 0 \\ l_{\text{act}} \\ l_{\text{obs}} \end{bmatrix}, \qquad
u = \begin{bmatrix} 0 \\ u_{\text{act}} \\ u_{\text{obs}} \end{bmatrix}
$$

---

## 8. MPC Node Implementation

### 8.1 ROS Interfaces

**Subscribers:**
- `/odom` (nav_msgs/Odometry) - robot's estimated position and orientation
- `/mpc_obstacles` (std_msgs/Float32MultiArray) - obstacle centroids

**Publishers:**
- `/cmd_vel` (geometry_msgs/Twist) - linear and angular velocities

### 8.2 Solver Configuration

The MPC uses CVXPY/OSQP with:
- Adaptive penalty parameter updates
- Sparse matrix factorization
- Box constraints for actuator bounds
- Warm-starting enabled

### 8.3 Integration with Hybrid Architecture

The MPC output is passed to the robot's actuation system. When selected by the supervisory logic, MPC ensures controls satisfy both motion constraints and collision-avoidance requirements.

---

## Authors

| Name | GitHub | Email |
|------|--------|-------|
| Kshitiz | [@Erebuzzz](https://github.com/Erebuzzz) | kshitiz23@iiserb.ac.in |
| Agolika | [@Agolika413](https://github.com/Agolika413) | agolika23@iiserb.ac.in |

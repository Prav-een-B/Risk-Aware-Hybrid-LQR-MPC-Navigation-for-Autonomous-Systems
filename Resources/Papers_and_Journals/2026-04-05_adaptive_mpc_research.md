# Research Packet
# Adaptive MPC, Checkpoint Tracking, and Dynamic-Obstacle Roadmap

Date: 2026-04-05

## 1. Scope

This research pass was tied to the next planned work items for the repository:

1. Keep the current hybrid LQR-MPC controller as the baseline
2. Add an adaptive-MPC-based comparison path
3. Move from full reference trajectories to checkpoint-driven tracking
4. Add moving obstacles in a bounded environment
5. Add random-walk obstacle motion with explicit safety and sensing margins
6. Add more complex trajectory families for realistic benchmarking

## 2. APIs and Primary Sources Used

### APIs

- arXiv API
- OpenAlex API
- Elsevier ScienceDirect API

### Representative query themes

- certainty-equivalent adaptive MPC nonlinear systems
- adaptive MPC wheeled mobile robot
- adaptive prediction horizon wheeled mobile robot obstacle avoidance
- dynamic control barrier function mobile robot
- moving obstacle avoidance mobile robot MPC
- distributionally robust risk-aware robot navigation
- waypoint guidance mobile robot

### Local theory sources

- `Resources/Books/Model_Predictive_Control_Theory_Computation_and_Design.pdf`
- `Resources/Books/Predictive_Control_for_Linear_and_Hybrid_Systems.pdf`
- `Resources/Books/Feedback_Systems_An_Introduction_for_Scientists_and_Engineers.pdf`

## 3. Current Codebase Alignment

The code audit shows the following current state:

- `lqr_controller.py`, `mpc_controller.py`, `hybrid_blender.py`, and
  `run_simulation.py` already support the baseline hybrid controller
- `adaptive_mpc_controller.py` already exists but is experimental and not wired
  into the runnable modes or evaluation framework
- `reference_generator.py` only supports full time-parameterized trajectories,
  mainly figure-8
- `risk_metrics.py` handles static obstacle distance risk and optional predicted
  state risk, but not sensing covariance or obstacle-motion prediction
- `evaluation/scenarios.py` only covers static obstacles

This means the research should inform integration and benchmarking work first,
not a full rewrite.

## 4. Selected Literature

| Topic | Reference | Why It Matters | Project Mapping | Relevance |
|---|---|---|---|---|
| Adaptive MPC foundation | Johannes Kohler (2026), "Certainty-equivalent adaptive MPC for uncertain nonlinear systems", arXiv:2603.17843 | Gives the most relevant theoretical template for certainty-equivalent adaptive MPC with least-mean-square adaptation, artificial references, soft state constraints, and robustness guarantees that scale with disturbance energy and parameter variation. | `adaptive_mpc_controller.py`, future adaptive hybrid mode, documentation of guarantees | Critical |
| Adaptive MPC for nonholonomic robot | Nikhil Potu Surya Prakash, Tamara Perreault, Trevor Voth, Zejun Zhong (2022), "Adaptive Model Predictive Control of Wheeled Mobile Robots", arXiv:2201.00863 | Directly studies an adaptive MPC controller for a two-wheeled mobile robot with unknown inertia and recursive parameter updates. | Adaptive mobile-robot identification law, parameter-update strategy | Critical |
| Linear MPC plus safety for diff-drive | Ali Mohamed Ali, Chao Shen, Hashim A. Hashim (2024), "A Linear MPC with Control Barrier Functions for Differential Drive Robots", arXiv:2404.10018 | Closest published analogue to the existing diff-drive safety stack. It supports recursive-feasibility and stability language for the current MPC branch. | `mpc_controller.py`, `risk_metrics.py`, future moving-obstacle safety layer | Critical |
| Adaptive horizon for tracking and obstacle avoidance | Peng Li, Shizhan Wang, Hongjiu Yang, Hai Zhao (2021), "Trajectory Tracking and Obstacle Avoidance for Wheeled Mobile Robots Based on EMPC With an Adaptive Prediction Horizon", IEEE Transactions on Cybernetics, DOI: 10.1109/TCYB.2021.3125333 | Shows that adapting the prediction horizon is useful for tracking plus obstacle avoidance. This is especially relevant when replacing a full reference trajectory with local checkpoint segments. | Checkpoint horizon manager, adaptive lookahead, comparative baseline | High |
| Dynamic obstacles with predicted envelopes | Zhuozhu Jian et al. (2023), "Dynamic Control Barrier Function-based Model Predictive Control to Safety-Critical Obstacle-Avoidance of Mobile Robot", ICRA 2023, DOI: 10.1109/ICRA48891.2023.10160857; arXiv:2209.08539 | Uses clustered obstacle ellipses, Kalman-filter-based obstacle prediction, and inflated semi-axes for safety. This is a direct bridge from static circular obstacles to predicted moving obstacles. | `evaluation/scenarios.py`, `risk_metrics.py`, dynamic-obstacle constraints | Critical |
| Efficient obstacle geometry for MPC | Mario Rosenfelder et al. (2025), "Efficient avoidance of ellipsoidal obstacles with model predictive control for mobile robots and vehicles", Mechatronics, DOI: 10.1016/j.mechatronics.2025.103386 | Supports using ellipsoidal obstacle models instead of only circles. This is attractive for predicted moving-obstacle envelopes and bounded-environment scenarios. | Obstacle representation, computationally lighter safe sets | High |
| Safe unicycle tracking with relaxed CBFs | Ali Rahmanian, Mohammad Hassan Asemani (2026), "Unified T-S fuzzy-relaxed control barrier function framework for unicycle Mobile robot safe trajectory tracking", ISA Transactions, DOI: 10.1016/j.isatra.2026.02.025 | This paper is newer and lightly cited, but the title and venue strongly suggest a unicycle-safe-tracking formulation that relaxes barrier constraints for better feasibility. This is an inference from metadata and title, not from a retrieved abstract. | Safety-margin relaxation for moving-obstacle tracking | Medium |
| Distributionally robust navigation with predicted motion uncertainty | Kanghyun Ryu, Negar Mehr (2024), "Integrating Predictive Motion Uncertainties with Distributionally Robust Risk-Aware Control for Safe Robot Navigation in Crowds", ICRA 2024, DOI: 10.1109/ICRA57147.2024.10610404 | Provides the right conceptual bridge for uncertainty-aware obstacle prediction and risk-aware safety margins. | Sensing factor, uncertainty-aware risk inflation, crowd-like moving obstacles | High |
| DR-MPC for multimodal disturbances | Jingyi Wu, Chao Ning (2025), "Distributionally Robust Model Predictive Control with Mixture of Gaussian Processes", arXiv:2502.05448 | Gives a strong route for adding principled robustness margins when obstacle or model uncertainty is not unimodal. | Safety factor design, robust adaptive MPC extension | High |
| Waypoint guidance | Himanshu, Raja Rout, Tarun Kumar Bera, Nizar Chatti (2025), "Fault tolerant self-reconfigurable waypoint guidance for mobile robots under actuator faults", Journal of the Franklin Institute, DOI: 10.1016/j.jfranklin.2025.107735 | The exact contribution was inferred from title and metadata. It still supports the shift from full trajectory playback to waypoint or checkpoint guidance as a serious control topic rather than a demo convenience. | Checkpoint manager and fault-aware tracking logic | Medium |
| Trajectory-learning and obstacle avoidance | Tao Ma, Burkhard Corves (2025), "Model Predictive Control-based dynamic movement primitives for trajectory learning and obstacle avoidance", Robotics and Autonomous Systems, DOI: 10.1016/j.robot.2025.105027 | Suggests a path to richer benchmark trajectories and data-driven motion primitives while staying inside an MPC framework. | More realistic trajectory library and checkpoint generation | Medium |
| Dynamic-obstacle CBF alternative | Zheng Zhang, Guang-Hong Yang (2025), "Dynamic obstacle avoidance for car-like mobile robots based on neurodynamic optimization with control barrier functions", Neurocomputing, DOI: 10.1016/j.neucom.2025.131252 | A second moving-obstacle CBF reference. It is useful as a cross-check that dynamic-obstacle CBF formulations are active and viable beyond one paper. | Dynamic-obstacle safety formulations | Medium |

## 5. Synthesis for This Repository

### 5.1 Adaptive MPC should be integrated as a new branch, not a replacement

The current repository already has an `adaptive_mpc_controller.py`, but it is
not exposed in the CLI or evaluation stack. The literature strongly supports
adding an adaptive MPC comparison path while keeping the current hybrid
controller intact.

Recommended comparison set:

1. LQR
2. MPC
3. Hybrid LQR-MPC
4. Adaptive MPC
5. Hybrid LQR-Adaptive-MPC

### 5.2 Checkpoint tracking is the right next abstraction

The current trajectory generator assumes a fully sampled reference path. That
is useful for controlled benchmarks, but it is too rigid for realistic
navigation.

Recommended change:

- represent the route as checkpoints or waypoints
- generate the local horizon reference online from the current checkpoint and
  the next few checkpoints
- optionally adapt the horizon length based on checkpoint geometry and risk

This keeps the controller local and makes complex real-world routes easier to
test.

### 5.3 Moving obstacles should be modeled through predicted safety envelopes

The strongest research-aligned implementation path is:

1. predict obstacle motion over the horizon
2. wrap each predicted obstacle as an inflated ellipse
3. use either MPC constraints or a D-CBF layer on top of those ellipses

This is better aligned with the literature than keeping only static circular
obstacles.

### 5.4 Random-walk obstacles need explicit sensing and safety inflation

For a random-walk obstacle model

`o_{k+1} = o_k + w_k`

with `w_k ~ N(0, Q_obs)`, the obstacle covariance grows over the horizon.
The practical safe-envelope rule for this repository should be:

`r_eff(k) = r_obs + d_safe + gamma_sensor * sqrt(lambda_max(Sigma_k)) + d_model`

where:

- `Sigma_k` is the predicted obstacle covariance
- `gamma_sensor` is the sensing-confidence multiplier
- `d_model` is an extra model-mismatch or actuation margin

This recommendation is a synthesis from the D-CBF moving-obstacle paper,
distributionally robust navigation literature, and the current repository
architecture. It is not copied from a single source.

### 5.5 Complex trajectories should be benchmark assets, not one-off demos

The current figure-8 path should become only one member of a trajectory suite.
The next suite should include:

- checkpoint polygons
- S-turns
- corridor slalom paths
- spline routes through sparse checkpoints
- dynamically re-routed waypoint paths near moving obstacles

## 6. Recommended Implementation Order

1. Add a checkpoint-based reference manager
2. Wire adaptive MPC into `run_simulation.py` and `evaluation/statistical_runner.py`
3. Add a bounded moving-obstacle scenario generator
4. Add obstacle prediction and uncertainty inflation in `risk_metrics.py`
5. Extend the obstacle representation in the MPC branch from circles to
   ellipses or inflated envelopes where needed
6. Run comparative evaluation across static and moving-obstacle scenarios

## 7. Concrete File Targets

| Goal | Primary Files |
|---|---|
| Checkpoint-based references | `src/hybrid_controller/hybrid_controller/trajectory/reference_generator.py`, `run_simulation.py` |
| Adaptive MPC integration | `src/hybrid_controller/hybrid_controller/controllers/adaptive_mpc_controller.py`, `run_simulation.py`, `evaluation/statistical_runner.py` |
| Moving-obstacle support | `evaluation/scenarios.py`, `src/hybrid_controller/hybrid_controller/controllers/risk_metrics.py`, `src/hybrid_controller/hybrid_controller/controllers/mpc_controller.py` |
| Comparative hybrid study | `run_simulation.py`, `evaluation/statistical_runner.py`, `docs/Work_Progress.md` |

## 8. Bottom Line

The literature supports the next phase clearly:

- yes to adaptive MPC as a comparative controller
- yes to checkpoint-based references
- yes to moving obstacles with predicted uncertainty-aware envelopes
- yes to additional safety and sensing factors grounded in predicted obstacle
  uncertainty
- yes to a richer trajectory library for realistic evaluation

The repository is already close enough structurally that the next step should
be targeted integration work rather than a redesign.

# Project Proposal

## Risk-Aware Hybrid LQR–MPC Navigation for Autonomous Systems

---

## 1. Introduction and Motivation

Autonomous robots must navigate safely and efficiently in environments that vary significantly in complexity. While linear feedback controllers (e.g., LQR) offer extreme computational efficiency, they fundamentally lack the predictive constraint enforcement required to remain collision-free in dynamic, cluttered spaces. Conversely, classical Model Predictive Control (MPC) elegantly enforces strict physical limits but imposes continuous non-linear optimization burdens that cripple real-time reactivity. 

To resolve this computational bottleneck without sacrificing structural safety guarantees, this research proposes a **Risk-Aware Hybrid LQR-MPC Navigation Controller**. By integrating the computational speed of LQR with the safety constraints of MPC, the system ensures optimal performance during nominal navigation while guaranteeing safe obstacle avoidance when risks are detected.

---

## 2. Research Objectives (Primary vs. Secondary)

### **Primary Objective: Risk-Aware Hybrid Integration**
Design and implement a unified hybrid navigation controller that dynamically switches between LQR and MPC based on a computed environmental risk metric. The primary focus is balancing computational efficiency with guaranteed collision avoidance in dynamic scenarios.

### **Secondary Objective & Research Novelty (Publishable Contributions) ✅ (Complete)**
To elevate the research beyond classical deterministic threshold switching (e.g., Euclidean distance limits) and traditional Tube-MPC robust forms, our advanced extensions focus on stochastic risk assessment and adaptive learning:

1. **Stochastic Model Predictive Control (SMPC)**: Incorporating probabilistically parameterized chance constraints ($\text{Pr}(\text{collision}) \le \epsilon$) instead of generic worst-case buffers.
2. **Covariance-Driven Mahalanobis Risk Metric**: An active probability-density-driven metric measuring the spatial and temporal density overlap between the robot's tracking estimation uncertainty ($\Sigma_x$) and stochastic obstacle motion ($\Sigma_{obs}$). This acts as a sophisticated, probabilistic supervisor for invoking SMPC only when multi-variate Gaussian collision probabilities breach safety thresholds.
3. **Adaptive Learning**: Utilizing online parameter-adapting mechanisms (Least Mean Squares tuning for velocity/steering mismatch) to enhance LQR tracking, preserving sub-millisecond execution times for >90% of nominal navigation.
4. **Dynamic Checkpoint Pathing**: Generating dynamic path spacing utilizing an obstacle-density exponential distribution equation to guarantee the MPC horizon remains kinematically feasible over varying constraints.

---

## 3. Problem Statement & Key Improvements

Develop a hybrid control architecture demonstrating measurable superiority over current State-Of-The-Art (SOTA) systems through the following steps:

- **Baseline Hybrid Navigation**: Create a seamless integration where LQR manages trajectory tracking and MPC solves real-time obstacle avoidance.
- **Advanced Probabilistic Safety**: Expand the standard hybrid logic to utilize real-time inverse-CDF parameterized chance constraints. 
- **Empirical Superiority Validation**: Test the unified architecture using Monte Carlo stress testing in dynamic (
andom_walk, moving) and high-clutter environments. The ultimate goal is to prove the hybrid design retains the 100% collision-free constraint guarantees of pure-SMPC, while drastically slashing cumulative compute cycles to match standard LQR speeds.

---

## 4. Engineering Phases

### Phase 1: Foundations ✅ (Complete)
- Select robot model and simulation environment
- Derive linearized dynamics
- Implement LQR baseline
- Achieve deterministic baseline tracking

### Phase 2: Safety Controller ✅ (Complete)
- Formulate MPC optimization problem
- Implement Checkpoint-Based pathing
- Basic obstacle avoidance frameworks

### Phase 3: Hybrid Integration ✅ (Complete)
- Formulate deterministic risk metrics (Euclidean distance base).
- Implement switching logic based on general risk thresholds.
- Output a baseline Hybrid Controller robust to static and dynamic obstacles.

### Phase 4: Stochastic Risk Supervisor (Advanced/Novelty) ✅ (Complete)
- Shift the underlying optimization to **Stochastic Model Predictive Control (SMPC)**.
- Implement Mahalanobis distance Covariance overlap for uncertainty-aware risk metrics.
- Predict constraint violations under Gaussian disturbance bounds automatically.

### Phase 5: Stochastic Experiments & Validation ✅ (Complete)
- Validate Mahalanobis-based SMPC and Adaptive LQR in highly cluttered and purely stochastic environments.
- Measure computational savings against pure SMPC execution.
- Experimental analysis via tuning of covariance thresholds for academic publication.

---

## 5. Performance Metrics

| Metric | Description |
|--------|-------------|
| **Computation Time** | Time per control step and comparative real-time feasibility vs pure MPC |
| **Collision Rate** | Frequency of safety-constraint violations under uncertainty |
| **Tracking Error** | Deviation from reference checkpoint trajectory |
| **Switching Statistics** | Activation frequency of the MPC vs LQR modes |

---

## 6. Implementation Environment

| Component | Choice |
|-----------|--------|
| **Simulation** | ROS 2 / Gazebo / Python environments |
| **Control Logic** | Hybrid LQR + MPC (with advanced SMPC extensions) |
| **Risk Metrics** | Multi-variate Mahalanobis Covariance-based approximations |

---

## Contact

| Name | GitHub | Email |
|------|--------|-------|
| Kshitiz | [@Erebuzzz](https://github.com/Erebuzzz) | kshitiz23@iiserb.ac.in |
| Agolika | [@Agolika413](https://github.com/Agolika413) | agolika23@iiserb.ac.in |

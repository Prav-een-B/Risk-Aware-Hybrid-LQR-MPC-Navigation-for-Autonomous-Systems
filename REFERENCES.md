# References

Academic and technical references that inform the current repository and the
next implementation phase.

Last updated: 2026-04-06

## 0. Implementation Update Note (2026-04-06)

- This integration pass (dynamic obstacles + adaptive and hybrid-adaptive
  standalone wiring) used the existing reference set already documented below.
- No new external papers were added in this pass.

## 1. Local Books

| Reference | Local Path | Main Use |
|---|---|---|
| Rawlings, J. B., Mayne, D. Q., Diehl, M. (2017). *Model Predictive Control: Theory, Computation, and Design* (2nd ed.) | `Resources/Books/Model_Predictive_Control_Theory_Computation_and_Design.pdf` | MPC cost design, terminal ingredients, soft constraints, robustness |
| Borrelli, F., Bemporad, A., Morari, M. (2017). *Predictive Control for Linear and Hybrid Systems* | `Resources/Books/Predictive_Control_for_Linear_and_Hybrid_Systems.pdf` | Hybrid MPC structure, constraint handling, stability |
| Astrom, K. J., Murray, R. M. (2008). *Feedback Systems: An Introduction for Scientists and Engineers* | `Resources/Books/Feedback_Systems_An_Introduction_for_Scientists_and_Engineers.pdf` | LQR, Lyapunov reasoning, tracking-system interpretation |

## 2. Core References for the Existing Repository

| Topic | Reference | Link | Main Use |
|---|---|---|---|
| Hybrid switching theory | Hespanha, J. P. et al. (2003). "Overcoming limitations of adaptive control by means of logic-based switching." | Systems and Control Letters 49(1) | Hysteresis and switching logic |
| Hybrid switching theory | Liberzon, D. (2003). *Switching in Systems and Control* | https://link.springer.com/book/10.1007/978-1-4612-0017-8 | Dwell-time and stability language |
| CVXPY code generation | Schaller, M., Banjac, G., Boyd, S. (2022). "Embedded Code Generation with CVXPY." | https://web.stanford.edu/~boyd/papers/cvxpygen.html | CVXPYgen acceleration path |
| Safety-critical diff-drive MPC | Ali Mohamed Ali, Chao Shen, Hashim A. Hashim (2024). "A Linear MPC with Control Barrier Functions for Differential Drive Robots." | https://arxiv.org/abs/2404.10018 | Diff-drive LMPC plus CBF safety reference |
| Robust stochastic MPC | Jingyi Wu, Chao Ning (2025). "Distributionally Robust Model Predictive Control with Mixture of Gaussian Processes." | https://arxiv.org/abs/2502.05448 | DR-CVaR and uncertainty-aware MPC extension |

## 3. Literature for the Next Planned Phase

### 3.1 Adaptive MPC

| Reference | Link | Why It Matters |
|---|---|---|
| Johannes Kohler (2026). "Certainty-equivalent adaptive MPC for uncertain nonlinear systems." | https://arxiv.org/abs/2603.17843 | Main theoretical anchor for adaptive MPC with least-mean-square adaptation, artificial references, and soft state constraints |
| Nikhil Potu Surya Prakash, Tamara Perreault, Trevor Voth, Zejun Zhong (2022). "Adaptive Model Predictive Control of Wheeled Mobile Robots." | https://arxiv.org/abs/2201.00863 | Direct adaptive MPC reference for a wheeled mobile robot |
| Peng Li, Shizhan Wang, Hongjiu Yang, Hai Zhao (2021). "Trajectory Tracking and Obstacle Avoidance for Wheeled Mobile Robots Based on EMPC With an Adaptive Prediction Horizon." | https://doi.org/10.1109/TCYB.2021.3125333 | Supports adaptive lookahead and local checkpoint horizon design |

### 3.2 Dynamic Obstacles and Safety Envelopes

| Reference | Link | Why It Matters |
|---|---|---|
| Zhuozhu Jian et al. (2023). "Dynamic Control Barrier Function-based Model Predictive Control to Safety-Critical Obstacle-Avoidance of Mobile Robot." | https://doi.org/10.1109/ICRA48891.2023.10160857 | Predicts moving obstacles and inflates ellipses for safe avoidance |
| Mario Rosenfelder et al. (2025). "Efficient avoidance of ellipsoidal obstacles with model predictive control for mobile robots and vehicles." | https://doi.org/10.1016/j.mechatronics.2025.103386 | Supports ellipsoidal obstacle models for efficient MPC constraints |
| Zheng Zhang, Guang-Hong Yang (2025). "Dynamic obstacle avoidance for car-like mobile robots based on neurodynamic optimization with control barrier functions." | https://doi.org/10.1016/j.neucom.2025.131252 | Additional dynamic-obstacle CBF reference |
| Ali Rahmanian, Mohammad Hassan Asemani (2026). "Unified T-S fuzzy-relaxed control barrier function framework for unicycle Mobile robot safe trajectory tracking." | https://doi.org/10.1016/j.isatra.2026.02.025 | Useful for feasibility-preserving safety relaxation in unicycle tracking |

### 3.3 Uncertainty-Aware Risk and Robust Safety

| Reference | Link | Why It Matters |
|---|---|---|
| Kanghyun Ryu, Negar Mehr (2024). "Integrating Predictive Motion Uncertainties with Distributionally Robust Risk-Aware Control for Safe Robot Navigation in Crowds." | https://doi.org/10.1109/ICRA57147.2024.10610404 | Strong basis for sensing-aware safety margins and crowd-like moving-obstacle prediction |
| Jingyi Wu, Chao Ning (2025). "Distributionally Robust Model Predictive Control with Mixture of Gaussian Processes." | https://arxiv.org/abs/2502.05448 | Gives a tractable robustification route for multimodal or poorly known disturbances |

### 3.4 Checkpoints, Waypoints, and Trajectory Diversity

| Reference | Link | Why It Matters |
|---|---|---|
| Himanshu, Raja Rout, Tarun Kumar Bera, Nizar Chatti (2025). "Fault tolerant self-reconfigurable waypoint guidance for mobile robots under actuator faults." | https://doi.org/10.1016/j.jfranklin.2025.107735 | Supports waypoint-guidance as a serious control problem rather than a visualization feature |
| Tao Ma, Burkhard Corves (2025). "Model Predictive Control-based dynamic movement primitives for trajectory learning and obstacle avoidance." | https://doi.org/10.1016/j.robot.2025.105027 | Relevant for building a broader trajectory and checkpoint benchmark suite |

## 4. Supporting APIs

| API | Endpoint | Current Use |
|---|---|---|
| arXiv API | https://info.arxiv.org/help/api/index.html | Exact paper metadata and recent abstracts |
| OpenAlex API | https://api.openalex.org | Citation counts, venue metadata, author metadata |
| Elsevier ScienceDirect API | https://api.elsevier.com/content/search/sciencedirect | Recent journal discovery in robotics, MPC, and safety-critical control |

## 5. Related Repository and Research Packet

| Resource | Link | Use |
|---|---|---|
| Adaptive MPC reference implementation | https://github.com/KohlerJohannes/Adaptive | Compare architecture and implementation choices |
| Current research packet | `Resources/Papers_and_Journals/2026-04-05_adaptive_mpc_research.md` | Project-aligned synthesis and file mapping |

## 6. Notes on Evidence

- Items backed by arXiv metadata include abstract-level findings.
- Items backed by OpenAlex plus DOI metadata include accurate venue and
  citation information.
- Some very recent journal papers were selected from Elsevier and OpenAlex
  metadata only. Where no abstract was retrieved, implementation suggestions are
  treated as informed inference rather than direct claims.

## Appendix A. Previously Accumulated Reference Archive

The following legacy reference blocks are retained from the earlier repository
documentation so that older context is not lost.

### Legacy Books

#### Model Predictive Control

| Citation | Location | Used For |
|----------|----------|----------|
| Rawlings, J.B., Mayne, D.Q., Diehl, M. (2017). *Model Predictive Control: Theory, Computation, and Design* (2nd ed.). Nob Hill Publishing. | [Local PDF](Resources/Books/Model_Predictive_Control_Theory_Computation_and_Design.pdf) | QP formulation, warm-start, move-blocking |
| Borrelli, F., Bemporad, A., Morari, M. (2017). *Predictive Control for Linear and Hybrid Systems*. Cambridge University Press. | [Local PDF](Resources/Books/Predictive_Control_for_Linear_and_Hybrid_Systems.pdf) | Tube MPC, constraint tightening, stability |

#### Control Theory

| Citation | Location | Used For |
|----------|----------|----------|
| Åström, K.J., Murray, R.M. (2008). *Feedback Systems: An Introduction for Scientists and Engineers*. Princeton University Press. | [Local PDF](Resources/Books/Feedback_Systems_An_Introduction_for_Scientists_and_Engineers.pdf) | PID tuning, LQR design, stability analysis |

### Legacy Research Papers

#### Tube MPC and Robustness

| Citation | DOI/Link | Used For |
|----------|----------|----------|
| Mayne, D.Q. et al. (2005). "Robust model predictive control of constrained linear systems with bounded disturbances." | Automatica 41(2) | Invariant tubes, constraint tightening |
| TU Munich (2024). "Tube-based MPC for autonomous vehicle path tracking." | [tum.de](https://www.tum.de) | Disturbance observer integration |

#### Move-Blocking

| Citation | DOI/Link | Used For |
|----------|----------|----------|
| Imperial College (2023). "Move-blocking strategies for computationally efficient MPC." | [imperial.ac.uk](https://www.imperial.ac.uk) | Decision variable reduction |
| Cagienard et al. (2004). "Move blocking strategies in receding horizon control." | IEEE CDC | Block parameterization |

#### Differential-Drive MPC

| Citation | DOI/Link | Used For |
|----------|----------|----------|
| SCS Europe (2023). "MPC for trajectory tracking in differential drive robots." | [scs-europe.net](https://scs-europe.net) | Kinematic model, cold-start handling |
| CEUR-WS (2023). "Nonlinear MPC for mobile robots: orientation error analysis." | [ceur-ws.org](https://ceur-ws.org) | Heading error minimization |

#### Control Rate Penalty and Adaptive Weights

| Citation | DOI/Link | Used For |
|----------|----------|----------|
| Rawlings, J.B., Mayne, D.Q., Diehl, M. (2017). *Model Predictive Control: Theory, Computation, and Design*, Ch. 1.3. | [Local PDF](Resources/Books/Model_Predictive_Control_Theory_Computation_and_Design.pdf) | Move suppression in the MPC cost |
| MDPI Sensors (2024). "Improved MPC with adaptive weight adjustment for autonomous vehicles." | MDPI | Adaptive Q-weight scheduling during transients |
| Åström, K.J., Murray, R.M. (2008). *Feedback Systems*, Ch. 6. | [Local PDF](Resources/Books/Feedback_Systems_An_Introduction_for_Scientists_and_Engineers.pdf) | Smooth trajectory generation, Hermite basis functions |

#### Supervisory Hybrid Blending and Jerk Control

| Citation | DOI/Link | Used For |
|----------|----------|----------|
| Hespanha, J.P. et al. (2003). "Overcoming limitations of adaptive control by means of logic-based switching." | Systems and Control Letters 49(1) | Supervisory switching and hysteresis |
| Liberzon, D. (2003). *Switching in Systems and Control*. Birkhäuser. | [springer.com](https://link.springer.com/book/10.1007/978-1-4612-0017-8) | Dwell-time stability and switching analysis |
| De Luca, A., Oriolo, G. (1998). "Modeling and control of nonholonomic mechanical systems." | Springer | Jerk-bounded control for differential-drive robots |
| Flash, T., Hogan, N. (1985). "The coordination of arm movements." | Journal of Neuroscience 5(7) | Minimum-jerk intuition for smooth control |
| Borrelli, F., Bemporad, A., Morari, M. (2017). *Predictive Control for Linear and Hybrid Systems*, Ch. 15. | [Local PDF](Resources/Books/Predictive_Control_for_Linear_and_Hybrid_Systems.pdf) | Hybrid MPC and supervisory control switching laws |

#### Statistical Validation and Robustness

| Citation | DOI/Link | Used For |
|----------|----------|----------|
| Tempo, R., Calafiore, G., Dabbene, F. (2013). *Randomized Algorithms for Analysis and Control of Uncertain Systems*. Springer. | [springer.com](https://link.springer.com/book/10.1007/978-1-4471-4610-0) | Monte Carlo validation methodology |
| Rubinstein, R.Y., Kroese, D.P. (2016). *Simulation and the Monte Carlo Method*. Wiley. | ISBN 978-1-118-63220-8 | Experimental design and confidence intervals |

#### Real-Time MPC Acceleration

| Citation | DOI/Link | Used For |
|----------|----------|----------|
| Schaller, M., Banjac, G., Boyd, S. (2022). "Embedded Code Generation with CVXPY." *IEEE Control Systems Letters*, 6:2653-2658. | [stanford.edu](https://web.stanford.edu/~boyd/papers/cvxpygen.html) | CVXPYgen C-code generation for faster MPC |
| Nguyen, K. et al. (2024). "TinyMPC: Model-Predictive Control on Resource-Constrained Microcontrollers." *ICRA 2024*. | [arxiv.org](https://arxiv.org) | Embedded MPC at high control rates |
| (2024). "TransformerMPC: Accelerating MPC via Transformer-based Active Constraint Selection." | [arxiv.org](https://arxiv.org) | Learned active-set prediction for runtime reduction |

#### CBF-MPC Safety Guarantees

| Citation | DOI/Link | Used For |
|----------|----------|----------|
| Ali, M.A., Shen, C., Hashim, H.A. (2024). "A Linear MPC with Control Barrier Functions for Differential Drive Robots." | [arXiv:2404.09325](https://arxiv.org/abs/2404.09325) | LMPC plus CBF for differential-drive safety |
| (2024). "NMPC-CLF-CBF with Relaxed Decay Rate for Nonholonomic Robots." | [arxiv.org](https://arxiv.org) | Feasibility-preserving barrier decay |
| (2024). "DRCC-MPC for Safe Robot Navigation in Crowds." | [emergentmind.com](https://www.emergentmind.com) | Distributionally robust chance-constrained MPC |

#### Hybrid Stability Theory

| Citation | DOI/Link | Used For |
|----------|----------|----------|
| (2025). "Hybrid Lyapunov and Barrier Function-Based Control." | [arxiv.org](https://arxiv.org) | Unified stability and safety framework |
| (2024). "Stability of Hybrid Systems in Closed-Loop with MPC." *IMT Lucca*. | [imtlucca.it](https://www.imtlucca.it) | MPC value function as Lyapunov candidate |

#### Key Related Papers From the Earlier Hybrid-LQR-MPC Review

| # | Citation | Venue | DOI/Link | Relevance to This Project |
|---|----------|-------|----------|--------------------------|
| 1 | Wu et al. (2021). "Composing MPC with LQR and Neural Networks for Amortized Efficiency and Stable Control." | arXiv | arXiv:2112.07238 | Same LQR plus MPC composition idea, with a different third-mode design |
| 2 | Awad et al. (2022). "Model Predictive Control with Fuzzy Logic Switching for Path Tracking of Autonomous Vehicles." | ISA Transactions 129A | 10.1016/j.isatra.2021.12.022 | Validates soft switching relative to hard switching |
| 3 | Kong et al. (2023). "Hybrid iLQR Model Predictive Control for Contact-Implicit Stabilization on Legged Robots." | IEEE T-RO 39(6) | IEEE T-RO 2023 | Relevant hybrid architecture reference |
| 4 | Le Cleac'h et al. (2024). "Fast Contact-Implicit Model Predictive Control." | IEEE T-RO 40 | 10.1109/TRO.2024.3351554 | Relevant for solver-structure and speedup ideas |

## Appendix B. Retained Earlier Research Artifact

The interrupted earlier literature sweep is also preserved in:

- `Resources/Papers_and_Journals/arxiv_literature_survey.json`

That file remains useful as a broader arXiv-centered evidence archive.

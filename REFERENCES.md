# References

Academic references for theoretical foundations and implementation decisions.

----

## Books

### Model Predictive Control

| Citation | Location | Used For |
|----------|----------|----------|
| Rawlings, J.B., Mayne, D.Q., Diehl, M. (2017). *Model Predictive Control: Theory, Computation, and Design* (2nd ed.). Nob Hill Publishing. | [Local PDF](Resources/Books/Model_Predictive_Control_Theory_Computation_and_Design.pdf) | QP formulation, warm-start, move-blocking |
| Borrelli, F., Bemporad, A., Morari, M. (2017). *Predictive Control for Linear and Hybrid Systems*. Cambridge University Press. | [Local PDF](Resources/Books/Predictive_Control_for_Linear_and_Hybrid_Systems.pdf) | Tube MPC, constraint tightening, stability |

### Control Theory

| Citation | Location | Used For |
|----------|----------|----------|
| Åström, K.J., Murray, R.M. (2008). *Feedback Systems: An Introduction for Scientists and Engineers*. Princeton University Press. | [Local PDF](Resources/Books/Feedback_Systems_An_Introduction_for_Scientists_and_Engineers.pdf) | PID tuning, LQR design, stability analysis |

----

## Research Papers

### Tube MPC & Robustness

| Citation | DOI/Link | Used For |
|----------|----------|----------|
| Mayne, D.Q. et al. (2005). "Robust model predictive control of constrained linear systems with bounded disturbances." | Automatica 41(2) | Invariant tubes, constraint tightening |
| TU Munich (2024). "Tube-based MPC for autonomous vehicle path tracking." | [tum.de](https://www.tum.de) | Disturbance observer integration |

### Move-Blocking

| Citation | DOI/Link | Used For |
|----------|----------|----------|
| Imperial College (2023). "Move-blocking strategies for computationally efficient MPC." | [imperial.ac.uk](https://www.imperial.ac.uk) | Decision variable reduction |
| Cagienard et al. (2004). "Move blocking strategies in receding horizon control." | IEEE CDC | Block parameterization |

### Differential-Drive MPC

| Citation | DOI/Link | Used For |
|----------|----------|----------|
| SCS Europe (2023). "MPC for trajectory tracking in differential drive robots." | [scs-europe.net](https://scs-europe.net) | Kinematic model, cold-start handling |
| CEUR-WS (2023). "Nonlinear MPC for mobile robots: orientation error analysis." | [ceur-ws.org](https://ceur-ws.org) | Heading error minimization |

### Control Rate Penalty & Adaptive Weights (v0.5.0)

| Citation | DOI/Link | Used For |
|----------|----------|----------|
| Rawlings, J.B., Mayne, D.Q., Diehl, M. (2017). *Model Predictive Control: Theory, Computation, and Design*, Ch. 1.3. | [Local PDF](Resources/Books/Model_Predictive_Control_Theory_Computation_and_Design.pdf) | Move suppression (Δu penalty) in MPC cost function |
| MDPI Sensors (2024). "Improved MPC with adaptive weight adjustment for autonomous vehicles." | MDPI | Adaptive Q weight scheduling during transients |
| Åström, K.J., Murray, R.M. (2008). *Feedback Systems*, Ch. 6. | [Local PDF](Resources/Books/Feedback_Systems_An_Introduction_for_Scientists_and_Engineers.pdf) | Smooth trajectory generation, Hermite basis functions |

### Supervisory Hybrid Blending & Jerk Control (v0.6.0)

| Citation | DOI/Link | Used For |
|----------|----------|----------|
| Hespanha, J.P. et al. (2003). "Overcoming limitations of adaptive control by means of logic-based switching." | Syst. Control Lett. 49(1) | Supervisory switching, hysteresis-based dwell time |
| Liberzon, D. (2003). *Switching in Systems and Control*. Birkhäuser. | [springer.com](https://link.springer.com/book/10.1007/978-1-4612-0017-8) | Dwell-time stability, Lyapunov-based switching |
| De Luca, A., Oriolo, G. (1998). "Modeling and control of nonholonomic mechanical systems." | Springer | Jerk-bounded control for differential drive robots |
| Flash, T., Hogan, N. (1985). "The coordination of arm movements." | J. Neuroscience 5(7) | Sigmoid-like minimum-jerk trajectory generation |
| Borrelli, F., Bemporad, A., Morari, M. (2017). *Predictive Control for Linear and Hybrid Systems*, Ch. 15. | [Local PDF](Resources/Books/Predictive_Control_for_Linear_and_Hybrid_Systems.pdf) | Hybrid MPC, supervisory control switching laws |

### Statistical Validation & Robustness (v0.6.1)

| Citation | DOI/Link | Used For |
|----------|----------|----------|
| Tempo, R., Calafiore, G., Dabbene, F. (2013). *Randomized Algorithms for Analysis and Control of Uncertain Systems*. Springer. | [springer.com](https://link.springer.com/book/10.1007/978-1-4471-4610-0) | Monte Carlo validation methodology, probabilistic performance guarantees |
| Rubinstein, R.Y., Kroese, D.P. (2016). *Simulation and the Monte Carlo Method*. Wiley. | ISBN 978-1-118-63220-8 | Monte Carlo experimental design, confidence intervals |

----

## APIs & Datasets

| Resource | Access | Used For |
|----------|--------|----------|
| arXiv API | [api.arxiv.org](https://info.arxiv.org/help/api/index.html) | Paper search |
| Elsevier ScienceDirect | API Key: `[REDACTED]` | Journal access |
| OpenAlex | API Key: `[REDACTED]` | Academic metadata |

----

## Code References

| Library | Version | Used For |
|---------|---------|----------|
| CVXPY | ≥1.4 | Convex optimization, MPC formulation |
| OSQP | ≥0.6 | QP solver, warm-start support |
| NumPy | ≥1.24 | Array operations, linearization |
| Matplotlib | ≥3.7 | Visualization |

----

*Last updated: 2026-02-16*
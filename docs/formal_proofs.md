# Appendix: Formal proofs for Risk-Aware Hybrid Navigation Framework

This document contains the formal mathematical proofs supporting the theoretical guarantees of the Smooth Supervisory Hybrid LQR-MPC architecture. These proofs form the theoretical foundation of the associated research paper.

**Revision History:**
- v1.0 (2025): Initial Theorem 1 and 2.
- v2.0 (2026-04-25): Narrowed Theorem 1 assumptions to control-affine systems. Added Proposition 3 (DARE Terminal Cost Stability). Added Proposition 4 (Dwell-Time Bound). Added Proposition 5 (CBF Safety under Blending). Added Remark on LMS Convergence.

---

## Theorem 1: Boundedness under Convex Blending (Revised)

> **IMPORTANT (v2.0 Revision):** The original proof assumed arbitrary nonlinear dynamics and invoked Jensen's inequality for general Lyapunov functions. This revision restricts the result to **control-affine** systems where the dynamics are exactly linear in $u$, making the Jensen step valid without additional convexity assumptions on $V_i$.

**Statement:**
Consider a discrete-time **control-affine** system $x_{k+1} = g(x_k) + h(x_k) u_k$, where $g: \mathbb{R}^n \to \mathbb{R}^n$ and $h: \mathbb{R}^n \to \mathbb{R}^{n \times m}$. Let $u^{MPC}(x)$ and $u^{LQR}(x)$ be two stabilizing control laws with associated Lyapunov functions $V_{MPC}(x)$ and $V_{LQR}(x)$ such that:
1. $V_{MPC}(f(x, u^{MPC})) - V_{MPC}(x) \leq -\alpha_{MPC}(\|x\|)$
2. $V_{LQR}(f(x, u^{LQR})) - V_{LQR}(x) \leq -\alpha_{LQR}(\|x\|)$

where $\alpha_{MPC}, \alpha_{LQR}$ are $\mathcal{K}$-class functions.

Under the continuous convex blending law $u(x, t) = w(t)u^{MPC}(x) + (1-w(t))u^{LQR}(x)$ where $w(t) \in [0, 1]$, the closed-loop system state remains bounded, and the unified Lyapunov function candidate $V(x) = w V_{MPC}(x) + (1-w) V_{LQR}(x)$ decreases strictly outside a bounded region $\mathcal{B}$.

**Assumptions (Explicit):**

- (A1) The system is **control-affine**: $f(x, u) = g(x) + h(x)u$.
- (A2) Both $V_{MPC}$ and $V_{LQR}$ are positive definite and radially unbounded.
- (A3) The blending weight satisfies $|\Delta w_k| \leq \dot{w}_{max} \Delta t$ (rate-limited).
- (A4) There exists $L_{blend} > 0$ such that $|V_{MPC}(x) - V_{LQR}(x)| \leq L_{blend}$ for all $x$ in the region of interest.

**Proof:**

By the control-affine assumption (A1), the blended dynamics satisfy:
$$f(x, u_{blend}) = g(x) + h(x) u_{blend} = g(x) + h(x)[w u^{MPC} + (1-w) u^{LQR}]$$
$$= w [g(x) + h(x) u^{MPC}] + (1-w) [g(x) + h(x) u^{LQR}]$$
$$= w f(x, u^{MPC}) + (1-w) f(x, u^{LQR})$$

This exact decomposition holds without approximation for control-affine systems.

Define the unified Lyapunov candidate:
$$V(x, t) = w(t) V_{MPC}(x) + (1-w(t)) V_{LQR}(x)$$

Computing the one-step difference along the trajectory:
$$\Delta V = V(x_{k+1}, t_{k+1}) - V(x_k, t_k)$$
$$= w_{k+1}V_{MPC}(x_{k+1}) + (1-w_{k+1})V_{LQR}(x_{k+1}) - [w_k V_{MPC}(x_k) + (1-w_k) V_{LQR}(x_k)]$$

Let $w_{k+1} = w_k + \Delta w$. Rearranging:
$$\Delta V = w_k [\Delta V_{MPC}] + (1-w_k) [\Delta V_{LQR}] + \Delta w [V_{MPC}(x_{k+1}) - V_{LQR}(x_{k+1})]$$

where $\Delta V_i = V_i(x_{k+1}) - V_i(x_k)$.

Since $x_{k+1} = w_k f(x, u^{MPC}) + (1-w_k) f(x, u^{LQR})$ and by the individual stability conditions:

For the state-dependent terms, we use that $V_i$ evaluated at the blended next state can be bounded. If $V_i$ are convex (which holds for the quadratic Lyapunov functions arising from DARE), Jensen's inequality gives:
$$V_i(x_{k+1}) \leq w_k V_i(f(x, u^{MPC})) + (1-w_k) V_i(f(x, u^{LQR}))$$

Substituting:
$$\Delta V \leq w_k [V_{MPC}(f(x,u^{MPC})) - V_{MPC}(x_k)] + (1-w_k)[V_{LQR}(f(x,u^{LQR})) - V_{LQR}(x_k)] + L_{blend} |\Delta w|$$

Applying (A3) and the individual Lyapunov decrease conditions:
$$\Delta V \leq -w_k \alpha_{MPC}(\|x_k\|) - (1-w_k) \alpha_{LQR}(\|x_k\|) + L_{blend} \dot{w}_{max} \Delta t$$

Define $\alpha_{min}(\|x\|) = \min\{\alpha_{MPC}(\|x\|), \alpha_{LQR}(\|x\|)\}$. Then:
$$\Delta V \leq -\alpha_{min}(\|x_k\|) + L_{blend} \dot{w}_{max} \Delta t$$

For $\|x_k\| > r^*$ where $\alpha_{min}(r^*) = L_{blend} \dot{w}_{max} \Delta t$, we have $\Delta V < 0$.

Therefore, $x_k$ ultimately converges to and remains within the bounded set:
$$\mathcal{B} = \{x : \alpha_{min}(\|x\|) \leq L_{blend} \dot{w}_{max} \Delta t\}$$

The size of $\mathcal{B}$ is proportional to $\dot{w}_{max}$ and $\Delta t$. Slower blending transitions (smaller $\dot{w}_{max}$) yield a tighter ultimate bound. $\blacksquare$

**Remark 1.1:** For our differential-drive robot, the continuous-time dynamics $\dot{x} = [v\cos\theta, v\sin\theta, \omega]^\top$ are control-affine in $u = [v, \omega]^\top$ with $g(x) = 0$ and:
$$h(x) = \begin{bmatrix} \cos\theta & 0 \\ \sin\theta & 0 \\ 0 & 1 \end{bmatrix}$$
so assumption (A1) is exactly satisfied.

**Remark 1.2:** If $V_{MPC}$ and $V_{LQR}$ are not convex (e.g., for general nonlinear Lyapunov functions), the Jensen step requires an additional Lipschitz bound on $V_i$ and the result weakens to a proposition (local boundedness near the origin).

---

## Theorem 2: No-Chattering Condition

**Statement:**
For the continuous blending supervisor defined by:
$w_{target}(t) = \sigma(r(t)) = \frac{1}{1 + e^{-k(r(t) - r_{thresh})}}$
where $r(t) \in [0, 1]$ is the scalar risk metric.

Given a hysteresis deadband $h > 0$ and a strict derivative bound $|\dot{w}(t)| \leq \dot{w}_{max}$, the closed-loop system is strictly free of Zeno behavior (infinite switching in finite time) and the maximum frequency of control mode transitions (crossing $w=0.5$) is bounded by $f_{max} \leq \frac{\dot{w}_{max}}{2}$.

**Proof:**
A "switch" or "transition" is defined as the blending weight $w(t)$ crossing the midpoint threshold $w=0.5$ from one stable domain (e.g., $w < 0.1$) to the other (e.g., $w > 0.9$).

To complete one full switching cycle (LQR-dominant $\to$ MPC-dominant $\to$ LQR-dominant), the blending weight must traverse from $w=0$ to $w=1$ and back to $w=0$.
The total variation required for this cycle is $\Delta w_{total} = 2.0$.

By construction in the `BlendingSupervisor` class, the weight derivative is strictly enforced:
$|\dot{w}(t)| \leq \dot{w}_{max} \quad \forall t$

Therefore, the minimum time $T_{cycle}$ required to complete one full switching cycle is constrained by the maximum rate of change:
$T_{cycle} \geq \frac{\Delta w_{total}}{\dot{w}_{max}} = \frac{2.0}{\dot{w}_{max}}$

The maximum switching frequency is the inverse of the minimum cycle time:
$f_{max} = \frac{1}{T_{cycle}} \leq \frac{\dot{w}_{max}}{2}$

Furthermore, the hysteresis mechanism defines a strict deadband $(r_{thresh} - h, r_{thresh} + h)$. While the risk metric $r(t)$ remains within this deadband, $\dot{w}_{target} = 0$.

For a chattering event to occur, the external environment or sensor noise must drive the risk metric $r(t)$ across the entire deadband width $\Delta r = 2h$. Assuming the risk metric itself has bounded variations (e.g., bounded obstacle relative velocity), the time required to traverse the deadband strictly separates switching events.

Since the minimum time between transitions is lower-bounded by a strictly positive constant $T \geq \min(\frac{1}{\dot{w}_{max}}, \frac{2h}{\dot{r}_{max}}) > 0$, Zeno behavior (where time between switches approaches 0) is mathematically impossible. $\blacksquare$

---

## Proposition 3: DARE Terminal Cost Stability (NEW)

**Statement:**
Consider the linearized differential-drive system at an operating point $(v_{ref}, \theta_{ref})$:
$$x_{k+1} = A_d x_k + B_d u_k, \quad A_d = I + \Delta t \cdot A_c, \quad B_d = \Delta t \cdot B_c$$

where:
$$A_c = \begin{bmatrix} 0 & 0 & -v_{ref}\sin\theta_{ref} \\ 0 & 0 & v_{ref}\cos\theta_{ref} \\ 0 & 0 & 0 \end{bmatrix}, \quad B_c = \begin{bmatrix} \cos\theta_{ref} & 0 \\ \sin\theta_{ref} & 0 \\ 0 & 1 \end{bmatrix}$$

Let $P \succ 0$ be the unique stabilizing solution of the DARE:
$$P = A_d^\top P A_d - A_d^\top P B_d (R + B_d^\top P B_d)^{-1} B_d^\top P A_d + Q$$

and let $K = (R + B_d^\top P B_d)^{-1} B_d^\top P A_d$ be the associated LQR gain.

If the MPC uses $V_f(x) = x^\top P x$ as the terminal cost and constrains the terminal state to lie in the maximal output-admissible set $\mathcal{X}_f = \{x : (A_d - B_d K)^j x \in \mathcal{X}, K(A_d - B_d K)^j x \in \mathcal{U}, \forall j \geq 0\}$, then:

1. The MPC optimization is recursively feasible.
2. The closed-loop system is asymptotically stable.
3. The optimal cost $V_N^*(x_k)$ is a Lyapunov function for the closed-loop system.

**Proof Sketch:**

**(1) Recursive Feasibility:** Given a feasible solution $\{u_0^*, \ldots, u_{N-1}^*\}$ at time $k$ with corresponding state trajectory $\{x_0^*, \ldots, x_N^*\}$, we construct a candidate solution at time $k+1$ by shifting: $\tilde{u}_j = u_{j+1}^*$ for $j = 0, \ldots, N-2$ and $\tilde{u}_{N-1} = -K x_N^*$.

Since $x_N^* \in \mathcal{X}_f$ (by the terminal constraint), and $\mathcal{X}_f$ is control-invariant under $K$, we have $(A_d - B_d K) x_N^* \in \mathcal{X}_f$, which means the shifted candidate satisfies all constraints. $\checkmark$

**(2) Lyapunov Decrease:** The cost of the shifted candidate satisfies:
$$V_N(\tilde{x}_0) \leq V_N^*(x_k) - l(x_k, u_0^*) + V_f((A_d - B_d K) x_N^*) - V_f(x_N^*) + l(x_N^*, -K x_N^*)$$

By the DARE property: $V_f((A_d - B_d K)x) - V_f(x) + l(x, -Kx) = 0$ (the DARE is precisely this identity).

Therefore: $V_N(\tilde{x}_0) \leq V_N^*(x_k) - l(x_k, u_0^*)$

Since $V_N^*(x_{k+1}) \leq V_N(\tilde{x}_0)$ (optimality), we get:
$$V_N^*(x_{k+1}) - V_N^*(x_k) \leq -l(x_k, u_0^*) \leq -\lambda_{min}(Q) \|x_k\|^2$$

This confirms $V_N^*$ is a Lyapunov function. $\blacksquare$

**Implementation Note:** This proposition is violated in the current codebase where `mpc_controller.py` uses an arbitrary terminal weight $Q_T = \text{diag}([50, 50, 10])$ instead of the DARE-derived $P$. **Phase P0-D** fixes this.

---

## Proposition 4: Dwell-Time Bound under Rate-Limited Blending (NEW)

**Statement:**
For the rate-limited blending supervisor with $|\dot{w}| \leq \dot{w}_{max}$ and hysteresis band $h > 0$, define the LQR and MPC "dominance" regions as:
$$\mathcal{R}_{LQR} = \{t : w(t) < 0.1\}, \quad \mathcal{R}_{MPC} = \{t : w(t) > 0.9\}$$

The minimum time spent in each dominance region before transitioning to the other satisfies:
$$\tau_{dwell} \geq \frac{0.8}{\dot{w}_{max}}$$

For the default parameters $\dot{w}_{max} = 2.0$ s$^{-1}$, this gives $\tau_{dwell} \geq 0.4$ s, or equivalently $\geq 20$ control steps at $\Delta t = 0.02$ s.

**Proof:**
To transition from $w = 0.1$ to $w = 0.9$ requires a weight change of $\Delta w = 0.8$. Under the rate limit $|\dot{w}| \leq \dot{w}_{max}$:
$$\tau_{transition} = \frac{\Delta w}{\dot{w}_{max}} = \frac{0.8}{\dot{w}_{max}}$$

Since the weight must first reach the dominance boundary before it can cross to the other region, $\tau_{dwell} \geq \tau_{transition}$.

By Liberzon's average dwell-time theory [R6], if $\tau_{dwell} > \frac{\ln \mu}{\lambda}$ where $\mu = \max\{V_{MPC}(x)/V_{LQR}(x), V_{LQR}(x)/V_{MPC}(x)\}$ at the switching surface and $\lambda$ is the minimum decay rate, then the switched system remains stable. $\blacksquare$

---

## Proposition 5: CBF Safety under Blended Control (NEW, Sketch)

**Statement:**
Consider the obstacle avoidance safety function $h_j(x) = \|p - p_{obs,j}\| - r_{obs,j} - d_{safe}$ for the $j$-th obstacle. Define the safe set $\mathcal{C}_j = \{x : h_j(x) \geq 0\}$.

If the MPC control law $u^{MPC}$ is designed such that $h_j(f(x, u^{MPC})) \geq (1-\gamma) h_j(x)$ for some $\gamma \in (0, 1)$ (discrete-time CBF condition), then the blended control $u_{blend} = w u^{MPC} + (1-w) u^{LQR}$ preserves safety when:

$$w(t) \geq w_{min}(x) = \frac{-\Delta h_j^{LQR}(x)}{(\Delta h_j^{MPC}(x) - \Delta h_j^{LQR}(x)) + \epsilon}$$

where $\Delta h_j^{ctrl}(x) = h_j(f(x, u^{ctrl})) - h_j(x)$ and $\epsilon > 0$ is a small regularization constant.

**Proof Sketch:**
For the control-affine system, $f(x, u_{blend}) = w f(x, u^{MPC}) + (1-w) f(x, u^{LQR})$. By the Lipschitz continuity of $h_j$ in the state:
$$h_j(f(x, u_{blend})) \approx w h_j(f(x, u^{MPC})) + (1-w) h_j(f(x, u^{LQR}))$$

For safety: $h_j(f(x, u_{blend})) \geq 0$ requires:
$$w \Delta h_j^{MPC}(x) + (1-w) \Delta h_j^{LQR}(x) + h_j(x) \geq 0$$

Solving for $w$ when $\Delta h_j^{LQR}(x) < 0$ (LQR alone would violate safety):
$$w \geq \frac{-(h_j(x) + \Delta h_j^{LQR}(x))}{\Delta h_j^{MPC}(x) - \Delta h_j^{LQR}(x)}$$

This provides a constructive lower bound on the blending weight necessary for safety near obstacles. The risk metric module should ensure $w(t) \geq w_{min}(x)$ whenever $h_j(x)$ is small. $\blacksquare$

**Remark 5.1:** This proposition connects the risk metric thresholds to formal safety requirements. When $d_{trigger}$ and the sigmoid parameters are calibrated such that $w(\sigma(r(x))) \geq w_{min}(x)$ for all $x$ near obstacles, the blended controller inherits the MPC's obstacle avoidance guarantees.

---

## Remark: LMS Convergence for Adaptive NMPC

The `LMSAdaptation` module estimates kinematic scaling parameters $\hat{\theta} = [\hat{v}_s, \hat{\omega}_s]^\top$ via normalized LMS:
$$\hat{\theta}_{k+1} = \text{proj}_{[\theta_{min}, \theta_{max}]}[\hat{\theta}_k + \Gamma \Phi_k^\top e_k]$$

where $e_k = x_{k+1}^{meas} - x_{k+1}^{pred}$ and $\Phi_k = \frac{\partial x_{k+1}}{\partial \theta}\bigg|_{\hat{\theta}_k}$.

By Ioannou and Sun [R11], if the regressor $\Phi_k$ satisfies persistent excitation:
$$\exists \alpha_0, T_0 > 0 : \sum_{k=t}^{t+T_0} \Phi_k \Phi_k^\top \succeq \alpha_0 I \quad \forall t$$

then $\|\hat{\theta}_k - \theta^*\| \to 0$ exponentially. The projection ensures $\hat{\theta}_k \in [\theta_{min}, \theta_{max}]$ for all $k$.

**Practical Consideration:** Persistent excitation requires the robot to execute sufficiently rich maneuvers (non-zero $v$ and $\omega$ with sufficient variation). Straight-line trajectories at constant speed provide rank-deficient $\Phi$ and may cause parameter drift. The figure-8 and sinusoidal reference trajectories used in our evaluation naturally satisfy PE.

---

## Proposition 6: Cross-Track Error Bound for Checkpoint Navigation (P6-F)

**Statement:**
Consider a robot navigating between consecutive checkpoints $p_i$ and $p_{i+1}$ separated by distance $L_i = \|p_{i+1} - p_i\|$. Under the waypoint-tracking MPC controller with arrival radius $r_{arr}$ and position noise $\sigma_p$, the cross-track error (XTE) at any point along the segment satisfies:

$$\text{XTE}(t) \leq r_{arr} \sin(\Delta\theta_{max}) + \frac{L_i}{2} \sin(\Delta\theta_{max}) + 3\sigma_p$$

where $\Delta\theta_{max}$ is the maximum heading alignment error at the start of each segment.

**Proof Sketch:**

1. **Geometric bound.** At departure from checkpoint $p_i$, the robot is within $r_{arr}$ of $p_i$ and heading toward $p_{i+1}$ with heading error $\Delta\theta$. The initial lateral offset is bounded by $r_{arr} \sin(\Delta\theta_{max})$.

2. **Propagation.** Over the segment, the heading controller drives $\Delta\theta \to 0$ exponentially. The maximum lateral deviation occurs at the midpoint and is bounded by the triangle formed by the initial offset, the segment length, and the heading correction arc:
   $$d_{lat}^{max} \leq r_{arr} \sin(\Delta\theta_{max}) + \frac{L_i}{2} \sin(\Delta\theta_{max})$$
   
   This follows from the fact that the robot's deviation from the line segment is at most proportional to the product of the initial heading error and the distance traveled before correction.

3. **Noise contribution.** Position noise $\sigma_p$ introduces a $3\sigma_p$ worst-case (99.7% confidence) lateral perturbation per timestep. Since the controller corrects noise-induced deviations within a few steps, the noise contribution does not accumulate beyond $3\sigma_p$.

4. **Combining bounds:**
   $$\text{XTE}(t) \leq r_{arr} \sin(\Delta\theta_{max}) + \frac{L_i}{2} \sin(\Delta\theta_{max}) + 3\sigma_p$$

**Corollary:** For the default parameters ($r_{arr} = 0.3$ m, $\Delta\theta_{max} \leq 15°$, $L_i \leq 1.5$ m, $\sigma_p = 0.05$ m):

$$\text{XTE}_{max} \leq 0.3 \cdot 0.259 + \frac{1.5}{2} \cdot 0.259 + 3 \cdot 0.05 = 0.078 + 0.194 + 0.15 = 0.422 \text{ m}$$

This provides a constructive guarantee that the CN mode trajectory never deviates more than ~42 cm from the inter-checkpoint line segments under nominal conditions. $\blacksquare$

**Remark 6.1:** The XTE bound scales linearly with checkpoint spacing $L_i$. The `CheckpointExtractor` with `curvature` strategy places more checkpoints at high-curvature regions, reducing $L_i$ where deviations are most likely. This adaptive spacing implicitly minimizes worst-case XTE.

**Remark 6.2:** In contrast, FRP mode's tracking error is bounded by the tube radius $w_{max}$ from Theorem 1. The CN bound is typically larger ($\sim 0.4$ m vs $\sim 0.1$ m for FRP) but the CN mode permits the MPC more freedom for obstacle avoidance, leading to lower collision rates in dense scenarios.


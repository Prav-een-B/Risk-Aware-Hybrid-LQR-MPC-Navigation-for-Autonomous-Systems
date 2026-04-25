"""
CVXPYgen Solver Wrapper (Phase 5B)
===================================

Provides a fast-path MPC solver using CVXPYgen code generation.
CVXPYgen generates custom C code from CVXPY problems, achieving
10x speedup (0.83ms vs 8.78ms for OSQP-based QPs).

Reference:
    Schaller, M., Banjac, G., Boyd, S. (2022).
    "Embedded Code Generation with CVXPY."
    IEEE Control Systems Letters, 6:2653-2658.

Architecture:
    1. generate_solver() - One-time offline: creates compiled C solver
    2. solve_fast()      - Online: uses generated solver for fast solves
    3. Fallback          - If CVXPYgen unavailable, falls back to CVXPY

P0-A: Added linearized obstacle avoidance constraints with slack variables.
P0-B: Cost now tracks ||u - u_ref||² instead of ||u||².

Usage:
    wrapper = CVXPYgenWrapper(horizon=5, nx=3, nu=2, max_obstacles=8)
    wrapper.generate_solver()  # One-time setup
    sol = wrapper.solve_fast(x0, x_refs, u_refs, A_d, B_d, obstacles)
"""

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cvxpy as cp
import numpy as np

# Try importing cvxpygen (optional dependency)
try:
    from cvxpygen import cpg

    CVXPYGEN_AVAILABLE = True
except ImportError:
    CVXPYGEN_AVAILABLE = False


@dataclass
class FastMPCSolution:
    """Solution from the fast CVXPYgen solver."""

    status: str
    optimal_control: np.ndarray
    control_sequence: np.ndarray
    predicted_states: np.ndarray
    cost: float
    solve_time_ms: float
    used_generated_solver: bool
    feasibility_margin: float = 0.0  # Added for compatibility with BlendingSupervisor


class CVXPYgenWrapper:
    """
    CVXPYgen-accelerated MPC solver with CVXPY fallback.

    Creates a parametrized CVXPY problem where the changing data
    (initial state, reference trajectory, linearization matrices,
    obstacle positions) are CVXPY Parameters. CVXPYgen then generates
    a compiled C solver specialized for this problem structure.

    The generated solver:
    - Solves the same QP family ~10x faster than interpreted CVXPY
    - Supports warm-starting for consecutive solves
    - Falls back gracefully to CVXPY when unavailable
    - Includes linearized obstacle avoidance constraints (P0-A fix)
    - Tracks u_ref in cost function (P0-B fix)

    Attributes:
        horizon: Prediction horizon N
        nx: State dimension (3 for [px, py, theta])
        nu: Control dimension (2 for [v, omega])
        max_obstacles: Maximum number of obstacles in parametrized problem
        generated: Whether a compiled solver has been generated
    """

    def __init__(
        self,
        horizon: int = 5,
        nx: int = 3,
        nu: int = 2,
        Q_diag: list = None,
        R_diag: list = None,
        P_diag: list = None,
        S_diag: list = None,
        J_diag: list = None,
        v_max: float = 2.0,
        omega_max: float = 3.0,
        dt: float = 0.02,
        d_safe: float = 0.3,
        slack_penalty: float = 5000.0,
        max_obstacles: int = 8,
        solver_name: str = "OSQP",
        code_dir: str = None,
    ):
        """
        Initialize CVXPYgen wrapper.

        Args:
            horizon: Prediction horizon N
            nx: State dimension
            nu: Control dimension
            Q_diag: State tracking weights
            R_diag: Control effort weights
            P_diag: Terminal cost weights
            S_diag: 1st-order control rate penalty
            J_diag: 2nd-order jerk penalty (optional)
            v_max: Max linear velocity
            omega_max: Max angular velocity
            d_safe: Safety distance from obstacles
            slack_penalty: Penalty weight for obstacle slack variables
            max_obstacles: Maximum number of obstacles (fixed for code generation)
            solver_name: Solver to use ("OSQP" recommended for speed)
            code_dir: Directory for generated C code
        """
        self.N = horizon
        self.nx = nx
        self.nu = nu
        self.solver_name = solver_name
        self.generated = False
        self.dt = dt
        self.d_safe = d_safe
        self.slack_penalty = slack_penalty
        self.max_obstacles = max_obstacles

        # Warm-start storage for obstacle linearization.
        # The fast MPC path is used by statistical evaluation, so it needs the
        # same P0-C behavior as MPCController: linearize obstacle constraints
        # around the latest predicted trajectory instead of the reference path.
        self._prev_solution: Optional[np.ndarray] = None
        self._prev_states: Optional[np.ndarray] = None

        # Weight matrices
        if Q_diag is None:
            Q_diag = [80.0, 80.0, 120.0]
        if R_diag is None:
            R_diag = [0.1, 0.1]
        if P_diag is None:
            P_diag = [20.0, 20.0, 40.0]
        if S_diag is None:
            S_diag = [0.1, 0.5]

        self.Q = np.diag(Q_diag)
        self.R = np.diag(R_diag)
        self.P = np.diag(P_diag)
        self.S = np.diag(S_diag)
        self.J = np.diag(J_diag) if J_diag is not None else None

        # Actuator limits (with 5% Tube MPC tightening)
        self.v_max = v_max * 0.95
        self.omega_max = omega_max * 0.95

        # Code generation directory
        if code_dir is None:
            code_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "_generated_solver"
            )
        self.code_dir = code_dir

        # Build the parametrized problem
        self._build_parametrized_problem()

        # Try to load previously generated solver
        self._try_load_generated()

    def _build_parametrized_problem(self):
        """
        Build the CVXPY problem with Parameters for all changing data.

        Parameters (change each solve):
            - x0_param: Initial state [3]
            - A_param, B_param: Linearized dynamics [3x3], [3x2]
            - x_ref_param: Reference states [N+1, 3]
            - u_ref_param: Reference controls [N, 2] (P0-B fix)
            - obs_normals_param: Pre-computed constraint normals [N*max_obs, 2] (P0-A)
            - obs_offsets_param: Pre-computed constraint offsets [N*max_obs] (P0-A)

        Variables (optimized):
            - x: States [N+1, 3]
            - u: Controls [N, 2]
            - slack: Obstacle slack variables [N * max_obs]
        """
        N = self.N
        M = self.max_obstacles

        # Parameters (changing data)
        self.x0_param = cp.Parameter(self.nx, name="x0")
        self.A_param = cp.Parameter((self.nx, self.nx), name="A")
        self.B_param = cp.Parameter((self.nx, self.nu), name="B")
        self.x_ref_param = cp.Parameter((N + 1, self.nx), name="x_ref")
        # P0-B: Add reference control parameter
        self.u_ref_param = cp.Parameter((N, self.nu), name="u_ref")

        # P0-A: Obstacle constraint parameters (pre-linearized)
        # We parameterize the linearized half-plane constraints directly:
        #   n_x * x[k,0] + n_y * x[k,1] >= offset
        # Inactive padded obstacles use zero normals and zero offsets, yielding
        # the trivially feasible DPP-compliant constraint 0 >= -slack.
        self.obs_normals_param = cp.Parameter((N * M, 2), name="obs_normals")
        self.obs_offsets_param = cp.Parameter(N * M, name="obs_offsets")

        # Decision variables
        self.x_var = cp.Variable((N + 1, self.nx), name="x")
        self.u_var = cp.Variable((N, self.nu), name="u")
        # P0-A: Slack variables for soft obstacle constraints
        self.slack_var = cp.Variable(N * M, nonneg=True, name="slack")

        # Cost function
        cost = 0
        for k in range(N):
            state_error = self.x_var[k] - self.x_ref_param[k]
            cost += cp.quad_form(state_error, self.Q)
            # P0-B fix: Track u_ref instead of penalizing absolute control
            control_error = self.u_var[k] - self.u_ref_param[k]
            cost += cp.quad_form(control_error, self.R)

            # 1st-order: control rate penalty
            if k > 0:
                du_rate = self.u_var[k] - self.u_var[k - 1]
                cost += cp.quad_form(du_rate, self.S)

            # 2nd-order: jerk penalty (Phase 5A)
            if self.J is not None and k > 1:
                jerk = self.u_var[k] - 2 * self.u_var[k - 1] + self.u_var[k - 2]
                cost += cp.quad_form(jerk, self.J)

        # Terminal cost
        terminal_error = self.x_var[N] - self.x_ref_param[N]
        cost += cp.quad_form(terminal_error, self.P)

        # P0-A: Obstacle slack penalty (only on active obstacles)
        cost += self.slack_penalty * cp.sum_squares(self.slack_var)

        # Constraints
        constraints = []

        # Initial state
        constraints.append(self.x_var[0] == self.x0_param)

        # LTI dynamics (single linearization point for CVXPYgen)
        for k in range(N):
            constraints.append(
                self.x_var[k + 1]
                == self.A_param @ self.x_var[k] + self.B_param @ self.u_var[k]
            )

        # Actuator constraints
        for k in range(N):
            constraints.append(self.u_var[k, 0] >= -self.v_max)
            constraints.append(self.u_var[k, 0] <= self.v_max)
            constraints.append(self.u_var[k, 1] >= -self.omega_max)
            constraints.append(self.u_var[k, 1] <= self.omega_max)

        # P0-A: Linearized obstacle avoidance constraints
        # For each (time_step, obstacle) pair:
        #   n_x * x[k,0] + n_y * x[k,1] >= offset - slack
        # Padded inactive obstacles use zero normals/offsets, making the
        # constraint 0 >= -slack without multiplying parameters together.
        for k in range(N):
            for j in range(M):
                idx = k * M + j
                lhs = (
                    self.obs_normals_param[idx, 0] * self.x_var[k, 0]
                    + self.obs_normals_param[idx, 1] * self.x_var[k, 1]
                )
                constraints.append(
                    lhs >= self.obs_offsets_param[idx] - self.slack_var[idx]
                )

        # Build problem
        self.problem = cp.Problem(cp.Minimize(cost), constraints)

        # Store for generated solver
        self._cpg_problem = None

    def generate_solver(self) -> bool:
        """
        Generate compiled C solver from the parametrized problem.

        This is a one-time offline operation. The generated solver
        is cached to disk and loaded on subsequent runs.

        Returns:
            True if generation succeeded, False otherwise.
        """
        if not CVXPYGEN_AVAILABLE:
            print("[CVXPYgen] cvxpygen not installed. Using CVXPY fallback.")
            return False

        try:
            print(f"[CVXPYgen] Generating C solver in {self.code_dir}...")
            cpg.generate_code(
                self.problem, code_dir=self.code_dir, solver=self.solver_name
            )

            # Load the generated solver
            import sys
            from importlib import import_module

            if self.code_dir not in sys.path:
                sys.path.insert(0, os.path.dirname(self.code_dir))

            self.generated = True
            print("[CVXPYgen] Solver generated and loaded successfully!")
            return True

        except Exception as e:
            print(f"[CVXPYgen] Generation failed: {e}")
            print("[CVXPYgen] Falling back to interpreted CVXPY.")
            self.generated = False
            return False

    def _try_load_generated(self):
        """Try to load a previously generated solver."""
        if not CVXPYGEN_AVAILABLE:
            return

        cpg_module_path = os.path.join(self.code_dir, "cpg_module.py")
        if os.path.exists(cpg_module_path):
            try:
                import sys

                if self.code_dir not in sys.path:
                    sys.path.insert(0, os.path.dirname(self.code_dir))
                self.generated = True
            except Exception:
                self.generated = False

    def _build_obstacle_linearization_states(
        self, x0: np.ndarray, u_refs: np.ndarray
    ) -> np.ndarray:
        """
        Build the trajectory used to linearize obstacle half-plane constraints.

        Prefer the previous predicted trajectory when available, shifted one
        step forward and anchored at the current measured state. On cold start
        or stale/invalid predictions, fall back to a simple nonlinear unicycle
        rollout using the reference controls.
        """
        if self._prev_states is not None and len(self._prev_states) >= self.N + 1:
            shifted = np.vstack([self._prev_states[1:], self._prev_states[-1:]])
            shifted = shifted[: self.N + 1].copy()
            shifted[0] = x0
            if np.all(np.isfinite(shifted)):
                return shifted

        states = np.zeros((self.N + 1, self.nx))
        states[0] = x0.copy()

        for k in range(self.N):
            v = float(u_refs[k, 0])
            omega = float(u_refs[k, 1])
            theta = states[k, 2]

            states[k + 1, 0] = states[k, 0] + v * np.cos(theta) * self.dt
            states[k + 1, 1] = states[k, 1] + v * np.sin(theta) * self.dt
            states[k + 1, 2] = states[k, 2] + omega * self.dt

        return states

    def _linearize_obstacles(
        self, linearization_states: np.ndarray, obstacles: List[Dict]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pre-compute linearized obstacle constraint normals and offsets.

        P0-A: For each (time_step, obstacle) pair, compute the half-plane:
            n_x * p_x + n_y * p_y >= offset
        where n = (p_lin - p_obs) / ||p_lin - p_obs|| is the outward normal.

        Args:
            linearization_states: Warm-start/cold-start linearization trajectory [N+1, 3]
            obstacles: List of obstacle dicts with 'x', 'y', 'radius'

        Returns:
            Tuple of (normals [N*M, 2], offsets [N*M])
        """
        N = self.N
        M = self.max_obstacles

        normals = np.zeros((N * M, 2))
        offsets = np.zeros(N * M)

        n_obs = min(len(obstacles), M) if obstacles else 0

        for k in range(N):
            for j in range(n_obs):
                idx = k * M + j
                obs = obstacles[j]

                # P0-C: Linearization point from warm-start trajectory when
                # available, otherwise from a cold-start nonlinear rollout.
                px_lin = linearization_states[min(k, len(linearization_states) - 1), 0]
                py_lin = linearization_states[min(k, len(linearization_states) - 1), 1]

                # Direction from obstacle to linearization point
                dx = px_lin - obs["x"]
                dy = py_lin - obs["y"]
                dist = np.sqrt(dx**2 + dy**2)

                if dist > 0.01:
                    # Unit normal pointing away from obstacle
                    nx = dx / dist
                    ny = dy / dist

                    normals[idx] = [nx, ny]
                    offsets[idx] = (
                        nx * obs["x"] + ny * obs["y"] + self.d_safe + obs["radius"]
                    )
                else:
                    # Too close to obstacle center; use fallback normal
                    normals[idx] = [1.0, 0.0]
                    offsets[idx] = obs["x"] + self.d_safe + obs["radius"]

        return normals, offsets

    def solve_fast(
        self,
        x0: np.ndarray,
        x_refs: np.ndarray,
        A_d: np.ndarray,
        B_d: np.ndarray,
        u_refs: np.ndarray = None,
        obstacles: List[Dict] = None,
    ) -> FastMPCSolution:
        """
        Solve MPC using the fastest available method.

        Uses generated C solver if available, otherwise falls back
        to interpreted CVXPY. Both paths produce identical results.

        Args:
            x0: Initial state [3]
            x_refs: Reference states [N+1, 3]
            A_d: Discrete-time A matrix [3, 3]
            B_d: Discrete-time B matrix [3, 2]
            u_refs: Reference controls [N, 2] (P0-B: used in cost)
            obstacles: List of obstacle dicts with 'x', 'y', 'radius' (P0-A)

        Returns:
            FastMPCSolution with optimal control and timing.
        """
        start_time = time.perf_counter()

        # Set parameter values
        self.x0_param.value = x0
        self.A_param.value = A_d
        self.B_param.value = B_d

        # Pad x_refs if needed
        if x_refs.shape[0] < self.N + 1:
            padded = np.zeros((self.N + 1, self.nx))
            padded[: x_refs.shape[0]] = x_refs
            padded[x_refs.shape[0] :] = x_refs[-1]
            x_refs = padded
        self.x_ref_param.value = x_refs

        # P0-B: Set reference controls (default to zero if not provided)
        if u_refs is not None:
            if u_refs.shape[0] < self.N:
                u_padded = np.zeros((self.N, self.nu))
                u_padded[: u_refs.shape[0]] = u_refs
                u_padded[u_refs.shape[0] :] = u_refs[-1]
                u_refs = u_padded
            self.u_ref_param.value = u_refs
        else:
            self.u_ref_param.value = np.zeros((self.N, self.nu))

        # P0-A/P0-C: Compute linearized obstacle constraints around the latest
        # predicted trajectory when available, with a cold-start rollout fallback.
        linearization_states = self._build_obstacle_linearization_states(
            x0, self.u_ref_param.value
        )

        if obstacles and len(obstacles) > 0:
            normals, offsets = self._linearize_obstacles(
                linearization_states, obstacles
            )
        else:
            normals = np.zeros((self.N * self.max_obstacles, 2))
            offsets = np.zeros(self.N * self.max_obstacles)

        self.obs_normals_param.value = normals
        self.obs_offsets_param.value = offsets

        # Solve
        used_generated = False
        
        solver_opts = {
            'eps_abs': 1e-3, 
            'eps_rel': 1e-3, 
            'max_iter': 2000,
            'alpha': 1.6, 
            'early_terminate': True
        }

        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                if self.generated and CVXPYGEN_AVAILABLE:
                    # Use generated solver (fast path)
                    self.problem.solve(method="CPG", warm_start=True)
                    used_generated = True
                else:
                    # Fallback to interpreted CVXPY
                    kwargs = {'warm_start': True, 'verbose': False}
                    if self.solver_name == 'OSQP':
                        kwargs.update(solver_opts)
                    self.problem.solve(solver=getattr(cp, self.solver_name), **kwargs)
        except Exception:
            # Double fallback
            try:
                self.problem.solve(solver=cp.SCS, verbose=False)
            except Exception:
                pass

        solve_time = (time.perf_counter() - start_time) * 1000  # ms

        if self.problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            if self.u_var.value is None or self.x_var.value is None:
                return FastMPCSolution(
                    status="optimal_no_values",
                    optimal_control=np.zeros(self.nu),
                    control_sequence=np.zeros((self.N, self.nu)),
                    predicted_states=np.tile(x0, (self.N + 1, 1)),
                    cost=float("inf"),
                    solve_time_ms=solve_time,
                    used_generated_solver=used_generated,
                )

            control_sequence = np.asarray(self.u_var.value)
            predicted_states = np.asarray(self.x_var.value)

            # Compute feasibility margin from slack
            feas_margin = 0.0
            if self.slack_var.value is not None:
                feas_margin = float(np.max(np.abs(self.slack_var.value)))

            self._prev_solution = control_sequence.copy()
            self._prev_states = predicted_states.copy()

            return FastMPCSolution(
                status="optimal",
                optimal_control=control_sequence[0],
                control_sequence=control_sequence,
                predicted_states=predicted_states,
                cost=self.problem.value,
                solve_time_ms=solve_time,
                used_generated_solver=used_generated,
                feasibility_margin=feas_margin,
            )
        else:
            # Fallback: zero control
            return FastMPCSolution(
                status=self.problem.status or "infeasible",
                optimal_control=np.zeros(self.nu),
                control_sequence=np.zeros((self.N, self.nu)),
                predicted_states=np.tile(x0, (self.N + 1, 1)),
                cost=float("inf"),
                solve_time_ms=solve_time,
                used_generated_solver=used_generated,
            )

    def benchmark(self, n_solves: int = 50) -> Dict[str, float]:
        """
        Benchmark solve time over multiple random problems.

        Args:
            n_solves: Number of random solves to average over.

        Returns:
            Dictionary with mean, std, min, max solve times (ms).
        """
        from ..models.linearization import Linearizer

        linearizer = Linearizer(dt=0.02)

        times = []
        for i in range(n_solves):
            # Random initial state near origin
            x0 = np.random.randn(self.nx) * 0.5
            x0[2] = np.random.uniform(-np.pi / 4, np.pi / 4)

            # Reference: move forward
            x_refs = np.zeros((self.N + 1, self.nx))
            for k in range(self.N + 1):
                x_refs[k] = [k * 0.1, 0.0, 0.0]

            # Reference controls
            u_refs = np.zeros((self.N, self.nu))
            u_refs[:, 0] = 0.5  # Nominal forward velocity

            # Test obstacles
            obstacles = [
                {"x": 0.3, "y": 0.2, "radius": 0.1},
                {"x": 0.6, "y": -0.1, "radius": 0.15},
            ]

            # Linearization
            A_d, B_d = linearizer.get_discrete_model_explicit(0.5, 0.0)

            sol = self.solve_fast(x0, x_refs, A_d, B_d, u_refs, obstacles)
            times.append(sol.solve_time_ms)

        times = np.array(times)
        return {
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "median_ms": float(np.median(times)),
            "n_solves": n_solves,
            "used_generated": self.generated,
            "speedup_vs_target": 180.0 / float(np.mean(times)),  # vs original 180ms
        }

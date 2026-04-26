"""Fix MPC obstacle avoidance with minimal, targeted changes.
Uses warm-start trajectory for obstacle linearization instead of reference."""

filepath = r"src\hybrid_controller\hybrid_controller\controllers\mpc_controller.py"

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Find the method body boundaries
docstring_end = '            where n_k is re-linearized each SQP iteration from predicted trajectory.\n        """'
if docstring_end not in content:
    docstring_end = docstring_end.replace('\n', '\r\n')

body_start = content.find(docstring_end) + len(docstring_end)

next_method = '    def get_warm_start'
body_end = content.find(next_method, body_start)

new_body = '''
        start_time = time.perf_counter()
        
        if obstacles is None:
            obstacles = []
        
        # Ensure reference arrays have correct shape
        if x_refs.shape[0] < self.N + 1:
            x_refs_padded = np.zeros((self.N + 1, 3))
            x_refs_padded[:x_refs.shape[0]] = x_refs
            x_refs_padded[x_refs.shape[0]:] = x_refs[-1]
            x_refs = x_refs_padded
        
        if u_refs.shape[0] < self.N:
            u_refs_padded = np.zeros((self.N, 2))
            u_refs_padded[:u_refs.shape[0]] = u_refs
            u_refs_padded[u_refs.shape[0]:] = u_refs[-1]
            u_refs = u_refs_padded
        
        # Decision variables: Deviations from reference
        dx = cp.Variable((self.N + 1, self.nx))
        
        # Move-blocking
        du_blocked = cp.Variable((self.N_blocks, self.nu))
        du_expanded = []
        for k in range(self.N):
            block_idx = k // self.block_size
            if block_idx < self.N_blocks:
                du_expanded.append(du_blocked[block_idx])
            else:
                du_expanded.append(du_blocked[-1])
        
        # Slack variables for ALL N+1 states (including terminal)
        if use_soft_constraints and len(obstacles) > 0:
            slack = cp.Variable((self.N + 1) * len(obstacles), nonneg=True)
        else:
            slack = None
        
        # Objective
        cost = 0
        
        # Unwrap reference orientation to ensure continuity
        x_refs_unwrapped = x_refs.copy()
        x_refs_unwrapped[:, 2] = np.unwrap(x_refs[:, 2])
        
        # Adjust initial state theta
        theta_ref_0 = x_refs_unwrapped[0, 2]
        diff = x0[2] - theta_ref_0
        diff_norm = self._normalize_angle(diff)
        x0_adjusted = x0.copy()
        x0_adjusted[2] = theta_ref_0 + diff_norm
        
        # Adaptive weight scheduling during startup
        Q_active = self.Q.copy()
        if self._step_count < self._ramp_up_steps:
            ramp = (self._step_count + 1) / self._ramp_up_steps
            Q_active[2, 2] *= (2.0 - ramp)
        
        for k in range(self.N):
            cost += cp.quad_form(dx[k], Q_active)
            u_k = u_refs[k] + du_expanded[k]
            cost += cp.quad_form(u_k, self.R)
            if k > 0:
                du_rate = du_expanded[k] - du_expanded[k - 1]
                cost += cp.quad_form(du_rate, self.S)
            if self.J is not None and k > 1:
                jerk = du_expanded[k] - 2 * du_expanded[k - 1] + du_expanded[k - 2]
                cost += cp.quad_form(jerk, self.J)
        
        # Terminal cost
        cost += cp.quad_form(dx[self.N], self.P)
        
        # Slack penalty: L1 + L2 for strong obstacle enforcement
        if slack is not None:
            cost += self.slack_penalty_l1 * cp.sum(slack)
            cost += self.slack_penalty * cp.sum_squares(slack)
        
        # Constraints
        constraints = []
        
        # Initial condition
        constraints.append(dx[0] == x0_adjusted - x_refs_unwrapped[0])
        
        # LTV dynamics
        for k in range(self.N):
            v_r = u_refs[k, 0] if abs(u_refs[k, 0]) > 0.01 else 0.1
            theta_r = x_refs_unwrapped[k, 2]
            A_d, B_d = self.linearizer.get_discrete_model_explicit(v_r, theta_r)
            constraints.append(dx[k + 1] == A_d @ dx[k] + B_d @ du_expanded[k])
        
        # Actuator constraints
        v_max_robust = self.v_max * 0.95
        omega_max_robust = self.omega_max * 0.95
        for k in range(self.N):
            u_total = u_refs[k] + du_expanded[k]
            constraints.append(u_total[0] >= -v_max_robust)
            constraints.append(u_total[0] <= v_max_robust)
            constraints.append(u_total[1] >= -omega_max_robust)
            constraints.append(u_total[1] <= omega_max_robust)
        
        # --- Obstacle avoidance constraints ---
        # Key improvement: use warm-start trajectory for linearization when available.
        # Previous predicted trajectory gives much better constraint normals than
        # the reference trajectory, especially when detouring around obstacles.
        if self._prev_states is not None and len(self._prev_states) >= self.N + 1:
            obs_lin_points = self._prev_states[:self.N + 1, :2]
        else:
            obs_lin_points = x_refs_unwrapped[:self.N + 1, :2]
        
        slack_idx = 0
        for obs in obstacles:
            for k in range(self.N + 1):  # Include terminal state
                # Use warm-start trajectory for better constraint normals
                px_lin = obs_lin_points[k, 0]
                py_lin = obs_lin_points[k, 1]
                
                dx_obs = px_lin - obs.x
                dy_obs = py_lin - obs.y
                dist = np.sqrt(dx_obs**2 + dy_obs**2)
                
                if dist > 0.01:
                    nx_dir = dx_obs / dist
                    ny_dir = dy_obs / dist
                    safe_dist = self.d_safe + obs.radius + self.w_max
                    
                    # Constraint on absolute position (x_ref + dx)
                    px_ref = x_refs_unwrapped[k, 0]
                    py_ref = x_refs_unwrapped[k, 1]
                    dpx = dx[k, 0]
                    dpy = dx[k, 1]
                    
                    lhs = nx_dir * (px_ref + dpx - obs.x) + ny_dir * (py_ref + dpy - obs.y)
                    
                    if slack is not None:
                        constraints.append(lhs >= safe_dist - slack[slack_idx])
                        slack_idx += 1
                    else:
                        constraints.append(lhs >= safe_dist)
                else:
                    if slack is not None:
                        slack_idx += 1
        
        # Solve
        problem = cp.Problem(cp.Minimize(cost), constraints)
        
        try:
            problem.solve(solver=getattr(cp, self.solver), verbose=False, warm_start=True)
        except:
            try:
                problem.solve(solver=cp.SCS, verbose=False)
            except:
                pass
        
        solve_time = (time.perf_counter() - start_time) * 1000
        
        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            slack_used = slack is not None and slack.value is not None and np.any(slack.value > 1e-6)
            slack_margin = 0.0
            if slack is not None and slack.value is not None:
                slack_margin = float(np.max(np.abs(slack.value)))
            
            # Reconstruct absolute states and controls
            dx_val = dx.value
            
            du_blocked_val = du_blocked.value
            du_val = np.zeros((self.N, self.nu))
            for k in range(self.N):
                block_idx = min(k // self.block_size, self.N_blocks - 1)
                du_val[k] = du_blocked_val[block_idx]
            
            x_pred = x_refs[:self.N+1] + dx_val
            u_pred = u_refs[:self.N] + du_val
            
            # Cold-start ramp-up
            if self._step_count < self._ramp_up_steps:
                ramp_factor = (self._step_count + 1) / self._ramp_up_steps
                omega_limit = self.omega_max * ramp_factor
                u_pred[0, 1] = np.clip(u_pred[0, 1], -omega_limit, omega_limit)
            
            self._step_count += 1
            self._prev_solution = u_pred
            self._prev_states = x_pred
            
            return MPCSolution(
                status="optimal",
                optimal_control=u_pred[0],
                control_sequence=u_pred,
                predicted_states=x_pred,
                cost=problem.value,
                solve_time_ms=solve_time,
                slack_used=slack_used,
                iterations=0,
                feasibility_margin=slack_margin
            )
        else:
            return self._get_fallback_solution(x0, x_refs, u_refs, solve_time)
    
'''

content = content[:body_start] + new_body + content[body_end:]

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("SUCCESS: solve_with_ltv rewritten with warm-start obstacle linearization")

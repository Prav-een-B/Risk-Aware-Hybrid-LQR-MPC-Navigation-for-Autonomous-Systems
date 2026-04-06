"""
Reference trajectory generation utilities.

The generator now supports both analytic benchmark trajectories and
checkpoint-driven paths so the project can transition away from a
full-reference-only workflow without breaking existing call sites.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class TrajectoryPoint:
    """Single point on a reference trajectory."""

    t: float
    px: float
    py: float
    theta: float
    v: float
    omega: float

    def get_state(self) -> np.ndarray:
        """Return [px, py, theta]."""
        return np.array([self.px, self.py, self.theta])

    def get_control(self) -> np.ndarray:
        """Return [v, omega]."""
        return np.array([self.v, self.omega])


@dataclass
class Checkpoint:
    """Checkpoint representation used for path-driven references."""

    px: float
    py: float

    def to_array(self) -> np.ndarray:
        """Return [px, py]."""
        return np.array([self.px, self.py], dtype=float)


class ReferenceTrajectoryGenerator:
    """
    Reference trajectory generator.

    Supported trajectory types:
    - figure8
    - circle
    - clover
    - slalom
    - checkpoint_path
    - lissajous
    - spiral
    - spline_path
    - urban_path
    - sinusoidal
    - random_waypoint
    - clothoid
    """

    TRAJECTORY_TYPES = (
        "figure8",
        "circle",
        "clover",
        "slalom",
        "checkpoint_path",
        "lissajous",
        "spiral",
        "spline_path",
        "urban_path",
        "sinusoidal",
        "random_waypoint",
        "clothoid",
    )

    CHECKPOINT_PRESETS = {
        "diamond": [
            (0.0, 0.0),
            (1.4, 1.0),
            (2.8, 0.0),
            (1.4, -1.0),
            (0.0, 0.0),
        ],
        "slalom_lane": [
            (0.0, 0.0),
            (1.0, 0.7),
            (2.0, -0.7),
            (3.0, 0.7),
            (4.0, -0.7),
            (5.0, 0.0),
        ],
        "warehouse": [
            (0.0, 0.0),
            (1.2, 0.0),
            (1.2, 1.0),
            (2.4, 1.0),
            (2.4, -0.8),
            (3.8, -0.8),
        ],
        "corridor_turn": [
            (0.0, 0.0),
            (1.0, 0.0),
            (2.0, 0.0),
            (2.5, 0.6),
            (2.5, 1.6),
            (2.5, 2.6),
        ],
    }

    def __init__(
        self,
        A: float = 2.0,
        a: float = 0.5,
        dt: float = 0.02,
        T_blend: float = 0.5,
        trajectory_type: str = "figure8",
        checkpoints: Optional[Sequence[Sequence[float]]] = None,
        checkpoint_preset: Optional[str] = None,
        nominal_speed: float = 0.8,
        checkpoint_mode: bool = False,
        # Lissajous parameters
        lissajous_b: float = 2.0,
        lissajous_c: float = 1.5,
        # Spiral parameters
        spiral_r0: float = 0.5,
        spiral_k: float = 0.3,
        spiral_omega: float = 1.0,
        # Spline path waypoints
        spline_waypoints: Optional[Sequence[Sequence[float]]] = None,
        # Urban path parameters
        urban_segment_length: float = 2.0,
        urban_num_segments: int = 4,
        # Sinusoidal parameters
        sinusoidal_v: float = 0.8,
        sinusoidal_omega: float = 1.0,
        # Random waypoint parameters
        random_num_waypoints: int = 8,
        random_bounds: Tuple[float, float, float, float] = (-3.0, 3.0, -3.0, 3.0),
        random_seed: Optional[int] = None,
        # Clothoid parameters
        clothoid_kappa0: float = 0.0,
        clothoid_k_rate: float = 0.5,
        clothoid_length: float = 10.0,
        # Checkpoint manager parameters
        base_switching_radius: float = 0.3,
        curvature_scaling: float = 0.2,
        hysteresis_margin: float = 0.1,
        forward_progress_timeout: float = 1.0,
    ):
        if trajectory_type not in self.TRAJECTORY_TYPES:
            raise ValueError(
                f"Unsupported trajectory_type '{trajectory_type}'. "
                f"Expected one of {self.TRAJECTORY_TYPES}."
            )

        self.A = A
        self.a = a
        self.dt = dt
        self.T_blend = T_blend
        self.trajectory_type = trajectory_type
        self.checkpoint_preset = checkpoint_preset
        self.nominal_speed = max(float(nominal_speed), 1e-3)
        self.checkpoint_mode = checkpoint_mode

        # Lissajous parameters
        self.lissajous_b = lissajous_b
        self.lissajous_c = lissajous_c

        # Spiral parameters
        self.spiral_r0 = spiral_r0
        self.spiral_k = spiral_k
        self.spiral_omega = spiral_omega

        # Spline path waypoints
        self.spline_waypoints = spline_waypoints

        # Urban path parameters
        self.urban_segment_length = urban_segment_length
        self.urban_num_segments = urban_num_segments

        # Sinusoidal parameters
        self.sinusoidal_v = sinusoidal_v
        self.sinusoidal_omega = sinusoidal_omega

        # Random waypoint parameters
        self.random_num_waypoints = random_num_waypoints
        self.random_bounds = random_bounds
        self.random_seed = random_seed

        # Clothoid parameters
        self.clothoid_kappa0 = clothoid_kappa0
        self.clothoid_k_rate = clothoid_k_rate
        self.clothoid_length = clothoid_length

        self._checkpoints = self._normalize_checkpoints(checkpoints)
        self._trajectory: Optional[np.ndarray] = None
        self._duration: float = 0.0
        
        # Initialize checkpoint manager if checkpoint mode is enabled
        self.checkpoint_manager: Optional['CheckpointManager'] = None
        if self.checkpoint_mode:
            from ..navigation.checkpoint_manager import CheckpointManager
            self.checkpoint_manager = CheckpointManager(
                base_switching_radius=base_switching_radius,
                curvature_scaling=curvature_scaling,
                hysteresis_margin=hysteresis_margin,
                forward_progress_timeout=forward_progress_timeout,
                dt=dt
            )

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _normalize_checkpoints(
        self, checkpoints: Optional[Sequence[Sequence[float]]]
    ) -> Optional[List[Checkpoint]]:
        """Normalize raw checkpoint input into Checkpoint objects."""
        if checkpoints is None:
            return None

        normalized: List[Checkpoint] = []
        for point in checkpoints:
            if len(point) != 2:
                raise ValueError("Each checkpoint must contain exactly two values.")
            normalized.append(Checkpoint(float(point[0]), float(point[1])))
        return normalized

    def _build_time_grid(self, duration: float) -> np.ndarray:
        """Build a time grid with at least two points."""
        duration = max(float(duration), self.dt)
        t = np.arange(0.0, duration, self.dt)
        if t.size < 2:
            t = np.array([0.0, self.dt])
        return t

    def _velocity_ramp(self, t: float) -> float:
        """Smooth start velocity ramp."""
        if self.T_blend <= 0.0 or t >= self.T_blend:
            return 1.0
        s = t / self.T_blend
        return 3.0 * s**2 - 2.0 * s**3

    def _analytic_position(self, t: float) -> Tuple[float, float]:
        """Evaluate an analytic trajectory at time t."""
        if self.trajectory_type == "figure8":
            px = self.A * np.sin(self.a * t)
            py = self.A * np.sin(self.a * t) * np.cos(self.a * t)
            return px, py

        if self.trajectory_type == "circle":
            px = self.A * (1.0 - np.cos(self.a * t))
            py = self.A * np.sin(self.a * t)
            return px, py

        if self.trajectory_type == "clover":
            phase = self.a * t
            px = self.A * np.cos(2.0 * phase) * np.cos(phase)
            py = self.A * np.cos(2.0 * phase) * np.sin(phase)
            return px, py

        if self.trajectory_type == "slalom":
            phase = self.a * t
            px = self.A * np.sin(phase)
            py = 0.45 * self.A * np.sin(2.0 * phase) + 0.2 * self.A * np.sin(3.0 * phase)
            return px, py

        if self.trajectory_type == "lissajous":
            # x = A*sin(a*t), y = A*sin(b*t)*cos(c*t)
            px = self.A * np.sin(self.a * t)
            py = self.A * np.sin(self.lissajous_b * t) * np.cos(self.lissajous_c * t)
            return px, py

        if self.trajectory_type == "spiral":
            # Polar coordinates: r = r0 + k*t, theta = omega*t
            r = self.spiral_r0 + self.spiral_k * t
            theta = self.spiral_omega * t
            px = r * np.cos(theta)
            py = r * np.sin(theta)
            return px, py

        if self.trajectory_type == "sinusoidal":
            # x = v*t, y = A*sin(omega*t)
            px = self.sinusoidal_v * t
            py = self.A * np.sin(self.sinusoidal_omega * t)
            return px, py

        if self.trajectory_type == "clothoid":
            # Euler spiral with linearly varying curvature
            # kappa(s) = kappa0 + k_rate*s
            # Using Fresnel integrals approximation
            s = t * self.nominal_speed  # Arc length parameter
            kappa = self.clothoid_kappa0 + self.clothoid_k_rate * s
            
            # Numerical integration for clothoid
            # theta(s) = kappa0*s + 0.5*k_rate*s^2
            theta = self.clothoid_kappa0 * s + 0.5 * self.clothoid_k_rate * s**2
            
            # Position via integration: dx/ds = cos(theta(s)), dy/ds = sin(theta(s))
            # Using small-step approximation
            n_steps = max(10, int(s / 0.1))
            if n_steps > 0:
                s_vals = np.linspace(0, s, n_steps)
                theta_vals = self.clothoid_kappa0 * s_vals + 0.5 * self.clothoid_k_rate * s_vals**2
                px = np.trapz(np.cos(theta_vals), s_vals)
                py = np.trapz(np.sin(theta_vals), s_vals)
            else:
                px, py = 0.0, 0.0
            
            return float(px), float(py)

        raise ValueError(
            "Analytic position is only defined for non-checkpoint trajectory types."
        )

    def _resolve_checkpoint_points(self) -> np.ndarray:
        """Resolve explicit checkpoints or a named preset."""
        # Handle spline_path trajectory type
        if self.trajectory_type == "spline_path":
            if self.spline_waypoints is not None:
                points = np.array(self.spline_waypoints, dtype=float)
            else:
                # Default spline waypoints from design doc
                points = np.array([
                    [0.0, 0.0],
                    [2.0, 1.0],
                    [3.0, -1.0],
                    [1.0, -2.0],
                    [0.0, 0.0]
                ], dtype=float)
            if len(points) < 2:
                raise ValueError("At least two waypoints are required for spline_path.")
            return points

        # Handle urban_path trajectory type
        if self.trajectory_type == "urban_path":
            # Create straight segments with 90° turns
            points = []
            x, y = 0.0, 0.0
            direction = 0  # 0: right, 1: up, 2: left, 3: down
            
            for i in range(self.urban_num_segments):
                if direction == 0:  # Move right
                    x += self.urban_segment_length
                elif direction == 1:  # Move up
                    y += self.urban_segment_length
                elif direction == 2:  # Move left
                    x -= self.urban_segment_length
                elif direction == 3:  # Move down
                    y -= self.urban_segment_length
                
                points.append([x, y])
                direction = (direction + 1) % 4  # 90° turn
            
            return np.array(points, dtype=float)

        # Handle random_waypoint trajectory type
        if self.trajectory_type == "random_waypoint":
            if self.random_seed is not None:
                np.random.seed(self.random_seed)
            
            x_min, x_max, y_min, y_max = self.random_bounds
            x_points = np.random.uniform(x_min, x_max, self.random_num_waypoints)
            y_points = np.random.uniform(y_min, y_max, self.random_num_waypoints)
            
            # Start and end at origin for closed loop
            points = [[0.0, 0.0]]
            for i in range(self.random_num_waypoints):
                points.append([x_points[i], y_points[i]])
            points.append([0.0, 0.0])
            
            return np.array(points, dtype=float)

        # Handle explicit checkpoints or presets
        if self._checkpoints is not None:
            points = np.array([cp.to_array() for cp in self._checkpoints], dtype=float)
        elif self.checkpoint_preset is not None:
            if self.checkpoint_preset not in self.CHECKPOINT_PRESETS:
                raise ValueError(
                    f"Unknown checkpoint preset '{self.checkpoint_preset}'. "
                    f"Expected one of {sorted(self.CHECKPOINT_PRESETS)}."
                )
            points = np.array(self.CHECKPOINT_PRESETS[self.checkpoint_preset], dtype=float)
        else:
            points = np.array(self.CHECKPOINT_PRESETS["diamond"], dtype=float)

        if len(points) < 2:
            raise ValueError("At least two checkpoints are required.")
        return points

    @staticmethod
    def _catmull_rom_point(
        p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, u: float
    ) -> np.ndarray:
        """Evaluate a Catmull-Rom spline point."""
        u2 = u * u
        u3 = u2 * u
        return 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * u
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3
        )

    def _sample_checkpoint_curve(self) -> np.ndarray:
        """Create a smooth path through checkpoints."""
        points = self._resolve_checkpoint_points()
        if len(points) == 2:
            steps = max(
                2,
                int(np.ceil(np.linalg.norm(points[1] - points[0]) / max(self.dt, 0.05))),
            )
            return np.linspace(points[0], points[1], steps)

        sampled: List[np.ndarray] = []
        spacing = max(self.nominal_speed * self.dt * 0.5, 0.05)

        for idx in range(len(points) - 1):
            p0 = points[max(idx - 1, 0)]
            p1 = points[idx]
            p2 = points[idx + 1]
            p3 = points[min(idx + 2, len(points) - 1)]

            segment_length = np.linalg.norm(p2 - p1)
            n_samples = max(10, int(np.ceil(segment_length / spacing)))
            u_values = np.linspace(0.0, 1.0, n_samples, endpoint=(idx == len(points) - 2))

            for u in u_values:
                sampled.append(self._catmull_rom_point(p0, p1, p2, p3, float(u)))

        curve = np.array(sampled, dtype=float)
        if curve.ndim != 2 or curve.shape[1] != 2:
            raise ValueError("Failed to build checkpoint curve.")
        return curve

    def _checkpoint_positions(self, duration: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate sampled positions for a checkpoint-driven path."""
        curve = self._sample_checkpoint_curve()
        diffs = np.diff(curve, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(seg_lengths)))
        total_length = max(float(cumulative[-1]), 1e-6)

        if duration is None:
            duration = total_length / self.nominal_speed

        t = self._build_time_grid(duration)
        arc_positions = np.linspace(0.0, total_length, t.size)
        px = np.interp(arc_positions, cumulative, curve[:, 0])
        py = np.interp(arc_positions, cumulative, curve[:, 1])
        return t, np.column_stack((px, py))

    def _analytic_positions(self, duration: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate sampled positions for an analytic trajectory."""
        # Check if this is a checkpoint-based trajectory type
        if self.trajectory_type in ("spline_path", "urban_path", "random_waypoint"):
            return self._checkpoint_positions(duration)
        
        t = self._build_time_grid(duration)
        positions = np.array([self._analytic_position(float(tk)) for tk in t], dtype=float)
        return t, positions

    def _build_trajectory(
        self, t: np.ndarray, positions: np.ndarray
    ) -> np.ndarray:
        """Build the full [t, px, py, theta, v, omega] trajectory array."""
        edge_order = 2 if t.size > 2 else 1
        raw_dx = np.gradient(positions[:, 0], self.dt, edge_order=edge_order)
        raw_dy = np.gradient(positions[:, 1], self.dt, edge_order=edge_order)

        sigma = np.array([self._velocity_ramp(float(tk)) for tk in t])
        dx = raw_dx * sigma
        dy = raw_dy * sigma

        theta_unwrapped = np.unwrap(np.arctan2(raw_dy, raw_dx))
        omega = np.gradient(theta_unwrapped, self.dt, edge_order=edge_order)
        theta = np.array([self._wrap_angle(angle) for angle in theta_unwrapped])
        v = np.sqrt(dx**2 + dy**2)

        trajectory = np.zeros((t.size, 6), dtype=float)
        trajectory[:, 0] = t
        trajectory[:, 1:3] = positions
        trajectory[:, 3] = theta
        trajectory[:, 4] = v
        trajectory[:, 5] = omega
        return trajectory

    def _interpolate_cached(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate a state and control from the cached trajectory."""
        if self._trajectory is None:
            raise ValueError("Trajectory not generated. Call generate() first.")

        t_clamped = np.clip(float(t), self._trajectory[0, 0], self._trajectory[-1, 0])
        px = np.interp(t_clamped, self._trajectory[:, 0], self._trajectory[:, 1])
        py = np.interp(t_clamped, self._trajectory[:, 0], self._trajectory[:, 2])

        theta_unwrapped = np.unwrap(self._trajectory[:, 3])
        theta = np.interp(t_clamped, self._trajectory[:, 0], theta_unwrapped)
        theta = self._wrap_angle(theta)

        v = np.interp(t_clamped, self._trajectory[:, 0], self._trajectory[:, 4])
        omega = np.interp(t_clamped, self._trajectory[:, 0], self._trajectory[:, 5])

        return np.array([px, py, theta]), np.array([v, omega])

    def position(self, t: float) -> Tuple[float, float]:
        """Return the reference position at time t."""
        # Checkpoint-based trajectories need cached trajectory
        if self.trajectory_type in ("checkpoint_path", "spline_path", "urban_path", "random_waypoint"):
            x_ref, _ = self._interpolate_cached(t)
            return float(x_ref[0]), float(x_ref[1])
        return self._analytic_position(t)

    def velocity(self, t: float) -> Tuple[float, float]:
        """Return the reference velocity components at time t."""
        if self._trajectory is not None:
            _, u_ref = self._interpolate_cached(t)
            theta = self.heading(t)
            return float(u_ref[0] * np.cos(theta)), float(u_ref[0] * np.sin(theta))

        # Checkpoint-based trajectories need cached trajectory
        if self.trajectory_type in ("checkpoint_path", "spline_path", "urban_path", "random_waypoint"):
            raise ValueError("Generate the checkpoint trajectory before requesting velocity.")

        eps = max(self.dt * 0.5, 1e-3)
        px0, py0 = self._analytic_position(max(0.0, t - eps))
        px1, py1 = self._analytic_position(t + eps)
        sigma = self._velocity_ramp(t)
        return sigma * (px1 - px0) / (2.0 * eps), sigma * (py1 - py0) / (2.0 * eps)

    def heading(self, t: float) -> float:
        """Return the reference heading at time t."""
        if self._trajectory is not None:
            x_ref, _ = self._interpolate_cached(t)
            return float(x_ref[2])

        # Checkpoint-based trajectories need cached trajectory
        if self.trajectory_type in ("checkpoint_path", "spline_path", "urban_path", "random_waypoint"):
            raise ValueError("Generate the checkpoint trajectory before requesting heading.")

        eps = max(self.dt * 0.5, 1e-3)
        px0, py0 = self._analytic_position(max(0.0, t - eps))
        px1, py1 = self._analytic_position(t + eps)
        return float(np.arctan2(py1 - py0, px1 - px0))

    def linear_velocity(self, t: float) -> float:
        """Return the reference linear speed at time t."""
        dx, dy = self.velocity(t)
        return float(np.sqrt(dx**2 + dy**2))

    def angular_velocity(self, t: float) -> float:
        """Return the reference angular velocity at time t."""
        if self._trajectory is not None:
            _, u_ref = self._interpolate_cached(t)
            return float(u_ref[1])

        eps = max(self.dt * 0.5, 1e-3)
        theta_prev = self.heading(max(0.0, t - eps))
        theta_next = self.heading(t + eps)
        dtheta = self._wrap_angle(theta_next - theta_prev)
        return float(dtheta / (2.0 * eps))

    def get_reference_at_time(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return (x_ref, u_ref) at time t."""
        if self._trajectory is not None:
            return self._interpolate_cached(t)

        px, py = self.position(t)
        theta = self.heading(t)
        v = self.linear_velocity(t)
        omega = self.angular_velocity(t)
        return np.array([px, py, theta]), np.array([v, omega])

    def generate(self, duration: Optional[float] = None) -> np.ndarray:
        """Generate and cache a discrete-time reference trajectory."""
        # Checkpoint-based trajectory types
        if self.trajectory_type in ("checkpoint_path", "spline_path", "urban_path", "random_waypoint"):
            t, positions = self._checkpoint_positions(duration)
        else:
            # Analytic trajectory types
            if duration is None:
                raise ValueError("duration must be provided for analytic trajectories.")
            t, positions = self._analytic_positions(duration)

        trajectory = self._build_trajectory(t, positions)
        self._trajectory = trajectory
        self._duration = float(trajectory[-1, 0]) if len(trajectory) > 1 else 0.0
        
        # If checkpoint mode is enabled, generate checkpoints with curvature
        if self.checkpoint_mode and self.checkpoint_manager is not None:
            checkpoints = self.generate_checkpoints_with_curvature()
            self.checkpoint_manager.set_checkpoints(checkpoints)
        
        return trajectory
    
    def generate_checkpoints_with_curvature(
        self,
        curvature_threshold_high: float = 2.0,
        curvature_threshold_low: float = 0.5,
        min_spacing: float = 0.1,
        max_spacing: float = 1.0
    ) -> List['Checkpoint']:
        """
        Generate checkpoints with curvature-aware spacing from cached trajectory.
        
        Args:
            curvature_threshold_high: High curvature threshold (1/m)
            curvature_threshold_low: Low curvature threshold (1/m)
            min_spacing: Minimum checkpoint spacing (m)
            max_spacing: Maximum checkpoint spacing (m)
            
        Returns:
            List of Checkpoint objects with curvature information
        """
        from ..navigation.checkpoint_manager import Checkpoint as CPCheckpoint
        
        if self._trajectory is None:
            raise ValueError("Trajectory not generated. Call generate() first.")
        
        # Compute curvature at each trajectory point
        curvature = self._compute_curvature()
        
        # Generate checkpoints with adaptive spacing
        checkpoints = []
        current_distance = 0.0
        last_checkpoint_idx = 0
        
        for i in range(1, len(self._trajectory)):
            # Compute distance traveled
            dx = self._trajectory[i, 1] - self._trajectory[i-1, 1]
            dy = self._trajectory[i, 2] - self._trajectory[i-1, 2]
            current_distance += np.sqrt(dx**2 + dy**2)
            
            # Compute required spacing based on curvature
            kappa = curvature[i]
            if kappa >= curvature_threshold_high:
                required_spacing = min_spacing
            elif kappa <= curvature_threshold_low:
                required_spacing = max_spacing
            else:
                # Linear interpolation
                alpha = (kappa - curvature_threshold_low) / (curvature_threshold_high - curvature_threshold_low)
                required_spacing = max_spacing - alpha * (max_spacing - min_spacing)
            
            # Add checkpoint if spacing threshold reached
            if current_distance >= required_spacing:
                checkpoint = CPCheckpoint(
                    x=float(self._trajectory[i, 1]),
                    y=float(self._trajectory[i, 2]),
                    theta=float(self._trajectory[i, 3]),
                    curvature=float(kappa),
                    index=i
                )
                checkpoints.append(checkpoint)
                current_distance = 0.0
                last_checkpoint_idx = i
        
        # Always add the final point as a checkpoint
        if last_checkpoint_idx < len(self._trajectory) - 1:
            i = len(self._trajectory) - 1
            checkpoint = CPCheckpoint(
                x=float(self._trajectory[i, 1]),
                y=float(self._trajectory[i, 2]),
                theta=float(self._trajectory[i, 3]),
                curvature=float(curvature[i]),
                index=i
            )
            checkpoints.append(checkpoint)
        
        return checkpoints
    
    def _compute_curvature(self) -> np.ndarray:
        """
        Compute curvature at each trajectory point using finite differences.
        
        Returns:
            curvature: (N,) array of curvature values (1/m)
        """
        if self._trajectory is None:
            raise ValueError("Trajectory not generated. Call generate() first.")
        
        # First derivatives (velocity)
        dx = np.gradient(self._trajectory[:, 1], self.dt)
        dy = np.gradient(self._trajectory[:, 2], self.dt)
        
        # Second derivatives (acceleration)
        d2x = np.gradient(dx, self.dt)
        d2y = np.gradient(dy, self.dt)
        
        # Curvature formula: kappa = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
        numerator = np.abs(dx * d2y - dy * d2x)
        denominator = (dx**2 + dy**2)**(3/2)
        
        # Avoid division by zero
        denominator = np.maximum(denominator, 1e-6)
        
        curvature = numerator / denominator
        return curvature

    def generate_figure_8(
        self, t_range: float, dt: float, A: float = None, a: float = None
    ) -> np.ndarray:
        """Retain the earlier helper used by older docs and examples."""
        if A is None:
            A = self.A
        if a is None:
            a = self.a

        t = np.arange(0, t_range, dt)
        xr = A * np.sin(a * t)
        yr = A * np.sin(a * t) * np.cos(a * t)
        dxr = a * A * np.cos(a * t)
        dyr = a * A * (np.cos(a * t) ** 2 - np.sin(a * t) ** 2)
        thetar = np.arctan2(dyr, dxr)
        vr = np.sqrt(dxr**2 + dyr**2)
        wr = np.diff(thetar, append=thetar[-1]) / dt
        wr = np.where(wr > np.pi / dt, wr - 2.0 * np.pi / dt, wr)
        wr = np.where(wr < -np.pi / dt, wr + 2.0 * np.pi / dt, wr)
        return np.stack([xr, yr, thetar, vr, wr], axis=1)

    def get_reference_at_index(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (x_ref, u_ref) from the cached trajectory at index k."""
        if self._trajectory is None:
            raise ValueError("Trajectory not generated. Call generate() first.")

        idx = min(max(int(k), 0), len(self._trajectory) - 1)
        point = self._trajectory[idx]
        x_ref = np.array([point[1], point[2], point[3]])
        u_ref = np.array([point[4], point[5]])
        return x_ref, u_ref

    def get_trajectory_segment(
        self, start_idx: int, horizon: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return a prediction segment from the cached trajectory."""
        if self._trajectory is None:
            raise ValueError("Trajectory not generated. Call generate() first.")

        x_refs = np.zeros((horizon, 3), dtype=float)
        u_refs = np.zeros((horizon, 2), dtype=float)
        n_points = len(self._trajectory)

        for offset in range(horizon):
            idx = min(max(start_idx + offset, 0), n_points - 1)
            point = self._trajectory[idx]
            x_refs[offset] = [point[1], point[2], point[3]]
            u_refs[offset] = [point[4], point[5]]

        return x_refs, u_refs

    def get_local_trajectory_segment(
        self, state: np.ndarray, horizon: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return a local forward horizon anchored to the nearest cached point.

        This is primarily intended for checkpoint-driven references where the
        controller should advance along the path based on the robot position
        rather than a fixed global time index.
        
        If checkpoint_mode is enabled, uses CheckpointManager for reference extraction.
        """
        # Use CheckpointManager if checkpoint mode is enabled
        if self.checkpoint_mode and self.checkpoint_manager is not None:
            return self.checkpoint_manager.get_local_trajectory_segment(state, horizon)
        
        # Otherwise use the original implementation
        if self._trajectory is None:
            raise ValueError("Trajectory not generated. Call generate() first.")

        if self._trajectory.shape[0] == 0:
            raise ValueError("Cached trajectory is empty.")

        position = np.asarray(state, dtype=float).reshape(-1)[:2]
        distances = np.linalg.norm(self._trajectory[:, 1:3] - position, axis=1)
        start_idx = int(np.argmin(distances))
        return self.get_trajectory_segment(start_idx, horizon)

    def get_checkpoints(self) -> List[Checkpoint]:
        """Return the active checkpoint list."""
        points = self._resolve_checkpoint_points()
        return [Checkpoint(float(point[0]), float(point[1])) for point in points]
    
    def update_checkpoint_manager(self, robot_position: np.ndarray, current_time: float) -> bool:
        """
        Update checkpoint manager with current robot position.
        
        Args:
            robot_position: Current robot position [x, y] or [x, y, theta]
            current_time: Current simulation time (s)
            
        Returns:
            True if checkpoint was switched, False otherwise
        """
        if self.checkpoint_mode and self.checkpoint_manager is not None:
            return self.checkpoint_manager.update(robot_position, current_time)
        return False
    
    def get_checkpoint_metrics(self) -> Dict[str, float]:
        """
        Get checkpoint tracking metrics.
        
        Returns:
            Dictionary of checkpoint metrics or empty dict if not in checkpoint mode
        """
        if self.checkpoint_mode and self.checkpoint_manager is not None:
            return self.checkpoint_manager.get_metrics()
        return {}

    @property
    def num_points(self) -> int:
        """Return the number of cached points."""
        return 0 if self._trajectory is None else len(self._trajectory)

    def get_trajectory_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Return ((x_min, x_max), (y_min, y_max))."""
        if self._trajectory is not None:
            x_min, x_max = self._trajectory[:, 1].min(), self._trajectory[:, 1].max()
            y_min, y_max = self._trajectory[:, 2].min(), self._trajectory[:, 2].max()
            return ((float(x_min), float(x_max)), (float(y_min), float(y_max)))

        # Checkpoint-based trajectories
        if self.trajectory_type in ("checkpoint_path", "spline_path", "urban_path", "random_waypoint"):
            points = self._resolve_checkpoint_points()
            return (
                (float(points[:, 0].min()), float(points[:, 0].max())),
                (float(points[:, 1].min()), float(points[:, 1].max())),
            )

        if self.trajectory_type == "circle":
            return ((0.0, 2.0 * self.A), (-self.A, self.A))

        # Default bounds for most analytic trajectories
        return ((-self.A, self.A), (-self.A, self.A))

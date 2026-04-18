from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from hybrid_controller.controllers.mpc_controller import Obstacle


DEFAULT_BOUNDS = (-2.5, 2.5, -1.8, 1.8)


@dataclass
class UncertaintyConfig:
    """Process and sensor noise parameters for uncertainty injection."""
    
    # Process noise (added to dynamics)
    process_noise_position_std: float = 0.0  # Position noise (m)
    process_noise_heading_std: float = 0.0   # Heading noise (rad)
    
    # Sensor noise (added to measurements)
    sensor_noise_position_std: float = 0.0   # Measurement noise (m)
    sensor_noise_heading_std: float = 0.0    # Measurement noise (rad)
    
    # Model mismatch (scaling factors)
    velocity_mismatch_factor: float = 1.0    # Actual/commanded velocity ratio
    angular_mismatch_factor: float = 1.0     # Actual/commanded angular velocity ratio
    
    # Control latency
    control_delay_steps: int = 0             # Pipeline delay (timesteps)

    # Covariances for Stochastic MPC and Full Covariance-Driven Risk
    estimation_covariance: np.ndarray = None  # Replaces fixed w_max scalar bounds
    
    def __post_init__(self):
        if self.estimation_covariance is None:
            self.estimation_covariance = np.zeros((2, 2))


class UncertaintyInjector:
    """Handles uncertainty injection for process noise, sensor noise, model mismatch, and control latency."""
    
    def __init__(self, config: UncertaintyConfig, seed: int = 0):
        """
        Initialize uncertainty injector.
        
        Args:
            config: Uncertainty configuration parameters
            seed: Random seed for reproducibility
        """
        self.config = config
        self.rng = np.random.RandomState(seed)
        
        # Control latency buffer
        self.control_buffer: List[np.ndarray] = []
    
    def inject_process_noise(self, state: np.ndarray) -> np.ndarray:
        """
        Add zero-mean Gaussian process noise to state.
        
        Args:
            state: Robot state [x, y, theta]
            
        Returns:
            State with process noise added
        """
        noisy_state = state.copy()
        
        if self.config.process_noise_position_std > 0:
            # Add noise to x and y position
            noisy_state[0] += self.rng.normal(0.0, self.config.process_noise_position_std)
            noisy_state[1] += self.rng.normal(0.0, self.config.process_noise_position_std)
        
        if self.config.process_noise_heading_std > 0:
            # Add noise to heading
            noisy_state[2] += self.rng.normal(0.0, self.config.process_noise_heading_std)
        
        return noisy_state
    
    def inject_sensor_noise(self, measurement: np.ndarray) -> np.ndarray:
        """
        Add zero-mean Gaussian sensor noise to measurement.
        
        Args:
            measurement: Measured state [x, y, theta]
            
        Returns:
            Measurement with sensor noise added
        """
        noisy_measurement = measurement.copy()
        
        if self.config.sensor_noise_position_std > 0:
            # Add noise to x and y measurement
            noisy_measurement[0] += self.rng.normal(0.0, self.config.sensor_noise_position_std)
            noisy_measurement[1] += self.rng.normal(0.0, self.config.sensor_noise_position_std)
        
        if self.config.sensor_noise_heading_std > 0:
            # Add noise to heading measurement
            noisy_measurement[2] += self.rng.normal(0.0, self.config.sensor_noise_heading_std)
        
        return noisy_measurement
    
    def apply_model_mismatch(self, control: np.ndarray) -> np.ndarray:
        """
        Scale commanded velocities by mismatch factors.
        
        Args:
            control: Commanded control [v, omega]
            
        Returns:
            Control with model mismatch applied
        """
        mismatched_control = control.copy()
        
        # Scale linear velocity
        mismatched_control[0] *= self.config.velocity_mismatch_factor
        
        # Scale angular velocity
        mismatched_control[1] *= self.config.angular_mismatch_factor
        
        return mismatched_control
    
    def buffer_control(self, control: np.ndarray) -> np.ndarray:
        """
        Buffer control command and return delayed control.
        
        Args:
            control: Current control command [v, omega]
            
        Returns:
            Delayed control command (or current if no delay)
        """
        if self.config.control_delay_steps <= 0:
            return control
        
        # Add current control to buffer
        self.control_buffer.append(control.copy())
        
        # If buffer is not full yet, return zero control
        if len(self.control_buffer) <= self.config.control_delay_steps:
            return np.zeros_like(control)
        
        # Return delayed control and remove from buffer
        delayed_control = self.control_buffer.pop(0)
        return delayed_control
    
    def reset_buffer(self):
        """Reset the control latency buffer."""
        self.control_buffer = []


@dataclass
class InflationConfig:
    """Explicit obstacle inflation settings used by risk and MPC."""

    safety_factor: float = 1.0
    sensing_factor: float = 0.0
    motion_lookahead: float = 0.5
    velocity_scaling_factor: float = 0.0
    sensing_range: float = float('inf')

    def normalized(self) -> "InflationConfig":
        return InflationConfig(
            safety_factor=max(1.0, float(self.safety_factor)),
            sensing_factor=max(0.0, float(self.sensing_factor)),
            motion_lookahead=max(0.0, float(self.motion_lookahead)),
            velocity_scaling_factor=max(0.0, float(self.velocity_scaling_factor)),
            sensing_range=max(0.0, float(self.sensing_range)) if not np.isinf(self.sensing_range) else float('inf'),
        )

    def compute_inflated_radius(
        self,
        base_radius: float,
        obstacle_speed: float,
        robot_speed: float = 0.0
    ) -> float:
        """
        Compute inflated obstacle radius.
        
        Formula:
            r_inflated = base_radius * safety_factor 
                       + sensing_factor
                       + motion_lookahead * obstacle_speed
                       + velocity_scaling_factor * robot_speed
        
        Args:
            base_radius: Base obstacle radius (m)
            obstacle_speed: Obstacle speed (m/s)
            robot_speed: Robot speed (m/s)
            
        Returns:
            Inflated radius (m)
        """
        return (
            base_radius * self.safety_factor
            + self.sensing_factor
            + self.motion_lookahead * obstacle_speed
            + self.velocity_scaling_factor * robot_speed
        )


@dataclass
class DynamicObstacle:
    """Obstacle state with optional bounded motion."""

    x: float
    y: float
    radius: float
    vx: float = 0.0
    vy: float = 0.0
    motion_model: str = "static"
    random_walk_std: float = 0.0
    max_speed: float = 0.5
    bounds: Tuple[float, float, float, float] = DEFAULT_BOUNDS

    @classmethod
    def from_dict(
        cls,
        payload: Dict[str, object],
        bounds: Tuple[float, float, float, float] = DEFAULT_BOUNDS,
    ) -> "DynamicObstacle":
        return cls(
            x=float(payload["x"]),
            y=float(payload["y"]),
            radius=float(payload["radius"]),
            vx=float(payload.get("vx", 0.0)),
            vy=float(payload.get("vy", 0.0)),
            motion_model=str(payload.get("motion_model", "static")),
            random_walk_std=float(payload.get("random_walk_std", 0.0)),
            max_speed=float(payload.get("max_speed", 0.5)),
            bounds=tuple(payload.get("bounds", bounds)),
        )

    @property
    def speed(self) -> float:
        return float(np.hypot(self.vx, self.vy))

    def clone(self) -> "DynamicObstacle":
        return DynamicObstacle(
            x=self.x,
            y=self.y,
            radius=self.radius,
            vx=self.vx,
            vy=self.vy,
            motion_model=self.motion_model,
            random_walk_std=self.random_walk_std,
            max_speed=self.max_speed,
            bounds=self.bounds,
        )

    def _clip_speed(self) -> None:
        speed = self.speed
        if speed <= self.max_speed or speed <= 1e-9:
            return
        scale = self.max_speed / speed
        self.vx *= scale
        self.vy *= scale

    def _apply_bounds(self) -> None:
        x_min, x_max, y_min, y_max = self.bounds
        min_x = x_min + self.radius
        max_x = x_max - self.radius
        min_y = y_min + self.radius
        max_y = y_max - self.radius

        if self.x < min_x:
            self.x = min_x
            self.vx = abs(self.vx)
        elif self.x > max_x:
            self.x = max_x
            self.vx = -abs(self.vx)

        if self.y < min_y:
            self.y = min_y
            self.vy = abs(self.vy)
        elif self.y > max_y:
            self.y = max_y
            self.vy = -abs(self.vy)

    def step(self, dt: float, rng: np.random.RandomState) -> None:
        """Advance the obstacle state inside its bounding box."""
        if self.motion_model == "static":
            return

        if self.motion_model == "random_walk":
            walk_scale = self.random_walk_std * np.sqrt(max(dt, 1e-6))
            self.vx += float(rng.normal(0.0, walk_scale))
            self.vy += float(rng.normal(0.0, walk_scale))
            self._clip_speed()
        elif self.motion_model == "linear":
            # Linear motion model - just enforce speed limits
            self._clip_speed()

        self.x += self.vx * dt
        self.y += self.vy * dt
        self._apply_bounds()

    def inflated_radius(
        self, 
        inflation: Optional[InflationConfig] = None,
        robot_velocity: float = 0.0
    ) -> float:
        if inflation is None:
            return self.radius
        cfg = inflation.normalized()
        return cfg.compute_inflated_radius(self.radius, self.speed, robot_velocity)

    def to_dict(
        self,
        inflation: Optional[InflationConfig] = None,
        use_base_radius: bool = False,
        robot_velocity: float = 0.0,
    ) -> Dict[str, object]:
        radius = self.radius if use_base_radius else self.inflated_radius(inflation, robot_velocity)
        return {
            "x": float(self.x),
            "y": float(self.y),
            "radius": float(radius),
            "base_radius": float(self.radius),
            "inflated_radius": float(self.inflated_radius(inflation, robot_velocity)),
            "vx": float(self.vx),
            "vy": float(self.vy),
            "speed": float(self.speed),
            "motion_model": self.motion_model,
        }

    def to_obstacle(
        self,
        inflation: Optional[InflationConfig] = None,
        use_base_radius: bool = False,
        robot_velocity: float = 0.0,
    ) -> Obstacle:
        radius = self.radius if use_base_radius else self.inflated_radius(inflation, robot_velocity)
        return Obstacle(
            x=float(self.x),
            y=float(self.y),
            radius=float(radius),
            vx=float(self.vx),
            vy=float(self.vy),
            motion_model=self.motion_model,
        )


class DynamicObstacleField:
    """Mutable obstacle field used during a single simulation rollout."""

    def __init__(
        self,
        obstacles: Sequence[DynamicObstacle],
        inflation: Optional[InflationConfig] = None,
        seed: int = 0,
    ):
        self.obstacles = [obs.clone() for obs in obstacles]
        self.inflation = (inflation or InflationConfig()).normalized()
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.robot_position: Optional[np.ndarray] = None
        self.robot_velocity: float = 0.0

    def step(self, dt: float) -> None:
        for obstacle in self.obstacles:
            obstacle.step(dt, self.rng)

    def update_robot_state(self, position: np.ndarray, velocity: float = 0.0) -> None:
        """Update robot position and velocity for sensing range and velocity-adaptive inflation."""
        self.robot_position = position
        self.robot_velocity = velocity

    def _is_within_sensing_range(self, obstacle: DynamicObstacle) -> bool:
        """Check if obstacle is within sensing range of robot."""
        if self.robot_position is None or np.isinf(self.inflation.sensing_range):
            return True
        
        distance = np.sqrt(
            (obstacle.x - self.robot_position[0])**2 + 
            (obstacle.y - self.robot_position[1])**2
        )
        return distance <= self.inflation.sensing_range

    def actual_obstacles(self) -> List[Obstacle]:
        """Return obstacles with base radius for collision detection (all obstacles, no filtering)."""
        return [obs.to_obstacle(use_base_radius=True) for obs in self.obstacles]

    def controller_obstacles(self) -> List[Obstacle]:
        """Return inflated obstacles for MPC constraints (filtered by sensing range)."""
        return [
            obs.to_obstacle(self.inflation, robot_velocity=self.robot_velocity) 
            for obs in self.obstacles 
            if self._is_within_sensing_range(obs)
        ]

    def risk_obstacles(self) -> List[Dict[str, object]]:
        """Return obstacles with metadata for risk assessment (filtered by sensing range)."""
        return [
            obs.to_dict(self.inflation, robot_velocity=self.robot_velocity) 
            for obs in self.obstacles 
            if self._is_within_sensing_range(obs)
        ]

    def snapshot(self, use_base_radius: bool = True) -> List[Dict[str, object]]:
        return [
            obs.to_dict(self.inflation, use_base_radius=use_base_radius, robot_velocity=self.robot_velocity)
            for obs in self.obstacles
        ]


@dataclass
class ObstacleConfig:
    """Obstacle layout configuration, including dynamic motion metadata."""

    obstacles: List[Dict[str, object]]
    seed: int
    name: str = "random"
    bounds: Tuple[float, float, float, float] = DEFAULT_BOUNDS

    def to_obstacle_list(
        self, inflation: Optional[InflationConfig] = None
    ) -> List[Obstacle]:
        return self.create_field(inflation).controller_obstacles()

    def create_field(
        self,
        inflation: Optional[InflationConfig] = None,
        seed: Optional[int] = None,
    ) -> DynamicObstacleField:
        obstacles = [
            DynamicObstacle.from_dict(payload, bounds=self.bounds)
            for payload in self.obstacles
        ]
        return DynamicObstacleField(
            obstacles=obstacles,
            inflation=inflation,
            seed=self.seed if seed is None else seed,
        )


@dataclass
class ScenarioConfig:
    """Complete scenario specification for evaluation."""
    name: str
    trajectory_type: str
    trajectory_params: Dict[str, Any]
    obstacle_config: ObstacleConfig
    inflation_config: InflationConfig
    uncertainty_config: UncertaintyConfig
    duration: float = 20.0
    dt: float = 0.02


class ScenarioGenerator:
    """Base class for scenario generation."""

    def generate(self, n_configs: int, base_seed: int) -> List[ObstacleConfig]:
        raise NotImplementedError


class RandomScenario(ScenarioGenerator):
    """Original randomized scatter with static obstacles."""

    def generate(self, n_configs: int, base_seed: int) -> List[ObstacleConfig]:
        configs: List[ObstacleConfig] = []
        for i in range(n_configs):
            rng = np.random.RandomState(base_seed + i)
            n_obs = rng.randint(2, 6)
            obstacles: List[Dict[str, object]] = []
            attempts = 0
            while len(obstacles) < n_obs and attempts < 120:
                x = rng.uniform(-2.0, 2.0)
                y = rng.uniform(-1.5, 1.5)
                radius = rng.uniform(0.1, 0.28)
                if x**2 + y**2 > 0.65**2:
                    obstacles.append({"x": float(x), "y": float(y), "radius": float(radius)})
                attempts += 1
            configs.append(
                ObstacleConfig(obstacles=obstacles, seed=base_seed + i, name=f"random_{i}")
            )
        return configs


class CorridorScenario(ScenarioGenerator):
    """Narrow corridor formed by two static obstacle walls."""

    def generate(self, n_configs: int, base_seed: int) -> List[ObstacleConfig]:
        configs: List[ObstacleConfig] = []
        for i in range(n_configs):
            rng = np.random.RandomState(base_seed + i)
            obstacles: List[Dict[str, object]] = []
            width = rng.uniform(0.6, 0.9)
            for x in np.linspace(-1.0, 1.0, 5):
                obstacles.append({"x": float(x), "y": float(width / 2 + rng.uniform(-0.05, 0.05)), "radius": 0.15})
                obstacles.append({"x": float(x), "y": float(-width / 2 + rng.uniform(-0.05, 0.05)), "radius": 0.15})
            configs.append(
                ObstacleConfig(obstacles=obstacles, seed=base_seed + i, name=f"corridor_{i}")
            )
        return configs


class BugTrapScenario(ScenarioGenerator):
    """U-shaped obstacle configuration for local-minima stress testing."""

    def generate(self, n_configs: int, base_seed: int) -> List[ObstacleConfig]:
        configs: List[ObstacleConfig] = []
        for i in range(n_configs):
            rng = np.random.RandomState(base_seed + i)
            obstacles: List[Dict[str, object]] = []
            center_x = rng.uniform(0.5, 1.5)
            center_y = rng.uniform(-0.5, 0.5)
            radius = 0.15
            offsets = [
                (0.0, 0.0),
                (0.0, 0.25),
                (0.0, -0.25),
                (-0.25, 0.25),
                (-0.25, -0.25),
            ]
            for dx, dy in offsets:
                obstacles.append({"x": center_x + dx, "y": center_y + dy, "radius": radius})
            configs.append(
                ObstacleConfig(obstacles=obstacles, seed=base_seed + i, name=f"bugtrap_{i}")
            )
        return configs


class DenseClutterScenario(ScenarioGenerator):
    """High-density static clutter that forces constant avoidance."""

    def generate(self, n_configs: int, base_seed: int) -> List[ObstacleConfig]:
        configs: List[ObstacleConfig] = []
        for i in range(n_configs):
            rng = np.random.RandomState(base_seed + i)
            obstacles: List[Dict[str, object]] = []
            n_obs = rng.randint(8, 15)
            attempts = 0
            while len(obstacles) < n_obs and attempts < 200:
                x = rng.uniform(-1.5, 1.5)
                y = rng.uniform(-1.0, 1.0)
                if x**2 + y**2 > 0.75**2:
                    obstacles.append(
                        {"x": float(x), "y": float(y), "radius": float(rng.uniform(0.1, 0.2))}
                    )
                attempts += 1
            configs.append(
                ObstacleConfig(obstacles=obstacles, seed=base_seed + i, name=f"dense_{i}")
            )
        return configs


class MovingObstacleScenario(ScenarioGenerator):
    """Bounded linear motion obstacles for dynamic avoidance tests."""

    def generate(self, n_configs: int, base_seed: int) -> List[ObstacleConfig]:
        configs: List[ObstacleConfig] = []
        bounds = DEFAULT_BOUNDS
        for i in range(n_configs):
            rng = np.random.RandomState(base_seed + i)
            obstacles = [
                {
                    "x": float(rng.uniform(0.8, 1.6)),
                    "y": float(rng.uniform(-0.8, 0.8)),
                    "radius": 0.18,
                    "vx": float(rng.choice([-0.35, 0.35])),
                    "vy": 0.0,
                    "motion_model": "linear",
                    "bounds": bounds,
                },
                {
                    "x": float(rng.uniform(-1.2, -0.4)),
                    "y": float(rng.uniform(-0.8, 0.8)),
                    "radius": 0.16,
                    "vx": 0.0,
                    "vy": float(rng.choice([-0.25, 0.25])),
                    "motion_model": "linear",
                    "bounds": bounds,
                },
                {
                    "x": float(rng.uniform(1.0, 1.8)),
                    "y": float(rng.uniform(-1.0, -0.2)),
                    "radius": 0.15,
                    "vx": float(rng.choice([-0.2, 0.2])),
                    "vy": float(rng.choice([-0.15, 0.15])),
                    "motion_model": "linear",
                    "bounds": bounds,
                },
            ]
            configs.append(
                ObstacleConfig(
                    obstacles=obstacles,
                    seed=base_seed + i,
                    name=f"moving_{i}",
                    bounds=bounds,
                )
            )
        return configs


class RandomWalkScenario(ScenarioGenerator):
    """Bounded random-walk obstacles with velocity diffusion."""

    def generate(self, n_configs: int, base_seed: int) -> List[ObstacleConfig]:
        configs: List[ObstacleConfig] = []
        bounds = DEFAULT_BOUNDS
        for i in range(n_configs):
            rng = np.random.RandomState(base_seed + i)
            obstacles: List[Dict[str, object]] = []
            for _ in range(4):
                obstacles.append(
                    {
                        "x": float(rng.uniform(-1.6, 1.6)),
                        "y": float(rng.uniform(-1.0, 1.0)),
                        "radius": float(rng.uniform(0.12, 0.2)),
                        "vx": float(rng.uniform(-0.15, 0.15)),
                        "vy": float(rng.uniform(-0.15, 0.15)),
                        "motion_model": "random_walk",
                        "random_walk_std": 0.2,
                        "max_speed": 0.45,
                        "bounds": bounds,
                    }
                )
            configs.append(
                ObstacleConfig(
                    obstacles=obstacles,
                    seed=base_seed + i,
                    name=f"random_walk_{i}",
                    bounds=bounds,
                )
            )
        return configs


def get_generator(name: str) -> ScenarioGenerator:
    if name == "corridor":
        return CorridorScenario()
    if name == "bugtrap":
        return BugTrapScenario()
    if name == "dense":
        return DenseClutterScenario()
    if name == "moving":
        return MovingObstacleScenario()
    if name == "random_walk":
        return RandomWalkScenario()
    return RandomScenario()


def build_demo_config(name: str, seed: int = 42) -> ObstacleConfig:
    """Small deterministic scenario presets used by the standalone runner."""
    bounds = DEFAULT_BOUNDS
    presets: Dict[str, List[Dict[str, object]]] = {
        "default": [
            {"x": 1.0, "y": 0.5, "radius": 0.2},
            {"x": -0.5, "y": -1.0, "radius": 0.25},
            {"x": 1.5, "y": -0.3, "radius": 0.15},
        ],
        "sparse": [
            {"x": 1.5, "y": 0.8, "radius": 0.2},
        ],
        "dense": [
            {"x": 1.0, "y": 0.5, "radius": 0.2},
            {"x": -0.5, "y": -1.0, "radius": 0.25},
            {"x": 1.5, "y": -0.3, "radius": 0.15},
            {"x": -1.5, "y": 0.5, "radius": 0.2},
            {"x": 0.0, "y": 0.8, "radius": 0.15},
        ],
        "corridor": [
            {"x": 1.0, "y": 0.3, "radius": 0.15},
            {"x": 1.0, "y": 0.7, "radius": 0.15},
            {"x": -0.8, "y": -0.7, "radius": 0.15},
            {"x": -0.3, "y": -1.2, "radius": 0.15},
        ],
        "moving": [
            {"x": 1.0, "y": 0.4, "radius": 0.18, "vx": 0.32, "vy": 0.0, "motion_model": "linear", "bounds": bounds},
            {"x": -0.7, "y": -0.8, "radius": 0.16, "vx": 0.0, "vy": 0.24, "motion_model": "linear", "bounds": bounds},
            {"x": 1.6, "y": -0.5, "radius": 0.14, "vx": -0.2, "vy": 0.12, "motion_model": "linear", "bounds": bounds},
        ],
        "random_walk": [
            {"x": 1.0, "y": 0.2, "radius": 0.18, "vx": 0.08, "vy": -0.04, "motion_model": "random_walk", "random_walk_std": 0.18, "max_speed": 0.4, "bounds": bounds},
            {"x": -0.8, "y": -0.9, "radius": 0.16, "vx": -0.05, "vy": 0.10, "motion_model": "random_walk", "random_walk_std": 0.18, "max_speed": 0.4, "bounds": bounds},
            {"x": 1.7, "y": -0.2, "radius": 0.14, "vx": 0.06, "vy": 0.06, "motion_model": "random_walk", "random_walk_std": 0.18, "max_speed": 0.4, "bounds": bounds},
        ],
    }

    if name not in presets:
        raise ValueError(f"Unknown demo scenario '{name}'.")

    return ObstacleConfig(obstacles=presets[name], seed=seed, name=name, bounds=bounds)


def get_baseline_static_scenario(seed: int = 42) -> ScenarioConfig:
    """
    Baseline static scenario: Lissajous trajectory with sparse static obstacles.
    
    Purpose: Establish baseline performance without uncertainty or dynamic obstacles.
    
    Expected behavior:
    - High checkpoint completion rate (>95%)
    - Low tracking error (<0.1m mean)
    - No collisions
    - Fast MPC solve times (<5ms)
    
    Args:
        seed: Random seed for obstacle generation
        
    Returns:
        ScenarioConfig for baseline_static scenario
    """
    # Create sparse static obstacles (3 obstacles)
    obstacles = [
        {"x": 1.5, "y": 0.8, "radius": 0.2},
        {"x": -0.5, "y": -1.0, "radius": 0.25},
        {"x": 1.0, "y": -0.5, "radius": 0.15},
    ]
    
    obstacle_config = ObstacleConfig(
        obstacles=obstacles,
        seed=seed,
        name="sparse_static",
        bounds=DEFAULT_BOUNDS
    )
    
    # Inflation config with baseline parameters
    inflation_config = InflationConfig(
        safety_factor=1.0,
        sensing_factor=0.05,
        motion_lookahead=0.5,
        velocity_scaling_factor=0.0,
        sensing_range=float('inf')
    )
    
    # No uncertainty (all noise/mismatch = 0)
    uncertainty_config = UncertaintyConfig(
        process_noise_position_std=0.0,
        process_noise_heading_std=0.0,
        sensor_noise_position_std=0.0,
        sensor_noise_heading_std=0.0,
        velocity_mismatch_factor=1.0,
        angular_mismatch_factor=1.0,
        control_delay_steps=0
    )
    
    # Lissajous trajectory parameters
    trajectory_params = {
        "amplitude": 2.0,
        "frequency": 0.5,
    }
    
    return ScenarioConfig(
        name="baseline_static",
        trajectory_type="lissajous",
        trajectory_params=trajectory_params,
        obstacle_config=obstacle_config,
        inflation_config=inflation_config,
        uncertainty_config=uncertainty_config,
        duration=20.0,
        dt=0.02
    )


def get_urban_dynamic_scenario(seed: int = 42) -> ScenarioConfig:
    """
    Urban dynamic scenario: Urban_path trajectory with moving obstacles.
    
    Purpose: Test navigation through city-block-like paths with dynamic obstacles.
    
    Expected behavior:
    - Moderate checkpoint completion rate (>80%)
    - Moderate tracking error (<0.2m mean)
    - Rare collisions (<2)
    
    Args:
        seed: Random seed for obstacle generation
        
    Returns:
        ScenarioConfig for urban_dynamic scenario
    """
    bounds = DEFAULT_BOUNDS
    
    # Create 5 moving obstacles with linear motion
    # Speed range: 0.2-0.4 m/s
    rng = np.random.RandomState(seed)
    obstacles = []
    
    for i in range(5):
        # Random position within bounds
        x = rng.uniform(-1.5, 1.5)
        y = rng.uniform(-1.0, 1.0)
        radius = rng.uniform(0.15, 0.22)
        
        # Random velocity with speed in range [0.2, 0.4]
        speed = rng.uniform(0.2, 0.4)
        angle = rng.uniform(0, 2 * np.pi)
        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)
        
        obstacles.append({
            "x": float(x),
            "y": float(y),
            "radius": float(radius),
            "vx": float(vx),
            "vy": float(vy),
            "motion_model": "linear",
            "bounds": bounds,
        })
    
    obstacle_config = ObstacleConfig(
        obstacles=obstacles,
        seed=seed,
        name="urban_dynamic",
        bounds=bounds
    )
    
    # Inflation config with moderate safety margins
    inflation_config = InflationConfig(
        safety_factor=1.2,
        sensing_factor=0.08,
        motion_lookahead=0.8,
        velocity_scaling_factor=0.0,
        sensing_range=float('inf')
    )
    
    # Moderate uncertainty
    uncertainty_config = UncertaintyConfig(
        process_noise_position_std=0.01,
        process_noise_heading_std=0.0,
        sensor_noise_position_std=0.02,
        sensor_noise_heading_std=0.0,
        velocity_mismatch_factor=0.95,
        angular_mismatch_factor=1.0,
        control_delay_steps=1
    )
    
    # Urban_path trajectory parameters
    trajectory_params = {
        "block_size": 2.0,
        "num_blocks": 4,
    }
    
    return ScenarioConfig(
        name="urban_dynamic",
        trajectory_type="urban_path",
        trajectory_params=trajectory_params,
        obstacle_config=obstacle_config,
        inflation_config=inflation_config,
        uncertainty_config=uncertainty_config,
        duration=25.0,
        dt=0.02
    )


def get_stochastic_navigation_scenario(seed: int = 42) -> ScenarioConfig:
    """
    Stochastic navigation scenario: Sinusoidal trajectory with random_walk obstacles.
    
    Purpose: Stress-test robustness to high process and sensor noise.
    
    Expected behavior:
    - Lower checkpoint completion rate (>70%)
    - Higher tracking error (<0.3m mean)
    - Occasional collisions (<5)
    - Hybrid blending adapts to high uncertainty
    
    Args:
        seed: Random seed for obstacle generation
        
    Returns:
        ScenarioConfig for stochastic_navigation scenario
    """
    bounds = DEFAULT_BOUNDS
    
    # Create 4 random_walk obstacles
    rng = np.random.RandomState(seed)
    obstacles = []
    
    for i in range(4):
        # Random position within bounds
        x = rng.uniform(-1.6, 1.6)
        y = rng.uniform(-1.0, 1.0)
        radius = rng.uniform(0.12, 0.2)
        
        # Initial velocity for random walk
        vx = rng.uniform(-0.15, 0.15)
        vy = rng.uniform(-0.15, 0.15)
        
        obstacles.append({
            "x": float(x),
            "y": float(y),
            "radius": float(radius),
            "vx": float(vx),
            "vy": float(vy),
            "motion_model": "random_walk",
            "random_walk_std": 0.2,
            "max_speed": 0.45,
            "bounds": bounds,
        })
    
    obstacle_config = ObstacleConfig(
        obstacles=obstacles,
        seed=seed,
        name="stochastic_navigation",
        bounds=bounds
    )
    
    # Inflation config with high safety margins and velocity-adaptive inflation
    inflation_config = InflationConfig(
        safety_factor=1.5,
        sensing_factor=0.1,
        motion_lookahead=1.0,
        velocity_scaling_factor=0.15,
        sensing_range=float('inf')
    )
    
    # High uncertainty: high process/sensor noise, model mismatch, 2-step delay
    uncertainty_config = UncertaintyConfig(
        process_noise_position_std=0.03,
        process_noise_heading_std=0.05,
        sensor_noise_position_std=0.04,
        sensor_noise_heading_std=0.06,
        velocity_mismatch_factor=0.90,
        angular_mismatch_factor=1.0,
        control_delay_steps=2
    )
    
    # Sinusoidal trajectory parameters
    trajectory_params = {
        "amplitude": 1.5,
        "frequency": 0.8,
    }
    
    return ScenarioConfig(
        name="stochastic_navigation",
        trajectory_type="sinusoidal",
        trajectory_params=trajectory_params,
        obstacle_config=obstacle_config,
        inflation_config=inflation_config,
        uncertainty_config=uncertainty_config,
        duration=20.0,
        dt=0.02
    )


def get_oscillatory_tracking_scenario(seed: int = 42) -> ScenarioConfig:
    """
    Oscillatory tracking scenario: Lissajous trajectory with high frequency and tight checkpoints.
    
    Purpose: Test controller responsiveness to high-frequency trajectory with dense checkpoints.
    
    Expected behavior:
    - High checkpoint density (>150 checkpoints)
    - Moderate completion rate (>75%)
    - Low overshoot (<0.05m mean)
    - Tests controller responsiveness
    
    Args:
        seed: Random seed for obstacle generation
        
    Returns:
        ScenarioConfig for oscillatory_tracking scenario
    """
    bounds = DEFAULT_BOUNDS
    
    # Create 8 dense static obstacles
    rng = np.random.RandomState(seed)
    obstacles = []
    
    # Generate 8 static obstacles with deterministic placement
    # Avoid center region to allow trajectory execution
    attempts = 0
    while len(obstacles) < 8 and attempts < 200:
        x = rng.uniform(-1.5, 1.5)
        y = rng.uniform(-1.0, 1.0)
        radius = rng.uniform(0.1, 0.2)
        
        # Avoid center region (radius > 0.75m from origin)
        if x**2 + y**2 > 0.75**2:
            obstacles.append({
                "x": float(x),
                "y": float(y),
                "radius": float(radius),
            })
        attempts += 1
    
    obstacle_config = ObstacleConfig(
        obstacles=obstacles,
        seed=seed,
        name="dense_static",
        bounds=bounds
    )
    
    # Inflation config with baseline parameters
    inflation_config = InflationConfig(
        safety_factor=1.0,
        sensing_factor=0.05,
        motion_lookahead=0.5,
        velocity_scaling_factor=0.0,
        sensing_range=float('inf')
    )
    
    # Low uncertainty: minimal process/sensor noise
    uncertainty_config = UncertaintyConfig(
        process_noise_position_std=0.005,
        process_noise_heading_std=0.0,
        sensor_noise_position_std=0.01,
        sensor_noise_heading_std=0.0,
        velocity_mismatch_factor=1.0,
        angular_mismatch_factor=1.0,
        control_delay_steps=0
    )
    
    # Lissajous trajectory with high-frequency parameters
    trajectory_params = {
        "amplitude": 1.5,
        "a": 2.0,
        "b": 3.0,
        "c": 2.5,
    }
    
    return ScenarioConfig(
        name="oscillatory_tracking",
        trajectory_type="lissajous",
        trajectory_params=trajectory_params,
        obstacle_config=obstacle_config,
        inflation_config=inflation_config,
        uncertainty_config=uncertainty_config,
        duration=20.0,
        dt=0.02
    )


def get_vehicle_realistic_scenario(seed: int = 42) -> ScenarioConfig:
    """
    Vehicle realistic scenario: Clothoid trajectory through corridor with realistic vehicle constraints.
    
    Purpose: Simulate realistic vehicle dynamics with model mismatch, control latency, and actuator lag.
    
    Expected behavior:
    - Realistic vehicle constraints
    - Moderate completion rate (>75%)
    - Adaptive MPC compensates for mismatch
    - Latency handled by predictive control
    
    Args:
        seed: Random seed for obstacle generation
        
    Returns:
        ScenarioConfig for vehicle_realistic scenario
    """
    bounds = DEFAULT_BOUNDS
    
    # Create corridor obstacles (width=1.2m, length=10.0m)
    # Corridor formed by two parallel walls of obstacles
    obstacles = []
    corridor_width = 1.2
    corridor_length = 10.0
    
    # Place obstacles along the corridor walls
    # Use 10 obstacles per side for a 10m corridor
    num_obstacles_per_side = 10
    for i in range(num_obstacles_per_side):
        x = -corridor_length / 2 + (i + 0.5) * (corridor_length / num_obstacles_per_side)
        
        # Top wall
        obstacles.append({
            "x": float(x),
            "y": float(corridor_width / 2),
            "radius": 0.15,
        })
        
        # Bottom wall
        obstacles.append({
            "x": float(x),
            "y": float(-corridor_width / 2),
            "radius": 0.15,
        })
    
    obstacle_config = ObstacleConfig(
        obstacles=obstacles,
        seed=seed,
        name="corridor",
        bounds=bounds
    )
    
    # Inflation config with higher safety margins and velocity-adaptive scaling
    inflation_config = InflationConfig(
        safety_factor=1.3,
        sensing_factor=0.1,
        motion_lookahead=0.6,
        velocity_scaling_factor=0.2,
        sensing_range=float('inf')
    )
    
    # High uncertainty: high model mismatch, high latency, actuator lag
    # Note: actuator_tau_v and actuator_tau_omega are mentioned in the design
    # but UncertaintyConfig doesn't have these fields yet. They may be handled
    # elsewhere in the system or added later.
    uncertainty_config = UncertaintyConfig(
        process_noise_position_std=0.02,
        process_noise_heading_std=0.0,
        sensor_noise_position_std=0.03,
        sensor_noise_heading_std=0.0,
        velocity_mismatch_factor=0.85,
        angular_mismatch_factor=0.90,
        control_delay_steps=3
    )
    
    # Clothoid trajectory parameters
    trajectory_params = {
        "kappa0": 0.0,
        "k_rate": 0.5,
    }
    
    return ScenarioConfig(
        name="vehicle_realistic",
        trajectory_type="clothoid",
        trajectory_params=trajectory_params,
        obstacle_config=obstacle_config,
        inflation_config=inflation_config,
        uncertainty_config=uncertainty_config,
        duration=25.0,
        dt=0.02
    )

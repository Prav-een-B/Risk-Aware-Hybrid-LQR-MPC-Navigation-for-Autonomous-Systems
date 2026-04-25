"""
Evaluation Scenarios (Phase 5-C)
==================================

Scenario generators implementing the four evaluation classes:
    - Class A: Static obstacles with density sweep
    - Class B: Dynamic obstacles (linear, sinusoidal, random walk)
    - Class C: Noise stress test
    - Class D: FRP vs CN head-to-head

Also includes:
    - P1-D: DynamicObstacle class with motion profiles
    - P2-B: Dynamic obstacle scenario generators

Reference:
    Risk-Aware Hybrid LQR-MPC Navigation, Phase 5 - Evaluation Overhaul
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from hybrid_controller.controllers.mpc_controller import Obstacle


# ── Existing base classes (kept for backward compatibility) ──────────

@dataclass
class ObstacleConfig:
    """Obstacle layout configuration."""
    obstacles: List[Dict[str, float]]
    seed: int
    name: str = "random"
    
    def to_obstacle_list(self) -> List[Obstacle]:
        return [Obstacle(x=o['x'], y=o['y'], radius=o['radius']) for o in self.obstacles]


class ScenarioGenerator:
    """Base class for scenario generation."""
    def generate(self, n_configs: int, base_seed: int) -> List[ObstacleConfig]:
        raise NotImplementedError


# ── P1-D: Dynamic obstacle with motion profiles ─────────────────────

@dataclass
class DynamicObstacle:
    """
    Dynamic obstacle with time-varying position (P1-D).
    
    Supports motion profiles:
        - 'static': No motion (fallback)
        - 'linear': Constant velocity in a direction
        - 'sinusoidal': Oscillates about a center
        - 'random_walk': Brownian motion with velocity clamping
    
    Attributes:
        x0, y0: Initial position
        radius: Obstacle radius
        profile: Motion profile type
        speed: Movement speed (m/s)
        direction: Direction angle for linear motion (radians)
        amplitude: Oscillation amplitude for sinusoidal (m)
        frequency: Oscillation frequency for sinusoidal (Hz)
    """
    x0: float
    y0: float
    radius: float = 0.2
    profile: str = 'static'
    speed: float = 0.3
    direction: float = 0.0       # For 'linear'
    amplitude: float = 0.5       # For 'sinusoidal'
    frequency: float = 0.2       # For 'sinusoidal' (Hz)
    _rng: Optional[np.random.RandomState] = field(default=None, repr=False)
    
    # Internal state for random walk
    _vx: float = field(default=0.0, init=False, repr=False)
    _vy: float = field(default=0.0, init=False, repr=False)
    
    def position_at(self, t: float) -> Tuple[float, float]:
        """Get obstacle position at time t."""
        if self.profile == 'static':
            return self.x0, self.y0
        
        elif self.profile == 'linear':
            dx = self.speed * np.cos(self.direction) * t
            dy = self.speed * np.sin(self.direction) * t
            return self.x0 + dx, self.y0 + dy
        
        elif self.profile == 'sinusoidal':
            dx = self.amplitude * np.sin(2 * np.pi * self.frequency * t)
            dy = self.amplitude * np.cos(2 * np.pi * self.frequency * t) * 0.5
            return self.x0 + dx, self.y0 + dy
        
        elif self.profile == 'random_walk':
            # Not time-based; use step() method instead
            return self.x0 + self._vx * t, self.y0 + self._vy * t
        
        return self.x0, self.y0
    
    def to_dict_at(self, t: float) -> Dict[str, float]:
        """Get obstacle dict at time t."""
        x, y = self.position_at(t)
        return {'x': float(x), 'y': float(y), 'radius': float(self.radius)}
    
    def to_obstacle_at(self, t: float) -> Obstacle:
        """Get Obstacle object at time t."""
        x, y = self.position_at(t)
        return Obstacle(x=float(x), y=float(y), radius=float(self.radius))


@dataclass
class DynamicObstacleConfig:
    """Configuration for a scene with dynamic obstacles."""
    dynamic_obstacles: List[DynamicObstacle]
    static_obstacles: List[Dict[str, float]] = field(default_factory=list)
    seed: int = 0
    name: str = "dynamic"
    
    def get_obstacles_at(self, t: float) -> List[Dict[str, float]]:
        """Get all obstacles as dicts at time t."""
        obs = [d.to_dict_at(t) for d in self.dynamic_obstacles]
        obs.extend(self.static_obstacles)
        return obs
    
    def get_obstacle_objects_at(self, t: float) -> List[Obstacle]:
        """Get all obstacles as Obstacle objects at time t."""
        obs = [d.to_obstacle_at(t) for d in self.dynamic_obstacles]
        obs.extend([Obstacle(x=o['x'], y=o['y'], radius=o['radius']) 
                     for o in self.static_obstacles])
        return obs


# ── Original scenario generators ─────────────────────────────────────

class RandomScenario(ScenarioGenerator):
    """Original randomized scatter."""
    def generate(self, n_configs: int, base_seed: int) -> List[ObstacleConfig]:
        configs = []
        for i in range(n_configs):
            rng = np.random.RandomState(base_seed + i)
            n_obs = rng.randint(2, 6)
            obstacles = []
            attempts = 0
            while len(obstacles) < n_obs and attempts < 100:
                x = rng.uniform(-2.0, 2.0)
                y = rng.uniform(-1.5, 1.5)
                r = rng.uniform(0.1, 0.3)
                
                # Simple check against origin
                if x**2 + y**2 > 0.6**2: 
                     obstacles.append({'x': float(x), 'y': float(y), 'radius': float(r)})
                attempts += 1
            configs.append(ObstacleConfig(obstacles=obstacles, seed=base_seed+i, name=f"random_{i}"))
        return configs

class CorridorScenario(ScenarioGenerator):
    """
    Narrow corridor formed by two walls of obstacles.
    Tests simultaneous constraint handling from both sides.
    """
    def generate(self, n_configs: int, base_seed: int) -> List[ObstacleConfig]:
        configs = []
        for i in range(n_configs):
            rng = np.random.RandomState(base_seed + i)
            obstacles = []
            
            # Corridor width varies slightly
            width = rng.uniform(0.6, 0.9)  
            length = 3.0
            
            # Left wall
            for x in np.linspace(-1.0, 1.0, 5):
                y = width/2 + rng.uniform(-0.05, 0.05)
                obstacles.append({'x': float(x), 'y': float(y), 'radius': 0.15})
                
            # Right wall
            for x in np.linspace(-1.0, 1.0, 5):
                y = -width/2 + rng.uniform(-0.05, 0.05)
                obstacles.append({'x': float(x), 'y': float(y), 'radius': 0.15})
                
            configs.append(ObstacleConfig(obstacles=obstacles, seed=base_seed+i, name=f"corridor_{i}"))
        return configs

class BugTrapScenario(ScenarioGenerator):
    """
    U-shaped obstacle configuration.
    Tests local minima and ability to navigate around complex geometry.
    Note: Standard MPC often fails this without a global planner or very long horizon.
    """
    def generate(self, n_configs: int, base_seed: int) -> List[ObstacleConfig]:
        configs = []
        for i in range(n_configs):
            rng = np.random.RandomState(base_seed + i)
            obstacles = []
            
            # U-shape parameters
            center_x = rng.uniform(0.5, 1.5)
            center_y = rng.uniform(-0.5, 0.5)
            r = 0.15
            
            # Back wall
            obstacles.append({'x': center_x, 'y': center_y, 'radius': r})
            obstacles.append({'x': center_x, 'y': center_y + 0.25, 'radius': r})
            obstacles.append({'x': center_x, 'y': center_y - 0.25, 'radius': r})
            
            # Side walls (forming the 'hook')
            obstacles.append({'x': center_x - 0.25, 'y': center_y + 0.25, 'radius': r})
            obstacles.append({'x': center_x - 0.25, 'y': center_y - 0.25, 'radius': r})
            
            configs.append(ObstacleConfig(obstacles=obstacles, seed=base_seed+i, name=f"bugtrap_{i}"))
        return configs

class DenseClutterScenario(ScenarioGenerator):
    """
    High-density field forcing almost continuous avoidance.
    Stress tests solver speed and feasibility logic.
    """
    def generate(self, n_configs: int, base_seed: int) -> List[ObstacleConfig]:
        configs = []
        for i in range(n_configs):
            rng = np.random.RandomState(base_seed + i)
            obstacles = []
            
            # Grid-like high density
            n_obs = rng.randint(8, 15)
            for _ in range(n_obs):
                x = rng.uniform(-1.5, 1.5)
                y = rng.uniform(-1.0, 1.0)
                
                # Clear start/goal
                if x**2 + y**2 > 0.8**2:
                    obstacles.append({'x': float(x), 'y': float(y), 'radius': rng.uniform(0.1, 0.2)})
            
            configs.append(ObstacleConfig(obstacles=obstacles, seed=base_seed+i, name=f"dense_{i}"))
        return configs


# ── P5-C: Class A-D Scenario Suite ──────────────────────────────────

class DensitySweepScenario(ScenarioGenerator):
    """
    Class A: Static obstacles with density sweep.
    
    Generates scenarios with obstacle counts of [2, 4, 8, 12, 20]
    for each configuration, with obstacles placed on a 10m x 10m arena.
    """
    
    DENSITY_COUNTS = [2, 4, 8, 12, 20]
    
    def __init__(self, obstacle_count: int = 8, arena_size: float = 5.0):
        """
        Args:
            obstacle_count: Number of obstacles (from DENSITY_COUNTS)
            arena_size: Half-width of the arena (meters)
        """
        self.obstacle_count = obstacle_count
        self.arena_size = arena_size
    
    def generate(self, n_configs: int, base_seed: int) -> List[ObstacleConfig]:
        configs = []
        for i in range(n_configs):
            rng = np.random.RandomState(base_seed + i)
            obstacles = []
            
            attempts = 0
            while len(obstacles) < self.obstacle_count and attempts < 500:
                x = rng.uniform(-self.arena_size, self.arena_size)
                y = rng.uniform(-self.arena_size, self.arena_size)
                r = rng.uniform(0.1, 0.25)
                
                # Clear zone around origin (start/goal)
                if x**2 + y**2 > 1.0**2:
                    # Check minimum spacing between obstacles
                    too_close = False
                    for obs in obstacles:
                        dist = np.sqrt((x - obs['x'])**2 + (y - obs['y'])**2)
                        if dist < r + obs['radius'] + 0.3:
                            too_close = True
                            break
                    if not too_close:
                        obstacles.append({'x': float(x), 'y': float(y), 'radius': float(r)})
                attempts += 1
            
            configs.append(ObstacleConfig(
                obstacles=obstacles, seed=base_seed+i,
                name=f"density_{self.obstacle_count}_{i}"
            ))
        return configs


class DynamicScenario:
    """
    Class B: Dynamic obstacles (P2-B).
    
    Generates dynamic obstacle configurations with specified motion profiles.
    """
    
    def __init__(self, motion_type: str = 'linear', velocity: float = 0.3,
                 n_dynamic: int = 3, n_static: int = 2):
        """
        Args:
            motion_type: 'linear', 'sinusoidal', or 'random_walk'
            velocity: Speed of moving obstacles (m/s)
            n_dynamic: Number of dynamic obstacles
            n_static: Number of static background obstacles
        """
        self.motion_type = motion_type
        self.velocity = velocity
        self.n_dynamic = n_dynamic
        self.n_static = n_static
    
    def generate(self, n_configs: int, base_seed: int) -> List[DynamicObstacleConfig]:
        configs = []
        for i in range(n_configs):
            rng = np.random.RandomState(base_seed + i)
            
            # Dynamic obstacles
            dynamic_obs = []
            for _ in range(self.n_dynamic):
                x0 = rng.uniform(-2.0, 2.0)
                y0 = rng.uniform(-1.5, 1.5)
                
                # Ensure not starting at origin
                while x0**2 + y0**2 < 0.8**2:
                    x0 = rng.uniform(-2.0, 2.0)
                    y0 = rng.uniform(-1.5, 1.5)
                
                dynamic_obs.append(DynamicObstacle(
                    x0=float(x0), y0=float(y0),
                    radius=rng.uniform(0.1, 0.25),
                    profile=self.motion_type,
                    speed=self.velocity,
                    direction=rng.uniform(0, 2 * np.pi),
                    amplitude=rng.uniform(0.3, 0.8),
                    frequency=rng.uniform(0.1, 0.3),
                ))
            
            # Static background obstacles
            static_obs = []
            for _ in range(self.n_static):
                x = rng.uniform(-2.0, 2.0)
                y = rng.uniform(-1.5, 1.5)
                if x**2 + y**2 > 0.6**2:
                    static_obs.append({'x': float(x), 'y': float(y),
                                       'radius': float(rng.uniform(0.1, 0.2))})
            
            configs.append(DynamicObstacleConfig(
                dynamic_obstacles=dynamic_obs,
                static_obstacles=static_obs,
                seed=base_seed + i,
                name=f"dynamic_{self.motion_type}_{i}"
            ))
        return configs


class NoiseStressScenario(ScenarioGenerator):
    """
    Class C: Noise stress test.
    
    Returns a fixed 8-obstacle layout; the caller varies noise parameters.
    """
    
    NOISE_LEVELS = {
        'position': [0.0, 0.05, 0.1, 0.2],
        'heading': [0.0, 0.01, 0.05],
    }
    
    def __init__(self, sigma_p: float = 0.05, sigma_theta: float = 0.01):
        self.sigma_p = sigma_p
        self.sigma_theta = sigma_theta
    
    def generate(self, n_configs: int, base_seed: int) -> List[ObstacleConfig]:
        """Generate configs with fixed obstacle layout but varied seeds."""
        configs = []
        # Fixed 8-obstacle layout
        fixed_obstacles = [
            {'x': 1.0, 'y': 0.5, 'radius': 0.2},
            {'x': -0.5, 'y': -1.0, 'radius': 0.2},
            {'x': 1.5, 'y': -0.3, 'radius': 0.15},
            {'x': -1.5, 'y': 0.5, 'radius': 0.2},
            {'x': 0.5, 'y': -0.8, 'radius': 0.15},
            {'x': -1.0, 'y': 0.8, 'radius': 0.2},
            {'x': 0.0, 'y': 1.2, 'radius': 0.15},
            {'x': 1.2, 'y': 1.0, 'radius': 0.2},
        ]
        
        for i in range(n_configs):
            configs.append(ObstacleConfig(
                obstacles=fixed_obstacles.copy(),
                seed=base_seed + i,
                name=f"noise_sp{self.sigma_p}_st{self.sigma_theta}_{i}"
            ))
        return configs


class FRPvsCNScenario(ScenarioGenerator):
    """
    Class D: FRP vs CN head-to-head.
    
    Uses 8-obstacle random scatter with sigma_p=0.05 noise.
    The caller runs both FRP and CN modes on each config.
    """
    
    def generate(self, n_configs: int, base_seed: int) -> List[ObstacleConfig]:
        """Generate 8-obstacle random scatters for FRP vs CN comparison."""
        density_gen = DensitySweepScenario(obstacle_count=8, arena_size=3.0)
        return density_gen.generate(n_configs, base_seed)


# ── Factory ──────────────────────────────────────────────────────────

def get_generator(name: str) -> ScenarioGenerator:
    if name == 'corridor': return CorridorScenario()
    if name == 'bugtrap': return BugTrapScenario()
    if name == 'dense': return DenseClutterScenario()
    if name == 'density_2': return DensitySweepScenario(obstacle_count=2)
    if name == 'density_4': return DensitySweepScenario(obstacle_count=4)
    if name == 'density_8': return DensitySweepScenario(obstacle_count=8)
    if name == 'density_12': return DensitySweepScenario(obstacle_count=12)
    if name == 'density_20': return DensitySweepScenario(obstacle_count=20)
    if name == 'noise_stress': return NoiseStressScenario()
    if name == 'frp_vs_cn': return FRPvsCNScenario()
    return RandomScenario()

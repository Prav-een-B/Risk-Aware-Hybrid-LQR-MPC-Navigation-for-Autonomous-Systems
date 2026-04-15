"""Checkpoint generation with curvature-adaptive and obstacle-aware spacing.

This module provides functionality for generating checkpoints from trajectories
with spacing that adapts based on local curvature and nearby obstacle density.
Denser checkpoints are placed in high-curvature regions and near obstacles for
better tracking accuracy and safer navigation.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class Checkpoint:
    """Waypoint in checkpoint-based navigation.
    
    Attributes:
        x: Position x coordinate (m)
        y: Position y coordinate (m)
        theta: Heading angle (rad)
        curvature: Local curvature at this point (1/m)
        index: Index in the original trajectory array
        reached: Whether this checkpoint has been reached
        time_reached: Simulation time when checkpoint was reached (s)
        overshoot: Distance traveled beyond checkpoint before switching (m)
        min_distance: Minimum distance achieved to this checkpoint (m)
    """
    x: float
    y: float
    theta: float
    curvature: float
    index: int
    reached: bool = False
    time_reached: float = 0.0
    overshoot: float = 0.0
    min_distance: float = float('inf')
    
    def distance_to(self, position: np.ndarray) -> float:
        """Compute Euclidean distance to a position.
        
        Args:
            position: 2D position array [x, y]
            
        Returns:
            Euclidean distance (m)
        """
        return np.sqrt((position[0] - self.x)**2 + (position[1] - self.y)**2)


def generate_checkpoints(
    trajectory: np.ndarray,
    curvature: np.ndarray,
    curvature_high: float = 2.0,
    curvature_low: float = 0.5,
    min_spacing: float = 0.1,
    max_spacing: float = 1.0
) -> List[Checkpoint]:
    """Generate checkpoints with adaptive spacing based on curvature.
    
    The spacing between checkpoints adapts to local curvature:
    - High curvature (>= curvature_high): min_spacing (dense checkpoints)
    - Low curvature (<= curvature_low): max_spacing (sparse checkpoints)
    - Intermediate curvature: linear interpolation between min and max spacing
    
    Spacing formula:
        spacing(kappa) = max_spacing - alpha * (max_spacing - min_spacing)
        where alpha = (kappa - curvature_low) / (curvature_high - curvature_low)
        clamped to [min_spacing, max_spacing]
    
    Args:
        trajectory: (N, 4) array with columns [t, x, y, theta]
        curvature: (N,) array of curvature values (1/m)
        curvature_high: High curvature threshold (1/m)
        curvature_low: Low curvature threshold (1/m)
        min_spacing: Minimum spacing between checkpoints (m)
        max_spacing: Maximum spacing between checkpoints (m)
        
    Returns:
        List of Checkpoint objects with adaptive spacing
        
    Notes:
        - At least one checkpoint is always generated (at trajectory start)
        - Spacing is computed based on cumulative distance traveled
        - High curvature regions get denser checkpoints for better tracking
    """
    if len(trajectory) == 0:
        return []
    
    checkpoints = []
    current_distance = 0.0
    last_checkpoint_idx = 0
    
    # Always add first checkpoint
    checkpoints.append(Checkpoint(
        x=trajectory[0, 1],
        y=trajectory[0, 2],
        theta=trajectory[0, 3],
        curvature=curvature[0],
        index=0
    ))
    
    for i in range(1, len(trajectory)):
        # Compute distance traveled since last checkpoint
        dx = trajectory[i, 1] - trajectory[i-1, 1]
        dy = trajectory[i, 2] - trajectory[i-1, 2]
        current_distance += np.sqrt(dx**2 + dy**2)
        
        # Compute required spacing based on curvature at current point
        kappa = curvature[i]
        
        if kappa >= curvature_high:
            # High curvature: use minimum spacing
            required_spacing = min_spacing
        elif kappa <= curvature_low:
            # Low curvature: use maximum spacing
            required_spacing = max_spacing
        else:
            # Intermediate curvature: linear interpolation
            alpha = (kappa - curvature_low) / (curvature_high - curvature_low)
            required_spacing = max_spacing - alpha * (max_spacing - min_spacing)
        
        # Add checkpoint if spacing threshold reached
        if current_distance >= required_spacing:
            checkpoint = Checkpoint(
                x=trajectory[i, 1],
                y=trajectory[i, 2],
                theta=trajectory[i, 3],
                curvature=kappa,
                index=i
            )
            checkpoints.append(checkpoint)
            current_distance = 0.0
            last_checkpoint_idx = i
    
    # Always add final checkpoint if not already added
    if last_checkpoint_idx < len(trajectory) - 1:
        checkpoints.append(Checkpoint(
            x=trajectory[-1, 1],
            y=trajectory[-1, 2],
            theta=trajectory[-1, 3],
            curvature=curvature[-1],
            index=len(trajectory) - 1
        ))
    
    return checkpoints


def _curvature_spacing(
    kappa: float,
    curvature_high: float,
    curvature_low: float,
    min_spacing: float,
    max_spacing: float,
) -> float:
    """Compute spacing from curvature using the standard linear interpolation."""
    if kappa >= curvature_high:
        return min_spacing
    if kappa <= curvature_low:
        return max_spacing
    alpha = (kappa - curvature_low) / (curvature_high - curvature_low)
    return max_spacing - alpha * (max_spacing - min_spacing)


def _count_obstacles_near(
    position: np.ndarray,
    obstacle_positions: np.ndarray,
    sensor_range: float,
) -> int:
    """Count how many obstacles are within *sensor_range* of *position*.
    
    Args:
        position: (2,) array [x, y]
        obstacle_positions: (M, 2) array of obstacle centres
        sensor_range: detection radius (m)
    """
    if len(obstacle_positions) == 0:
        return 0
    diffs = obstacle_positions - position
    dists = np.sqrt(np.sum(diffs ** 2, axis=1))
    return int(np.sum(dists <= sensor_range))


def generate_checkpoints_obstacle_aware(
    trajectory: np.ndarray,
    curvature: np.ndarray,
    obstacle_positions: np.ndarray,
    sensor_range: float = 5.0,
    gamma: float = 0.5,
    S_min: float = 0.1,
    S_max: float = 1.0,
    curvature_high: float = 2.0,
    curvature_low: float = 0.5,
) -> List[Checkpoint]:
    """Generate checkpoints with spacing driven by both curvature and obstacle density.
    
    For every candidate position along the trajectory two spacing values are
    computed and the *tighter* (smaller) one governs checkpoint placement:
    
    1. **Curvature spacing** — identical to :func:`generate_checkpoints`.
    2. **Obstacle-density spacing**::
    
           S_obs = S_min + (S_max - S_min) * exp(-gamma * N_obs)
    
       where *N_obs* is the number of obstacles within *sensor_range* of the
       candidate point.  Zero obstacles → ``S_obs ≈ S_max`` (sparse); many
       obstacles → ``S_obs → S_min`` (dense).  The exponential avoids the
       divergence problems of polynomial formulas for small obstacle counts.
    
    Args:
        trajectory: (N, 4) array with columns [t, x, y, theta]
        curvature: (N,) array of curvature values (1/m)
        obstacle_positions: (M, 2) array of obstacle centre coordinates, or
            an empty (0, 2) array when no obstacles are present.
        sensor_range: lookahead range for obstacle detection (m)
        gamma: exponential decay rate controlling obstacle sensitivity
        S_min: minimum spacing between checkpoints (m)
        S_max: maximum spacing between checkpoints (m)
        curvature_high: high curvature threshold (1/m)
        curvature_low: low curvature threshold (1/m)
        
    Returns:
        List of Checkpoint objects with adaptive spacing
    """
    if len(trajectory) == 0:
        return []

    if obstacle_positions is None or len(obstacle_positions) == 0:
        obstacle_positions = np.empty((0, 2))

    checkpoints: List[Checkpoint] = []
    current_distance = 0.0
    last_checkpoint_idx = 0

    checkpoints.append(Checkpoint(
        x=trajectory[0, 1],
        y=trajectory[0, 2],
        theta=trajectory[0, 3],
        curvature=curvature[0],
        index=0,
    ))

    for i in range(1, len(trajectory)):
        dx = trajectory[i, 1] - trajectory[i - 1, 1]
        dy = trajectory[i, 2] - trajectory[i - 1, 2]
        current_distance += np.sqrt(dx ** 2 + dy ** 2)

        kappa = curvature[i]
        s_curv = _curvature_spacing(kappa, curvature_high, curvature_low, S_min, S_max)

        pos = np.array([trajectory[i, 1], trajectory[i, 2]])
        n_obs = _count_obstacles_near(pos, obstacle_positions, sensor_range)
        s_obs = S_min + (S_max - S_min) * np.exp(-gamma * n_obs)

        required_spacing = min(s_curv, s_obs)

        if current_distance >= required_spacing:
            checkpoints.append(Checkpoint(
                x=trajectory[i, 1],
                y=trajectory[i, 2],
                theta=trajectory[i, 3],
                curvature=kappa,
                index=i,
            ))
            current_distance = 0.0
            last_checkpoint_idx = i

    if last_checkpoint_idx < len(trajectory) - 1:
        checkpoints.append(Checkpoint(
            x=trajectory[-1, 1],
            y=trajectory[-1, 2],
            theta=trajectory[-1, 3],
            curvature=curvature[-1],
            index=len(trajectory) - 1,
        ))

    return checkpoints

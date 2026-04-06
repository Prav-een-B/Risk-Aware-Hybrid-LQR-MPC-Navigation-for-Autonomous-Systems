"""Checkpoint generation with curvature-adaptive spacing.

This module provides functionality for generating checkpoints from trajectories
with spacing that adapts based on local curvature. Denser checkpoints are placed
in high-curvature regions for better tracking accuracy.
"""

from dataclasses import dataclass
from typing import List

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

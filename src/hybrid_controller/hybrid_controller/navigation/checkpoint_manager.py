"""Checkpoint-based navigation manager with adaptive switching logic."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import numpy as np


@dataclass
class Checkpoint:
    """Waypoint in checkpoint-based navigation."""
    # Position and orientation
    x: float
    y: float
    theta: float
    
    # Trajectory properties
    curvature: float
    index: int  # Index in original trajectory
    
    # Tracking state
    reached: bool = False
    time_reached: float = 0.0
    overshoot: float = 0.0
    min_distance: float = float('inf')
    
    def distance_to(self, position: np.ndarray) -> float:
        """Compute Euclidean distance to position."""
        return np.sqrt((position[0] - self.x)**2 + (position[1] - self.y)**2)


class CheckpointManager:
    """Manages checkpoint-based navigation with adaptive switching."""
    
    def __init__(self,
                 base_switching_radius: float = 0.3,
                 curvature_scaling: float = 0.2,
                 hysteresis_margin: float = 0.1,
                 forward_progress_timeout: float = 1.0,
                 dt: float = 0.02):
        """
        Initialize checkpoint manager.
        
        Args:
            base_switching_radius: Base radius for checkpoint switching (m)
            curvature_scaling: Curvature-dependent radius adjustment (m)
            hysteresis_margin: Hysteresis deadband (m)
            forward_progress_timeout: Max time without progress (s)
            dt: Timestep (s)
        """
        self.base_radius = base_switching_radius
        self.curvature_scaling = curvature_scaling
        self.hysteresis_margin = hysteresis_margin
        self.forward_progress_timeout = forward_progress_timeout
        self.dt = dt
        
        self.checkpoints: List[Checkpoint] = []
        self.current_idx: int = 0
        self.switching_radius: float = base_switching_radius
        self.hysteresis_active: bool = False
        self.last_distance: float = float('inf')
        self.no_progress_time: float = 0.0
        
        # Metrics
        self.checkpoints_reached: int = 0
        self.checkpoints_missed: int = 0
        self.total_overshoot: float = 0.0
        self.checkpoint_times: List[float] = []
    
    def set_checkpoints(self, checkpoints: List[Checkpoint]) -> None:
        """
        Set the checkpoint queue for navigation.
        
        Args:
            checkpoints: List of checkpoints to follow
        """
        self.checkpoints = checkpoints
        self.reset()
    
    def reset(self) -> None:
        """Reset checkpoint manager to initial state."""
        self.current_idx = 0
        self.switching_radius = self.base_radius
        self.hysteresis_active = False
        self.last_distance = float('inf')
        self.no_progress_time = 0.0
        
        # Reset metrics
        self.checkpoints_reached = 0
        self.checkpoints_missed = 0
        self.total_overshoot = 0.0
        self.checkpoint_times = []
        
        # Reset checkpoint states
        for cp in self.checkpoints:
            cp.reached = False
            cp.time_reached = 0.0
            cp.overshoot = 0.0
            cp.min_distance = float('inf')
    
    def get_current_checkpoint(self) -> Optional[Checkpoint]:
        """
        Get the current active checkpoint.
        
        Returns:
            Current checkpoint or None if all checkpoints reached
        """
        if self.current_idx < len(self.checkpoints):
            return self.checkpoints[self.current_idx]
        return None
    
    def update(self, robot_position: np.ndarray, current_time: float) -> bool:
        """
        Update checkpoint manager with current robot position.
        
        Args:
            robot_position: Current robot position [x, y] or [x, y, theta]
            current_time: Current simulation time (s)
        
        Returns:
            True if checkpoint was switched, False otherwise
        """
        if self.current_idx >= len(self.checkpoints):
            return False
        
        current_cp = self.checkpoints[self.current_idx]
        
        # Compute distance to current checkpoint
        distance = np.sqrt(
            (robot_position[0] - current_cp.x)**2 + 
            (robot_position[1] - current_cp.y)**2
        )
        
        # Update minimum distance achieved
        current_cp.min_distance = min(current_cp.min_distance, distance)
        
        # Compute adaptive switching radius
        self._update_switching_radius(current_cp.curvature)
        
        # Check forward progress
        if distance > self.last_distance:
            # Moving away from checkpoint
            self.no_progress_time += self.dt
            if not self.hysteresis_active:
                # Activate hysteresis: increase radius
                self.switching_radius += self.hysteresis_margin
                self.hysteresis_active = True
        else:
            # Moving toward checkpoint
            self.no_progress_time = 0.0
            if self.hysteresis_active:
                # Deactivate hysteresis: restore radius
                self.switching_radius -= self.hysteresis_margin
                self.hysteresis_active = False
        
        self.last_distance = distance
        
        # Check for forward progress timeout
        if self.no_progress_time > self.forward_progress_timeout:
            self.checkpoints_missed += 1
            return self._advance_checkpoint(current_time, distance)
        
        # Check switching condition
        if distance <= self.switching_radius:
            return self._advance_checkpoint(current_time, distance)
        
        return False
    
    def _update_switching_radius(self, curvature: float) -> None:
        """
        Update switching radius based on curvature.
        
        Formula:
            radius = base_radius - curvature_scaling * curvature
        
        High curvature -> smaller radius (tighter tracking)
        Low curvature -> larger radius (more tolerance)
        
        Args:
            curvature: Local curvature at checkpoint (1/m)
        """
        adjustment = self.curvature_scaling * curvature
        self.switching_radius = max(
            self.base_radius - adjustment,
            0.1  # Minimum radius
        )
    
    def _advance_checkpoint(self, current_time: float, distance: float) -> bool:
        """
        Advance to next checkpoint and record metrics.
        
        Args:
            current_time: Current simulation time (s)
            distance: Distance to checkpoint at time of advancement (m)
        
        Returns:
            True (checkpoint was advanced)
        """
        current_cp = self.checkpoints[self.current_idx]
        
        # Mark as reached
        current_cp.reached = True
        current_cp.time_reached = current_time
        current_cp.overshoot = max(0.0, distance - self.switching_radius)
        
        # Record metrics
        self.checkpoints_reached += 1
        self.total_overshoot += current_cp.overshoot
        if self.current_idx > 0:
            time_delta = current_time - self.checkpoints[self.current_idx - 1].time_reached
            self.checkpoint_times.append(time_delta)
        
        # Advance to next checkpoint
        self.current_idx += 1
        self.last_distance = float('inf')
        self.no_progress_time = 0.0
        self.hysteresis_active = False
        
        return True
    
    def get_local_trajectory_segment(self, 
                                     robot_state: np.ndarray,
                                     horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract local reference horizon for MPC.
        
        Args:
            robot_state: Current robot state [x, y, theta]
            horizon: Number of future steps (N+1 for MPC)
            
        Returns:
            x_refs: (horizon, 3) reference states
            u_refs: (horizon-1, 2) reference controls
        """
        x_refs = np.zeros((horizon, 3))
        u_refs = np.zeros((horizon - 1, 2))
        
        # Fill from current checkpoint forward
        for i in range(horizon):
            cp_idx = min(self.current_idx + i, len(self.checkpoints) - 1)
            cp = self.checkpoints[cp_idx]
            
            x_refs[i] = [cp.x, cp.y, cp.theta]
            
            if i < horizon - 1:
                # Compute reference velocity from checkpoint spacing
                next_cp_idx = min(cp_idx + 1, len(self.checkpoints) - 1)
                next_cp = self.checkpoints[next_cp_idx]
                
                dx = next_cp.x - cp.x
                dy = next_cp.y - cp.y
                distance = np.sqrt(dx**2 + dy**2)
                
                # Reference velocity (assume constant speed)
                v_ref = min(distance / self.dt, 1.0)  # Cap at 1 m/s
                
                # Reference angular velocity from heading change
                dtheta = next_cp.theta - cp.theta
                # Normalize to [-pi, pi]
                while dtheta > np.pi: 
                    dtheta -= 2*np.pi
                while dtheta < -np.pi: 
                    dtheta += 2*np.pi
                omega_ref = dtheta / self.dt
                
                u_refs[i] = [v_ref, omega_ref]
        
        return x_refs, u_refs
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Export checkpoint tracking metrics.
        
        Returns:
            Dictionary of metrics including completion rate, mean time, and overshoot
        """
        completion_rate = self.checkpoints_reached / len(self.checkpoints) if self.checkpoints else 0.0
        mean_time = np.mean(self.checkpoint_times) if self.checkpoint_times else 0.0
        mean_overshoot = self.total_overshoot / max(1, self.checkpoints_reached)
        
        return {
            'checkpoints_total': len(self.checkpoints),
            'checkpoints_reached': self.checkpoints_reached,
            'checkpoints_missed': self.checkpoints_missed,
            'completion_rate': completion_rate,
            'mean_time_to_checkpoint': mean_time,
            'mean_overshoot': mean_overshoot,
        }

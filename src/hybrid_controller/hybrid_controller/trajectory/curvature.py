"""Curvature computation for trajectory analysis.

This module provides functions for computing local curvature at trajectory points
using finite difference methods.
"""

import numpy as np


def compute_curvature(trajectory: np.ndarray) -> np.ndarray:
    """
    Compute curvature at each trajectory point using finite differences.
    
    Uses the formula: kappa = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
    where x', y' are first derivatives (velocity) and x'', y'' are second derivatives.
    
    Args:
        trajectory: (N, 4) array with columns [t, x, y, theta]
        
    Returns:
        curvature: (N,) array of curvature values (1/m), always non-negative
        
    Notes:
        - Curvature is clamped to avoid division by zero using epsilon = 1e-6
        - Returns non-negative values (absolute value in numerator)
        - Units: 1/meters (inverse radius of curvature)
        - High curvature (>2.0) indicates sharp turns
        - Low curvature (<0.5) indicates straight segments
    """
    # First derivatives (velocity)
    dx = np.gradient(trajectory[:, 1])
    dy = np.gradient(trajectory[:, 2])
    
    # Second derivatives (acceleration)
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)
    
    # Curvature formula: kappa = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
    numerator = np.abs(dx * d2y - dy * d2x)
    denominator = (dx**2 + dy**2)**(3/2)
    
    # Avoid division by zero with epsilon clamping
    denominator = np.maximum(denominator, 1e-6)
    
    curvature = numerator / denominator
    return curvature

"""
Property-based tests for trajectory formula correctness.

**Validates: Requirements 12.1-12.9**
"""

import os
import sys

import numpy as np
from hypothesis import given, strategies as st, settings

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src", "hybrid_controller")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from hybrid_controller.trajectory.reference_generator import ReferenceTrajectoryGenerator


# Tolerance for numerical comparisons
STANDARD_TOLERANCE = 1e-6
INTEGRATION_TOLERANCE = 1e-3  # For clothoid (numerical integration)


@given(
    A=st.floats(min_value=0.5, max_value=5.0),
    a=st.floats(min_value=0.1, max_value=2.0),
    b=st.floats(min_value=0.5, max_value=3.0),
    c=st.floats(min_value=0.5, max_value=3.0),
    t=st.floats(min_value=0.1, max_value=5.0),
)
@settings(max_examples=50, deadline=2000)
def test_property_32_lissajous_formula(A, a, b, c, t):
    """
    Property 32: Trajectory Formula Correctness - Lissajous
    
    For a Lissajous trajectory, the generated positions SHALL match 
    the mathematical formula: x = A*sin(a*t), y = A*sin(b*t)*cos(c*t)
    within numerical tolerance.
    
    **Validates: Requirements 12.1**
    """
    generator = ReferenceTrajectoryGenerator(
        A=A,
        a=a,
        lissajous_b=b,
        lissajous_c=c,
        dt=0.05,
        trajectory_type="lissajous",
    )
    
    # Get position at time t
    x_gen, y_gen = generator.position(t)
    
    # Compute expected position from formula
    x_expected = A * np.sin(a * t)
    y_expected = A * np.sin(b * t) * np.cos(c * t)
    
    # Verify positions match within tolerance
    assert np.abs(x_gen - x_expected) < STANDARD_TOLERANCE, \
        f"Lissajous x mismatch: got {x_gen}, expected {x_expected}, diff {abs(x_gen - x_expected)}"
    assert np.abs(y_gen - y_expected) < STANDARD_TOLERANCE, \
        f"Lissajous y mismatch: got {y_gen}, expected {y_expected}, diff {abs(y_gen - y_expected)}"


@given(
    r0=st.floats(min_value=0.1, max_value=2.0),
    k=st.floats(min_value=0.1, max_value=1.0),
    omega=st.floats(min_value=0.5, max_value=2.0),
    t=st.floats(min_value=0.1, max_value=5.0),
)
@settings(max_examples=50, deadline=2000)
def test_property_32_spiral_formula(r0, k, omega, t):
    """
    Property 32: Trajectory Formula Correctness - Spiral
    
    For a spiral trajectory, the generated positions SHALL match 
    the mathematical formula in polar coordinates: r = r0 + k*t, theta = omega*t
    within numerical tolerance.
    
    **Validates: Requirements 12.2**
    """
    generator = ReferenceTrajectoryGenerator(
        spiral_r0=r0,
        spiral_k=k,
        spiral_omega=omega,
        dt=0.05,
        trajectory_type="spiral",
    )
    
    # Get position at time t
    x_gen, y_gen = generator.position(t)
    
    # Compute expected position from formula
    r = r0 + k * t
    theta = omega * t
    x_expected = r * np.cos(theta)
    y_expected = r * np.sin(theta)
    
    # Verify positions match within tolerance
    assert np.abs(x_gen - x_expected) < STANDARD_TOLERANCE, \
        f"Spiral x mismatch: got {x_gen}, expected {x_expected}, diff {abs(x_gen - x_expected)}"
    assert np.abs(y_gen - y_expected) < STANDARD_TOLERANCE, \
        f"Spiral y mismatch: got {y_gen}, expected {y_expected}, diff {abs(y_gen - y_expected)}"


@given(
    A=st.floats(min_value=0.5, max_value=5.0),
    v=st.floats(min_value=0.3, max_value=2.0),
    omega=st.floats(min_value=0.5, max_value=2.0),
    t=st.floats(min_value=0.1, max_value=5.0),
)
@settings(max_examples=50, deadline=2000)
def test_property_32_sinusoidal_formula(A, v, omega, t):
    """
    Property 32: Trajectory Formula Correctness - Sinusoidal
    
    For a sinusoidal trajectory, the generated positions SHALL match 
    the mathematical formula: x = v*t, y = A*sin(omega*t)
    within numerical tolerance.
    
    **Validates: Requirements 12.5**
    """
    generator = ReferenceTrajectoryGenerator(
        A=A,
        sinusoidal_v=v,
        sinusoidal_omega=omega,
        dt=0.05,
        trajectory_type="sinusoidal",
    )
    
    # Get position at time t
    x_gen, y_gen = generator.position(t)
    
    # Compute expected position from formula
    x_expected = v * t
    y_expected = A * np.sin(omega * t)
    
    # Verify positions match within tolerance
    assert np.abs(x_gen - x_expected) < STANDARD_TOLERANCE, \
        f"Sinusoidal x mismatch: got {x_gen}, expected {x_expected}, diff {abs(x_gen - x_expected)}"
    assert np.abs(y_gen - y_expected) < STANDARD_TOLERANCE, \
        f"Sinusoidal y mismatch: got {y_gen}, expected {y_expected}, diff {abs(y_gen - y_expected)}"


@given(
    kappa0=st.floats(min_value=-1.0, max_value=1.0),
    k_rate=st.floats(min_value=0.1, max_value=1.0),
    t=st.floats(min_value=0.1, max_value=3.0),
    nominal_speed=st.floats(min_value=0.5, max_value=2.0),
)
@settings(max_examples=30, deadline=5000)
def test_property_32_clothoid_formula(kappa0, k_rate, t, nominal_speed):
    """
    Property 32: Trajectory Formula Correctness - Clothoid
    
    For a clothoid trajectory, the generated positions SHALL match 
    the Euler spiral equations with linearly varying curvature
    within numerical tolerance (relaxed for numerical integration).
    
    **Validates: Requirements 12.7**
    """
    generator = ReferenceTrajectoryGenerator(
        clothoid_kappa0=kappa0,
        clothoid_k_rate=k_rate,
        nominal_speed=nominal_speed,
        dt=0.05,
        trajectory_type="clothoid",
    )
    
    # Get position at time t
    x_gen, y_gen = generator.position(t)
    
    # Compute expected position from formula (same numerical integration as implementation)
    s = t * nominal_speed  # Arc length parameter
    
    # Numerical integration for clothoid
    n_steps = max(10, int(s / 0.1))
    if n_steps > 0:
        s_vals = np.linspace(0, s, n_steps)
        theta_vals = kappa0 * s_vals + 0.5 * k_rate * s_vals**2
        x_expected = np.trapz(np.cos(theta_vals), s_vals)
        y_expected = np.trapz(np.sin(theta_vals), s_vals)
    else:
        x_expected, y_expected = 0.0, 0.0
    
    # Verify positions match within relaxed tolerance (numerical integration)
    assert np.abs(x_gen - x_expected) < INTEGRATION_TOLERANCE, \
        f"Clothoid x mismatch: got {x_gen}, expected {x_expected}, diff {abs(x_gen - x_expected)}"
    assert np.abs(y_gen - y_expected) < INTEGRATION_TOLERANCE, \
        f"Clothoid y mismatch: got {y_gen}, expected {y_expected}, diff {abs(y_gen - y_expected)}"


@given(
    A=st.floats(min_value=0.5, max_value=5.0),
    a=st.floats(min_value=0.1, max_value=2.0),
    t=st.floats(min_value=0.1, max_value=5.0),
)
@settings(max_examples=50, deadline=2000)
def test_property_32_figure8_formula(A, a, t):
    """
    Property 32: Trajectory Formula Correctness - Figure8
    
    For a figure8 trajectory, the generated positions SHALL match 
    the mathematical formula: x = A*sin(a*t), y = A*sin(a*t)*cos(a*t)
    within numerical tolerance.
    
    **Validates: Requirements 12.1-12.9** (figure8 is a specific Lissajous case)
    """
    generator = ReferenceTrajectoryGenerator(
        A=A,
        a=a,
        dt=0.05,
        trajectory_type="figure8",
    )
    
    # Get position at time t
    x_gen, y_gen = generator.position(t)
    
    # Compute expected position from formula
    x_expected = A * np.sin(a * t)
    y_expected = A * np.sin(a * t) * np.cos(a * t)
    
    # Verify positions match within tolerance
    assert np.abs(x_gen - x_expected) < STANDARD_TOLERANCE, \
        f"Figure8 x mismatch: got {x_gen}, expected {x_expected}, diff {abs(x_gen - x_expected)}"
    assert np.abs(y_gen - y_expected) < STANDARD_TOLERANCE, \
        f"Figure8 y mismatch: got {y_gen}, expected {y_expected}, diff {abs(y_gen - y_expected)}"


@given(
    A=st.floats(min_value=0.5, max_value=5.0),
    a=st.floats(min_value=0.1, max_value=2.0),
    t=st.floats(min_value=0.1, max_value=5.0),
)
@settings(max_examples=50, deadline=2000)
def test_property_32_circle_formula(A, a, t):
    """
    Property 32: Trajectory Formula Correctness - Circle
    
    For a circle trajectory, the generated positions SHALL match 
    the mathematical formula: x = A*(1 - cos(a*t)), y = A*sin(a*t)
    within numerical tolerance.
    
    **Validates: Requirements 12.1-12.9**
    """
    generator = ReferenceTrajectoryGenerator(
        A=A,
        a=a,
        dt=0.05,
        trajectory_type="circle",
    )
    
    # Get position at time t
    x_gen, y_gen = generator.position(t)
    
    # Compute expected position from formula
    x_expected = A * (1.0 - np.cos(a * t))
    y_expected = A * np.sin(a * t)
    
    # Verify positions match within tolerance
    assert np.abs(x_gen - x_expected) < STANDARD_TOLERANCE, \
        f"Circle x mismatch: got {x_gen}, expected {x_expected}, diff {abs(x_gen - x_expected)}"
    assert np.abs(y_gen - y_expected) < STANDARD_TOLERANCE, \
        f"Circle y mismatch: got {y_gen}, expected {y_expected}, diff {abs(y_gen - y_expected)}"


@given(
    A=st.floats(min_value=0.5, max_value=5.0),
    a=st.floats(min_value=0.1, max_value=2.0),
    t=st.floats(min_value=0.1, max_value=5.0),
)
@settings(max_examples=50, deadline=2000)
def test_property_32_clover_formula(A, a, t):
    """
    Property 32: Trajectory Formula Correctness - Clover
    
    For a clover trajectory, the generated positions SHALL match 
    the mathematical formula: x = A*cos(2*phase)*cos(phase), y = A*cos(2*phase)*sin(phase)
    where phase = a*t, within numerical tolerance.
    
    **Validates: Requirements 12.1-12.9**
    """
    generator = ReferenceTrajectoryGenerator(
        A=A,
        a=a,
        dt=0.05,
        trajectory_type="clover",
    )
    
    # Get position at time t
    x_gen, y_gen = generator.position(t)
    
    # Compute expected position from formula
    phase = a * t
    x_expected = A * np.cos(2.0 * phase) * np.cos(phase)
    y_expected = A * np.cos(2.0 * phase) * np.sin(phase)
    
    # Verify positions match within tolerance
    assert np.abs(x_gen - x_expected) < STANDARD_TOLERANCE, \
        f"Clover x mismatch: got {x_gen}, expected {x_expected}, diff {abs(x_gen - x_expected)}"
    assert np.abs(y_gen - y_expected) < STANDARD_TOLERANCE, \
        f"Clover y mismatch: got {y_gen}, expected {y_expected}, diff {abs(y_gen - y_expected)}"


@given(
    A=st.floats(min_value=0.5, max_value=5.0),
    a=st.floats(min_value=0.1, max_value=2.0),
    t=st.floats(min_value=0.1, max_value=5.0),
)
@settings(max_examples=50, deadline=2000)
def test_property_32_slalom_formula(A, a, t):
    """
    Property 32: Trajectory Formula Correctness - Slalom
    
    For a slalom trajectory, the generated positions SHALL match 
    the mathematical formula: x = A*sin(phase), y = 0.45*A*sin(2*phase) + 0.2*A*sin(3*phase)
    where phase = a*t, within numerical tolerance.
    
    **Validates: Requirements 12.1-12.9**
    """
    generator = ReferenceTrajectoryGenerator(
        A=A,
        a=a,
        dt=0.05,
        trajectory_type="slalom",
    )
    
    # Get position at time t
    x_gen, y_gen = generator.position(t)
    
    # Compute expected position from formula
    phase = a * t
    x_expected = A * np.sin(phase)
    y_expected = 0.45 * A * np.sin(2.0 * phase) + 0.2 * A * np.sin(3.0 * phase)
    
    # Verify positions match within tolerance
    assert np.abs(x_gen - x_expected) < STANDARD_TOLERANCE, \
        f"Slalom x mismatch: got {x_gen}, expected {x_expected}, diff {abs(x_gen - x_expected)}"
    assert np.abs(y_gen - y_expected) < STANDARD_TOLERANCE, \
        f"Slalom y mismatch: got {y_gen}, expected {y_expected}, diff {abs(y_gen - y_expected)}"

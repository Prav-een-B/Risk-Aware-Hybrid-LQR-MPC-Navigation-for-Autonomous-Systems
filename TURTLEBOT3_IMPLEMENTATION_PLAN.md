# TurtleBot3 Hybrid LQR-MPC — Implementation Plan

---

## Quick Start

```bash
source /opt/ros/jazzy/setup.bash
source ~/hybrid_ws/install/setup.bash
export TURTLEBOT3_MODEL=burger

ros2 launch hybrid_nav turtlebot3_hybrid.launch.py
```

Monitor in second terminal:
```bash
source /opt/ros/jazzy/setup.bash
source ~/hybrid_ws/install/setup.bash
ros2 topic echo /hybrid/controller_mode
```

---

## Components

| # | Component | File | Status |
|---|-----------|------|--------|
| 1 | Robot Model | TurtleBot3 Burger (pre-installed) | ✅ |
| 2 | Gazebo World + Walls + Obstacles | `worlds/hybrid_obstacle_world.sdf` | ✅ |
| 3 | Hybrid Controller Node | `hybrid_nav/nodes/hybrid_controller_node.py` | ✅ |
| 4 | Trajectory | Built into controller (figure-8) | ✅ |
| 5 | Obstacle Publisher | `hybrid_nav/nodes/obstacle_publisher_node.py` | ✅ |
| 6 | Launch File | `launch/turtlebot3_hybrid.launch.py` | ✅ |
| 7 | RViz Config | `rviz/hybrid_nav.rviz` | ✅ |

---

## File Locations

All ROS2 files are in: `~/hybrid_ws/src/hybrid_nav/`

```
hybrid_nav/
├── hybrid_nav/
│   ├── nodes/
│   │   ├── hybrid_controller_node.py    ← MAIN CONTROLLER
│   │   ├── obstacle_publisher_node.py
│   │   └── test_node.py
│   ├── controllers -> (symlink to your Windows controllers/)
│   ├── models -> (symlink to your Windows models/)
│   ├── trajectory -> (symlink)
│   └── logging -> (symlink)
├── launch/
│   └── turtlebot3_hybrid.launch.py      ← LAUNCH FILE (has parameters)
├── worlds/
│   └── hybrid_obstacle_world.sdf       ← GAZEBO WORLD
├── rviz/
│   └── hybrid_nav.rviz
├── package.xml
├── setup.py
└── setup.cfg
```

---

## TUNABLE PARAMETERS — Where to Change Them

### File: `~/hybrid_ws/src/hybrid_nav/launch/turtlebot3_hybrid.launch.py`

Lines ~103-115, inside the `parameters` dict:

```python
parameters=[{
    'use_sim_time': True,
    'control_rate': 20.0,              # Hz — control loop rate
    'trajectory_amplitude': 0.5,       # ← TUNE THIS (meters, figure-8 size)
    'trajectory_frequency': 0.2,       # ← TUNE THIS (rad/s, figure-8 speed)
    'trajectory_duration': 120.0,      # seconds
    'dt': 0.05,                        # must match 1/control_rate
    'mpc_horizon': 10,                 # MPC lookahead steps
    'd_safe': 0.35,                    # safety distance from obstacles
    'v_max': 0.22,                     # TB3 Burger max (DO NOT EXCEED)
    'omega_max': 2.84,                 # TB3 Burger max
    'use_hybrid': True,                # False = LQR only
}],
```

### File: `~/hybrid_ws/src/hybrid_nav/hybrid_nav/nodes/hybrid_controller_node.py`

LQR weight matrices (line ~84-85 in __init__):

```python
self.lqr = LQRController(
    Q_diag=[10.0, 10.0, 1.0],    # ← TUNE: [position_x, position_y, heading]
    R_diag=[1.0, 0.5],           # ← TUNE: [velocity_cost, angular_cost]
    v_max=self.v_max,
    omega_max=self.omega_max,
    dt=self.dt,
)
```

### What Each Parameter Does

| Parameter | Effect | Increase → | Decrease → |
|-----------|--------|------------|------------|
| **Q_diag[0,1]** (position) | How aggressively robot chases position | Faster correction, may oscillate | Slower, smoother tracking |
| **Q_diag[2]** (heading) | How aggressively robot corrects heading | Snaps to heading fast | Lazy heading correction |
| **R_diag[0]** (velocity) | Cost of using velocity | Slower, gentler | Faster, more aggressive |
| **R_diag[1]** (angular) | Cost of using angular velocity | Less turning | More turning |
| **trajectory_amplitude** | Size of figure-8 | Bigger path, needs more speed | Smaller path, easier to track |
| **trajectory_frequency** | Speed of figure-8 | Faster movement, harder to track | Slower, easier to track |
| **d_safe** | Safety buffer around obstacles | More avoidance, less direct | Closer passes |
| **use_hybrid** | MPC+LQR vs LQR only | — | Set False for LQR-only testing |

### Tuning Tips

1. **If robot goes in circles**: Lower `Q_diag` (try `[5.0, 5.0, 0.5]`) and raise `R_diag` (try `[2.0, 1.0]`)
2. **If robot barely moves**: Lower `R_diag` (try `[0.1, 0.1]`)  
3. **If robot oscillates**: Raise `R_diag[1]` (angular cost)
4. **If robot can't keep up**: Lower `trajectory_frequency` (try `0.1`)
5. **Start simple**: Set `use_hybrid: False` to test LQR only first

### IMPORTANT: After changing parameters, rebuild!

```bash
cd ~/hybrid_ws
rm -rf build install log
source /opt/ros/jazzy/setup.bash
colcon build --packages-select hybrid_nav
source install/setup.bash
```

---

## Velocity Limits

TurtleBot3 Burger max speed is **0.22 m/s**.

The figure-8 max velocity is approximately: `v_max ≈ A × freq × sqrt(2)`

| A | freq | Max velocity | Fits TB3? |
|---|------|-------------|-----------|
| 1.0 | 0.3 | 0.42 m/s | ❌ Too fast |
| 0.5 | 0.2 | 0.14 m/s | ✅ |
| 0.5 | 0.15 | 0.11 m/s | ✅ (safest) |
| 0.3 | 0.2 | 0.085 m/s | ✅ (very safe) |

---

## Architecture

```
obstacle_publisher ──/obstacles──▶ hybrid_controller_node
                                         │
                                    reads /odom from Gazebo
                                         │
                                    LQR + MPC + Blend
                                         │
                                    publishes /cmd_vel (TwistStamped)
                                         │
                                         ▼
                                   TurtleBot3 (Gazebo)
```

---

## Troubleshooting

| Problem | Check | Fix |
|---------|-------|-----|
| Robot doesn't move at all | `ros2 topic hz /cmd_vel` | Rebuild, check for import errors |
| Robot moves randomly | Terminal logs show growing error | Lower trajectory speed (freq) or tune Q/R |
| Robot spins in circles | Heading error > 1.0 rad | Lower Q_diag[2], raise R_diag[1] |
| "FALLBACK mode" | Symlinks broken | Check: `ls -la ~/hybrid_ws/src/hybrid_nav/hybrid_nav/ | grep "^l"` |
| Build fails | Stale build dirs | `rm -rf build install log` then rebuild |

---

## Bug History

| Bug | Root Cause | Fix Applied |
|-----|-----------|-------------|
| Build fails with `--editable` | Setuptools version | Use `colcon build` without `--symlink-install` |
| `No module named 'hybrid_nav.models'` | Symlink named `models_src` | Renamed to `models` |
| Wrong API calls | 3 method mismatches | Fixed all method signatures |
| Robot doesn't move | `Twist` instead of `TwistStamped` | Changed to `TwistStamped` |
| Robot moves randomly | Trajectory 2x faster than TB3 | Reduced A=0.5, freq=0.2 |
| Trajectory desync | Callback counter instead of sim time | Now uses simulation clock |

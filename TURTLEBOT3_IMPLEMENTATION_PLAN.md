# TurtleBot3 Hybrid LQR-MPC — Implementation Plan

## Status: Fixed & Ready ✅

### Root Cause Found

The diagnostic revealed the robot **WAS following the trajectory** (waypoints 1→25 advancing correctly), but `AVOID(w=1.00)` was permanently at 100% because:

1. **Obstacles were placed ON the figure-8 path** → always "near" an obstacle  
2. **`d_trigger=1.5m`** covered the entire 3m arena → avoidance never turned off
3. **MPC constantly fighting the base controller** → made motion look "back and forth"

### Fixes Applied

| What | Before | After |
|------|--------|-------|
| Obstacle positions | ON path `(0.35, 0.15)` | OFF path `(0.30, 0.35)` |
| `d_trigger` | 1.5m (whole arena) | **0.6m** (only near obstacles) |
| Gazebo world SDF | Obstacles blocking path | Obstacles beside path |

---

## How to Run

```bash
source /opt/ros/jazzy/setup.bash
source ~/hybrid_ws/install/setup.bash
export TURTLEBOT3_MODEL=burger

ros2 launch hybrid_nav turtlebot3_hybrid.launch.py
```

Monitor in a second terminal:
```bash
source /opt/ros/jazzy/setup.bash
source ~/hybrid_ws/install/setup.bash
ros2 topic echo /hybrid/controller_mode
```

---

## Architecture

```
obstacle_publisher ──/obstacles──▶ hybrid_controller_node
                                        │
                                   reads /odom from Gazebo
                                        │
                              ┌─── risk < threshold? ──┐
                              │                        │
                         DRIVE mode              AVOID mode
                    (turn-then-drive)         (MPC + hybrid_blender)
                              │                        │
                              └────── /cmd_vel ────────┘
                                    (TwistStamped)
                                        │
                                        ▼
                                  TurtleBot3 (Gazebo)
```

---

## Expected Log Output

```
[wp=5/219]  pos=(+0.10,+0.09)  | d=0.06m | v=0.090 ω=+0.15 | DRIVE
[wp=40/219] pos=(+0.48,+0.05)  | d=0.08m | v=0.085 ω=-0.20 | DRIVE
[wp=55/219] pos=(+0.35,+0.20)  | d=0.11m | v=0.070 ω=-0.30 | AVOID(w=0.35)
[wp=60/219] pos=(+0.30,+0.28)  | d=0.09m | v=0.060 ω=-0.40 | AVOID(w=0.80)
[wp=65/219] pos=(+0.22,+0.20)  | d=0.07m | v=0.088 ω=-0.12 | DRIVE
```

- `DRIVE` = normal trajectory tracking (turn-then-drive)
- `AVOID(w=X)` = near obstacle, hybrid_blender activates MPC with weight X

---

## Files

| File | Location | Purpose |
|------|----------|---------|
| **Controller** | `~/hybrid_ws/src/hybrid_nav/hybrid_nav/nodes/hybrid_controller_node.py` | Turn-then-drive + MPC overlay |
| **Obstacle pub** | `~/hybrid_ws/src/hybrid_nav/hybrid_nav/nodes/obstacle_publisher_node.py` | Obstacle positions |
| **Launch** | `~/hybrid_ws/src/hybrid_nav/launch/turtlebot3_hybrid.launch.py` | All parameters |
| **World** | `~/hybrid_ws/src/hybrid_nav/worlds/hybrid_obstacle_world.sdf` | Gazebo arena |
| **RViz** | `~/hybrid_ws/src/hybrid_nav/rviz/hybrid_nav.rviz` | Visualization |

---

## Tunable Parameters

### In launch file (lines ~103-115):
| Parameter | Value | Effect |
|-----------|-------|--------|
| `trajectory_amplitude` | 0.5m | Figure-8 size |
| `trajectory_frequency` | 0.2 rad/s | Path speed (max v ≈ 0.14 m/s) |
| `v_max` | 0.22 m/s | TB3 Burger hardware limit |
| `use_hybrid` | True | Enable MPC for obstacle avoidance |

### In controller node (line ~84):
| Parameter | Value | Effect |
|-----------|-------|--------|
| `d_trigger` | **0.6m** | Distance at which blender activates |
| `d_safe` | 0.35m | Hard safety margin |
| `cruise_speed` | 0.10 m/s | Forward tracking speed |
| `turn_threshold` | 0.35 rad | When to turn-in-place vs drive |

### In obstacle publisher (lines ~32-36):
| Obstacle | Position (x, y) | Radius |
|----------|-----------------|--------|
| Red | (0.30, 0.35) | 0.10m |
| Orange | (-0.35, -0.35) | 0.10m |
| Blue | (0.55, 0.00) | 0.08m |

---

## Rebuild Command

```bash
cd ~/hybrid_ws
rm -rf build install log
source /opt/ros/jazzy/setup.bash
colcon build --packages-select hybrid_nav
source install/setup.bash
```

---

## Bug History

| # | Bug | Root Cause | Fix |
|---|-----|-----------|-----|
| 1 | Build fails | `--symlink-install` unsupported | Use plain `colcon build` |
| 2 | Import errors | Symlinks named `models_src` | Renamed to `models` |
| 3 | API mismatches | Wrong method signatures | Fixed all 3 |
| 4 | Robot doesn't move | `Twist` instead of `TwistStamped` | Changed to `TwistStamped` |
| 5 | Random movement | Trajectory 2x faster than TB3 | Reduced A=0.5, freq=0.2 |
| 6 | Back-and-forth | **AVOID(w=1.0) always on** | Moved obstacles off path, d_trigger=0.6 |

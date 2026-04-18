#!/usr/bin/env python3
"""Generate plot outputs for all trajectory families, checkpoint presets, and bounded moving scenarios."""

from __future__ import annotations

import json
import argparse
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
BATCH_DIR = OUTPUTS / "all_cases"


MODE_OUTPUT_FILES: Dict[str, List[str]] = {
    "lqr": [
        "lqr_tracking.png",
        "lqr_error.png",
        "lqr_control.png",
    ],
    "mpc": [
        "mpc_obstacle_avoidance.png",
        "mpc_error.png",
        "mpc_control.png",
    ],
    "hybrid": [
        "hybrid_trajectory.png",
        "hybrid_error.png",
        "hybrid_control.png",
        "hybrid_blending.png",
    ],
    "adaptive": [
        "adaptive_obstacle_avoidance.png",
        "adaptive_error.png",
        "adaptive_control.png",
        "adaptive_parameter_convergence.png",
    ],
    "hybrid_adaptive": [
        "hybrid_adaptive_trajectory.png",
        "hybrid_adaptive_error.png",
        "hybrid_adaptive_control.png",
        "hybrid_adaptive_blending.png",
        "hybrid_adaptive_parameter_convergence.png",
    ],
}


@dataclass
class Case:
    name: str
    mode: str
    duration: float
    trajectory: str
    checkpoint_preset: str
    scenario: str

    def to_command(self) -> List[str]:
        return [
            "python",
            "run_simulation.py",
            "--mode",
            self.mode,
            "--duration",
            str(self.duration),
            "--trajectory",
            self.trajectory,
            "--checkpoint-preset",
            self.checkpoint_preset,
            "--scenario",
            self.scenario,
        ]


def build_cases(duration: float) -> List[Case]:
    modes_all = ["lqr", "mpc", "hybrid", "adaptive", "hybrid_adaptive"]
    cases: List[Case] = []

    analytic_trajectories = ["figure8", "circle", "clover", "slalom"]
    checkpoint_presets = ["diamond", "slalom_lane", "warehouse", "corridor_turn"]

    for trajectory in analytic_trajectories:
        for mode in modes_all:
            cases.append(
                Case(
                    name=f"{mode}__traj-{trajectory}__scn-default",
                    mode=mode,
                    duration=duration,
                    trajectory=trajectory,
                    checkpoint_preset="diamond",
                    scenario="default",
                )
            )

    for preset in checkpoint_presets:
        for mode in modes_all:
            cases.append(
                Case(
                    name=f"{mode}__traj-checkpoint_path-{preset}__scn-default",
                    mode=mode,
                    duration=duration,
                    trajectory="checkpoint_path",
                    checkpoint_preset=preset,
                    scenario="default",
                )
            )

    dynamic_modes = ["mpc", "hybrid", "adaptive", "hybrid_adaptive"]
    dynamic_scenarios = ["moving", "random_walk"]

    for scenario in dynamic_scenarios:
        for trajectory in analytic_trajectories:
            for mode in dynamic_modes:
                cases.append(
                    Case(
                        name=f"{mode}__traj-{trajectory}__scn-{scenario}",
                        mode=mode,
                        duration=duration,
                        trajectory=trajectory,
                        checkpoint_preset="diamond",
                        scenario=scenario,
                    )
                )

        for preset in checkpoint_presets:
            for mode in dynamic_modes:
                cases.append(
                    Case(
                        name=f"{mode}__traj-checkpoint_path-{preset}__scn-{scenario}",
                        mode=mode,
                        duration=duration,
                        trajectory="checkpoint_path",
                        checkpoint_preset=preset,
                        scenario=scenario,
                    )
                )

    return cases


def copy_mode_outputs(mode: str, destination: Path) -> List[str]:
    destination.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []

    for filename in MODE_OUTPUT_FILES.get(mode, []):
        src = OUTPUTS / filename
        if src.exists():
            shutil.copy2(src, destination / filename)
            copied.append(filename)

    return copied


def run_case(case: Case) -> Dict[str, object]:
    start = time.time()
    command = case.to_command()

    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    case_dir = BATCH_DIR / case.name
    copied = []
    if completed.returncode == 0:
        copied = copy_mode_outputs(case.mode, case_dir)

    return {
        "case": asdict(case),
        "command": command,
        "returncode": completed.returncode,
        "elapsed_s": round(time.time() - start, 3),
        "copied_files": copied,
        "stdout_tail": completed.stdout[-1200:],
        "stderr_tail": completed.stderr[-1200:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duration",
        type=float,
        default=6.0,
        help="Simulation duration per case (seconds).",
    )
    args = parser.parse_args()

    duration = args.duration
    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    cases = build_cases(duration)
    records: List[Dict[str, object]] = []

    print(f"Running {len(cases)} cases...")
    for idx, case in enumerate(cases, start=1):
        print(f"[{idx:02d}/{len(cases)}] {case.name}")
        records.append(run_case(case))

    success = [r for r in records if r["returncode"] == 0]
    failed = [r for r in records if r["returncode"] != 0]

    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_cases": len(records),
        "succeeded": len(success),
        "failed": len(failed),
        "records": records,
    }

    manifest_path = BATCH_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nDone")
    print(f"Success: {len(success)}")
    print(f"Failed: {len(failed)}")
    print(f"Manifest: {manifest_path}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

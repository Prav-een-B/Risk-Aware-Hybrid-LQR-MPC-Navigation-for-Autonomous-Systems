#!/usr/bin/env python3
"""Collect logs, plots, evaluation tables, and ROS artifacts into one bundle."""

import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path


def sanitize_label(label: str) -> str:
    """Create a filesystem-safe artifact label."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", label.strip())
    return sanitized.strip("_") or "run"


def copy_source(source: Path, repo_root: Path, run_dir: Path) -> dict:
    """Copy one file or directory into the artifact bundle."""
    try:
        relative_name = source.relative_to(repo_root)
    except ValueError:
        relative_name = Path(source.name)

    destination = run_dir / relative_name
    destination.parent.mkdir(parents=True, exist_ok=True)

    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
        kind = "directory"
    else:
        shutil.copy2(source, destination)
        kind = "file"

    return {
        "source": str(source),
        "destination": str(destination),
        "type": kind,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("artifacts"),
        help="Root artifact directory",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="validation",
        help="Human-readable label for this collection pass",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="File or directory to copy into the bundle. Can be used multiple times.",
    )
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        help="Extra manifest metadata in key=value form. Can be used multiple times.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.destination / f"{timestamp}_{sanitize_label(args.label)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "timestamp": timestamp,
        "label": args.label,
        "repo_root": str(repo_root),
        "copied": [],
        "missing": [],
        "metadata": {},
    }

    for item in args.metadata:
        key, _, value = item.partition("=")
        if not key:
            continue
        manifest["metadata"][key] = value

    for raw_source in args.source:
        source = Path(raw_source)
        if not source.is_absolute():
            source = repo_root / source

        if not source.exists():
            manifest["missing"].append(str(source))
            continue

        manifest["copied"].append(copy_source(source, repo_root, run_dir))

    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

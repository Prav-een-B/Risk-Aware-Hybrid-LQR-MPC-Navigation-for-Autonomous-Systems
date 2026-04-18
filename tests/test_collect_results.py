import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_collect_results_builds_manifest_and_copies_sources(tmp_path):
    source_dir = tmp_path / "logs"
    source_dir.mkdir()
    (source_dir / "sample.log").write_text("ok", encoding="utf-8")

    destination = tmp_path / "artifacts"
    script = REPO_ROOT / "tools" / "collect_results.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--destination",
            str(destination),
            "--label",
            "pytest run",
            "--source",
            str(source_dir),
            "--metadata",
            "stage=unit_test",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    run_dir = Path(completed.stdout.strip())
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["metadata"]["stage"] == "unit_test"
    assert manifest["copied"]
    copied_log = run_dir / source_dir.name / "sample.log"
    assert copied_log.exists()

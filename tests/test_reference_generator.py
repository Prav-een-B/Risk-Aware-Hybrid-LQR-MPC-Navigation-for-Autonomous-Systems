import os
import sys

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src", "hybrid_controller")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from hybrid_controller.trajectory.reference_generator import ReferenceTrajectoryGenerator


def test_all_supported_trajectory_families_generate_finite_samples():
    for trajectory_type in ReferenceTrajectoryGenerator.TRAJECTORY_TYPES:
        kwargs = {
            "A": 2.0,
            "a": 0.5,
            "dt": 0.05,
            "trajectory_type": trajectory_type,
        }
        if trajectory_type == "checkpoint_path":
            kwargs["checkpoint_preset"] = "warehouse"

        generator = ReferenceTrajectoryGenerator(**kwargs)
        trajectory = generator.generate(4.0)

        assert trajectory.shape[1] == 6
        assert trajectory.shape[0] >= 2
        assert np.isfinite(trajectory).all()


def test_checkpoint_path_exposes_prediction_segments():
    generator = ReferenceTrajectoryGenerator(
        dt=0.05,
        trajectory_type="checkpoint_path",
        checkpoint_preset="warehouse",
    )
    generator.generate(5.0)

    x_ref, u_ref = generator.get_reference_at_index(0)
    x_seg, u_seg = generator.get_trajectory_segment(0, 6)

    assert x_ref.shape == (3,)
    assert u_ref.shape == (2,)
    assert x_seg.shape == (6, 3)
    assert u_seg.shape == (6, 2)

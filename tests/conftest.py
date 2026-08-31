import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def ensure_dataset():
    """
    Most tests exercise the pipeline end-to-end and need `dataset/normal.csv`
    / `dataset/attack.csv` to exist. The real SWaT dataset is NDA-gated, so
    CI (and any contributor without access to it) falls back to the
    synthetic generator. An existing dataset (real or previously generated)
    is left untouched.
    """
    normal_path = os.path.join(Config.DATASET_DIR, Config.NORMAL_DATA_FILE)
    attack_path = os.path.join(Config.DATASET_DIR, Config.ATTACK_DATA_FILE)
    if not (os.path.exists(normal_path) and os.path.exists(attack_path)):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        subprocess.run(
            [sys.executable, os.path.join(repo_root, "scripts", "generate_synthetic_dataset.py")],
            check=True,
            cwd=repo_root,
        )
    yield

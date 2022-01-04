import random
import subprocess
from pathlib import Path
from typing import Tuple

import numpy as np
import torch


def get_project_paths() -> Tuple[Path, Path, Path]:
    """Returns path to project, venv_python, and scripts"""
    project_path = Path(__file__).resolve().parent.parent
    venv_python = project_path / "venv" / "bin" / "python"
    scripts_path = project_path / "scripts"
    return project_path, venv_python, scripts_path


def get_git_hash():
    """Get short git hash, with "+" suffix if local files modified"""
    h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode("utf-8")

    # Add '+' suffix if local files are modified
    exitcode, _ = subprocess.getstatusoutput("git diff-index --quiet HEAD")
    if exitcode != 0:
        h += "+"
    return "git" + h


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    h = get_git_hash()
    print("git hash: ", h)

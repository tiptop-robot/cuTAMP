# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

import omegaconf
import yaml

from cutamp.config import TAMPConfiguration
from cutamp.envs import TAMPEnvironment
from cutamp.envs.utils import get_env_dict

_log = logging.getLogger(__name__)
_GIT_CWD = Path(__file__).parent


def _collect_git_info() -> dict:
    """Return git commit hash and dirty status for metadata."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_GIT_CWD,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=_GIT_CWD,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (FileNotFoundError, subprocess.CalledProcessError):
        _log.warning("Failed to collect git info", exc_info=True)
        return {"commit": None, "dirty": None}


def _get_git_diff() -> str | None:
    """Return the full git diff against HEAD, or None if unavailable or empty."""
    try:
        diff = subprocess.check_output(
            ["git", "diff", "HEAD"],
            cwd=_GIT_CWD,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return diff if diff else None
    except (FileNotFoundError, subprocess.CalledProcessError):
        _log.warning("Failed to get git diff", exc_info=True)
        return None


class _OmegaConfEncoder(json.JSONEncoder):
    """Encode OmegaConf objects for JSON serialization."""

    def default(self, obj):
        if isinstance(obj, (omegaconf.ListConfig, omegaconf.DictConfig)):
            return omegaconf.OmegaConf.to_container(obj, resolve=True)
        return super().default(obj)


class ExperimentLogger:
    """Simple experiment logger."""

    def __init__(self, name: str, config: TAMPConfiguration, experiment_dir: Optional[Path] = None):
        self.exp_dir = experiment_dir if experiment_dir is not None else Path(config.experiment_root) / name
        if self.exp_dir.exists():
            _log.warning(f"Experiment directory {self.exp_dir} already exists")
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        _log.info(f"Logging experiment to {self.exp_dir}")

        # Save the config
        with open(self.exp_dir / "config.yml", "w") as f:
            yaml.dump(config.__dict__, f, sort_keys=False)

        # Save git info and diff if dirty
        git_info = _collect_git_info()
        with open(self.exp_dir / "git_info.json", "w") as f:
            json.dump(git_info, f, indent=2)
        if git_info["dirty"]:
            diff = _get_git_diff()
            if diff:
                (self.exp_dir / "git.diff").write_text(diff, encoding="utf-8")

    def log_dict(self, name: str, data: dict) -> Path:
        path = self.exp_dir / f"{name}.json"
        if path.exists():
            raise ValueError(f"File {path} already exists")
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save as JSON, YAML is too slow to load
        with open(path, "w") as f:
            json.dump(data, f, indent=2, cls=_OmegaConfEncoder)
        _log.info(f"Logged {name} to {path}")
        return path

    def save_env(self, env: TAMPEnvironment, filename: str = "tamp_env.yml") -> Path:
        """Save the TAMP environment as a YAML file."""
        env_dict = get_env_dict(env)
        env_path = self.exp_dir / filename
        with open(env_path, "w") as f:
            yaml.dump(env_dict, f, sort_keys=False)
        _log.info(f"Saved environment to {env_path}")
        return env_path

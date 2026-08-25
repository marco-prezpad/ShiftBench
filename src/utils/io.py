"""
io.py

I/O utility functions.

Author: Marco Pérez Padilla
Date:   09-08-2026
"""

from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    """Load a YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist and return Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory

"""
test_io.py

Test suite that checks the functionality of the I/O utility functions.

Author: Marco Pérez Padilla
Date:   09-08-2026
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.utils.io import ensure_dir, load_yaml


class TestEnsureDir:
    def test_creates_directory_that_does_not_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            new_dir = Path(tmp) / "a" / "b" / "c"
            result = ensure_dir(new_dir)
            assert result == new_dir
            assert new_dir.exists()
            assert new_dir.is_dir()

    def test_succeeds_if_directory_already_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp)
            result = ensure_dir(existing)
            assert result == existing
            assert existing.exists()

    def test_returns_path_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = ensure_dir(tmp)
            assert isinstance(result, Path)


class TestLoadYAML:
    def test_loads_valid_yaml_file(self):
        data = {"key": "value", "list": [1, 2, 3]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            path = f.name

        try:
            loaded = load_yaml(path)
            assert loaded == data
        finally:
            Path(path).unlink()

    def test_loads_empty_yaml_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            path = f.name

        try:
            loaded = load_yaml(path)
            assert loaded is None
        finally:
            Path(path).unlink()

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_yaml("/tmp/does_not_exist_xyz123.yaml")

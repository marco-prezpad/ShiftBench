"""
test_protocol.py

Integration test for BenchmarkProtocol.

Author: Marco Pérez Padilla
Date:   10-08-2026
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.data.load import load_adult
from src.evaluation.protocol import BenchmarkProtocol

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture
def adult_sample_path():
    """Crea un subset pequeño del Adult para no saturar la GPU."""
    path = DATA_DIR / "adult.data"
    if not path.exists():
        pytest.skip("adult.data not found")
    df = load_adult(path).sample(n=300, random_state=42)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f, index=False)
        return f.name


class TestBenchmarkProtocol:
    def test_run_creates_outputs(self, adult_sample_path):
        with tempfile.TemporaryDirectory() as tmp:
            protocol = BenchmarkProtocol(
                data_path=adult_sample_path,
                results_dir=tmp,
                n_bootstrap=2,
                alphas=[0.0, 0.5],
                random_state=123,
            )
            protocol.run()

            assert (Path(tmp) / "scores.json").exists()
            assert (Path(tmp) / "metrics.csv").exists()


@pytest.fixture
def small_dataset_path():
    """Creates a small temporary CSV file for the benchmark."""
    df = load_adult(DATA_DIR / "adult.data").sample(n=300, random_state=42)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f, index=False)
        yield f.name
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def large_dataset_path():
    """Creates a temporary CSV with >1000 rows to trigger max_kernel_ref_size."""
    df = load_adult(DATA_DIR / "adult.data").sample(n=1500, random_state=42)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f, index=False)
        yield f.name
    Path(f.name).unlink(missing_ok=True)


def test_resume_from_checkpoint(small_dataset_path):
    """Interrumpt a run, then resume and verify it completes without re-running done alphas."""
    with tempfile.TemporaryDirectory() as tmp:
        protocol = BenchmarkProtocol(
            data_path=small_dataset_path,
            results_dir=tmp,
            n_bootstrap=2,
            alphas=[0.0, 0.5],
            random_state=123,
        )
        protocol.run()
        with open(Path(tmp) / "scores_partial.json", "w") as f:
            scores = {"mmd": {"0.0": [0.1, 0.2]}}
            json.dump(scores, f)

        resumed = BenchmarkProtocol(
            data_path=small_dataset_path,
            results_dir=tmp,
            n_bootstrap=2,
            alphas=[0.0, 0.5],
            random_state=123,
        )
        resumed.run()

        with open(Path(tmp) / "scores.json") as f:
            final_scores = json.load(f)
        assert "0.0" in final_scores["mmd"]
        assert "0.5" in final_scores["mmd"]
        assert not (Path(tmp) / "scores_partial.json").exists()


def test_force_flag(small_dataset_path):
    """Force flag should ignore existing checkpoint and start from scratch."""
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp) / "scores_partial.json"
        checkpoint.write_text('{"mmd": {"0.0": [0.1, 0.2]}}')

        protocol = BenchmarkProtocol(
            data_path=small_dataset_path,
            results_dir=tmp,
            n_bootstrap=2,
            alphas=[0.0, 0.5],
            random_state=123,
            force=True,
        )
        protocol.run()

        with open(Path(tmp) / "scores.json") as f:
            final_scores = json.load(f)
        assert len(final_scores["mmd"]["0.0"]) == 2
        assert len(final_scores["mmd"]["0.5"]) == 2


def test_all_alphas_already_completed(small_dataset_path):
    """If checkpoint has all alphas done, the protocol should exit early."""
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp) / "scores_partial.json"
        scores = {"mmd": {"0.0": [0.1, 0.2], "0.5": [0.3, 0.4]}}
        checkpoint.write_text(json.dumps(scores))

        protocol = BenchmarkProtocol(
            data_path=small_dataset_path,
            results_dir=tmp,
            n_bootstrap=2,
            alphas=[0.0, 0.5],
            random_state=123,
        )
        protocol.run()
        assert checkpoint.exists()


def test_max_kernel_ref_size(large_dataset_path):
    """Protocol runs with max_kernel_ref_size without memory errors."""
    with tempfile.TemporaryDirectory() as tmp:
        protocol = BenchmarkProtocol(
            data_path=large_dataset_path,
            results_dir=tmp,
            n_bootstrap=1,
            alphas=[0.0, 0.5],
            random_state=123,
            max_kernel_ref_size=200, 
        )
        protocol.run()
        assert (Path(tmp) / "scores.json").exists()
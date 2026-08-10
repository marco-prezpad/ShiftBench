"""
test_protocol.py

Integration test for BenchmarkProtocol.

Author: Marco Pérez Padilla
Date:   10-08-2026
"""

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
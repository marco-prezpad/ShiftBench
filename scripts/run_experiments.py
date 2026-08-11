#!/usr/bin/env python3
"""
run_experiments.py

Run the full ShiftBench benchmark using the configuration files.

Usage:
    python scripts/run_experiments.py

Author: Marco Pérez Padilla
Date:   11-08-2026
"""

import sys

from src.evaluation.protocol import BenchmarkProtocol
from src.utils.io import load_yaml

CONFIG_PATH = "configs/config.yaml"


def main() -> None:
    force = "--force" in sys.argv
    config = load_yaml(CONFIG_PATH)
    exp = config["experiment"]
    ds = config["datasets"]["adult"]

    protocol = BenchmarkProtocol(
        data_path=ds["file"],
        results_dir=config["output"]["results_dir"],
        random_state=exp["random_state"],
        reference_frac=exp.get("reference_frac", 0.5),
        subgroup_col=ds["shift"]["subgroup_col"],
        subgroup_value=ds["shift"]["subgroup_value"],
        alphas=ds["alphas"],
        n_bootstrap=exp["n_bootstrap_runs"],
        significance_level=exp["significance_level"],
        max_kernel_ref_size=exp.get("max_kernel_ref_size", 1000),
        force=force,
    )
    protocol.run()


if __name__ == "__main__":
    main()
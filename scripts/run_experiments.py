#!/usr/bin/env python3
"""
run_experiments.py

Run ShiftBench benchmark for a given domain.

Usage:
    python scripts/run_experiments.py [--domain adult|cifar10c|timeseries] [--force]

Author: Marco Pérez Padilla
Date:   17-08-2026
"""

import argparse

from src.evaluation.protocol import BenchmarkProtocol
from src.utils.io import load_yaml

CONFIG_PATH = "configs/config.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ShiftBench benchmark.")
    parser.add_argument("--domain", default="adult", choices=["adult", "cifar10c", "timeseries"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_yaml(CONFIG_PATH)
    exp = config["experiment"]

    if args.domain == "adult":
        ds = config["datasets"]["adult"]
        protocol = BenchmarkProtocol(
            domain="adult",
            data_path=ds["file"],
            results_dir=ds["results_dir"],
            random_state=exp["random_state"],
            reference_frac=exp.get("reference_frac", 0.5),
            subgroup_col=ds["shift"]["subgroup_col"],
            subgroup_value=ds["shift"]["subgroup_value"],
            alphas=ds["alphas"],
            n_bootstrap=exp["n_bootstrap_runs"],
            significance_level=exp["significance_level"],
            max_kernel_ref_size=exp.get("max_kernel_ref_size", 1000),
            force=args.force,
        )
    elif args.domain == "cifar10c":
        img = config.get("images", {})
        protocol = BenchmarkProtocol(
            domain="cifar10c",
            embeddings_dir=img.get("embeddings_dir", "embeddings/cifar10"),
            results_dir=img.get("results_dir", "results/cifar10c"),
            class_a=img.get("class_a", 0),
            class_b=img.get("class_b", 1),
            alphas=img.get("alphas", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            n_bootstrap=exp["n_bootstrap_runs"],
            significance_level=exp["significance_level"],
            max_kernel_ref_size=exp.get("max_kernel_ref_size", 1000),
            force=args.force,
        )
    elif args.domain == "timeseries":
        ts = config.get("timeseries", {})
        protocol = BenchmarkProtocol(
            domain="timeseries",
            results_dir=ts.get("results_dir", "results/timeseries"),
            series_n_samples=ts.get("series_n_samples", 1000),
            series_length=ts.get("series_length", 50),
            alphas=ts.get("alphas", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            n_bootstrap=exp["n_bootstrap_runs"],
            significance_level=exp["significance_level"],
            max_kernel_ref_size=exp.get("max_kernel_ref_size", 1000),
            force=args.force,
        )

    protocol.run()


if __name__ == "__main__":
    main()
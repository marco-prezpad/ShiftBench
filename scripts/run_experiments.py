#!/usr/bin/env python3
"""
run_experiments.py

Run ShiftBench benchmark for a given domain.

Usage:
    python scripts/run_experiments.py [--domain adult|cifar10c|timeseries|text] [--force]
                                       [--use-detector-params]

Author: Marco Pérez Padilla
Date:   17-08-2026
"""

import argparse

from src.evaluation.protocol import BenchmarkProtocol
from src.utils.io import load_yaml

CONFIG_PATH = "configs/config.yaml"
DETECTOR_PARAMS_PATH = "configs/detector_params.yaml"

# configs/config.yaml uses "embedding_drift" for readability; the detector
# registry (see src/detectors/factory.py) uses the shorter "embedding".
CONFIG_DETECTOR_NAME_TO_REGISTRY_NAME = {
    "mmd": "mmd",
    "lsdd": "lsdd",
    "kl": "kl",
    "embedding_drift": "embedding",
    "evidently": "evidently",
}


def enabled_detectors_from_config(config: dict) -> list[str] | None:
    """Read configs/config.yaml's `detectors.<name>.enabled` flags.

    Returns None (meaning "use BenchmarkProtocol's default list") if the
    config has no `detectors` section, so behavior is unchanged for
    configs that predate this option.
    """
    detectors_config = config.get("detectors")
    if not detectors_config:
        return None
    enabled = [
        CONFIG_DETECTOR_NAME_TO_REGISTRY_NAME[name]
        for name, settings in detectors_config.items()
        if settings.get("enabled", True) and name in CONFIG_DETECTOR_NAME_TO_REGISTRY_NAME
    ]
    return enabled or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ShiftBench benchmark.")
    parser.add_argument(
        "--domain", default="adult", choices=["adult", "cifar10c", "timeseries", "text"]
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--use-detector-params",
        action="store_true",
        help=(
            "Opt in to configs/detector_params.yaml hyperparameters. Off by "
            "default: some of those keys aren't accepted by every detector "
            "yet, and applying them can change results (see README_REFACTOR.md)."
        ),
    )
    args = parser.parse_args()

    config = load_yaml(CONFIG_PATH)
    experiment_config = config["experiment"]
    enabled_detectors = enabled_detectors_from_config(config)
    detector_params = load_yaml(DETECTOR_PARAMS_PATH) if args.use_detector_params else None

    common_kwargs = dict(
        n_bootstrap=experiment_config["n_bootstrap_runs"],
        significance_level=experiment_config["significance_level"],
        max_kernel_ref_size=experiment_config.get("max_kernel_ref_size", 1000),
        force=args.force,
        enabled_detectors=enabled_detectors,
        detector_params=detector_params,
    )

    if args.domain == "adult":
        dataset_config = config["datasets"]["adult"]
        protocol = BenchmarkProtocol(
            domain="adult",
            data_path=dataset_config["file"],
            results_dir=dataset_config["results_dir"],
            random_state=experiment_config["random_state"],
            reference_frac=experiment_config.get("reference_frac", 0.5),
            subgroup_col=dataset_config["shift"]["subgroup_col"],
            subgroup_value=dataset_config["shift"]["subgroup_value"],
            alphas=dataset_config["alphas"],
            **common_kwargs,
        )
    elif args.domain == "cifar10c":
        images_config = config.get("images", {})
        protocol = BenchmarkProtocol(
            domain="cifar10c",
            embeddings_dir=images_config.get("embeddings_dir", "embeddings/cifar10"),
            results_dir=images_config.get("results_dir", "results/cifar10c"),
            random_state=experiment_config["random_state"],
            class_a=images_config.get("class_a", 0),
            class_b=images_config.get("class_b", 1),
            alphas=images_config.get(
                "alphas", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            ),
            **common_kwargs,
        )
    elif args.domain == "timeseries":
        timeseries_config = config.get("timeseries", {})
        protocol = BenchmarkProtocol(
            domain="timeseries",
            results_dir=timeseries_config.get("results_dir", "results/timeseries"),
            random_state=experiment_config["random_state"],
            series_n_samples=timeseries_config.get("series_n_samples", 1000),
            series_length=timeseries_config.get("series_length", 50),
            alphas=timeseries_config.get(
                "alphas", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            ),
            **common_kwargs,
        )
    elif args.domain == "text":
        text_config = config.get("text", {})
        protocol = BenchmarkProtocol(
            domain="text",
            embeddings_dir=text_config.get("embeddings_dir", "embeddings/newsgroups"),
            results_dir=text_config.get("results_dir", "results/text"),
            random_state=experiment_config["random_state"],
            text_test_size=text_config.get("text_test_size", 3000),
            text_kl_n_components=text_config.get("text_kl_n_components", 50),
            alphas=text_config.get(
                "alphas", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            ),
            **common_kwargs,
        )

    protocol.run()


if __name__ == "__main__":
    main()
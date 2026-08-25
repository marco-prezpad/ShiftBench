"""
benchmark_protocol.py

Generic benchmark protocol for ShiftBench.
Supports domains: 'adult' (tabular), 'cifar10c' (image embeddings),
'timeseries' (synthetic series), and 'text' (TF-IDF embeddings).

All domain-specific data preparation, shift generation, and
detector-input formatting is delegated to a BaseDomainHandler obtained
from DomainHandlerFactory (see domain_handlers/), so this class only
knows how to drive the bootstrap loop, train detectors, handle GPU OOM
fallback, and persist results. 

Author: Marco Pérez Padilla
Date:   19-08-2026 (refactor)
"""
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.detectors.factory import DetectorFactory
from src.evaluation.metrics import compute_detection_metrics
from src.evaluation.protocol.domain_handlers import DomainHandlerFactory
from src.utils.io import ensure_dir
from src.utils.logging import setup_logging

# Import detectors to trigger registration with DetectorFactory.
import src.detectors.mmd_detector  # noqa: F401,E402
import src.detectors.lsdd_detector  # noqa: F401,E402
import src.detectors.kl_detector  # noqa: F401,E402
import src.detectors.embedding_drift_detector  # noqa: F401,E402
import src.detectors.evidently_detector  # noqa: F401,E402

logger = setup_logging()

# Silence alibi_detect warnings about GPU memory fragmentation and
# RuntimeWarnings from sklearn's GaussianMixture when fitting with small datasets.
logging.getLogger("alibi_detect.utils.pytorch.distance").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=RuntimeWarning)

DEFAULT_ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
DEFAULT_DETECTOR_NAMES = ["mmd", "lsdd", "kl", "embedding", "evidently"]


class TqdmHandler(logging.Handler):
    def emit(self, record):
        tqdm.write(self.format(record))


alibi_logger = logging.getLogger("alibi_detect")
alibi_logger.addHandler(TqdmHandler())
alibi_logger.setLevel(logging.INFO)
alibi_logger.propagate = False


class BenchmarkProtocol:
    """Run distribution shift benchmark for different data domains."""

    def __init__(
        self,
        domain: str = "adult",
        data_path: str | Path = "data/adult.data",
        embeddings_dir: str | Path = "embeddings/cifar10",
        results_dir: str | Path = "results",
        random_state: int = 42,
        reference_frac: float = 0.5,
        subgroup_col: str = "race",
        subgroup_value: str = "Black",
        noise_std: float = 0.5,
        alphas: list[float] | None = None,
        n_bootstrap: int = 30,
        significance_level: float = 0.05,
        max_kernel_ref_size: int = 1000,
        force: bool = False,
        cifar_test_size: int = 3000,
        cifar_kl_n_components: int = 50,
        class_a: int = 0,
        class_b: int = 1,
        series_n_samples: int = 1000,
        series_length: int = 50,
        text_test_size: int = 3000,
        text_kl_n_components: int = 50,
        enabled_detectors: list[str] | None = None,
        detector_params: dict[str, dict] | None = None,
    ):
        self.domain = domain
        self.results_dir = Path(results_dir)
        self.random_state = random_state
        self.n_bootstrap = n_bootstrap
        self.significance_level = significance_level
        self.max_kernel_ref_size = max_kernel_ref_size
        self.force = force
        self.alphas = alphas or DEFAULT_ALPHAS
        self.enabled_detectors = enabled_detectors or DEFAULT_DETECTOR_NAMES
        self.detector_params = detector_params or {}

        self.scores: dict[str, dict[float, list[float]]] = {}
        self.X_ref_numeric: np.ndarray | None = None
        self.X_pool_numeric: np.ndarray | None = None
        self._rng: np.random.Generator | None = None
        self._checkpoint_path = self.results_dir / "scores_partial.json"

        self._handler = DomainHandlerFactory.create(
            domain,
            data_path=data_path,
            embeddings_dir=embeddings_dir,
            random_state=random_state,
            reference_frac=reference_frac,
            subgroup_col=subgroup_col,
            subgroup_value=subgroup_value,
            noise_std=noise_std,
            max_kernel_ref_size=max_kernel_ref_size,
            cifar_test_size=cifar_test_size,
            cifar_kl_n_components=cifar_kl_n_components,
            class_a=class_a,
            class_b=class_b,
            series_n_samples=series_n_samples,
            series_length=series_length,
            text_test_size=text_test_size,
            text_kl_n_components=text_kl_n_components,
        )

    # Detector lifecycle
    def _init_detectors(self) -> dict:
        """Create detector instances with appropriate device/kwargs."""
        detectors = {}
        for name in self.enabled_detectors:
            explicit_kwargs = {}
            if name in ("mmd", "lsdd"):
                explicit_kwargs["device"] = "cuda" if torch.cuda.is_available() else "cpu"
            if name == "embedding" and self.domain in ("cifar10c", "text"):
                explicit_kwargs["n_components"] = 64
            # Optional hyperparameters from configs/detector_params.yaml (opt-in).
            # explicit_kwargs always wins so this never changes default behavior.
            extra_params = self.detector_params.get(name, {})
            combined_params = {**extra_params, **explicit_kwargs}
            detectors[name] = DetectorFactory.create_from_config(name, combined_params)
        return detectors

    def _fit_detector(self, name: str, detector) -> None:
        """Fit a single detector on the representation its family expects."""
        if name == "evidently":
            detector.fit(self._handler.reference_evidently)
        elif name == "kl":
            detector.fit(self._handler.reference_kl)
        elif name in ("mmd", "lsdd"):
            detector.fit(self._kernel_detector_reference())
        else:
            detector.fit(self.X_ref_numeric)

    def _kernel_detector_reference(self) -> np.ndarray:
        """Reference sample for MMD/LSDD, subsampled to max_kernel_ref_size."""
        X_ref = self.X_ref_numeric
        if len(X_ref) > self.max_kernel_ref_size:
            idx = self._rng.choice(len(X_ref), size=self.max_kernel_ref_size, replace=False)
            return X_ref[idx]
        return X_ref

    def _score_detector(
        self, name: str, detector, X_test_numeric, X_test_evidently, X_test_kl
    ) -> float:
        """Score a single detector on the representation its family expects."""
        if name == "evidently":
            return detector.score(X_test_evidently)
        elif name == "kl":
            return detector.score(X_test_kl)
        else:
            return detector.score(X_test_numeric)

    def _fit_all_detectors(self, detectors: dict) -> None:
        """Fit every detector, falling back to CPU on GPU OOM."""
        for name, detector in list(detectors.items()):
            try:
                self._fit_detector(name, detector)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                logger.warning(f"GPU OOM for {name}, falling back to CPU")
                detectors[name] = DetectorFactory.create_from_config(name, {"device": "cpu"})
                self._fit_detector(name, detectors[name])

    def _score_with_oom_fallback(
        self, name: str, detectors: dict, X_test_numeric, X_test_evidently, X_test_kl
    ) -> float:
        """Score a detector, retraining it on CPU and retrying once on GPU OOM."""
        try:
            return self._score_detector(
                name, detectors[name], X_test_numeric, X_test_evidently, X_test_kl
            )
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            logger.warning(f"GPU OOM during score for {name}. Switching to CPU for this detector.")
            torch.cuda.empty_cache()
            detectors[name] = DetectorFactory.create_from_config(name, {"device": "cpu"})
            self._fit_detector(name, detectors[name])
            return self._score_detector(
                name, detectors[name], X_test_numeric, X_test_evidently, X_test_kl
            )

    # Checkpointing
    def _load_checkpoint(self) -> set[float]:
        """Load a partial checkpoint (if any) and return the set of finished alphas."""
        completed_alphas: set[float] = set()
        if self._checkpoint_path.exists() and not self.force:
            with open(self._checkpoint_path) as f:
                raw = json.load(f)
            self.scores = {
                det: {float(a): v for a, v in alpha_dict.items()}
                for det, alpha_dict in raw.items()
            }
            first_detector = next(iter(self.scores.keys()), None)
            if first_detector:
                for alpha, alpha_scores in self.scores[first_detector].items():
                    if len(alpha_scores) >= self.n_bootstrap:
                        completed_alphas.add(float(alpha))
            if completed_alphas:
                logger.info(
                    f"Resuming from checkpoint. {len(completed_alphas)} alpha(s) already done."
                )
        elif self.force:
            logger.info("Force flag set – starting from scratch.")
        return completed_alphas

    def _save_alpha_checkpoint(self, alpha: float) -> None:
        alpha_dir = self.results_dir / "alpha_scores"
        alpha_dir.mkdir(parents=True, exist_ok=True)
        alpha_file = alpha_dir / f"alpha_{alpha:.2f}.json"
        alpha_scores = {det: {alpha: self.scores[det][alpha]} for det in self.scores}
        with open(alpha_file, "w") as f:
            json.dump(alpha_scores, f, indent=2)

    def _save_partial_metrics(self) -> None:
        first_detector = next(iter(self.scores), None)
        if first_detector and 0.0 in self.scores[first_detector]:
            completed_alphas = [a for a in self.alphas if a in self.scores[first_detector]]
            partial_metrics = compute_detection_metrics(
                self.scores,
                completed_alphas,
                self.n_bootstrap,
                threshold_percentile=(1 - self.significance_level) * 100,
            )
            partial_metrics.to_csv(self.results_dir / "metrics_partial.csv", index=False)

    def _save_global_checkpoint(self) -> None:
        with open(self._checkpoint_path, "w") as f:
            json.dump(self.scores, f, indent=2)

    def _save_final_results(self) -> None:
        with open(self.results_dir / "scores.json", "w") as f:
            json.dump(self.scores, f, indent=2)

        metrics_df = compute_detection_metrics(
            self.scores,
            self.alphas,
            self.n_bootstrap,
            threshold_percentile=(1 - self.significance_level) * 100,
        )
        metrics_df.to_csv(self.results_dir / "metrics.csv", index=False)

        if self._checkpoint_path.exists():
            self._checkpoint_path.unlink()

    # Main execution
    def _run_alpha(self, alpha: float, detectors: dict) -> None:
        """Run every bootstrap iteration for a single shift intensity."""
        tqdm.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} Running alpha={alpha}")
        for name in detectors:
            self.scores.setdefault(name, {}).setdefault(alpha, [])

        for run_id in tqdm(range(self.n_bootstrap), desc=f"  α={alpha}", leave=False):
            seed = self.random_state + run_id
            X_test_shifted, _ = self._handler.generate_shifted_test_set(alpha=alpha, seed=seed)

            X_test_numeric = self._handler.build_numeric_test_view(X_test_shifted)
            X_test_evidently = self._handler.build_evidently_test_view(X_test_shifted)
            X_test_kl = self._handler.build_kl_test_view(X_test_shifted)

            for name in detectors:
                score = self._score_with_oom_fallback(
                    name, detectors, X_test_numeric, X_test_evidently, X_test_kl
                )
                self.scores[name][alpha].append(score)

        self._save_alpha_checkpoint(alpha)
        self._save_partial_metrics()
        self._save_global_checkpoint()

    def run(self) -> None:
        logger.info(f"Starting benchmark protocol for domain '{self.domain}'...")
        ensure_dir(self.results_dir)
        ensure_dir(self.results_dir / "figures")

        final_metrics_path = self.results_dir / "metrics.csv"
        if final_metrics_path.exists() and not self.force:
            logger.info("metrics.csv already exists. Skipping (use --force to re-run).")
            return

        completed_alphas = self._load_checkpoint()
        remaining_alphas = [a for a in self.alphas if a not in completed_alphas]
        if not remaining_alphas:
            logger.info("All alphas already completed.")
            return

        self.X_ref_numeric, self.X_pool_numeric = self._handler.prepare_reference_and_pool()
        logger.info(
            f"Reference shape: {self.X_ref_numeric.shape}, Pool shape: {self.X_pool_numeric.shape}"
        )

        self._rng = np.random.default_rng(self.random_state)
        detectors = self._init_detectors()
        self._fit_all_detectors(detectors)

        for alpha in tqdm(remaining_alphas, desc="Alphas"):
            self._run_alpha(alpha, detectors)

        self._save_final_results()
        logger.info("Benchmark finished.")
        logger.info(f"Results saved to {self.results_dir}")

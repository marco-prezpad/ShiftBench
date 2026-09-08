
# ShiftBench

Benchmark for comparing unsupervised distribution-shift detectors under
controlled shift intensities, across four different data domains.

Author: Marco Pérez Padilla


## Table of contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Data and embeddings](#data-and-embeddings)
- [Configuration](#configuration)
- [Usage](#usage)
- [Architecture](#architecture)
- [How to extend the benchmark](#how-to-extend-the-benchmark)
- [Tests](#tests)
- [Results layout](#results-layout)
- [Project structure](#project-structure)


## Introduction

ShiftBench measures, for five unsupervised drift detectors (MMD, LSDD,
KL, a projection + distance-based detector, and Evidently AI), how well
each one detects a distribution shift whose intensity is controlled by
a parameter `alpha ∈ [0, 1]`:

- `alpha = 0.0` → test set drawn from the same distribution as the
  reference (no detector should fire).
- `alpha = 1.0` → the maximum shift defined for that domain.
- Intermediate values interpolate between the two extremes.

For each `alpha`, `n_bootstrap` independent resamples are run, each
detector's score is computed, and detection metrics (TPR, FPR, AUC-TPR,
AUC-ROC...) are aggregated against the threshold calibrated at
`alpha = 0`.

The benchmark runs on four domains, each with its own type of synthetic
shift:

| Domain       | Data                                     | Shift type                                    |
|--------------|-------------------------------------------|-------------------------------------------------|
| `adult`      | UCI Adult (tabular)                       | Over-representation of a demographic subgroup   |
| `cifarc`   | ResNet18 embeddings of CIFAR-10           | Mixture of two classes                          |
| `timeseries` | Synthetic series (sine vs. square wave)   | Mixture of two classes                          |
| `text`       | TF-IDF of 20 Newsgroups (2 categories)    | Mixture of two classes                          |


## Installation

Requires Python 3.11+.

```bash
git clone <repo>
cd shiftbench
pip install -e .
```

Main dependencies: `numpy`, `pandas`, `scikit-learn`, `scipy`, `torch`,
`alibi-detect`, `evidently`, `matplotlib`, `seaborn`, `pyyaml`, `tqdm`.
All of them are pinned in `pyproject.toml`.

`torch` uses a GPU automatically if one is available
(`torch.cuda.is_available()`); otherwise it falls back to CPU with no
extra configuration needed.

For development (tests, lint):

```bash
pip install -e ".[dev]"
```

## Docker

Recommended alternativo to reproduce the exact same environment:

```bash
docker compose up -d shiftbench
docker compose exec shiftbench python scripts/run_experiments.py --domain adult
```

For Jupyter support:

```bash
docker compose up -d jupyter
```

Access `http://localhost:8888`

> By using this alternative, you must run all commands written in this file inside the docker itself as follows: `docker compose exec shiftbench <command>`


## Data and embeddings

Each domain needs its own preparation step before the benchmark can run.

**UCI Adult** (`adult` domain):

```bash
python scripts/download_datasets.py
```

Downloads `data/adult.data`, `data/adult.test`, and `data/adult.names`.

**CIFAR-10** (`cifarc` domain):

```bash
python scripts/extract_cifar10_embeddings.py
```

Downloads CIFAR-10 via `torchvision` (if not already in `data/`) and
extracts embeddings with an ImageNet-pretrained ResNet18, saving them
to `embeddings/cifar10/`.

**20 Newsgroups** (`text` domain):

```bash
python scripts/extract_newsgroups_embeddings.py
```

Downloads the two-category subset (`rec.sport.baseball`, `sci.med`),
vectorizes it with TF-IDF (500 features), and saves it to
`embeddings/newsgroups/`.

**Synthetic time series** (`timeseries` domain): no download needed —
generated synthetically at benchmark run time.


## Configuration

All configuration lives under `configs/`:

- **`config.yaml`** — the single source of truth for running
  experiments: global parameters (`experiment`: random seed, number of
  bootstrap runs, significance level...), and one section per domain
  (`datasets.adult`, `images`, `timeseries`, `text`) with its data /
  embeddings path, results directory, and the grid of `alphas` to
  evaluate. It also has a `detectors` section to enable/disable
  detectors by name (`enabled: true/false`).

- **`detector_params.yaml`** — optional per-detector hyperparameters
  (e.g. `kernel`, `sigma` for MMD/LSDD). This is **opt-in**: by default
  nothing here is applied, so as not to change already-calibrated
  results. It's enabled explicitly with `--use-detector-params` (see
  below). Any key a detector's constructor doesn't accept is silently
  ignored.


## Usage

**Run the benchmark** for a domain:

```bash
python scripts/run_experiments.py --domain adult
python scripts/run_experiments.py --domain cifarc
python scripts/run_experiments.py --domain timeseries
python scripts/run_experiments.py --domain text
```

Available flags:

- `--force` — ignore any previous results/checkpoints and start over
  from scratch.
- `--use-detector-params` — apply `configs/detector_params.yaml` (see
  above).

> Selective re-runs: if you disable one or more detectors in config.yaml and run with --force, ShiftBench will only re-run the enabled detectors. The scores and calibration of the disabled detectors are preserved from the previous scores.json and calibration_scores.json, so you don't lose results for detectors you didn't want to recompute.

The benchmark is **resumable**: if interrupted, it saves a checkpoint
after every `alpha` (`results/<domain>/scores_partial.json`) and, when
relaunched, picks up where it left off instead of redoing work. If
`results/<domain>/metrics.csv` already exists, nothing is re-run unless
`--force` is passed.

**Generate figures** for an already-evaluated domain:

```bash
python scripts/generate_figures.py --domain adult
```

Generates, under `results/<domain>/figures/`: TPR vs. alpha per
detector, AUC-TPR bar chart, score distribution under H0, and FPR
calibration against the nominal significance level.

```bash
python scripts/paper_figures.py
```

Generates the combined 2×2 TPR figure and the non-monotonic score
figure used in the paper, under results/figures/.

## Architecture

```
src/
├── data/
│   ├── loaders/         # how raw data is loaded for each source
│   └── shifts/           # how each type of synthetic shift is generated
├── detectors/             # one detector per file + registration factory
└── evaluation/
    ├── metrics.py          # TPR/FPR/AUC computation from raw scores
    ├── visualize/           # figure generation
    └── protocol/
        ├── benchmark_protocol.py     # orchestrator: bootstrap loop, checkpoints, CPU fallback
        └── domain_handlers/           # one handler per domain + registration factory
```

The design leans on two ideas that repeat on purpose:

### 1. Factory + decorator-based registration

Both detectors and domains register themselves this way, instead of
keeping an `if/elif` with the supported names:

```python
@DetectorFactory.register("mmd")
class MMDDetector(BaseDetector):
    ...

detector = DetectorFactory.create("mmd", device="cpu")
```

```python
@DomainHandlerFactory.register("cifarc")
class Cifar10cDomainHandler(BaseDomainHandler):
    ...

handler = DomainHandlerFactory.create("cifarc", embeddings_dir=...)
```

To see what detectors or domains exist at runtime:
`DetectorFactory.available()` / `DomainHandlerFactory.available()`.

### 2. `BenchmarkProtocol` knows nothing about any specific domain or detector

`benchmark_protocol.py` orchestrates the bootstrap loop, checkpoint
handling, and the CPU fallback when the GPU runs out of memory — but it
delegates to:

- **`BaseDomainHandler`** (the interface each domain implements) for
  everything data-specific: how it's loaded, how the shift is
  generated, and what representation each detector family needs
  (`build_numeric_test_view`, `build_evidently_test_view`,
  `build_kl_test_view`).
- **`DetectorFactory`** to instantiate detectors by name.

### Flow of a run

1. `run_experiments.py` reads `config.yaml`, assembles the chosen
   domain's arguments, and creates a `BenchmarkProtocol`.
2. `BenchmarkProtocol` asks `DomainHandlerFactory` for the domain's
   handler, and `DetectorFactory` for the enabled detectors.
3. The handler prepares the reference and pool data
   (`prepare_reference_and_pool`).
4. Each detector is trained on the representation it needs
   (`reference_evidently`, `reference_kl`, or the general numeric
   view).
5. For each `alpha` and each bootstrap repetition: the handler
   generates a shifted test set and builds the three views the
   detectors need; each detector scores that view.
6. After each `alpha`, a checkpoint and partial metrics are saved.
7. Once every `alpha` is done, final metrics are computed
   (`compute_detection_metrics`) and `scores.json` + `metrics.csv` are
   saved.


## How to extend the benchmark

**Adding a new detector:**

1. Create `src/detectors/my_detector.py` with a class that inherits
   from `BaseDetector` (implementing `fit` and `score`) and decorate it
   with `@DetectorFactory.register("my_detector")`.
2. Import it once (so the decorator runs) alongside the other detector
   imports in `benchmark_protocol.py`.
3. Add its enabled flag to the detectors section of config.yaml
   (or include it in the default detector list if you prefer to run it
   everywhere).

**Adding a new domain:**

1. Create
   `src/evaluation/protocol/domain_handlers/my_domain_handler.py` with
   a class that inherits from `BaseDomainHandler` and decorate it with
   `@DomainHandlerFactory.register("my_domain")`.
2. Implement `prepare_reference_and_pool`, `generate_shifted_test_set`,
   `build_numeric_test_view`, `build_evidently_test_view`, and
   `reference_evidently`. `reference_kl` / `build_kl_test_view` are
   optional — if not overridden, they reuse the general numeric view
   (see `AdultDomainHandler` or `TimeseriesDomainHandler` as an example
   of a domain that doesn't need PCA before KL).
3. Add the registration import to `domain_handlers/__init__.py`.
4. Add the matching branch in `scripts/run_experiments.py` to read its
   section of `config.yaml`.

## Tests

```bash
pytest
```

The `tests/` layout mirrors `src/` (`tests/data`, `tests/detectors`,
`tests/evaluation`, `tests/utils`). Tests that depend on downloaded data
(`data/adult.data`, `embeddings/...`) are automatically skipped
(`pytest.skip`) if those files aren't present, instead of failing.


## Results layout

Each domain writes to `results/<domain>/`:

```
results/<domain>/
├── scores.json           # raw score for each detector, per alpha and repetition
├── scores_partial.json   # checkpoint (deleted once a run finishes successfully)
├── metrics.csv            # TPR/FPR/AUC-TPR/AUC-ROC/... per detector and alpha
├── metrics_partial.csv    # metrics computed after the last completed alpha
├── alpha_scores/           # one JSON file per alpha, for incremental inspection
└── figures/                 # PNGs generated by generate_figures.py
```


## Project structure

```
configs/
    config.yaml            # per-domain experiment configuration
    detector_params.yaml   # optional detector hyperparameters (opt-in)
notebooks/                 # Jupyter notebooks. EDA
scripts/
    download_datasets.py             # downloads UCI Adult
    extract_cifar10_embeddings.py    # downloads CIFAR-10 and extracts ResNet18 embeddings
    extract_newsgroups_embeddings.py # downloads 20 Newsgroups and vectorizes with TF-IDF
    run_experiments.py               # entry point for running the benchmark
    generate_figures.py              # entry point for generating figures
    paper_figures.py                 # combined TPR figure + non-monotonic score figure
src/
    data/
        loaders/            # raw data loading (adult, cifar10)
        shifts/               # synthetic shift generators
    detectors/               # MMD, LSDD, KL, embedding, Evidently + factory
    evaluation/
        metrics.py
        visualize/
        protocol/
            benchmark_protocol.py
            domain_handlers/
    utils/                   # logging, I/O (YAML, directories)
tests/                       # mirrors src/
README.md                    # this file
```
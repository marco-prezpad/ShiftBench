#!/usr/bin/env python3
"""
download_datasets.py

Download all datasets required by ShiftBench.

Usage:
    python scripts/download_datasets.py

Author: Marco Pérez Padilla
Date:   02-08-2026
"""
import sys
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

ADULT_FILES = {
    "adult.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    "adult.test": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
    "adult.names": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names",
}


def download_file(url: str, dest: Path) -> None:
    """Download a file from url to dest with progress feedback."""
    print(f"  Downloading {dest.name}...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        print("OK")
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)


def main() -> None:
    print(f"Creating data directory at {DATA_DIR}...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading UCI Adult dataset...")
    for filename, url in ADULT_FILES.items():
        dest = DATA_DIR / filename
        if dest.exists():
            print(f"  {filename} already exists, skipping.")
            continue
        download_file(url, dest)
    print("  UCI Adult done.")

    print("Note: CIFAR-10-C will be auto-downloaded by torchvision on first use.")
    print("Note: NLP dataset must be downloaded separately (see README).")
    print("Note: Time series dataset must be downloaded separately (see README).")

    print(f"\nAll available datasets downloaded to {DATA_DIR}")


if __name__ == "__main__":
    main()
"""
detector_styles.py

Per-detector plotting styles for ShiftBench figures.

Colors are all CSS/matplotlib named colors -- no hex codes -- so the
palette stays readable and easy to tweak by name.

Author: Marco Pérez Padilla
Date:   11-08-2026
"""

DETECTOR_STYLES = {
    "mmd": {"color": "steelblue", "marker": "o", "linestyle": "-"},
    "lsdd": {"color": "orange", "marker": "s", "linestyle": "--"},
    "kl": {"color": "green", "marker": "^", "linestyle": "-."},
    "embedding": {"color": "red", "marker": "D", "linestyle": ":"},
    "evidently": {"color": "purple", "marker": "v", "linestyle": "-"},
}

"""First-run settings and help shared by the notebook and command-line workflow."""

import os
import struct
import sys
from importlib import metadata
from pathlib import Path


ANALYSIS_PINS = {
    "numpy": "2.3.5",
    "scipy": "1.17.0",
    "pandas": "2.2.3",
    "scikit-learn": "1.8.0",
    "statsmodels": "0.14.6",
    "h5py": "3.16.0",
    "matplotlib": "3.10.8",
    "Pillow": "13.0.0",
    "threadpoolctl": "3.6.0",
}
NOTEBOOK_PINS = {
    "ipykernel": "7.4.0",
    "nbformat": "5.11.1",
    "nbclient": "0.11.0",
    "jupyterlab": "4.6.6",
}


def check_environment():
    """Diagnose missing libraries before downloading any expression data."""
    # The analyses already use two BLAS threads. Give joblib the same default
    # on managed hosts that cannot report a physical core count.
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
    if sys.version_info < (3, 11) or struct.calcsize("P") < 8:
        raise RuntimeError("Use 64-bit Python 3.12 for this analysis.")
    if sys.flags.optimize:
        raise RuntimeError("Start Python without -O so the scientific checks run.")
    pins = ANALYSIS_PINS | NOTEBOOK_PINS
    installed = {}
    for name in pins:
        try:
            installed[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed[name] = "unavailable"
    missing = [name for name in ANALYSIS_PINS if installed[name] == "unavailable"]
    if missing:
        packages = " ".join(f"{name}=={pins[name]}" for name in ANALYSIS_PINS)
        raise RuntimeError(
            "Missing analysis packages: "
            + ", ".join(missing)
            + "\nIn a terminal for this environment run: python -m pip install -r requirements.txt"
            + "\nWith only the notebook, use a new cell: %pip install "
            + packages
            + "\nRestart the kernel after installation."
        )
    different = [
        f"{n} {installed[n]} (reference {v})" for n, v in ANALYSIS_PINS.items() if installed[n] != v
    ]
    if different or sys.version_info[:2] != (3, 12):
        print(
            "Reference environment: 64-bit Python 3.12. Results will be checked at manuscript precision."
        )
        if different:
            print("Different package versions: " + ", ".join(different))
    return installed


def resolve_paths(base, input_dir=None, output_dir=None):
    """Explicit folder options take precedence over environment variables."""
    cache = (
        Path(input_dir or os.environ.get("GHIR_INPUTS") or base / "inputs").expanduser().resolve()
    )
    output = (
        Path(output_dir or os.environ.get("GHIR_OUTPUT") or base / "GHIR_reproduction")
        .expanduser()
        .resolve()
    )
    return cache, output


def offline_mode(override=None):
    if override is not None:
        return override
    value = os.environ.get("GHIR_OFFLINE", "").strip().lower()
    if value not in ("", "0", "false", "no", "1", "true", "yes"):
        raise ValueError("Set GHIR_OFFLINE to 1 (offline) or 0 (download missing data).")
    return value in ("1", "true", "yes")


def display_path(path):
    """Use short relative paths when the run is inside the current directory."""
    try:
        return str(Path(path).resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)

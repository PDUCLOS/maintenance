"""NASA CMAPSS dataset loader.

Source: NASA Ames Prognostics Data Repository
        https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

The dataset contains multivariate sensor readings from a fleet of turbofan
engines, used to predict Remaining Useful Life (RUL). Four sub-datasets
(FD001..FD004) vary in operating conditions and fault modes.

Layout of train_FD001.txt (space-separated, no header):
    unit_nr, time_cycles, op_setting_1, op_setting_2, op_setting_3,
    sensor_01, sensor_02, ..., sensor_21

This loader raises a clear error if the dataset is missing — no fabricated
data, no fallback to "demo" rows. The rule is: real data or nothing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import settings
from src.utils.logger import logger

# Canonical column names (from CMAPSS readme.txt)
COLUMN_NAMES: list[str] = [
    "unit_nr",
    "time_cycles",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
    "sensor_01",
    "sensor_02",
    "sensor_03",
    "sensor_04",
    "sensor_05",
    "sensor_06",
    "sensor_07",
    "sensor_08",
    "sensor_09",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21",
]

SUBSETS: tuple[str, ...] = ("FD001", "FD002", "FD003", "FD004")


def expected_files(subset: str) -> dict[str, Path]:
    """Return the expected (train, test, rul) paths for a given subset."""
    if subset not in SUBSETS:
        raise ValueError(f"Unknown CMAPSS subset: {subset}. Expected one of {SUBSETS}.")
    base = settings.cmapss_dir
    return {
        "train": base / f"train_{subset}.txt",
        "test": base / f"test_{subset}.txt",
        "rul": base / f"RUL_{subset}.txt",
    }


def assert_cmapss_present(subset: str | None = None) -> None:
    """Raise FileNotFoundError if the requested CMAPSS files are missing.

    This is called before any parsing — it prevents the pipeline from
    silently producing empty indexes when the user forgot to download data.
    """
    subsets = (subset,) if subset else SUBSETS
    missing: list[str] = []
    for s in subsets:
        for kind, path in expected_files(s).items():
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            "NASA CMAPSS files are missing. Did you run 'make data'?\n"
            "  Missing files:\n    - " + "\n    - ".join(missing)
        )


def load_train(subset: str) -> pd.DataFrame:
    """Load a CMAPSS training file into a typed DataFrame.

    Returns a DataFrame indexed by (unit_nr, time_cycles).
    """
    raise NotImplementedError(
        f"CMAPSS loader: to be implemented in W1 (load {subset} train file, "
        "apply COLUMN_NAMES, set (unit_nr, time_cycles) index)."
    )


def load_test(subset: str) -> pd.DataFrame:
    """Load a CMAPSS test file. The RUL for the last cycle of each unit
    is in the corresponding RUL_{subset}.txt file (loaded separately)."""
    raise NotImplementedError(
        f"CMAPSS loader: to be implemented in W1 (load {subset} test file)."
    )


def load_rul(subset: str) -> pd.Series:
    """Load the ground-truth Remaining Useful Life per unit for the test set.

    Returns a Series indexed by unit_nr, values = remaining cycles at last
    observed time_cycles in the test set.
    """
    raise NotImplementedError(
        f"CMAPSS loader: to be implemented in W1 (load {subset} RUL file)."
    )


def discover_readme() -> Path | None:
    """Return the path to the CMAPSS readme.txt if it exists, else None.

    The readme is valuable for the RAG knowledge base: it explains what
    each sensor measures, the operating conditions, and the data splits.
    """
    candidate = settings.cmapss_dir / "readme.txt"
    if candidate.is_file():
        return candidate
    logger.warning("CMAPSS readme.txt not found at {}", candidate)
    return None


__all__ = [
    "COLUMN_NAMES",
    "SUBSETS",
    "expected_files",
    "assert_cmapss_present",
    "load_train",
    "load_test",
    "load_rul",
    "discover_readme",
]

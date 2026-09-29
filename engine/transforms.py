"""
Feature-engineering transforms — all from scratch.
Each function returns a NEW Dataset (immutable pattern).
"""

from __future__ import annotations

import math
from typing import Any

from .loader import DataColumn, Dataset
from .statistics import mean as _mean, std_dev as _std


def _clone_dataset(ds: Dataset) -> Dataset:
    return ds.clone()


def _find_col(ds: Dataset, name: str) -> DataColumn:
    col = ds.get_column(name)
    if col is None:
        raise ValueError(f"Column '{name}' not found in dataset.")
    return col


def log_transform(ds: Dataset, col_name: str, base: str = "ln") -> Dataset:
    """
    Logarithmic transform: ln, log2, log10, or log1p (ln(1+x)).
    Non-positive values become None (undefined for log).
    """
    out = _clone_dataset(ds)
    col = _find_col(out, col_name)
    if col.dtype != "numeric":
        raise ValueError(f"Log transform requires numeric column, got '{col.dtype}'.")

    log_fn = {
        "ln": math.log,
        "log2": math.log2,
        "log10": math.log10,
        "log1p": math.log1p,
    }.get(base)

    if log_fn is None:
        raise ValueError(
            f"Unsupported log base: '{base}'. Use ln, log2, log10, or log1p."
        )

    new_name = f"{col_name}_{base}"
    new_values: list = []
    missing = 0
    for v in col.values:
        if v is None:
            new_values.append(None)
            missing += 1
        elif base == "log1p":
            if v < -1:
                new_values.append(None)
                missing += 1
            else:
                new_values.append(log_fn(v))
        else:
            if v <= 0:
                new_values.append(None)
                missing += 1
            else:
                new_values.append(log_fn(v))

    out.columns.append(
        DataColumn(
            name=new_name,
            dtype="numeric",
            values=new_values,
            missing_count=missing,
        )
    )
    return out


def normalize_min_max(ds: Dataset, col_name: str) -> Dataset:
    """x_norm = (x - min) / (max - min) -> [0, 1]"""
    out = _clone_dataset(ds)
    col = _find_col(out, col_name)
    if col.dtype != "numeric":
        raise ValueError("Min-max normalization requires a numeric column.")

    valid = [v for v in col.values if v is not None]
    if not valid:
        raise ValueError("Column has no valid numeric values.")

    lo = min(valid)
    hi = max(valid)
    rng = hi - lo

    new_values: list = []
    missing = 0
    for v in col.values:
        if v is None:
            new_values.append(None)
            missing += 1
        elif rng == 0:
            new_values.append(0.0)
        else:
            new_values.append((v - lo) / rng)

    out.columns.append(
        DataColumn(
            name=f"{col_name}_norm",
            dtype="numeric",
            values=new_values,
            missing_count=missing,
        )
    )
    return out


def standardize_zscore(ds: Dataset, col_name: str) -> Dataset:
    """z = (x - mean) / std"""
    out = _clone_dataset(ds)
    col = _find_col(out, col_name)
    if col.dtype != "numeric":
        raise ValueError("Z-score standardization requires a numeric column.")

    valid = [v for v in col.values if v is not None]
    if not valid:
        raise ValueError("Column has no valid numeric values.")

    m = _mean(valid)
    s = _std(valid, ddof=1)

    new_values: list = []
    missing = 0
    for v in col.values:
        if v is None:
            new_values.append(None)
            missing += 1
        elif s == 0:
            new_values.append(0.0)
        else:
            new_values.append((v - m) / s)

    out.columns.append(
        DataColumn(
            name=f"{col_name}_zscore",
            dtype="numeric",
            values=new_values,
            missing_count=missing,
        )
    )
    return out


def one_hot_encode(
    ds: Dataset, col_name: str, drop_original: bool = True, max_categories: int = 50
) -> Dataset:
    out = _clone_dataset(ds)
    col = _find_col(out, col_name)

    unique_vals = sorted(set(v for v in col.values if v is not None))
    if len(unique_vals) > max_categories:
        raise ValueError(
            f"Column '{col_name}' has {len(unique_vals)} unique values "
            f"(limit {max_categories}). Reduce cardinality first."
        )

    for uv in unique_vals:
        safe_name = str(uv).replace(" ", "_").replace("/", "_")
        new_values: list = []
        missing = 0
        for v in col.values:
            if v is None:
                new_values.append(None)
                missing += 1
            elif v == uv:
                new_values.append(1.0)
            else:
                new_values.append(0.0)
        out.columns.append(
            DataColumn(
                name=f"{col_name}_{safe_name}",
                dtype="numeric",
                values=new_values,
                missing_count=missing,
            )
        )

    if drop_original:
        out.columns = [c for c in out.columns if c.name != col_name]

    return out


def fill_missing(ds: Dataset, col_name: str, strategy: str = "mean") -> Dataset:
    """
    Fill missing values.
    Strategies: mean, median, mode (any type), zero, ffill (forward-fill).
    """
    from .statistics import mean as s_mean, median as s_median, mode as s_mode

    out = _clone_dataset(ds)
    col = _find_col(out, col_name)

    valid = [v for v in col.values if v is not None]
    if not valid:
        return out

    fill_val = None

    if strategy == "mean":
        if col.dtype != "numeric":
            raise ValueError("'mean' fill requires numeric column.")
        fill_val = s_mean(valid)
    elif strategy == "median":
        if col.dtype != "numeric":
            raise ValueError("'median' fill requires numeric column.")
        fill_val = s_median(valid)
    elif strategy == "mode":
        fill_val = s_mode(valid)
    elif strategy == "zero":
        fill_val = 0.0
    elif strategy == "ffill":
        last = None
        for i, v in enumerate(col.values):
            if v is not None:
                last = v
            else:
                col.values[i] = last
        col.missing_count = sum(1 for v in col.values if v is None)
        return out
    else:
        raise ValueError(f"Unknown fill strategy: '{strategy}'")

    for i, v in enumerate(col.values):
        if v is None:
            col.values[i] = fill_val

    col.missing_count = sum(1 for v in col.values if v is None)
    return out


def drop_missing(ds: Dataset, col_name: str | None = None) -> Dataset:
    """Drop rows with missing values (in specific column or any column)."""
    out = _clone_dataset(ds)

    keep = []
    for i in range(out.row_count):
        if col_name:
            col = _find_col(out, col_name)
            if col.values[i] is not None:
                keep.append(i)
        else:
            if all(c.values[i] is not None for c in out.columns):
                keep.append(i)

    for col in out.columns:
        col.values = [col.values[i] for i in keep]
        col.missing_count = sum(1 for v in col.values if v is None)

    out.row_count = len(keep)
    return out

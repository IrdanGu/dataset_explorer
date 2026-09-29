"""
CSV loader with automatic type detection.
Parses CSV from scratch (no pandas). Uses only the csv stdlib module for
RFC-4180 quoting rules, but all type inference is hand-rolled.
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DataColumn:
    """Metadata + values for a single column."""

    name: str
    dtype: str  # "numeric" | "categorical" | "datetime" | "text"
    values: list  # raw parsed values (float | str | None)
    missing_count: int = 0


@dataclass
class Dataset:
    """In-memory tabular dataset."""

    columns: list[DataColumn] = field(default_factory=list)
    row_count: int = 0
    file_name: str = ""

    # Convenience ----------------------------------------------------------

    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def get_column(self, name: str) -> Optional[DataColumn]:
        for c in self.columns:
            if c.name == name:
                return c
        return None

    def numeric_columns(self) -> list[DataColumn]:
        return [c for c in self.columns if c.dtype == "numeric"]

    def categorical_columns(self) -> list[DataColumn]:
        return [c for c in self.columns if c.dtype == "categorical"]

    def to_rows(self) -> list[dict]:
        """Return list-of-dicts representation."""
        rows = []
        for i in range(self.row_count):
            row = {}
            for col in self.columns:
                row[col.name] = col.values[i]
            rows.append(row)
        return rows

    def to_csv_string(self) -> str:
        """Serialize back to CSV."""
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(self.column_names())
        for i in range(self.row_count):
            row = []
            for col in self.columns:
                v = col.values[i]
                row.append("" if v is None else str(v))
            writer.writerow(row)
        return buf.getvalue()

    def clone(self) -> "Dataset":
        """Deep copy."""
        cols = []
        for c in self.columns:
            cols.append(
                DataColumn(
                    name=c.name,
                    dtype=c.dtype,
                    values=list(c.values),
                    missing_count=c.missing_count,
                )
            )
        return Dataset(columns=cols, row_count=self.row_count, file_name=self.file_name)


# ---------------------------------------------------------------------------
# Type detection helpers
# ---------------------------------------------------------------------------

_MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none", "-", "?", "missing"}


def _is_missing(raw: str) -> bool:
    return raw.strip().lower() in _MISSING_TOKENS


def _try_float(raw: str) -> Optional[float]:
    """Attempt to parse a string as float. Returns None on failure."""
    s = raw.strip().replace(",", "")  # handle "1,234.5"
    if not s:
        return None
    try:
        v = float(s)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except ValueError:
        return None


def detect_types(raw_columns: dict[str, list[str]]) -> dict[str, str]:
    """
    Heuristic type detection for each column.

    Rules:
      - If >= 80% of non-missing values parse as float -> "numeric"
      - If unique-ratio < 0.3 of total rows -> "categorical"
      - Otherwise -> "text"
    """
    result: dict[str, str] = {}

    for name, values in raw_columns.items():
        non_missing = [v for v in values if not _is_missing(v)]
        total = len(non_missing)

        if total == 0:
            result[name] = "text"
            continue

        # Try numeric
        numeric_count = sum(1 for v in non_missing if _try_float(v) is not None)
        if numeric_count / total >= 0.80:
            result[name] = "numeric"
            continue

        # Categorical heuristic
        unique = len(set(v.strip().lower() for v in non_missing))
        if total > 0 and unique / total < 0.30:
            result[name] = "categorical"
            continue

        # If few unique values overall (<= 20), still treat as categorical
        if unique <= 20:
            result[name] = "categorical"
            continue

        result[name] = "text"

    return result


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------


def load_csv(source: str | Path | io.IOBase, file_name: str = "") -> Dataset:
    """
    Load a CSV file into a Dataset.

    *source* can be a file path, a Path object, or a file-like object.
    """
    # Open / read
    if isinstance(source, (str, Path)):
        path = Path(source)
        file_name = file_name or path.name
        with open(path, "r", newline="", encoding="utf-8-sig") as f:
            text = f.read()
    else:
        text = source.read()
        if isinstance(text, bytes):
            text = text.decode("utf-8-sig")

    reader = csv.reader(io.StringIO(text))
    rows_iter = iter(reader)

    # Header
    try:
        headers = next(rows_iter)
    except StopIteration:
        return Dataset(file_name=file_name)

    headers = [h.strip() for h in headers]

    # Deduplicate header names
    seen: dict[str, int] = {}
    clean_headers: list[str] = []
    for h in headers:
        if h in seen:
            seen[h] += 1
            clean_headers.append(f"{h}_{seen[h]}")
        else:
            seen[h] = 0
            clean_headers.append(h)

    # Read all rows into column-major lists
    raw_cols: dict[str, list[str]] = {h: [] for h in clean_headers}
    row_count = 0
    for row in rows_iter:
        row_count += 1
        for i, h in enumerate(clean_headers):
            raw_cols[h].append(row[i] if i < len(row) else "")

    # Detect types
    types = detect_types(raw_cols)

    # Build DataColumns with parsed values
    columns: list[DataColumn] = []
    for h in clean_headers:
        dtype = types[h]
        raw_values = raw_cols[h]
        parsed: list[Any] = []
        missing = 0

        for rv in raw_values:
            if _is_missing(rv):
                parsed.append(None)
                missing += 1
            elif dtype == "numeric":
                v = _try_float(rv)
                if v is not None:
                    parsed.append(v)
                else:
                    parsed.append(None)
                    missing += 1
            else:
                parsed.append(rv.strip())

        columns.append(
            DataColumn(
                name=h,
                dtype=dtype,
                values=parsed,
                missing_count=missing,
            )
        )

    return Dataset(
        columns=columns,
        row_count=row_count,
        file_name=file_name,
    )

"""
Descriptive statistics computed from scratch (no numpy / pandas).
Every formula is implemented manually so you learn the math behind EDA.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

from .loader import DataColumn


def _valid(values: list) -> list[float]:
    return [v for v in values if v is not None]


def mean(xs: list[float]) -> float:
    if not xs:
        return 0.0
    return sum(xs) / len(xs)


def median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def mode(xs: list) -> Any:
    if not xs:
        return None
    freq: dict[Any, int] = {}
    for v in xs:
        freq[v] = freq.get(v, 0) + 1
    best: Any = xs[0]
    best_count = 0
    for k, c in freq.items():
        if c > best_count:
            best = k
            best_count = c
    return best


def variance(xs: list[float], ddof: int = 1) -> float:
    """Sample variance (ddof=1) or population variance (ddof=0)."""
    n = len(xs)
    if n <= ddof:
        return 0.0
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / (n - ddof)


def std_dev(xs: list[float], ddof: int = 1) -> float:
    return math.sqrt(variance(xs, ddof))


def percentile(xs: list[float], p: float) -> float:
    """Linear-interpolation percentile (matches numpy). p in [0, 100]."""
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    if n == 1:
        return s[0]
    k = (p / 100.0) * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def iqr(xs: list[float]) -> float:
    return percentile(xs, 75) - percentile(xs, 25)


def skewness(xs: list[float]) -> float:
    """Fisher-Pearson adjusted: G1 = [n/((n-1)(n-2))] * sum[((xi-mean)/std)^3]"""
    n = len(xs)
    if n < 3:
        return 0.0
    m = mean(xs)
    s = std_dev(xs, ddof=1)
    if s == 0:
        return 0.0
    m3 = sum(((x - m) / s) ** 3 for x in xs)
    return (n / ((n - 1) * (n - 2))) * m3


def kurtosis(xs: list[float]) -> float:
    """Excess kurtosis (Fisher). Normal distribution = 0."""
    n = len(xs)
    if n < 4:
        return 0.0
    m = mean(xs)
    s = std_dev(xs, ddof=1)
    if s == 0:
        return 0.0
    adj = ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))) * sum(
        ((x - m) / s) ** 4 for x in xs
    )
    correction = (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    return adj - correction


def min_val(xs: list[float]) -> float:
    return min(xs) if xs else 0.0


def max_val(xs: list[float]) -> float:
    return max(xs) if xs else 0.0


def count_unique(xs: list) -> int:
    return len(set(x for x in xs if x is not None))


def value_counts(xs: list) -> dict:
    freq: dict = {}
    for v in xs:
        if v is None:
            continue
        freq[v] = freq.get(v, 0) + 1
    return dict(sorted(freq.items(), key=lambda kv: kv[1], reverse=True))


def histogram_bins(xs: list[float], n_bins: int | None = None) -> list[dict]:
    """Sturges' rule binning: k = ceil(log2(n)) + 1"""
    if not xs:
        return []
    if n_bins is None:
        n_bins = max(1, int(math.ceil(math.log2(len(xs))) + 1))
    lo = min(xs)
    hi = max(xs)
    if lo == hi:
        return [{"lo": lo, "hi": hi, "count": len(xs)}]
    width = (hi - lo) / n_bins
    bins = [
        {"lo": lo + i * width, "hi": lo + (i + 1) * width, "count": 0}
        for i in range(n_bins)
    ]
    for x in xs:
        idx = int((x - lo) / width)
        if idx >= n_bins:
            idx = n_bins - 1
        bins[idx]["count"] += 1
    return bins


@dataclass
class ColumnStats:
    name: str
    dtype: str
    count: int
    missing: int
    missing_pct: float
    unique: int
    mean: Optional[float] = None
    median: Optional[float] = None
    mode_val: Optional[Any] = None
    std: Optional[float] = None
    variance: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    q25: Optional[float] = None
    q50: Optional[float] = None
    q75: Optional[float] = None
    iqr: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    histogram: Optional[list[dict]] = None
    top_values: Optional[dict] = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "name": self.name,
            "dtype": self.dtype,
            "count": self.count,
            "missing": self.missing,
            "missing_pct": round(self.missing_pct, 2),
            "unique": self.unique,
        }
        if self.dtype == "numeric":
            d.update(
                {
                    "mean": _r(self.mean),
                    "median": _r(self.median),
                    "mode": _r(self.mode_val),
                    "std": _r(self.std),
                    "variance": _r(self.variance),
                    "min": _r(self.min),
                    "max": _r(self.max),
                    "q25": _r(self.q25),
                    "q50": _r(self.q50),
                    "q75": _r(self.q75),
                    "iqr": _r(self.iqr),
                    "skewness": _r(self.skewness),
                    "kurtosis": _r(self.kurtosis),
                    "histogram": self.histogram,
                }
            )
        else:
            d["top_values"] = dict(list((self.top_values or {}).items())[:15])
            d["mode"] = self.mode_val
        return d


def _r(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float):
        return round(v, 4)
    return v


def compute_statistics(col: DataColumn) -> ColumnStats:
    total = len(col.values)
    vals = _valid(col.values)

    base = ColumnStats(
        name=col.name,
        dtype=col.dtype,
        count=total,
        missing=col.missing_count,
        missing_pct=(col.missing_count / total * 100) if total > 0 else 0,
        unique=count_unique(col.values),
    )

    if col.dtype == "numeric" and vals:
        base.mean = mean(vals)
        base.median = median(vals)
        base.mode_val = mode(vals)
        base.std = std_dev(vals)
        base.variance = variance(vals)
        base.min = min_val(vals)
        base.max = max_val(vals)
        base.q25 = percentile(vals, 25)
        base.q50 = percentile(vals, 50)
        base.q75 = percentile(vals, 75)
        base.iqr = iqr(vals)
        base.skewness = skewness(vals)
        base.kurtosis = kurtosis(vals)
        base.histogram = histogram_bins(vals)
    else:
        base.top_values = value_counts(col.values)
        base.mode_val = mode([v for v in col.values if v is not None]) if vals else None

    return base

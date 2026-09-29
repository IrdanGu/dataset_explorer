"""
Pearson correlation matrix — from scratch.
r = sum((xi-mx)(yi-my)) / sqrt(sum((xi-mx)^2) * sum((yi-my)^2))
"""

from __future__ import annotations

import math
from typing import Optional

from .loader import Dataset
from .statistics import mean as _mean


def _valid_pairs(xs: list, ys: list) -> tuple[list[float], list[float]]:
    px, py = [], []
    for x, y in zip(xs, ys):
        if x is not None and y is not None:
            px.append(x)
            py.append(y)
    return px, py


def pearson(xs: list[float], ys: list[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None

    mx = _mean(xs)
    my = _mean(ys)

    num = 0.0
    dx2 = 0.0
    dy2 = 0.0

    for x, y in zip(xs, ys):
        dx = x - mx
        dy = y - my
        num += dx * dy
        dx2 += dx * dx
        dy2 += dy * dy

    denom = math.sqrt(dx2 * dy2)
    if denom == 0:
        return None

    r = num / denom
    return max(-1.0, min(1.0, r))


def correlation_matrix(dataset: Dataset) -> dict:
    """Pairwise Pearson correlation for all numeric columns."""
    num_cols = dataset.numeric_columns()
    names = [c.name for c in num_cols]
    n = len(num_cols)

    matrix: list[list[float | None]] = [[0.0] * n for _ in range(n)]

    for i in range(n):
        matrix[i][i] = 1.0
        for j in range(i + 1, n):
            xs, ys = _valid_pairs(num_cols[i].values, num_cols[j].values)
            r = pearson(xs, ys)
            val = round(r, 4) if r is not None else None
            matrix[i][j] = val
            matrix[j][i] = val

    return {"columns": names, "matrix": matrix}

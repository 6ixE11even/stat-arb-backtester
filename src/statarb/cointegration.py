"""
Pair selection by the Engle-Granger cointegration test.

Two price series are cointegrated if some linear combination of them is stationary —
i.e. they wander together and the *spread* mean-reverts. That spread is the thing we
trade. For each candidate pair we run the test and, if it clears the p-value, fit the
hedge ratio (the spread's slope) by OLS.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


def hedge_ratio(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """OLS of y on x -> (slope beta, intercept). Spread = y - (beta*x + intercept)."""
    beta, intercept = np.polyfit(x, y, 1)
    return float(beta), float(intercept)


def find_pairs(prices: pd.DataFrame, pvalue_threshold: float = 0.05) -> pd.DataFrame:
    """Every pair that passes the cointegration test, sorted most-significant first."""
    rows = []
    for a, b in combinations(prices.columns, 2):
        _, pvalue, _ = coint(prices[a], prices[b])
        if pvalue < pvalue_threshold:
            beta, intercept = hedge_ratio(prices[a].to_numpy(), prices[b].to_numpy())
            rows.append({"asset_a": a, "asset_b": b, "pvalue": pvalue, "beta": beta, "intercept": intercept})

    cols = ["asset_a", "asset_b", "pvalue", "beta", "intercept"]
    return pd.DataFrame(rows, columns=cols).sort_values("pvalue").reset_index(drop=True)

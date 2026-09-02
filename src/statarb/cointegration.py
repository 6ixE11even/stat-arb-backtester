"""
Pair selection by the Engle-Granger cointegration test.

Two price series are cointegrated if some linear combination of them is stationary —
i.e. they wander together and the *spread* mean-reverts. That spread is the thing we
trade. For each candidate pair we run the test and, if it clears the p-value, fit the
hedge ratio (the spread's slope) by OLS.

Two things this has to get right or the backtest is fiction:

*Selection is in-sample.* The test and the OLS see only the first `train_frac` of the
history. Fitting beta on all 600 days and then trading it from day 1 hands the
strategy the answer; the spread is stationary by construction because you solved for
the coefficient that made it so.

*Fifteen pairs is fifteen tests.* Six assets give 15 combinations, and at p < 0.05
you expect roughly one false positive from noise alone. The p-values go through a
Benjamini-Hochberg step so the reported set controls the false-discovery rate rather
than the per-test error.
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


def benjamini_hochberg(pvalues: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Boolean mask of the p-values that survive BH at false-discovery rate `q`."""
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    cutoffs = q * np.arange(1, n + 1) / n
    passing = np.nonzero(ranked <= cutoffs)[0]
    keep = np.zeros(n, dtype=bool)
    if passing.size:
        keep[order[: passing[-1] + 1]] = True
    return keep


def find_pairs(prices: pd.DataFrame, pvalue_threshold: float = 0.05,
               train_frac: float = 0.6, correct_fdr: bool = True) -> pd.DataFrame:
    """Pairs that pass the cointegration test on the training window.

    Engle-Granger is not symmetric — regressing A on B and B on A give different
    p-values — so both directions are tested and the stronger one is kept, with
    `asset_a` set to the dependent leg that won.

    The returned frame carries `train_end` in `.attrs`: the positional index where
    the training window stops and out-of-sample begins.
    """
    if not 0.0 < train_frac <= 1.0:
        raise ValueError(f"train_frac must be in (0, 1], got {train_frac}")
    train_end = max(int(len(prices) * train_frac), 2)
    train = prices.iloc[:train_end]

    cands = []
    for a, b in combinations(prices.columns, 2):
        p_ab = coint(train[a], train[b])[1]
        p_ba = coint(train[b], train[a])[1]
        dep, ind, pvalue = (a, b, p_ab) if p_ab <= p_ba else (b, a, p_ba)
        beta, intercept = hedge_ratio(train[dep].to_numpy(), train[ind].to_numpy())
        cands.append({"asset_a": dep, "asset_b": ind, "pvalue": float(pvalue),
                      "beta": beta, "intercept": intercept})

    cols = ["asset_a", "asset_b", "pvalue", "beta", "intercept"]
    if not cands:
        out = pd.DataFrame(columns=cols)
        out.attrs["train_end"] = train_end
        return out

    frame = pd.DataFrame(cands, columns=cols)
    keep = frame["pvalue"] < pvalue_threshold
    if correct_fdr:
        keep &= benjamini_hochberg(frame["pvalue"].to_numpy(), q=pvalue_threshold)

    out = frame[keep].sort_values("pvalue").reset_index(drop=True)
    out.attrs["train_end"] = train_end
    return out

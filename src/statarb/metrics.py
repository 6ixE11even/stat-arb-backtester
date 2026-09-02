"""Performance statistics for a daily return series."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def performance(returns: pd.Series, turnover: pd.Series | None = None) -> dict[str, float]:
    r = returns.dropna()
    ann_ret = r.mean() * TRADING_DAYS
    ann_vol = r.std() * np.sqrt(TRADING_DAYS)
    equity = (1.0 + r).cumprod()
    max_dd = (equity / equity.cummax() - 1.0).min()

    # The strategy is flat most of the time and a flat day returns exactly zero.
    # Counting those as losses put the hit rate somewhere near 20% and said
    # nothing about the trades. Measure it over the days with money at risk.
    active = r[r != 0.0]
    out = {
        "ann_return_%": ann_ret * 100,
        "ann_vol_%": ann_vol * 100,
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0.0,
        "max_drawdown_%": max_dd * 100,
        "hit_rate_%": (active > 0).mean() * 100 if len(active) else float("nan"),
        "days_at_risk_%": len(active) / len(r) * 100 if len(r) else float("nan"),
    }
    if turnover is not None:
        out["avg_daily_turnover"] = float(turnover.dropna().mean())
    return out


def equity_curve(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()


def split_performance(returns: pd.Series, train_end: int,
                      turnover: pd.Series | None = None) -> dict[str, dict[str, float]]:
    """In-sample vs out-of-sample stats, split at positional index `train_end`.

    The pairs and their hedge ratios were fitted on the first slice. Only the
    second slice is evidence.
    """
    def _slice(x):
        return None if x is None else x.iloc[train_end:]
    return {
        "in_sample": performance(returns.iloc[:train_end], turnover if turnover is None
                                 else turnover.iloc[:train_end]),
        "out_of_sample": performance(returns.iloc[train_end:], _slice(turnover)),
    }

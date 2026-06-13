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

    out = {
        "ann_return_%": ann_ret * 100,
        "ann_vol_%": ann_vol * 100,
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0.0,
        "max_drawdown_%": max_dd * 100,
        "hit_rate_%": (r > 0).mean() * 100,
    }
    if turnover is not None:
        out["avg_daily_turnover"] = float(turnover.dropna().mean())
    return out


def equity_curve(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()

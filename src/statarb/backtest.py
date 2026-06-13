"""
Event-driven backtest.

Each pair trades a fixed gross notional, so its daily P&L becomes a *return* (spread
change ÷ notional), comparable across pairs. Yesterday's position earns today's move —
no look-ahead. Turnover is charged at `cost_bps` each time the position flips. The
portfolio is an equal-weight blend of the pair returns.
"""
from __future__ import annotations

import pandas as pd

from statarb.strategy import positions_from_z, spread_and_zscore


def backtest_pair(prices: pd.DataFrame, a: str, b: str, beta: float, intercept: float = 0.0,
                  window: int = 30, entry: float = 2.0, exit: float = 0.5,
                  cost_bps: float = 1.0) -> pd.DataFrame:
    spread, z = spread_and_zscore(prices, a, b, beta, intercept, window)
    pos = positions_from_z(z, entry, exit)

    notional = prices[a].abs() + abs(beta) * prices[b].abs()   # gross $ per unit spread
    spread_ret = spread.diff() / notional.shift(1)             # spread move as a return
    turnover = pos.diff().abs().fillna(0.0)

    ret = pos.shift(1).fillna(0.0) * spread_ret - turnover * (cost_bps / 1e4)
    return pd.DataFrame({"position": pos, "spread": spread, "z": z, "turnover": turnover, "return": ret})


def backtest_portfolio(prices: pd.DataFrame, pairs: pd.DataFrame, **kwargs):
    """Backtest every selected pair and equal-weight them. Returns (portfolio_return,
    per_pair_returns, {pair: full_frame})."""
    if pairs.empty:
        raise ValueError("No cointegrated pairs to backtest.")

    frames, rets = {}, {}
    for r in pairs.itertuples():
        bt = backtest_pair(prices, r.asset_a, r.asset_b, r.beta, r.intercept, **kwargs)
        name = f"{r.asset_a}-{r.asset_b}"
        frames[name] = bt
        rets[name] = bt["return"]

    per_pair = pd.DataFrame(rets)
    portfolio = per_pair.mean(axis=1)   # equal weight across pairs
    return portfolio, per_pair, frames

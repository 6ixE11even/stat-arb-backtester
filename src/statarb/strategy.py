"""
The trading rule: trade the spread's z-score.

When the spread stretches far from its rolling mean (|z| above an entry band) we bet
on reversion; we close once it's back near the mean. A simple state machine avoids
flip-flopping on every wiggle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def spread_and_zscore(prices: pd.DataFrame, a: str, b: str, beta: float,
                      intercept: float = 0.0, window: int = 30) -> tuple[pd.Series, pd.Series]:
    spread = prices[a] - (beta * prices[b] + intercept)
    z = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()
    return spread, z


def positions_from_z(z: pd.Series, entry: float = 2.0, exit: float = 0.5) -> pd.Series:
    """+1 = long the spread (long a / short b), -1 = short it, 0 = flat.

    Enter when |z| >= entry, exit when |z| <= exit. Holding in between, so we trade
    on band crossings, not on every bar.
    """
    zv = z.to_numpy()
    pos = np.zeros(len(zv))
    state = 0
    for t in range(len(zv)):
        if np.isnan(zv[t]):
            pos[t] = 0
            continue
        if state == 0:
            if zv[t] >= entry:
                state = -1            # spread too high -> short it
            elif zv[t] <= -entry:
                state = 1             # spread too low -> long it
        elif abs(zv[t]) <= exit:
            state = 0                 # reverted -> close
        pos[t] = state
    return pd.Series(pos, index=z.index)

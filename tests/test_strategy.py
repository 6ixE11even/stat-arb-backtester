"""Logic checks on a small cointegrated fixture (a unit-test input, not product data).

The live data layer pulls real crypto prices from Deribit; these tests exercise the
cointegration / signal / backtest logic deterministically and offline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from statarb.backtest import backtest_portfolio
from statarb.cointegration import find_pairs
from statarb.metrics import performance
from statarb.strategy import positions_from_z


def _cointegrated_fixture(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0, 1, n)) + 100.0
    y = 2.0 * x + rng.normal(0, 1.0, n)        # cointegrated with X (stationary spread)
    z = np.cumsum(rng.normal(0, 1, n)) + 80.0  # independent random walk
    return pd.DataFrame({"X": x, "Y": y, "Z": z}, index=pd.bdate_range("2022-01-01", periods=n))


def test_find_pairs_detects_cointegration():
    pairs = find_pairs(_cointegrated_fixture())
    found = {frozenset([r.asset_a, r.asset_b]) for r in pairs.itertuples()}
    assert frozenset(["X", "Y"]) in found


def test_position_state_machine():
    z = pd.Series([0.0, 2.5, 1.0, 0.2, -2.5, 0.0])
    pos = positions_from_z(z, entry=2.0, exit=0.5).to_numpy()
    assert pos[1] == -1 and pos[3] == 0 and pos[4] == 1


def test_backtest_produces_finite_sharpe():
    prices = _cointegrated_fixture()
    portfolio, _, _ = backtest_portfolio(prices, find_pairs(prices))
    assert np.isfinite(performance(portfolio)["sharpe"])

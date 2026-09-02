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


def test_pairs_are_selected_without_seeing_the_test_window():
    """Beta must come from the training slice only, or the spread is stationary by
    construction and the backtest is just describing the fit."""
    prices = _cointegrated_fixture()
    pairs = find_pairs(prices, train_frac=0.6)
    train_end = pairs.attrs["train_end"]
    assert train_end == int(len(prices) * 0.6)
    row = pairs[pairs.asset_a.isin(["X", "Y"]) & pairs.asset_b.isin(["X", "Y"])].iloc[0]
    from statarb.cointegration import hedge_ratio
    train_beta, _ = hedge_ratio(prices[row.asset_a].to_numpy()[:train_end],
                                prices[row.asset_b].to_numpy()[:train_end])
    assert abs(row.beta - train_beta) < 1e-9


def test_multiple_testing_is_corrected():
    """15 pairs at p<0.05 throws off false positives on pure noise."""
    from statarb.cointegration import benjamini_hochberg
    rng = np.random.default_rng(7)
    noise = pd.DataFrame({c: np.cumsum(rng.normal(0, 1, 400)) + 100 for c in "ABCDEF"},
                         index=pd.bdate_range("2022-01-01", periods=400))
    assert len(find_pairs(noise, correct_fdr=True)) <= len(find_pairs(noise, correct_fdr=False))
    # BH keeps nothing when every p-value is uniform-ish and large
    assert not benjamini_hochberg(np.array([0.4, 0.6, 0.9]), q=0.05).any()
    assert benjamini_hochberg(np.array([0.001, 0.9, 0.95]), q=0.05)[0]


def test_missing_signal_does_not_book_a_phantom_round_trip():
    """A NaN z used to write position 0 for one bar, so diff() charged 2x turnover."""
    z = pd.Series([0.0, 2.5, np.nan, 2.4, 0.1])
    pos = positions_from_z(z, entry=2.0, exit=0.5)
    assert pos.to_numpy().tolist() == [0.0, -1.0, -1.0, -1.0, 0.0]
    assert pos.diff().abs().fillna(0.0).sum() == 2.0     # one entry, one exit


def test_hit_rate_ignores_flat_days():
    from statarb.metrics import performance as perf
    r = pd.Series([0.0] * 90 + [0.01] * 6 + [-0.01] * 4)
    assert abs(perf(r)["hit_rate_%"] - 60.0) < 1e-9
    assert abs(perf(r)["days_at_risk_%"] - 10.0) < 1e-9

"""
End-to-end:  python scripts/run_backtest.py

Generate the price panel, find cointegrated pairs, backtest the z-score strategy,
print performance, and write the equity curve + a spread chart to reports/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from statarb.backtest import backtest_portfolio          # noqa: E402
from statarb.cointegration import find_pairs             # noqa: E402
from statarb.data import load_prices                     # noqa: E402
from statarb.metrics import equity_curve, performance    # noqa: E402
from statarb.viz import plot_equity, plot_pair           # noqa: E402


def main() -> None:
    print("Fetching real crypto prices from Deribit (cached after first run)...")
    prices = load_prices()
    print(f"  {prices.shape[0]} days x {prices.shape[1]} assets: {', '.join(prices.columns)}\n")
    pairs = find_pairs(prices)
    print("Cointegrated pairs:")
    print(pairs.to_string(index=False), "\n")

    portfolio, _per_pair, frames = backtest_portfolio(prices, pairs, cost_bps=1.0)
    avg_turnover = sum(frames[k]["turnover"] for k in frames) / len(frames)
    stats = performance(portfolio, avg_turnover)

    print("Portfolio performance:")
    for k, v in stats.items():
        print(f"  {k:<20} {v:>9.3f}")

    reports = ROOT / "reports"
    (reports / "figures").mkdir(parents=True, exist_ok=True)
    equity_curve(portfolio).to_csv(reports / "equity.csv")
    pd.Series(stats).to_csv(reports / "performance.csv")
    plot_equity(equity_curve(portfolio), reports / "figures" / "equity_curve.png")
    top = pairs.iloc[0]
    name = f"{top.asset_a}-{top.asset_b}"
    plot_pair(frames[name], name, reports / "figures" / "pair_spread.png")
    print(f"\nwrote -> {reports}/ (equity.csv, performance.csv, figures/)")


if __name__ == "__main__":
    main()

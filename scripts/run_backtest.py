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
from statarb.metrics import equity_curve, split_performance  # noqa: E402
from statarb.viz import plot_equity, plot_pair           # noqa: E402


def main() -> None:
    print("Fetching real crypto prices from Deribit (cached after first run)...")
    prices = load_prices()
    print(f"  {prices.shape[0]} days x {prices.shape[1]} assets: {', '.join(prices.columns)}\n")
    pairs = find_pairs(prices)
    train_end = pairs.attrs["train_end"]
    print(f"Cointegrated pairs (selected on the first {train_end} days, "
          f"BH-corrected across {len(prices.columns) * (len(prices.columns) - 1) // 2} tests):")
    if pairs.empty:
        print("  none survived.\n")
        near = find_pairs(prices, pvalue_threshold=1.1, correct_fdr=False)
        print("  Ranked candidates (NOT traded — shown so the gate is visible):")
        print(near.head(3)[["asset_a", "asset_b", "pvalue"]].to_string(index=False))
        print("\n  Nothing to trade is a result. On this panel the surviving pair set is")
        print("  unstable across training windows, which is what you would expect if the")
        print("  relationships are sample artefacts rather than structure.\n")
        return
    print(pairs.to_string(index=False), "\n")

    portfolio, _per_pair, frames = backtest_portfolio(prices, pairs, cost_bps=1.0)
    avg_turnover = sum(frames[k]["turnover"] for k in frames) / len(frames)
    split = split_performance(portfolio, train_end, avg_turnover)
    stats = split["out_of_sample"]

    print(f"{'':<20} {'in-sample':>12} {'out-of-sample':>14}")
    for k in split["in_sample"]:
        print(f"  {k:<18} {split['in_sample'][k]:>12.3f} {split['out_of_sample'][k]:>14.3f}")
    print("\n  Only the out-of-sample column is evidence: the pairs and their hedge")
    print("  ratios were fitted on the in-sample window.")

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

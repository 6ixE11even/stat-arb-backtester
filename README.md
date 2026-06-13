# Statistical Arbitrage — Crypto Pairs Trading Backtester

Pull **real crypto prices from the Deribit public API**, find cointegrated pairs,
trade the mean-reverting spread on its z-score, and measure it with an **event-driven
backtester** that charges transaction costs and never peeks at the future. End to end:
live data → pair selection → signal → P&L.

## How it works

1. **Data** (`data.py`) — daily close prices for a basket of crypto perpetuals
   (BTC, ETH, SOL, XRP, …) straight from Deribit (`get_tradingview_chart_data`,
   public, no key). Cached to `data/` after the first pull.
2. **Pair selection** (`cointegration.py`) — the Engle-Granger test on every pair;
   survivors get a hedge ratio by OLS. A cointegrated pair has a *stationary* spread,
   which is what makes it tradeable.
3. **Signal** (`strategy.py`) — standardise the spread to a rolling z-score; a state
   machine shorts the spread at z ≥ +2, longs at z ≤ −2, flattens near 0.
4. **Backtest** (`backtest.py`) — each pair trades a fixed gross notional so its P&L
   is a return; yesterday's position earns today's move (no look-ahead); flips cost
   `cost_bps`. The portfolio equal-weights the pairs.
5. **Metrics** (`metrics.py`) — annualised return/vol, Sharpe, max drawdown, hit rate,
   turnover; equity curve + spread charts in `viz.py`.

## Run

```bash
uv sync
uv run python scripts/run_backtest.py   # fetches live Deribit data -> metrics + reports/figures/
uv run pytest                           # cointegration / signal / backtest logic (offline)
```

The first run pulls live prices from Deribit and caches them; reruns use the cache.
`reports/` then holds `equity.csv`, `performance.csv`, and the figures.

## Structure

```
stat-arb-backtester/
├── src/statarb/
│   ├── data.py           # real Deribit crypto price client (cached)
│   ├── cointegration.py  # Engle-Granger pair selection + hedge ratio
│   ├── strategy.py       # spread z-score -> positions
│   ├── backtest.py       # event-driven engine with costs
│   ├── metrics.py        # Sharpe / drawdown / turnover
│   └── viz.py            # equity curve + spread chart
├── scripts/run_backtest.py
└── tests/                # logic checks on a cointegrated fixture
```

## Notes

- **Real data, no synthetic.** Prices come live from Deribit; the unit tests use a
  small cointegrated fixture so the cointegration/signal/backtest logic is checked
  deterministically and offline.
- Backtest is gross of financing and assumes fills at the daily close — conservative
  enough to trust the *shape* of the result. Swap the instrument list in `data.py` to
  trade a different universe.

---

*Built by Tejas Pandya — NYU MSFE.*

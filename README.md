# Statistical Arbitrage — Crypto Pairs Trading Backtester

Pull **real crypto prices from the Deribit public API**, find cointegrated pairs,
trade the mean-reverting spread on its z-score, and measure it with a daily backtest
that charges transaction costs and never peeks at the future. End to end: live data →
out-of-sample pair selection → signal → P&L.

## Result

On 366 daily closes (2025-09-02 to 2026-09-02) of BTC, ETH, SOL, XRP and AVAX, **no pair
survives** selection on the first 60% of the sample once the ten pairwise tests are
corrected for multiple testing. Two pairs clear p < 0.05 on their own (ETH-BTC 0.042,
BTC-XRP 0.047) and none clears Benjamini-Hochberg, so nothing is traded.

The candidates are not stable either. Re-selecting on the first 50 / 60 / 70 / 80% of
the sample gives four different raw survivor sets: none; ETH-BTC and BTC-XRP; AVAX-ETH and
XRP-SOL; AVAX-ETH, XRP-SOL and AVAX-SOL. AVAX-ETH survives the correction only once the
training window reaches 70%. A relationship that appears and disappears with the
training window is a sample artefact, and the backtest declines to trade it.

## How it works

1. **Data** (`data.py`) — daily close prices for a basket of crypto perpetuals
   (BTC, ETH, SOL, XRP, …) straight from Deribit (`get_tradingview_chart_data`,
   public, no key). Cached to `data/` after the first pull.
2. **Pair selection** (`cointegration.py`) — the Engle-Granger test on every pair, in
   both regression directions, using only the first `train_frac` (60%) of the history;
   the p-values go through Benjamini-Hochberg so the selected set controls the
   false-discovery rate. Survivors get a hedge ratio by OLS on the same training slice.
   A cointegrated pair has a *stationary* spread, which is what makes it tradeable.
3. **Signal** (`strategy.py`) — standardise the spread to a rolling z-score; a state
   machine shorts the spread at z ≥ +2, longs at z ≤ −2, flattens near 0.
4. **Backtest** (`backtest.py`) — vectorised over daily bars: each pair trades a fixed
   gross notional so its P&L is a return; yesterday's position earns today's move (no
   look-ahead); every unit of position change costs `cost_bps`. The portfolio
   equal-weights the pairs, and performance is reported in-sample and out-of-sample
   separately; only the second is evidence.
5. **Metrics** (`metrics.py`) — annualised return/vol, Sharpe, max drawdown, hit rate over
   days with money at risk, turnover; equity curve + spread charts in `viz.py`.

## The math

**Cointegration.** Two non-stationary (I(1)) price series $y_t, x_t$ are cointegrated
if some linear combination is stationary. Engle-Granger tests this in two steps:
regress $y_t = \beta x_t + c + u_t$ by OLS, then run an ADF unit-root test on the
residuals $\hat u_t$ — using Engle-Granger critical values, which are stricter than
plain ADF because $\beta$ was estimated. Rejection means the spread

$$s_t = y_t - \hat\beta x_t$$

is mean-reverting, and $\hat\beta$ is the hedge ratio that makes the pair
market-neutral in the cointegrating direction.

**Signal.** The spread is standardised on a rolling window,
$z_t = (s_t - \mu_t)/\sigma_t$, and traded as a state machine: short the spread at
$z \ge +2$, long at $z \le -2$, flatten near zero. Under an Ornstein-Uhlenbeck view
of the spread, $ds_t = \theta(\mu - s_t)\,dt + \sigma\,dW_t$, the entry threshold is
a bet that $|z|=2$ deviations decay with half-life $\ln 2 / \theta$. The tests pin
the state machine's entry and exit thresholds; nothing tests the half-life itself.

**P&L.** Yesterday's position earns today's spread change (no look-ahead); each
position flip is charged `cost_bps` of the notional traded. Reported Sharpe is
$\sqrt{252}\,\bar r / \hat\sigma_r$ on daily portfolio returns.

## References

- Engle, R. & Granger, C. (1987), *Co-integration and Error Correction*, Econometrica 55(2) — the two-step test.
- Gatev, E., Goetzmann, W. & Rouwenhorst, K.G. (2006), *Pairs Trading: Performance of a Relative-Value Arbitrage Rule*, Review of Financial Studies 19(3).
- Avellaneda, M. & Lee, J.-H. (2010), *Statistical Arbitrage in the US Equities Market*, Quantitative Finance 10(7) — the OU/z-score framing.

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

- **Live data for the backtest, fixtures for the tests.** Prices come live from Deribit
  (cached in `data/`); the 7 unit tests use small fixtures so the selection, signal and
  backtest logic is checked deterministically and offline.
- MATIC_USDC-PERPETUAL is in the instrument list but Deribit returns one constant price
  for all 366 days, so `load_prices` drops it: six instruments, five live assets, ten
  pair tests.
- The backtest is gross of financing and fills at the daily close. Swap the instrument
  list in `data.py` to trade a different universe.

---

*Built by Tejas Pandya — NYU MSFE.*

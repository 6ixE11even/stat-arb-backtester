"""
Real crypto price data from the Deribit public API (no key, no synthetic data).

Pulls daily close prices for a basket of crypto instruments and caches them to
data/ so a backtest is reproducible offline after the first pull. Deribit's
`get_tradingview_chart_data` returns clean OHLC arrays; we keep the close.

Deribit answers status="ok" for instruments that no longer trade and hands back a
flat line — MATIC_USDC-PERPETUAL currently returns the same 0.4277 for every one of
366 days. A dead series is not a price series: it has no variance, so the
cointegration regression against it is degenerate and the p-values it produces are
noise. `load_prices` drops those columns and says so.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

DERIBIT_CHART = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"

# Liquid Deribit instruments. Inverse perps for BTC/ETH, linear (USDC) for the alts.
DEFAULT_INSTRUMENTS = {
    "BTC": "BTC-PERPETUAL",
    "ETH": "ETH-PERPETUAL",
    "SOL": "SOL_USDC-PERPETUAL",
    "XRP": "XRP_USDC-PERPETUAL",
    "AVAX": "AVAX_USDC-PERPETUAL",
    "MATIC": "MATIC_USDC-PERPETUAL",
}

CACHE = Path(__file__).resolve().parents[2] / "data" / "crypto_prices.csv"


def fetch_close(instrument: str, days: int = 365, resolution: str = "1D",
                session: requests.Session | None = None) -> pd.Series:
    """Daily close series for one Deribit instrument."""
    end = int(time.time() * 1000)
    start = end - days * 86_400_000
    get = (session or requests).get
    resp = get(DERIBIT_CHART, params={"instrument_name": instrument, "resolution": resolution,
                                       "start_timestamp": start, "end_timestamp": end}, timeout=30)
    resp.raise_for_status()
    result = resp.json()["result"]
    if result.get("status") != "ok":
        raise RuntimeError(f"Deribit returned status={result.get('status')} for {instrument}")
    return pd.Series(result["close"], index=pd.to_datetime(result["ticks"], unit="ms"))


MIN_DISTINCT_PRICES = 10


def _drop_dead_series(prices: pd.DataFrame) -> pd.DataFrame:
    """Drop instruments whose price never moves — delisted or never-traded perps."""
    dead = [c for c in prices.columns if prices[c].nunique() < MIN_DISTINCT_PRICES]
    for c in dead:
        print(f"  (dropped {c}: {prices[c].nunique()} distinct prices — instrument is not trading)")
    return prices.drop(columns=dead)


def load_prices(instruments: dict[str, str] | None = None, days: int = 365,
                use_cache: bool = True) -> pd.DataFrame:
    """Price panel (one column per asset). Reads the cache if present, else fetches
    live from Deribit and writes the cache."""
    instruments = instruments or DEFAULT_INSTRUMENTS
    if use_cache and CACHE.exists():
        cached = pd.read_csv(CACHE, index_col=0, parse_dates=True)
        # A cache is only a cache if it answers the question. One holding 365 days
        # of six assets used to satisfy a request for 730 days of nine, silently.
        live = _drop_dead_series(cached)
        if set(live.columns) & set(instruments) and len(live) >= days * 0.9:
            return live[[c for c in instruments if c in live.columns]]

    session = requests.Session()
    columns = {}
    for symbol, instrument in instruments.items():
        try:
            columns[symbol] = fetch_close(instrument, days=days, session=session)
        except Exception as exc:  # noqa: BLE001 — skip anything that isn't trading
            print(f"  (skipped {symbol}: {exc})")

    prices = _drop_dead_series(pd.DataFrame(columns).dropna())
    if prices.empty:
        raise RuntimeError("No price data fetched — check network access to deribit.com.")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(CACHE)
    return prices

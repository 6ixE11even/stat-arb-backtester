"""Equity curve and a spread/z-score chart with the trading bands."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_equity(equity: pd.Series, out_path: str | Path, title: str = "Stat-arb portfolio equity") -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(equity.index, equity.to_numpy(), color="#3b6ea5", lw=1.7)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("growth of $1")
    ax.grid(alpha=0.25)
    _save(fig, out_path)


def plot_pair(frame: pd.DataFrame, name: str, out_path: str | Path,
              entry: float = 2.0, exit: float = 0.5) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(frame.index, frame["spread"], color="#2c3e50", lw=1.1)
    ax1.set_title(f"{name}: spread", fontweight="bold")
    ax1.grid(alpha=0.25)

    ax2.plot(frame.index, frame["z"], color="#3b6ea5", lw=1.0)
    for lvl, c in [(entry, "#c0392b"), (-entry, "#c0392b"), (exit, "#27ae60"), (-exit, "#27ae60")]:
        ax2.axhline(lvl, color=c, ls="--", lw=0.8)
    ax2.set_title("z-score with entry (red) / exit (green) bands")
    ax2.grid(alpha=0.25)
    _save(fig, out_path)


def _save(fig, out_path: str | Path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

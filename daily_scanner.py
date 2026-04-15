#!/usr/bin/env python3
"""
daily_scanner.py  —  E ∩ FVG5 Daily Signal Scanner
====================================================
Run once per day after updating your stock CSV files.
Scans all tickers and prints two lists:

  ENTRY  (buy)  : Supertrend recently flipped to bullish
                  AND vol-weighted FVG POC spread is at an extreme low
                  → price is well below the FVG activity centre  (mean-reversion up)

  EXIT   (sell) : Supertrend recently flipped to bearish
                  AND vol-weighted FVG POC spread is at an extreme high
                  → price is well above the FVG activity centre  (mean-reversion down)

Why no argrelextrema here?
--------------------------
argrelextrema(np.less_equal, order=10) needs 10 FUTURE bars to confirm a local
minimum, so it cannot be used on the last (today's) bar.  Instead we use the
Supertrend flip itself as the trough/peak proxy:
  • A buy  flip (trend -1 → +1) always happens near a local low.
  • A sell flip (trend +1 → -1) always happens near a local high.
The FVG5 spread condition is then checked at today's close.
This faithfully mirrors the E ∩ FVG5 backtest logic.

Usage
-----
  python daily_scanner.py                         # uses ./price_data/
  DATA_DIR=/path/to/csvs python daily_scanner.py  # custom folder

CSV format expected
-------------------
  Row 0  : column names  (Price, Close, High, Low, Open, Volume)
  Row 1  : ticker row    (skipped)
  Row 2  : label row     (skipped)
  Row 3+ : daily bars    (date index, OHLCV values)

Parameters (edit below if needed)
----------------------------------
  ST_PERIOD  = 10    Supertrend ATR length
  ST_MULT    = 3.0   Supertrend ATR multiplier
  CONF_W     = 5     bars to look back for a recent ST flip
  FVG_PERIOD = 100   rolling lookback for FVG POC (bars)
  FVG_SMOOTH = 10    SMA smoothing on raw POC
  SIGMA      = 2.0   threshold = mean ± SIGMA × std
  MIN_BARS   = 150   minimum history bars required to compute reliable thresholds
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Parameters  (edit here)
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR   = os.environ.get("DATA_DIR",
             os.path.join(os.path.dirname(os.path.abspath(__file__)), "price_data"))
ST_PERIOD  = 10
ST_MULT    = 3.0
CONF_W     = 5       # look back this many bars for a recent ST flip
FVG_PERIOD = 100     # rolling FVG POC window (bars)
FVG_SMOOTH = 10      # SMA smoothing on raw POC
SIGMA      = 2.0     # entry/exit threshold in standard deviations
MIN_BARS   = FVG_PERIOD + FVG_SMOOTH + CONF_W + 20   # minimum usable bars


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_csv(path: str) -> pd.DataFrame:
    """Load a yfinance-style CSV with 3 header rows."""
    df = pd.read_csv(path, header=0, skiprows=[1, 2],
                     index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Close"]).sort_index()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Supertrend
# ─────────────────────────────────────────────────────────────────────────────
def supertrend(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               period: int = 10, mult: float = 3.0):
    """
    Returns (trend, buy_sig, sell_sig).
      trend    : +1 = uptrend, -1 = downtrend
      buy_sig  : True on the bar where trend flips  -1 → +1
      sell_sig : True on the bar where trend flips  +1 → -1
    """
    N   = len(close)
    hl2 = (high + low) / 2.0

    tr    = np.empty(N); tr[0] = high[0] - low[0]
    for i in range(1, N):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i]  - close[i-1]))

    atr   = np.zeros(N); atr[0] = tr[0]; alpha = 1.0 / period
    for i in range(1, N):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]

    bu = hl2 - mult * atr
    bd = hl2 + mult * atr

    up    = np.empty(N); dn = np.empty(N)
    trend = np.ones(N, dtype=int)
    up[0] = bu[0]; dn[0] = bd[0]

    for i in range(1, N):
        up[i] = bu[i] if close[i-1] <= up[i-1] else max(bu[i], up[i-1])
        dn[i] = bd[i] if close[i-1] >= dn[i-1] else min(bd[i], dn[i-1])
        if   trend[i-1] == -1 and close[i] > dn[i-1]: trend[i] =  1
        elif trend[i-1] ==  1 and close[i] < up[i-1]: trend[i] = -1
        else:                                           trend[i] = trend[i-1]

    buy_sig  = np.zeros(N, bool); buy_sig[1:]  = (trend[1:] ==  1) & (trend[:-1] == -1)
    sell_sig = np.zeros(N, bool); sell_sig[1:] = (trend[1:] == -1) & (trend[:-1] ==  1)

    return trend, buy_sig, sell_sig


# ─────────────────────────────────────────────────────────────────────────────
# FVG Profile + Rolling POC  (volume-weighted spread)
# ─────────────────────────────────────────────────────────────────────────────
def compute_vw_spread(df: pd.DataFrame,
                      period: int = 100,
                      sma_smooth: int = 10) -> np.ndarray:
    """
    Returns vw_spread array (same length as df):
        vw_spread[i] = (close[i] - vw_poc[i]) / close[i] * 100

    Negative → price below FVG activity centre  (potential buy zone)
    Positive → price above FVG activity centre  (potential sell zone)
    """
    close  = df["Close"].values.astype(float)
    high   = df["High"].values.astype(float)
    low    = df["Low"].values.astype(float)
    volume = df["Volume"].values.astype(float)
    n      = len(close)

    # Detect FVGs (vectorised)
    # Bull FVG at bar t: high[t-2] < low[t]   → midpoint = (high[t-2]+low[t])/2
    # Bear FVG at bar t: low[t-2]  > high[t]  → midpoint = (low[t-2]+high[t])/2
    bull_mask = np.zeros(n, bool); bear_mask = np.zeros(n, bool)
    bull_mask[2:] = high[:-2] < low[2:]
    bear_mask[2:] = low[:-2]  > high[2:]

    bull_mid = np.zeros(n); bear_mid = np.zeros(n)
    bull_mid[2:] = (high[:-2] + low[2:])  / 2.0
    bear_mid[2:] = (low[:-2]  + high[2:]) / 2.0

    S = pd.Series

    # Volume-weighted rolling sums
    vw_bv  = S(np.where(bull_mask, bull_mid * volume, 0.0)).rolling(period).sum().values
    vw_dv  = S(np.where(bear_mask, bear_mid * volume, 0.0)).rolling(period).sum().values
    vw_bw  = S(np.where(bull_mask, volume,            0.0)).rolling(period).sum().values
    vw_dw  = S(np.where(bear_mask, volume,            0.0)).rolling(period).sum().values
    vw_tot = vw_bw + vw_dw

    raw_vw_poc = np.where(vw_tot > 0, (vw_bv + vw_dv) / vw_tot, np.nan)
    vw_poc     = S(raw_vw_poc).rolling(sma_smooth).mean().values

    vw_spread  = np.where(~np.isnan(vw_poc),
                          (close - vw_poc) / close * 100.0,
                          np.nan)
    return vw_spread


# ─────────────────────────────────────────────────────────────────────────────
# Signal checker  (last bar only)
# ─────────────────────────────────────────────────────────────────────────────
def check_signals(df: pd.DataFrame, ticker: str) -> dict | None:
    """
    Evaluate E ∩ FVG5 entry and exit conditions on the last bar.

    Returns a dict with keys:
        entry (bool), exit (bool),
        last_date, last_close,
        vw_spread_val, vw_mean, vw_std,
        st_trend, buy_flip_bars_ago, sell_flip_bars_ago
    Returns None if not enough data.
    """
    if len(df) < MIN_BARS:
        return None

    close  = df["Close"].values.astype(float)
    high   = df["High"].values.astype(float)
    low    = df["Low"].values.astype(float)

    # Indicators
    trend, buy_sig, sell_sig = supertrend(high, low, close, ST_PERIOD, ST_MULT)
    vw_spread                = compute_vw_spread(df, FVG_PERIOD, FVG_SMOOTH)

    # Thresholds: computed on the valid (non-warm-up) part of history
    warm      = FVG_PERIOD + FVG_SMOOTH + 1
    valid_sp  = vw_spread[warm:-1]           # exclude the very last bar from stats
    vw_mean   = float(np.nanmean(valid_sp))
    vw_std    = float(np.nanstd(valid_sp))
    thresh_lo = vw_mean - SIGMA * vw_std     # entry threshold (price below POC)
    thresh_hi = vw_mean + SIGMA * vw_std     # exit  threshold (price above POC)

    # Last-bar values
    last_close    = float(close[-1])
    last_date     = df.index[-1]
    last_spread   = float(vw_spread[-1]) if not np.isnan(vw_spread[-1]) else np.nan
    last_trend    = int(trend[-1])

    # Recent ST flip: look back CONF_W bars from today (inclusive)
    window        = buy_sig[-CONF_W:]
    buy_flip      = bool(window.any())
    buy_bars_ago  = int(CONF_W - 1 - np.where(window)[0][-1]) if buy_flip else None

    window        = sell_sig[-CONF_W:]
    sell_flip     = bool(window.any())
    sell_bars_ago = int(CONF_W - 1 - np.where(window)[0][-1]) if sell_flip else None

    # Signal conditions
    entry_ok = (buy_flip and
                not np.isnan(last_spread) and
                last_spread < thresh_lo)

    exit_ok  = (sell_flip and
                not np.isnan(last_spread) and
                last_spread > thresh_hi)

    return dict(
        ticker         = ticker,
        entry          = entry_ok,
        exit           = exit_ok,
        last_date      = last_date,
        last_close     = last_close,
        vw_spread      = last_spread,
        vw_mean        = vw_mean,
        vw_std         = vw_std,
        thresh_lo      = thresh_lo,
        thresh_hi      = thresh_hi,
        sigma_distance = ((last_spread - vw_mean) / vw_std
                          if vw_std > 0 and not np.isnan(last_spread) else np.nan),
        st_trend       = last_trend,
        buy_bars_ago   = buy_bars_ago,
        sell_bars_ago  = sell_bars_ago,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not os.path.isdir(DATA_DIR):
        sys.exit(f"[ERROR] DATA_DIR not found: {DATA_DIR}")

    csv_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv"))
    if not csv_files:
        sys.exit(f"[ERROR] No CSV files found in {DATA_DIR}")

    tickers = [os.path.splitext(f)[0] for f in csv_files]

    print("=" * 70)
    print(f"  E ∩ FVG5 Daily Scanner")
    print(f"  Run date : {datetime.now().strftime('%Y-%m-%d  %H:%M')}")
    print(f"  Universe : {len(tickers)} tickers  |  DATA_DIR: {DATA_DIR}")
    print(f"  Settings : ST({ST_PERIOD},{ST_MULT})  FVG({FVG_PERIOD},{FVG_SMOOTH})"
          f"  Sigma={SIGMA}  ConfWindow={CONF_W} bars")
    print("=" * 70)

    entry_signals = []
    exit_signals  = []
    skipped       = []

    for ticker in tickers:
        path = os.path.join(DATA_DIR, f"{ticker}.csv")
        try:
            df  = load_csv(path)
            res = check_signals(df, ticker)
            if res is None:
                skipped.append(ticker)
                continue
            if res["entry"]:
                entry_signals.append(res)
            if res["exit"]:
                exit_signals.append(res)
        except Exception as e:
            skipped.append(ticker)

    # ── Sort by how extreme the spread is ────────────────────────────────────
    # Entry: most negative spread first (biggest discount to POC)
    entry_signals.sort(key=lambda r: r["sigma_distance"])
    # Exit: most positive spread first (biggest premium to POC)
    exit_signals.sort(key=lambda r: r["sigma_distance"], reverse=True)

    # ── ENTRY list ────────────────────────────────────────────────────────────
    print(f"\n{'━'*70}")
    print(f"  ENTRY SIGNALS  (buy)  —  {len(entry_signals)} ticker(s)")
    print(f"  Condition: ST flipped BULLISH within last {CONF_W} bars")
    print(f"             AND vw_spread < mean − {SIGMA}σ  (price below FVG centre)")
    print(f"{'━'*70}")

    if entry_signals:
        hdr = (f"  {'Ticker':<8}  {'Date':<12}  {'Close':>8}  "
               f"{'Spread%':>9}  {'Threshold':>10}  {'σ-dist':>7}  "
               f"{'ST flip':>10}  ST")
        print(hdr)
        print(f"  {'-'*68}")
        for r in entry_signals:
            flip_str = (f"{r['buy_bars_ago']}d ago"
                        if r["buy_bars_ago"] is not None else "–")
            trend_str = "▲ up" if r["st_trend"] == 1 else "▼ dn"
            print(f"  {r['ticker']:<8}  "
                  f"{str(r['last_date'])[:10]:<12}  "
                  f"{r['last_close']:>8.2f}  "
                  f"{r['vw_spread']:>8.2f}%  "
                  f"{r['thresh_lo']:>9.2f}%  "
                  f"{r['sigma_distance']:>+6.2f}σ  "
                  f"{flip_str:>10}  {trend_str}")
    else:
        print("  (none today)")

    # ── EXIT list ─────────────────────────────────────────────────────────────
    print(f"\n{'━'*70}")
    print(f"  EXIT SIGNALS  (sell)  —  {len(exit_signals)} ticker(s)")
    print(f"  Condition: ST flipped BEARISH within last {CONF_W} bars")
    print(f"             AND vw_spread > mean + {SIGMA}σ  (price above FVG centre)")
    print(f"{'━'*70}")

    if exit_signals:
        hdr = (f"  {'Ticker':<8}  {'Date':<12}  {'Close':>8}  "
               f"{'Spread%':>9}  {'Threshold':>10}  {'σ-dist':>7}  "
               f"{'ST flip':>10}  ST")
        print(hdr)
        print(f"  {'-'*68}")
        for r in exit_signals:
            flip_str = (f"{r['sell_bars_ago']}d ago"
                        if r["sell_bars_ago"] is not None else "–")
            trend_str = "▲ up" if r["st_trend"] == 1 else "▼ dn"
            print(f"  {r['ticker']:<8}  "
                  f"{str(r['last_date'])[:10]:<12}  "
                  f"{r['last_close']:>8.2f}  "
                  f"{r['vw_spread']:>8.2f}%  "
                  f"{r['thresh_hi']:>9.2f}%  "
                  f"{r['sigma_distance']:>+6.2f}σ  "
                  f"{flip_str:>10}  {trend_str}")
    else:
        print("  (none today)")

    # ── Plain ticker lists (easy to copy) ─────────────────────────────────────
    print(f"\n{'━'*70}")
    print("  QUICK COPY  —  plain ticker lists")
    print(f"{'━'*70}")
    entry_tickers = [r["ticker"] for r in entry_signals]
    exit_tickers  = [r["ticker"] for r in exit_signals]
    print(f"  ENTRY ({len(entry_tickers)}): {', '.join(entry_tickers) if entry_tickers else '–'}")
    print(f"  EXIT  ({len(exit_tickers)}): {', '.join(exit_tickers)  if exit_tickers  else '–'}")

    # ── Footer ────────────────────────────────────────────────────────────────
    print(f"\n{'━'*70}")
    print(f"  Scanned {len(tickers)} tickers  |  "
          f"entry={len(entry_signals)}  exit={len(exit_signals)}  "
          f"skipped={len(skipped)}")
    if skipped:
        print(f"  Skipped: {', '.join(skipped[:15])}"
              + (" …" if len(skipped) > 15 else ""))
    print(f"{'━'*70}\n")

    print("Column guide")
    print("  Spread%   : (close − VW POC) / close × 100  "
          "(negative = price below POC)")
    print("  Threshold : mean ± 2σ of Spread% over full history")
    print("  σ-dist    : how many standard deviations from the mean")
    print("  ST flip   : how many bars ago the Supertrend direction changed")


if __name__ == "__main__":
    main()

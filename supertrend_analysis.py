"""
Supertrend Pattern Analysis
============================
Ports the PineScript v4 Supertrend indicator to Python and tests several
candidate patterns for predicting forward price returns on daily US stock data.

Patterns tested
---------------
A  Baseline        : any price trough (argrelextrema min, order=10)
B  Trough-in-UT    : trough while Supertrend = +1 (uptrend dip-to-buy)
C  Trough-in-DT    : trough while Supertrend = -1 (downtrend, catching knife)
D  ST Buy Signal   : Supertrend flips +1, regardless of extrema
E  Confluence      : ST buy signal fires within ±5 bars of a trough
F  ST Buy + Fisher : ST buy signal AND Fisher < 0 at that bar (oversold tilt)

Forward windows: 1 / 5 / 10 / 20 trading days.
Dataset: price_data/<TICKER>.csv  (505 US stocks, 5 years daily)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR     = os.environ.get("DATA_DIR",
               os.path.join(os.path.dirname(__file__), "price_data"))
ORDER        = 10       # argrelextrema neighbourhood
ST_PERIOD    = 10       # Supertrend ATR period
ST_MULT      = 3.0      # Supertrend ATR multiplier
CONFLUENCE_W = 5        # ±bars window for trough ↔ buy-signal confluence
LEN_PRICE    = 9        # Fisher price lookback (for pattern F)
FORWARD_DAYS = [1, 5, 10, 20]

_csv_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv")) \
             if os.path.isdir(DATA_DIR) else []
TICKERS = [os.path.splitext(f)[0] for f in _csv_files]
if not TICKERS:
    raise SystemExit(f"No CSV files found in {DATA_DIR}. "
                     "Run yfinance download first.")


# ─────────────────────────────────────────────────────────────────────────────
# Supertrend — Python port of PineScript v4
# ─────────────────────────────────────────────────────────────────────────────
def supertrend(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               period: int = 10, multiplier: float = 3.0):
    """
    Returns (trend, up_band, dn_band, buy_signal, sell_signal).

    trend      : +1 = uptrend, -1 = downtrend
    up_band    : support level when in uptrend
    dn_band    : resistance level when in downtrend
    buy_signal : True on the bar where trend flips to +1
    sell_signal: True on the bar where trend flips to -1

    Porting notes
    -------------
    * changeATR=True → Wilder ATR  (RMA, alpha=1/period)
    * up  ratchets upward  only when close[i-1] > up[i-1]
    * dn  ratchets downward only when close[i-1] < dn[i-1]
    * trend flip: -1→+1 when close > dn[i-1]; +1→-1 when close < up[i-1]
    """
    n = len(close)
    hl2 = (high + low) / 2.0

    # True Range
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i]  - close[i - 1]))

    # Wilder ATR (RMA, alpha = 1/period)
    atr = np.zeros(n)
    atr[0] = tr[0]
    alpha = 1.0 / period
    for i in range(1, n):
        atr[i] = alpha * tr[i] + (1.0 - alpha) * atr[i - 1]

    # Basic bands
    basic_up = hl2 - multiplier * atr
    basic_dn = hl2 + multiplier * atr

    # Ratcheted bands + trend
    up    = np.empty(n)
    dn    = np.empty(n)
    trend = np.ones(n, dtype=int)

    up[0] = basic_up[0]
    dn[0] = basic_dn[0]

    for i in range(1, n):
        # Support ratchets up; resistance ratchets down
        up[i] = basic_up[i] if close[i - 1] <= up[i - 1] \
                else max(basic_up[i], up[i - 1])
        dn[i] = basic_dn[i] if close[i - 1] >= dn[i - 1] \
                else min(basic_dn[i], dn[i - 1])

        if   trend[i - 1] == -1 and close[i] > dn[i - 1]:
            trend[i] = 1
        elif trend[i - 1] ==  1 and close[i] < up[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]

    buy_signal  = np.zeros(n, dtype=bool)
    sell_signal = np.zeros(n, dtype=bool)
    buy_signal[1:]  = (trend[1:] ==  1) & (trend[:-1] == -1)
    sell_signal[1:] = (trend[1:] == -1) & (trend[:-1] ==  1)

    return trend, up, dn, buy_signal, sell_signal


# ─────────────────────────────────────────────────────────────────────────────
# Fisher Transform (same as fisher_extrema_analysis.py — needed for pattern F)
# ─────────────────────────────────────────────────────────────────────────────
def fisher_transform(close: np.ndarray, len_price: int = 9):
    """Returns (fisher,) — only the raw Fisher series needed here."""
    n = len(close)
    s = pd.Series(close)
    max_h = s.rolling(len_price).max().values
    min_l = s.rolling(len_price).min().values

    v = np.zeros(n)
    for i in range(1, n):
        rng = max(max_h[i] - min_l[i], 1e-10)
        raw = 0.33 * 2.0 * ((close[i] - min_l[i]) / rng - 0.5) + 0.67 * v[i - 1]
        v[i] = max(-0.999, min(0.999, raw))

    return 0.5 * np.log((1.0 + v) / np.maximum(1.0 - v, 1e-10))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def fwd_ret(close: np.ndarray, locs: np.ndarray, fwd: int) -> np.ndarray:
    valid = locs[locs + fwd < len(close)]
    if len(valid) == 0:
        return np.array([])
    return (close[valid + fwd] - close[valid]) / close[valid] * 100.0


def stats(rets: np.ndarray) -> dict:
    if len(rets) == 0:
        return dict(n=0, mean=np.nan, median=np.nan, win_rate=np.nan)
    return dict(n=len(rets),
                mean    =float(np.mean(rets)),
                median  =float(np.median(rets)),
                win_rate=float(np.mean(rets > 0) * 100))


def load_csv(path: str) -> pd.DataFrame:
    """Load a yfinance-exported CSV (3-row header)."""
    df = pd.read_csv(path, header=0, skiprows=[1, 2],
                     index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=["Close"])


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
PATTERNS = ["A_baseline", "B_trough_UT", "C_trough_DT",
            "D_ST_buy",   "E_confluence", "F_ST_buy_fisher"]

all_rows = []
warm = ST_PERIOD + LEN_PRICE + 5   # enough warm-up for both indicators
max_fwd = max(FORWARD_DAYS)

print(f"[INFO] {len(TICKERS)} tickers  |  ST({ST_PERIOD},{ST_MULT})  "
      f"|  extrema order={ORDER}  |  confluence ±{CONFLUENCE_W} bars\n")

for idx, ticker in enumerate(TICKERS, 1):
    csv_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    try:
        df    = load_csv(csv_path)
        close = df["Close"].values.astype(float)
        high  = df["High"].values.astype(float)
        low   = df["Low"].values.astype(float)
        if len(close) < warm + max_fwd + ORDER:
            continue
    except Exception:
        continue

    # ── Indicators ──────────────────────────────────────────────────────────
    trend, up_band, dn_band, buy_sig, sell_sig = supertrend(
        high, low, close, ST_PERIOD, ST_MULT)
    fish = fisher_transform(close, LEN_PRICE)

    # ── Price extrema ────────────────────────────────────────────────────────
    all_min = argrelextrema(close, np.less_equal,   order=ORDER)[0]
    all_max = argrelextrema(close, np.greater_equal, order=ORDER)[0]

    # Remove bars too close to the edges (no forward data, insufficient warmup)
    edge_ok = lambda locs: locs[(locs >= warm) & (locs <= len(close) - max_fwd - 1)]
    troughs = edge_ok(all_min)

    # ── Signal sets for each pattern ─────────────────────────────────────────
    # A: all troughs
    sig_A = troughs

    # B: trough while ST = +1
    sig_B = troughs[trend[troughs] == 1]

    # C: trough while ST = -1
    sig_C = troughs[trend[troughs] == -1]

    # D: ST buy signal (any bar, no extrema requirement)
    buy_locs = np.where(buy_sig)[0]
    sig_D = edge_ok(buy_locs)

    # E: confluence — trough AND ST buy signal within ±CONFLUENCE_W bars
    conf_mask = np.zeros(len(close), dtype=bool)
    for b in buy_locs:
        lo, hi = max(0, b - CONFLUENCE_W), min(len(close), b + CONFLUENCE_W + 1)
        conf_mask[lo:hi] = True
    sig_E = troughs[conf_mask[troughs]]

    # F: ST buy signal AND Fisher < 0 at that bar
    sig_F = sig_D[fish[sig_D] < 0]

    signals = dict(A=sig_A, B=sig_B, C=sig_C, D=sig_D, E=sig_E, F=sig_F)

    for fwd in FORWARD_DAYS:
        for pat_key, locs in signals.items():
            s = stats(fwd_ret(close, locs, fwd))
            all_rows.append(dict(
                ticker   = ticker,
                pattern  = pat_key,
                fwd_days = fwd,
                n        = s["n"],
                mean_ret = s["mean"],
                median   = s["median"],
                win_rate = s["win_rate"],
            ))

    if idx % 50 == 0 or idx == len(TICKERS):
        print(f"  … {idx}/{len(TICKERS)} done")


# ─────────────────────────────────────────────────────────────────────────────
# Aggregated results
# ─────────────────────────────────────────────────────────────────────────────
df_all = pd.DataFrame(all_rows)
df_all.to_csv("supertrend_results_full.csv", index=False)
print(f"\n[INFO] Full results → supertrend_results_full.csv  "
      f"({df_all['ticker'].nunique()} tickers)")

PAT_LABELS = {
    "A": "A  Baseline (any trough)",
    "B": "B  Trough in Uptrend   (ST=+1)",
    "C": "C  Trough in Downtrend (ST=-1)",
    "D": "D  ST Buy Signal       (any bar)",
    "E": "E  Confluence          (trough ±5b of ST buy)",
    "F": "F  ST Buy + Fisher<0   (oversold)",
}

agg = (df_all
       .groupby(["pattern", "fwd_days"])
       .agg(
           tickers   = ("ticker",   "nunique"),
           total_n   = ("n",        "sum"),
           mean_ret  = ("mean_ret", "mean"),
           median    = ("median",   "mean"),
           win_rate  = ("win_rate", "mean"),
       )
       .round(2))

print("\n" + "=" * 75)
print(f"CROSS-TICKER PATTERN SUMMARY  ({df_all['ticker'].nunique()} US stocks)")
print("=" * 75)

for pat in sorted(PAT_LABELS):
    print(f"\n{PAT_LABELS[pat]}")
    print(f"  {'Days':>4}  {'Signals':>8}  {'MeanRet%':>9}  {'Median%':>8}  {'WinRate%':>9}")
    print("  " + "-" * 45)
    sub = agg.loc[pat] if pat in agg.index.get_level_values(0) else pd.DataFrame()
    for fwd in FORWARD_DAYS:
        if (pat, fwd) in agg.index:
            r = agg.loc[(pat, fwd)]
            print(f"  {fwd:>4}d  {int(r['total_n']):>8,}  "
                  f"{r['mean_ret']:>+9.2f}%  {r['median']:>+8.2f}%  "
                  f"{r['win_rate']:>8.1f}%")

# ── Head-to-head comparison table at each horizon ────────────────────────────
print("\n\n" + "=" * 75)
print("HEAD-TO-HEAD  (win rate % by pattern × forward window)")
print("=" * 75)
pivot_wr  = agg["win_rate"].unstack("fwd_days")
pivot_ret = agg["mean_ret"].unstack("fwd_days")
pivot_n   = agg["total_n"].unstack("fwd_days")

pivot_wr.index  = [PAT_LABELS[p] for p in pivot_wr.index]
pivot_ret.index = [PAT_LABELS[p] for p in pivot_ret.index]
pivot_n.index   = [PAT_LABELS[p] for p in pivot_n.index]

print("\nWin Rate (%)")
print(pivot_wr.to_string())
print("\nMean Return (%)")
print(pivot_ret.to_string())
print("\nTotal Signals (across all tickers)")
print(pivot_n.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# Summary chart  — win rate & mean return for all patterns at each horizon
# ─────────────────────────────────────────────────────────────────────────────
patterns_ordered = list(PAT_LABELS.keys())          # A … F
short_labels     = ["A\nBaseline", "B\nTrough+UT", "C\nTrough+DT",
                    "D\nST Buy", "E\nConfluence", "F\nST+Fisher"]
colors = ["#999999", "#2ca02c", "#d62728", "#1f77b4", "#ff7f0e", "#9467bd"]

fig, axes = plt.subplots(2, len(FORWARD_DAYS), figsize=(18, 9), sharey="row")
fig.suptitle(
    f"Supertrend Pattern Analysis  –  {df_all['ticker'].nunique()} US Stocks  (daily, 5y)\n"
    f"ST({ST_PERIOD}, {ST_MULT})  |  Extrema order={ORDER}  |  Confluence ±{CONFLUENCE_W} bars",
    fontsize=12, fontweight="bold"
)

for col, fwd in enumerate(FORWARD_DAYS):
    wr_vals  = [agg.loc[(p, fwd), "win_rate"]  if (p, fwd) in agg.index else np.nan
                for p in patterns_ordered]
    ret_vals = [agg.loc[(p, fwd), "mean_ret"]  if (p, fwd) in agg.index else np.nan
                for p in patterns_ordered]
    n_vals   = [int(agg.loc[(p, fwd), "total_n"]) if (p, fwd) in agg.index else 0
                for p in patterns_ordered]

    x = np.arange(len(patterns_ordered))

    # Win rate row
    ax = axes[0, col]
    bars = ax.bar(x, wr_vals, color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(50, color="gray", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_title(f"Forward {fwd}d", fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(short_labels, fontsize=8)
    if col == 0:
        ax.set_ylabel("Win Rate (%)", fontsize=9)
    ax.set_ylim(0, 115)
    for bar, wr, n in zip(bars, wr_vals, n_vals):
        if not np.isnan(wr):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{wr:.1f}%\n(N={n:,})", ha="center", va="bottom",
                    fontsize=6.5, linespacing=1.3)

    # Mean return row
    ax = axes[1, col]
    bars = ax.bar(x, ret_vals, color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(0, color="gray", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(short_labels, fontsize=8)
    if col == 0:
        ax.set_ylabel("Avg Return (%)", fontsize=9)
    for bar, rt in zip(bars, ret_vals):
        if not np.isnan(rt):
            ypos = bar.get_height() + (0.05 if rt >= 0 else -0.35)
            ax.text(bar.get_x() + bar.get_width()/2, ypos,
                    f"{rt:+.2f}%", ha="center", va="bottom", fontsize=7)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("supertrend_summary_chart.png", dpi=130, bbox_inches="tight")
plt.close()
print("\n[INFO] Chart saved → supertrend_summary_chart.png")

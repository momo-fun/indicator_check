"""
CVD (Cumulative Volume Delta) Pattern Analysis
===============================================
Approximates the CVD Profiles [TradingIQ] indicator from daily OHLCV data and
tests whether CVD delta / acceleration at price troughs adds predictive power —
both standalone and in combination with Pattern E (ST flip + trough confluence).

CVD approximation from daily OHLCV
------------------------------------
The original Pine indicator uses tick-level data via request.security_lower_tf.
With only daily bars we estimate:

    delta[i]  = volume[i] * (2*close[i] - high[i] - low[i]) / (high[i] - low[i])

This is the standard "bar delta" formula:
    buy_vol  ≈ volume * (close - low)  / (high - low)
    sell_vol ≈ volume * (high - close) / (high - low)
    delta    = buy_vol - sell_vol

    cvd[i]   = cumulative sum of delta    (same as Pine's `cvd`)
    accel[i] = delta[i] - delta[i-1]     (same as Pine's `cvdAccel`)

Patterns tested
---------------
L  Baseline            : any price trough (argrelextrema, order=10)
M  Trough + δ<0        : trough with net selling pressure (negative delta)
N  Trough + δ>0        : trough with net buying already present
O  Trough + accel>0    : trough where buying is accelerating
E  ST+Trough           : Supertrend flip ±5b of trough  ← previous best
P  E + δ<0 at flip     : Pattern E where delta is still negative (capitulation)
Q  E + δ>0 at flip     : Pattern E where delta already turned positive (confirmed)
R  E + accel>0 at flip : Pattern E with positive CVD acceleration at flip bar
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
ORDER        = 10     # argrelextrema neighbourhood
ST_PERIOD    = 10     # Supertrend ATR period
ST_MULT      = 3.0    # Supertrend multiplier
CONFLUENCE_W = 5      # ±bars for ST↔trough confluence (Pattern E)
FORWARD_DAYS = [1, 5, 10, 20]

_csv_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv")) \
             if os.path.isdir(DATA_DIR) else []
TICKERS = [os.path.splitext(f)[0] for f in _csv_files]
if not TICKERS:
    raise SystemExit(f"No CSV files in {DATA_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# CVD approximation from daily OHLCV
# ─────────────────────────────────────────────────────────────────────────────
def compute_cvd(high: np.ndarray, low: np.ndarray,
                close: np.ndarray, volume: np.ndarray):
    """
    Returns (cvd, delta, accel) — all shape (N,).

    delta[i]  : bar delta  = vol*(2*close-high-low)/(high-low)
    cvd[i]    : cumulative delta (∑ delta[0..i])
    accel[i]  : delta[i] - delta[i-1]  (Pine's cvdAccel)
    """
    hl    = np.where(high - low < 1e-10, 1e-10, high - low)
    delta = volume * (2.0 * close - high - low) / hl
    cvd   = np.cumsum(delta)
    accel = np.empty(len(delta))
    accel[0]  = 0.0
    accel[1:] = delta[1:] - delta[:-1]
    return cvd, delta, accel


# ─────────────────────────────────────────────────────────────────────────────
# Supertrend (same as previous scripts)
# ─────────────────────────────────────────────────────────────────────────────
def supertrend(high, low, close, period=10, multiplier=3.0):
    N   = len(close)
    hl2 = (high + low) / 2.0
    tr  = np.empty(N); tr[0] = high[0] - low[0]
    for i in range(1, N):
        tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    atr = np.zeros(N); atr[0] = tr[0]; alpha = 1.0 / period
    for i in range(1, N):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
    basic_up = hl2 - multiplier * atr
    basic_dn = hl2 + multiplier * atr
    up = np.empty(N); dn = np.empty(N); trend = np.ones(N, dtype=int)
    up[0] = basic_up[0]; dn[0] = basic_dn[0]
    for i in range(1, N):
        up[i] = basic_up[i] if close[i-1] <= up[i-1] else max(basic_up[i], up[i-1])
        dn[i] = basic_dn[i] if close[i-1] >= dn[i-1] else min(basic_dn[i], dn[i-1])
        if   trend[i-1] == -1 and close[i] > dn[i-1]: trend[i] =  1
        elif trend[i-1] ==  1 and close[i] < up[i-1]: trend[i] = -1
        else:                                           trend[i] = trend[i-1]
    buy_sig = np.zeros(N, dtype=bool)
    buy_sig[1:] = (trend[1:] == 1) & (trend[:-1] == -1)
    return trend, buy_sig


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def fwd_ret(close, locs, fwd):
    valid = locs[locs + fwd < len(close)]
    if len(valid) == 0:
        return np.array([])
    return (close[valid + fwd] - close[valid]) / close[valid] * 100.0

def stats(rets):
    if len(rets) == 0:
        return dict(n=0, mean=np.nan, median=np.nan, win_rate=np.nan)
    return dict(n=len(rets), mean=float(np.mean(rets)),
                median=float(np.median(rets)),
                win_rate=float(np.mean(rets > 0) * 100))

def load_csv(path):
    df = pd.read_csv(path, header=0, skiprows=[1, 2],
                     index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=["Close"])

def edge_ok(locs, warm, n, max_fwd):
    return locs[(locs >= warm) & (locs <= n - max_fwd - 1)]


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
warm    = ST_PERIOD + ORDER + 5
max_fwd = max(FORWARD_DAYS)
all_rows = []

print(f"[INFO] {len(TICKERS)} tickers  |  ST({ST_PERIOD},{ST_MULT})  "
      f"|  extrema order={ORDER}\n")

for idx, ticker in enumerate(TICKERS, 1):
    csv_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    try:
        df     = load_csv(csv_path)
        close  = df["Close"].values.astype(float)
        high   = df["High"].values.astype(float)
        low    = df["Low"].values.astype(float)
        volume = df["Volume"].values.astype(float)
        if len(close) < warm + max_fwd + 10:
            continue
    except Exception:
        continue

    N = len(close)

    # ── Indicators ──────────────────────────────────────────────────────────
    _, buy_sig           = supertrend(high, low, close, ST_PERIOD, ST_MULT)
    cvd, delta, accel    = compute_cvd(high, low, close, volume)

    # ── Price troughs ────────────────────────────────────────────────────────
    all_min = argrelextrema(close, np.less_equal, order=ORDER)[0]
    troughs = edge_ok(all_min, warm, N, max_fwd)

    # ── Pattern E: ST buy within ±CONFLUENCE_W bars of a trough ──────────────
    buy_locs  = np.where(buy_sig)[0]
    conf_mask = np.zeros(N, dtype=bool)
    for b in buy_locs:
        lo, hi = max(0, b - CONFLUENCE_W), min(N, b + CONFLUENCE_W + 1)
        conf_mask[lo:hi] = True
    sig_E = edge_ok(troughs[conf_mask[troughs]], warm, N, max_fwd)

    # ── CVD-based trough patterns ─────────────────────────────────────────────
    # L: all troughs (baseline)
    sig_L = troughs

    # M: trough with negative delta (net selling pressure)
    sig_M = troughs[delta[troughs] < 0]

    # N: trough with positive delta (net buying already present)
    sig_N = troughs[delta[troughs] >= 0]

    # O: trough with positive CVD acceleration (buying pressure accelerating)
    sig_O = troughs[accel[troughs] > 0]

    # ── CVD filters on Pattern E ──────────────────────────────────────────────
    # The E signal fires at the ST flip bar (buy_sig), not necessarily the trough.
    # For E signals, we check the delta/accel at the Pattern E bar itself.
    # We find the ST flip bar closest to (or at) each E trough.
    def e_signal_bar(e_trough_locs):
        """Return the ST flip bar closest to each Pattern E trough."""
        bars = []
        for t in e_trough_locs:
            nearby = buy_locs[np.abs(buy_locs - t) <= CONFLUENCE_W]
            if len(nearby):
                bars.append(int(nearby[np.argmin(np.abs(nearby - t))]))
        return np.array(bars, dtype=int)

    e_flip_bars = e_signal_bar(sig_E)

    # P: Pattern E where delta < 0 at the ST flip bar (net selling at flip)
    if len(e_flip_bars):
        p_mask = delta[e_flip_bars] < 0
        sig_P  = edge_ok(sig_E[p_mask], warm, N, max_fwd)
    else:
        sig_P = np.array([], dtype=int)

    # Q: Pattern E where delta > 0 at the ST flip bar (net buying at flip)
    if len(e_flip_bars):
        q_mask = delta[e_flip_bars] >= 0
        sig_Q  = edge_ok(sig_E[q_mask], warm, N, max_fwd)
    else:
        sig_Q = np.array([], dtype=int)

    # R: Pattern E where CVD acceleration > 0 at the ST flip bar
    if len(e_flip_bars):
        r_mask = accel[e_flip_bars] > 0
        sig_R  = edge_ok(sig_E[r_mask], warm, N, max_fwd)
    else:
        sig_R = np.array([], dtype=int)

    # ── Forward returns ──────────────────────────────────────────────────────
    signals = dict(L=sig_L, M=sig_M, N=sig_N, O=sig_O,
                   E=sig_E, P=sig_P, Q=sig_Q, R=sig_R)

    for fwd in FORWARD_DAYS:
        for pat, locs in signals.items():
            s = stats(fwd_ret(close, locs, fwd))
            all_rows.append(dict(
                ticker=ticker, pattern=pat, fwd_days=fwd,
                n=s["n"], mean_ret=s["mean"],
                median=s["median"], win_rate=s["win_rate"],
            ))

    if idx % 50 == 0 or idx == len(TICKERS):
        print(f"  … {idx}/{len(TICKERS)} done")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
df_all = pd.DataFrame(all_rows)
df_all.to_csv("cvd_results_full.csv", index=False)
print(f"\n[INFO] Results → cvd_results_full.csv  "
      f"({df_all['ticker'].nunique()} tickers)")

PAT_LABELS = {
    "L": "L  Baseline              (any trough)",
    "M": "M  Trough + δ<0          (net selling at trough — capitulation)",
    "N": "N  Trough + δ>0          (net buying already present at trough)",
    "O": "O  Trough + accel>0      (buying acceleration at trough)",
    "E": "E  ST+Trough             (Supertrend flip ±5b of trough)  ← prev best",
    "P": "P  E + δ<0 at flip       (still selling when ST flips)",
    "Q": "Q  E + δ>0 at flip       (buyers in when ST flips)",
    "R": "R  E + accel>0 at flip   (CVD acceleration positive at flip)",
}

agg = (df_all
       .groupby(["pattern", "fwd_days"])
       .agg(tickers  =("ticker",   "nunique"),
            total_n  =("n",        "sum"),
            mean_ret =("mean_ret", "mean"),
            median   =("median",   "mean"),
            win_rate =("win_rate", "mean"))
       .round(2))

print("\n" + "=" * 78)
print(f"CROSS-TICKER SUMMARY  ({df_all['ticker'].nunique()} US stocks, daily 5y)")
print("=" * 78)

for pat in ["L", "M", "N", "O", "E", "P", "Q", "R"]:
    print(f"\n{PAT_LABELS[pat]}")
    print(f"  {'Days':>4}  {'Signals':>8}  {'MeanRet%':>9}  {'Median%':>8}  {'WinRate%':>9}")
    print("  " + "-" * 46)
    for fwd in FORWARD_DAYS:
        if (pat, fwd) in agg.index:
            r = agg.loc[(pat, fwd)]
            print(f"  {fwd:>4}d  {int(r['total_n']):>8,}  "
                  f"{r['mean_ret']:>+9.2f}%  {r['median']:>+8.2f}%  "
                  f"{r['win_rate']:>8.1f}%")

# Pivot tables
print("\n\n" + "=" * 78)
print("HEAD-TO-HEAD")
print("=" * 78)
piv_wr  = agg["win_rate"].unstack("fwd_days")
piv_ret = agg["mean_ret"].unstack("fwd_days")
piv_n   = agg["total_n"].unstack("fwd_days")
for p in list(PAT_LABELS.keys()):
    for tbl in [piv_wr, piv_ret, piv_n]:
        if p in tbl.index:
            tbl.rename(index={p: PAT_LABELS[p]}, inplace=True)
print("\nWin Rate (%)\n" + piv_wr.to_string())
print("\nMean Return (%)\n" + piv_ret.to_string())
print("\nTotal Signals\n" + piv_n.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# Summary chart — two rows of panels
# ─────────────────────────────────────────────────────────────────────────────
patterns_ordered = ["L", "M", "N", "O", "E", "P", "Q", "R"]
short_labels = ["L\nBaseline", "M\nTrgh+δ<0", "N\nTrgh+δ>0",
                "O\nTrgh+acc", "E\nST+Trgh", "P\nE+δ<0",
                "Q\nE+δ>0",    "R\nE+acc>0"]
colors = ["#aec7e8", "#d62728", "#2ca02c", "#9467bd",
          "#ff7f0e", "#8c564b", "#17becf", "#e377c2"]

fig, axes = plt.subplots(2, len(FORWARD_DAYS), figsize=(20, 9), sharey="row")
fig.suptitle(
    f"CVD Pattern Analysis  –  {df_all['ticker'].nunique()} US Stocks (daily, 5y)\n"
    f"CVD delta/accel approximated from OHLCV  |  "
    f"ST({ST_PERIOD},{ST_MULT})  |  Extrema order={ORDER}",
    fontsize=11, fontweight="bold"
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
    ax.axhline(50, color="gray", lw=1, ls="--", alpha=0.5)
    ax.axhline(90, color="green", lw=0.8, ls=":", alpha=0.5, label="90%")
    ax.set_title(f"Forward {fwd}d", fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(short_labels, fontsize=7.5)
    if col == 0: ax.set_ylabel("Win Rate (%)", fontsize=9)
    ax.set_ylim(0, 120)
    for bar, wr, n in zip(bars, wr_vals, n_vals):
        if not np.isnan(wr):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                    f"{wr:.1f}%\nN={n:,}", ha="center", va="bottom",
                    fontsize=6, linespacing=1.3)

    # Mean return row
    ax = axes[1, col]
    bars = ax.bar(x, ret_vals, color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(0, color="gray", lw=1, ls="--", alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(short_labels, fontsize=7.5)
    if col == 0: ax.set_ylabel("Avg Return (%)", fontsize=9)
    for bar, rt in zip(bars, ret_vals):
        if not np.isnan(rt):
            ypos = bar.get_height() + (0.1 if rt >= 0 else -0.5)
            ax.text(bar.get_x() + bar.get_width()/2, ypos,
                    f"{rt:+.2f}%", ha="center", va="bottom", fontsize=6.5)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("cvd_summary_chart.png", dpi=130, bbox_inches="tight")
plt.close()
print("\n[INFO] Chart → cvd_summary_chart.png")

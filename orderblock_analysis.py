"""
Order Block Detector + Combination Pattern Analysis
====================================================
Ports LuxAlgo's PineScript v5 Order Block Detector to Python, then tests it
standalone and in combination with the best previous patterns.

Patterns tested
---------------
G  Bull OB formed        : Bullish OB confirmed (volume pivot bar, os=1)
H  Bull OB first touch   : Price first retraces into an active bullish OB zone
E  Confluence (ST+trough): Supertrend flips +1 within ±5 bars of argrelextrema trough
I  E + OB formed         : Pattern E signal AND a bull OB formed in the past 15 bars
J  OB touch + E nearby   : Bull OB first touch AND Pattern E fired in the past 15 bars

Porting notes (verified against plan agent output)
----------------------------------------------------
* ta.highest(length)   → rolling(length).max()  — current bar IS included, no shift
* high[length] at bar i → high[i-length]  (Pine [k] = k bars ago)
* phv fires at bar i when volume[i-length] is a strict pivot high (leftbars=rightbars=length)
  → in Python: phv_arr[pivot_idx] is set; formation loop checks phv_arr[i-length]
* os uses ffill to replicate Pine's var stateful carry-forward
* Bull OB: top=hl2[pivot], btm=low[pivot]; mitigated when target_bull < btm
* Bear OB: top=high[pivot], btm=hl2[pivot]; mitigated when target_bear > top
* Touch (bull): first bar after formation where low[j] <= ob_top, before mitigation
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
ORDER        = 10      # argrelextrema neighbourhood
OB_LENGTH    = 5       # LuxAlgo default: Volume Pivot Length
OB_MULT      = 'Wick'  # mitigation method
ST_PERIOD    = 10      # Supertrend ATR period
ST_MULT      = 3.0     # Supertrend multiplier
CONFLUENCE_W = 5       # ±bars for ST↔trough confluence (Pattern E)
COMBO_W      = 15      # look-back bars for OB↔E combination (Patterns I & J)
FORWARD_DAYS = [1, 5, 10, 20]

_csv_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv")) \
             if os.path.isdir(DATA_DIR) else []
TICKERS = [os.path.splitext(f)[0] for f in _csv_files]
if not TICKERS:
    raise SystemExit(f"No CSV files in {DATA_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# Order Block Detector — Python port of LuxAlgo PineScript v5
# ─────────────────────────────────────────────────────────────────────────────
def order_block_detector(high, low, close, volume,
                         length: int = 5, mitigation: str = 'Wick'):
    """
    Parameters
    ----------
    high, low, close, volume : np.ndarray shape (N,)
    length     : Volume Pivot Length (default 5)
    mitigation : 'Wick' or 'Close'

    Returns
    -------
    bull_formed      : int array — confirmation bar indices of bullish OBs
    bull_first_touch : int array — first bar price enters each bullish OB zone
    at_bull_ob       : bool array (N,) — True when price is inside any active bull OB
    bull_obs_detail  : list of dicts for detailed inspection
    """
    N   = len(close)
    hl2 = (high + low) / 2.0

    # ── Rolling highest / lowest  (window includes current bar) ───────────────
    upper = pd.Series(high).rolling(length).max().values   # max(high[i-L+1 .. i])
    lower = pd.Series(low).rolling(length).min().values    # min(low [i-L+1 .. i])

    if mitigation == 'Close':
        target_bull = pd.Series(close).rolling(length).min().values
        target_bear = pd.Series(close).rolling(length).max().values
    else:                                                   # 'Wick'
        target_bull = lower
        target_bear = upper

    # ── os : order structure ──────────────────────────────────────────────────
    # high_lag[i] = high[i-length]  (Pine: high[length] at bar i)
    high_lag        = np.empty(N); high_lag[:] = np.nan
    low_lag         = np.empty(N); low_lag[:]  = np.nan
    high_lag[length:] = high[:N - length]
    low_lag[length:]  = low[:N  - length]

    # os=0 (bearish structure): anchor high > current rolling max → price declining
    # os=1 (bullish structure): anchor low  < current rolling min → price rising
    raw_os  = np.where(high_lag > upper, 0.0,
              np.where(low_lag  < lower, 1.0, np.nan))
    os_arr  = (pd.Series(raw_os)
                 .ffill()
                 .fillna(0)
                 .astype(int)
                 .values)

    # ── Volume pivot high ─────────────────────────────────────────────────────
    # phv_arr[i] is set when volume[i] strictly exceeds all `length` bars on each side.
    # In Pine this fires at bar i, but Pine's OB formation accesses phv at bar i-length
    # (i.e., phv_arr[pivot_idx] where pivot_idx = i - length in the formation loop).
    phv_arr = np.full(N, np.nan)
    for i in range(length, N - length):
        v = volume[i]
        if v > volume[i - length:i].max() and v > volume[i + 1:i + length + 1].max():
            phv_arr[i] = v

    # ── OB Formation ──────────────────────────────────────────────────────────
    # Pine evaluates formation at bar i; pivot is at pivot_idx = i - length.
    # We use i as formed_at (the "confirmation" bar, earliest actionable bar).
    bull_obs = []
    bear_obs = []

    for i in range(2 * length, N):
        pivot_idx = i - length
        if np.isnan(phv_arr[pivot_idx]):
            continue

        if os_arr[i] == 1:                          # Bullish OB
            bull_obs.append({
                'formed_at':    i,
                'pivot_bar':    pivot_idx,
                'top':          hl2[pivot_idx],     # hl2 of pivot candle
                'btm':          low[pivot_idx],     # low  of pivot candle
                'mitigated_at': None,
                'touches':      [],
            })
        else:                                       # Bearish OB  (os == 0)
            bear_obs.append({
                'formed_at':    i,
                'pivot_bar':    pivot_idx,
                'top':          high[pivot_idx],    # high of pivot candle
                'btm':          hl2[pivot_idx],     # hl2  of pivot candle
                'mitigated_at': None,
                'touches':      [],
            })

    # ── Mitigation + Touch (vectorised per OB) ────────────────────────────────
    for ob in bull_obs:
        f   = ob['formed_at']
        end = N

        tb_slice = target_bull[f + 1:]              # target_bull after formation
        cross    = np.where(tb_slice < ob['btm'])[0]
        if len(cross):
            ob['mitigated_at'] = f + 1 + cross[0]
            end = ob['mitigated_at']

        # Touch: low enters OB zone (low ≤ ob_top) before mitigation
        t_idxs = np.where(low[f + 1:end] <= ob['top'])[0]
        ob['touches'] = (f + 1 + t_idxs).tolist()

    for ob in bear_obs:
        f   = ob['formed_at']
        end = N

        tb_slice = target_bear[f + 1:]
        cross    = np.where(tb_slice > ob['top'])[0]
        if len(cross):
            ob['mitigated_at'] = f + 1 + cross[0]
            end = ob['mitigated_at']

        t_idxs = np.where(high[f + 1:end] >= ob['btm'])[0]
        ob['touches'] = (f + 1 + t_idxs).tolist()

    # ── Build output arrays ───────────────────────────────────────────────────
    bull_formed = np.array([ob['formed_at'] for ob in bull_obs], dtype=int) \
                  if bull_obs else np.array([], dtype=int)

    bull_first_touch = np.array(
        [ob['touches'][0] for ob in bull_obs if ob['touches']], dtype=int)

    at_bull_ob = np.zeros(N, dtype=bool)
    for ob in bull_obs:
        for t in ob['touches']:
            at_bull_ob[t] = True

    return bull_formed, bull_first_touch, at_bull_ob, bull_obs


# ─────────────────────────────────────────────────────────────────────────────
# Supertrend (reused from supertrend_analysis.py)
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
warm    = max(2 * OB_LENGTH, ST_PERIOD) + ORDER + 5
max_fwd = max(FORWARD_DAYS)

all_rows = []
skipped  = []

print(f"[INFO] {len(TICKERS)} tickers  |  OB length={OB_LENGTH}  "
      f"|  ST({ST_PERIOD},{ST_MULT})  |  combo window={COMBO_W}b\n")

for idx, ticker in enumerate(TICKERS, 1):
    csv_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    try:
        df     = load_csv(csv_path)
        close  = df["Close"].values.astype(float)
        high   = df["High"].values.astype(float)
        low    = df["Low"].values.astype(float)
        volume = df["Volume"].values.astype(float)
        if len(close) < warm + max_fwd + 10:
            skipped.append(ticker); continue
    except Exception as e:
        skipped.append(ticker); continue

    N = len(close)

    # ── Indicators ──────────────────────────────────────────────────────────
    trend, buy_sig = supertrend(high, low, close, ST_PERIOD, ST_MULT)
    bull_formed, bull_first_touch, at_bull_ob, _ = order_block_detector(
        high, low, close, volume, OB_LENGTH, OB_MULT)

    # ── Price troughs ────────────────────────────────────────────────────────
    all_min  = argrelextrema(close, np.less_equal, order=ORDER)[0]
    troughs  = edge_ok(all_min, warm, N, max_fwd)

    # ── Pattern E: ST buy signal within ±CONFLUENCE_W bars of a trough ───────
    buy_locs   = np.where(buy_sig)[0]
    conf_mask  = np.zeros(N, dtype=bool)
    for b in buy_locs:
        lo, hi = max(0, b - CONFLUENCE_W), min(N, b + CONFLUENCE_W + 1)
        conf_mask[lo:hi] = True
    sig_E = edge_ok(troughs[conf_mask[troughs]], warm, N, max_fwd)

    # ── Pattern G: Bull OB formed ─────────────────────────────────────────────
    sig_G = edge_ok(bull_formed, warm, N, max_fwd)

    # ── Pattern H: Bull OB first touch ───────────────────────────────────────
    sig_H = edge_ok(bull_first_touch, warm, N, max_fwd)

    # ── Pattern I: Pattern E signal AND bull OB formed in past COMBO_W bars ──
    # Build a boolean mask: at each bar, was a bull OB formed within last COMBO_W bars?
    ob_formed_mask = np.zeros(N, dtype=bool)
    ob_formed_mask[bull_formed[bull_formed < N]] = True
    # Rolling window: any bull OB formed in [i-COMBO_W .. i]
    ob_recent = pd.Series(ob_formed_mask.astype(int)).rolling(COMBO_W + 1).max().fillna(0).values.astype(bool)
    sig_I = edge_ok(sig_E[ob_recent[sig_E]], warm, N, max_fwd)

    # ── Pattern J: OB first touch AND Pattern E in past COMBO_W bars ─────────
    # Build a boolean mask: was a Pattern E signal in past COMBO_W bars?
    e_mask  = np.zeros(N, dtype=bool)
    e_mask[sig_E[sig_E < N]] = True
    e_recent = pd.Series(e_mask.astype(int)).rolling(COMBO_W + 1).max().fillna(0).values.astype(bool)
    sig_J = edge_ok(sig_H[e_recent[sig_H]], warm, N, max_fwd)

    # ── Forward returns ──────────────────────────────────────────────────────
    signals = dict(G=sig_G, H=sig_H, E=sig_E, I=sig_I, J=sig_J)

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

if skipped:
    print(f"\n[WARN] Skipped {len(skipped)} tickers")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
df_all = pd.DataFrame(all_rows)
df_all.to_csv("orderblock_results_full.csv", index=False)
print(f"\n[INFO] Results → orderblock_results_full.csv  "
      f"({df_all['ticker'].nunique()} tickers)")

PAT_LABELS = {
    "G": "G  Bull OB Formed          (LuxAlgo OB, standalone)",
    "H": "H  Bull OB First Touch     (price retraces into OB zone)",
    "E": "E  ST+Trough Confluence    (Supertrend flip ±5b of trough)  ← prev best",
    "I": "I  E + OB formed ≤15b ago  (Pattern E while OB is fresh)",
    "J": "J  OB Touch + E ≤15b ago   (OB retracement after E signal)",
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

for pat in ["G", "H", "E", "I", "J"]:
    print(f"\n{PAT_LABELS[pat]}")
    print(f"  {'Days':>4}  {'Signals':>8}  {'MeanRet%':>9}  {'Median%':>8}  {'WinRate%':>9}")
    print("  " + "-" * 46)
    for fwd in FORWARD_DAYS:
        if (pat, fwd) in agg.index:
            r = agg.loc[(pat, fwd)]
            print(f"  {fwd:>4}d  {int(r['total_n']):>8,}  "
                  f"{r['mean_ret']:>+9.2f}%  {r['median']:>+8.2f}%  "
                  f"{r['win_rate']:>8.1f}%")

# Head-to-head pivot tables
print("\n\n" + "=" * 78)
print("HEAD-TO-HEAD  (win rate %)")
print("=" * 78)
pivot_wr  = agg["win_rate"].unstack("fwd_days")
pivot_ret = agg["mean_ret"].unstack("fwd_days")
pivot_n   = agg["total_n"].unstack("fwd_days")
for p in pivot_wr.index:
    if p in PAT_LABELS:
        pivot_wr  = pivot_wr.rename(index={p: PAT_LABELS[p]})
        pivot_ret = pivot_ret.rename(index={p: PAT_LABELS[p]})
        pivot_n   = pivot_n.rename(index={p: PAT_LABELS[p]})
print("\nWin Rate (%)\n" + pivot_wr.to_string())
print("\nMean Return (%)\n" + pivot_ret.to_string())
print("\nTotal Signals\n" + pivot_n.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# Summary chart
# ─────────────────────────────────────────────────────────────────────────────
patterns_ordered = ["G", "H", "E", "I", "J"]
short_labels = ["G\nOB Formed", "H\nOB Touch", "E\nST+Trough",
                "I\nE+OBformed", "J\nOBtouch+E"]
colors = ["#aec7e8", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

fig, axes = plt.subplots(2, len(FORWARD_DAYS), figsize=(18, 9), sharey="row")
fig.suptitle(
    f"Order Block + Combination Pattern Analysis  –  "
    f"{df_all['ticker'].nunique()} US Stocks (daily, 5y)\n"
    f"OB(length={OB_LENGTH}, {OB_MULT})  |  ST({ST_PERIOD},{ST_MULT})  "
    f"|  Extrema order={ORDER}  |  Combo window={COMBO_W}b",
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

    ax = axes[0, col]
    bars = ax.bar(x, wr_vals, color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(50,  color="gray", lw=1, ls="--", alpha=0.5)
    ax.axhline(90,  color="green", lw=0.8, ls=":", alpha=0.5)
    ax.set_title(f"Forward {fwd}d", fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(short_labels, fontsize=8)
    if col == 0: ax.set_ylabel("Win Rate (%)", fontsize=9)
    ax.set_ylim(0, 118)
    for bar, wr, n in zip(bars, wr_vals, n_vals):
        if not np.isnan(wr):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{wr:.1f}%\nN={n:,}", ha="center", va="bottom",
                    fontsize=6.5, linespacing=1.3)

    ax = axes[1, col]
    bars = ax.bar(x, ret_vals, color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(0, color="gray", lw=1, ls="--", alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(short_labels, fontsize=8)
    if col == 0: ax.set_ylabel("Avg Return (%)", fontsize=9)
    for bar, rt in zip(bars, ret_vals):
        if not np.isnan(rt):
            ypos = bar.get_height() + (0.1 if rt >= 0 else -0.5)
            ax.text(bar.get_x() + bar.get_width()/2, ypos,
                    f"{rt:+.2f}%", ha="center", va="bottom", fontsize=7)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("orderblock_summary_chart.png", dpi=130, bbox_inches="tight")
plt.close()
print("\n[INFO] Chart → orderblock_summary_chart.png")

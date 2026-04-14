"""
Fair Value Gap Profile + Rolling POC  —  Predictive Pattern Analysis
=====================================================================
Tests whether the BigBeluga FVG Profile + Rolling POC indicator has
predictive ability at price extrema detected by argrelextrema.

Indicator logic (ported from Pine Script)
-----------------------------------------
Bull FVG at bar t : high[t-2] < low[t]       → midpoint = (high[t-2] + low[t])  / 2
Bear FVG at bar t : low[t-2]  > high[t]      → midpoint = (low[t-2]  + high[t]) / 2

Rolling POC ≈ centre of FVG activity over the last PERIOD bars.
True POC = mode bin of midpoints.  Here we use a fast vectorised proxy:
    weighted mean of all FVG midpoints in the window
    (equal-weight and volume-weight variants).
The proxy is highly correlated with the true mode for typical price distributions.

Derived oscillator
------------------
    poc_spread = (close − poc) / close × 100   [percent]

Negative ⇒ price below FVG activity centre  (potential mean-reversion buy)
Positive ⇒ price above FVG activity centre  (potential mean-reversion sell)

Patterns tested at price TROUGHS
---------------------------------
  Baseline  : every argrelextrema trough
  P1        : poc_spread  <  mean − 2 σ          (extreme low spread)
  P2        : fvg_imbalance  >  0                (bull FVGs outnumber bear)
  P3        : POC divergence — price lower low,  POC same or higher
  P4        : P1 AND P2                           (strict combined filter)
  P1b       : poc_spread  <  mean − 1 σ          (moderate low spread)
  P5        : volume-weighted poc_spread  <  mean − 2 σ

Asymmetry check at price PEAKS (mirror — "win" = price falls after signal)
---------------------------------------------------------------------------
  Peak_base : every trough
  P1_peak   : poc_spread  >  mean + 2 σ
  P2_peak   : fvg_imbalance  <  0  (bear FVGs dominate)

Tickers : all CSVs found in price_data/  (505 US stocks, daily 5-year data)
Extrema : scipy.signal.argrelextrema, order = ORDER
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
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
PERIOD       = 100      # rolling POC lookback bars (Pine default 500; use 100 for speed)
SMA_SMOOTH   = 10       # SMA smoothing on raw POC  (matches Pine Script)
ORDER        = 10       # argrelextrema neighbourhood
# NOTE: with ORDER=10, bars [loc+1 .. loc+10] are all higher than the trough by
# definition, so 1/5/10-day forward returns are almost always positive regardless
# of indicator value — those windows are biased and uninformative.
# Use windows > ORDER to test genuine out-of-neighbourhood predictive ability.
FORWARD_DAYS = [10, 20, 30, 60]
CHART_TICKER_LIMIT = 20  # skip per-ticker charts when more tickers than this

DATA_DIR = os.environ.get(
    "DATA_DIR", os.path.join(os.path.dirname(__file__), "price_data"))

_csv_files = (sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv"))
              if os.path.isdir(DATA_DIR) else [])
TICKERS    = [os.path.splitext(f)[0] for f in _csv_files] or ["AAPL", "MSFT"]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_csv(path: str) -> pd.DataFrame:
    """Load a yfinance-format CSV (3 header rows: Price / Ticker / Date)."""
    df = pd.read_csv(path, header=0, skiprows=[1, 2],
                     index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Close"])
    df = df.sort_index()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FVG + Rolling POC computation  (fully vectorised, O(N) per stock)
# ─────────────────────────────────────────────────────────────────────────────
def compute_fvg_poc(df: pd.DataFrame,
                    period: int = 100,
                    sma_smooth: int = 10):
    """
    Returns a dict with arrays (all length N, aligned to df rows):
        poc          : equal-weight rolling POC price level  (SMA-smoothed)
        vw_poc       : volume-weight rolling POC              (SMA-smoothed)
        poc_spread   : (close − poc)    / close × 100
        vw_spread    : (close − vw_poc) / close × 100
        fvg_imbalance: (bull_count − bear_count) / total_count  ∈ [−1, +1]
        bull_cnt_roll: rolling bull FVG count (for further analysis)
        bear_cnt_roll: rolling bear FVG count
    """
    close  = df["Close"].values.astype(float)
    high   = df["High"].values.astype(float)
    low    = df["Low"].values.astype(float)
    volume = df["Volume"].values.astype(float)
    n      = len(close)

    # ── Detect FVGs (vectorised, no Python loop) ────────────────────────────
    # Bull FVG at bar t: high[t-2] < low[t]   (price gap up, 2-bar gap)
    # Bear FVG at bar t: low[t-2]  > high[t]  (price gap down, 2-bar gap)
    bull_mask = np.zeros(n, dtype=bool)
    bear_mask = np.zeros(n, dtype=bool)
    bull_mask[2:] = high[:-2] < low[2:]
    bear_mask[2:] = low[:-2]  > high[2:]

    # Midpoints
    bull_mid = np.zeros(n, dtype=float)
    bear_mid = np.zeros(n, dtype=float)
    bull_mid[2:] = (high[:-2] + low[2:])  / 2.0
    bear_mid[2:] = (low[:-2]  + high[2:]) / 2.0

    # ── Pandas rolling sums (fully vectorised) ───────────────────────────────
    s = pd.Series

    # Equal-weight
    ew_bull_val = s(np.where(bull_mask, bull_mid,          0.0)).rolling(period).sum().values
    ew_bear_val = s(np.where(bear_mask, bear_mid,          0.0)).rolling(period).sum().values
    ew_bull_cnt = s(bull_mask.astype(float)).rolling(period).sum().values
    ew_bear_cnt = s(bear_mask.astype(float)).rolling(period).sum().values
    ew_total    = ew_bull_cnt + ew_bear_cnt

    raw_poc = np.where(ew_total > 0,
                       (ew_bull_val + ew_bear_val) / ew_total,
                       np.nan)
    poc = s(raw_poc).rolling(sma_smooth).mean().values

    # Volume-weighted
    vw_bull_val = s(np.where(bull_mask, bull_mid * volume, 0.0)).rolling(period).sum().values
    vw_bear_val = s(np.where(bear_mask, bear_mid * volume, 0.0)).rolling(period).sum().values
    vw_bull_wt  = s(np.where(bull_mask, volume,            0.0)).rolling(period).sum().values
    vw_bear_wt  = s(np.where(bear_mask, volume,            0.0)).rolling(period).sum().values
    vw_total    = vw_bull_wt + vw_bear_wt

    raw_vw_poc = np.where(vw_total > 0,
                          (vw_bull_val + vw_bear_val) / vw_total,
                          np.nan)
    vw_poc = s(raw_vw_poc).rolling(sma_smooth).mean().values

    # ── Derived metrics ──────────────────────────────────────────────────────
    poc_spread    = np.where(~np.isnan(poc),    (close - poc)    / close * 100.0, np.nan)
    vw_spread     = np.where(~np.isnan(vw_poc), (close - vw_poc) / close * 100.0, np.nan)
    fvg_imbalance = np.where(ew_total > 0,
                             (ew_bull_cnt - ew_bear_cnt) / ew_total,
                             np.nan)

    return dict(
        poc=poc, vw_poc=vw_poc,
        poc_spread=poc_spread, vw_spread=vw_spread,
        fvg_imbalance=fvg_imbalance,
        bull_cnt_roll=ew_bull_cnt, bear_cnt_roll=ew_bear_cnt,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Forward-return helpers
# ─────────────────────────────────────────────────────────────────────────────
def forward_returns(close: np.ndarray, locs: np.ndarray, fwd: int) -> np.ndarray:
    """% return from close[loc] to close[loc+fwd]."""
    valid = locs[locs + fwd < len(close)]
    if len(valid) == 0:
        return np.array([])
    return (close[valid + fwd] - close[valid]) / close[valid] * 100.0


def stats_block(returns: np.ndarray) -> dict:
    if len(returns) == 0:
        return dict(n=0, mean=np.nan, median=np.nan, win_rate=np.nan)
    return dict(
        n        = len(returns),
        mean     = float(np.mean(returns)),
        median   = float(np.median(returns)),
        win_rate = float(np.mean(returns > 0) * 100.0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# POC Divergence detection
# ─────────────────────────────────────────────────────────────────────────────
def find_poc_divergence(close: np.ndarray,
                        poc:   np.ndarray,
                        trough_locs: np.ndarray) -> np.ndarray:
    """
    Bullish POC divergence: a PAIR of consecutive troughs (j, i) where
        close[i] < close[j]   (price lower low)
        poc[i]   >= poc[j]    (POC holds or rises  →  institutional support)
    Returns the bar indices of the second trough (the signal bar).
    """
    div_locs = []
    for k in range(1, len(trough_locs)):
        j = trough_locs[k - 1]
        i = trough_locs[k]
        if np.isnan(poc[i]) or np.isnan(poc[j]):
            continue
        if close[i] < close[j] and poc[i] >= poc[j]:
            div_locs.append(i)
    return np.array(div_locs, dtype=int)


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis loop
# ─────────────────────────────────────────────────────────────────────────────
all_results = []
skipped     = []
do_charts   = len(TICKERS) <= CHART_TICKER_LIMIT

min_bars = PERIOD + SMA_SMOOTH + ORDER + max(FORWARD_DAYS) + 20

print(f"[INFO] FVG Profile + Rolling POC  —  Prediction Analysis")
print(f"[INFO] Period={PERIOD}, SMA={SMA_SMOOTH}, extrema order={ORDER}")
print(f"[INFO] Processing {len(TICKERS)} tickers "
      f"({'with' if do_charts else 'without'} per-ticker charts) …\n")

if do_charts:
    fig_all = plt.figure(figsize=(22, len(TICKERS) * 8))
    gs_all  = gridspec.GridSpec(len(TICKERS), 2,
                                figure=fig_all, hspace=0.6, wspace=0.3)

for idx, ticker in enumerate(TICKERS, 1):
    csv_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if not os.path.isfile(csv_path):
        skipped.append((ticker, "no CSV"))
        continue
    try:
        df = load_csv(csv_path)
    except Exception as e:
        skipped.append((ticker, str(e)))
        continue

    if len(df) < min_bars:
        skipped.append((ticker, f"only {len(df)} bars < {min_bars}"))
        continue

    close = df["Close"].values.astype(float)

    # ── Compute indicator ────────────────────────────────────────────────────
    ind = compute_fvg_poc(df, PERIOD, SMA_SMOOTH)
    poc          = ind["poc"]
    vw_poc       = ind["vw_poc"]
    poc_spread   = ind["poc_spread"]
    vw_spread    = ind["vw_spread"]
    fvg_imbalance= ind["fvg_imbalance"]

    # ── Price extrema ─────────────────────────────────────────────────────────
    ilocs_min_all = argrelextrema(close, np.less_equal,    order=ORDER)[0]
    ilocs_max_all = argrelextrema(close, np.greater_equal, order=ORDER)[0]

    warm    = PERIOD + SMA_SMOOTH + 1
    max_fwd = max(FORWARD_DAYS)

    valid_min = ilocs_min_all[(ilocs_min_all >= warm) &
                              (ilocs_min_all <= len(close) - max_fwd - 1)]
    valid_max = ilocs_max_all[(ilocs_max_all >= warm) &
                              (ilocs_max_all <= len(close) - max_fwd - 1)]

    if len(valid_min) < 3:
        skipped.append((ticker, "too few troughs"))
        continue

    # ── Indicator statistics (exclude warm-up) ────────────────────────────────
    spread_vals = poc_spread[warm:]
    sp_mean = np.nanmean(spread_vals)
    sp_std  = np.nanstd(spread_vals)

    vw_spread_vals = vw_spread[warm:]
    vw_mean = np.nanmean(vw_spread_vals)
    vw_std  = np.nanstd(vw_spread_vals)

    # Thresholds
    thresh_lo    = sp_mean - 2.0 * sp_std   # P1 extreme trough
    thresh_lo_1s = sp_mean - 1.0 * sp_std   # P1b moderate trough
    thresh_hi    = sp_mean + 2.0 * sp_std   # P1_peak extreme peak
    vw_thresh_lo = vw_mean - 2.0 * vw_std   # P5 volume-weighted

    # ── Pattern masks at TROUGHS ──────────────────────────────────────────────
    spread_at_min  = poc_spread[valid_min]
    vw_at_min      = vw_spread[valid_min]
    imbal_at_min   = fvg_imbalance[valid_min]

    p1_mask  = spread_at_min  < thresh_lo         # extreme low spread
    p1b_mask = spread_at_min  < thresh_lo_1s      # moderate low spread (1σ)
    p2_mask  = imbal_at_min   > 0                 # bull FVGs dominant
    p4_mask  = p1_mask & p2_mask                  # P1 AND P2
    p5_mask  = vw_at_min < vw_thresh_lo           # volume-weighted extreme

    # P3: POC divergence
    div_locs  = find_poc_divergence(close, poc, valid_min)
    p3_mask   = np.isin(valid_min, div_locs)

    # ── Pattern masks at PEAKS (asymmetry / sell-side check) ─────────────────
    spread_at_max = poc_spread[valid_max]
    imbal_at_max  = fvg_imbalance[valid_max]

    p1_peak_mask = spread_at_max > thresh_hi       # price far above POC
    p2_peak_mask = imbal_at_max  < 0               # bear FVGs dominant at peak

    # ── Collect forward returns for every pattern ─────────────────────────────
    trough_patterns = {
        "baseline_trough" : valid_min,
        "P1_ExtremeLow"   : valid_min[p1_mask],
        "P1b_ModerateLow" : valid_min[p1b_mask],
        "P2_BullDominant" : valid_min[p2_mask],
        "P3_POCDivergence": valid_min[p3_mask],
        "P4_P1andP2"      : valid_min[p4_mask],
        "P5_VW_ExtremeLow": valid_min[p5_mask],
    }
    peak_patterns = {
        "baseline_peak"   : valid_max,
        "P1_ExtremeHigh"  : valid_max[p1_peak_mask],
        "P2p_BearDominant": valid_max[p2_peak_mask],
    }

    for fwd in FORWARD_DAYS:
        row = dict(
            ticker   = ticker,
            fwd_days = fwd,
            n_bars   = len(close),
            sp_mean  = round(sp_mean,  4),
            sp_std   = round(sp_std,   4),
            thresh_lo= round(thresh_lo,4),
            thresh_hi= round(thresh_hi,4),
        )
        for pat, locs in trough_patterns.items():
            ret = forward_returns(close, locs, fwd)
            sb  = stats_block(ret)
            row[f"{pat}_n"      ] = sb["n"]
            row[f"{pat}_mean"   ] = sb["mean"]
            row[f"{pat}_median" ] = sb["median"]
            row[f"{pat}_winrate"] = sb["win_rate"]

        # For peaks, "win" = price FALLS  ⇒ flip return sign
        for pat, locs in peak_patterns.items():
            ret = -forward_returns(close, locs, fwd)   # negative = win for short
            sb  = stats_block(ret)
            row[f"{pat}_n"      ] = sb["n"]
            row[f"{pat}_mean"   ] = sb["mean"]   # avg gain from being short
            row[f"{pat}_median" ] = sb["median"]
            row[f"{pat}_winrate"] = sb["win_rate"]

        all_results.append(row)

    # ── Per-ticker charts (only for small ticker sets) ────────────────────────
    if do_charts:
        ci = idx - 1
        ax1 = fig_all.add_subplot(gs_all[ci, 0])
        ax1.set_title(f"{ticker}  –  Price + POC", fontsize=10, fontweight="bold")
        ax1.plot(df.index, close,  color="black",  lw=0.9, label="Close")
        ax1.plot(df.index, poc,    color="orange",  lw=1.4, ls="--", label="Rolling POC")
        ax1.plot(df.index, vw_poc, color="magenta", lw=1.0, ls=":",  label="VW POC")
        if len(valid_min) > 0:
            ax1.scatter(df.index[valid_min], close[valid_min],
                        color="lime", marker="^", s=40, zorder=5, label="Trough")
        if p1_mask.any():
            ax1.scatter(df.index[valid_min[p1_mask]], close[valid_min[p1_mask]],
                        color="blue", marker="*", s=120, zorder=6,
                        label=f"P1 Extreme (n={p1_mask.sum()})")
        if p3_mask.any():
            ax1.scatter(df.index[valid_min[p3_mask]], close[valid_min[p3_mask]],
                        color="purple", marker="D", s=60, zorder=6,
                        label=f"P3 Diverge (n={p3_mask.sum()})")
        if len(valid_max) > 0:
            ax1.scatter(df.index[valid_max], close[valid_max],
                        color="red", marker="v", s=40, zorder=5, label="Peak")
        if p1_peak_mask.any():
            ax1.scatter(df.index[valid_max[p1_peak_mask]],
                        close[valid_max[p1_peak_mask]],
                        color="darkred", marker="x", s=100, zorder=6,
                        label=f"P1_peak (n={p1_peak_mask.sum()})")
        ax1.legend(fontsize=7, loc="upper left")
        ax1.set_ylabel("Price")
        ax1.tick_params(axis="x", labelrotation=30, labelsize=7)

        ax2 = fig_all.add_subplot(gs_all[ci, 1])
        ax2.set_title(f"{ticker}  –  POC Spread  (close−poc)/close×100",
                      fontsize=10, fontweight="bold")
        ax2.plot(df.index, poc_spread, color="steelblue", lw=1.0, label="EW spread")
        ax2.plot(df.index, vw_spread,  color="darkorchid", lw=0.8,
                 alpha=0.7, label="VW spread")
        ax2.axhline(sp_mean,  color="gray",       lw=0.8, ls="--", label="mean")
        ax2.axhline(thresh_lo, color="dodgerblue", lw=1.3, ls=":",
                    label=f"mean−2σ = {thresh_lo:+.1f}%")
        ax2.axhline(thresh_hi, color="tomato",    lw=1.3, ls=":",
                    label=f"mean+2σ = {thresh_hi:+.1f}%")
        ax2.axhline(0, color="black", lw=0.5)
        if len(valid_min) > 0:
            ax2.scatter(df.index[valid_min], poc_spread[valid_min],
                        color="lime", marker="^", s=40, zorder=5)
        if p1_mask.any():
            ax2.scatter(df.index[valid_min[p1_mask]], poc_spread[valid_min[p1_mask]],
                        color="blue", marker="*", s=120, zorder=6)
        if len(valid_max) > 0:
            ax2.scatter(df.index[valid_max], poc_spread[valid_max],
                        color="red", marker="v", s=40, zorder=5)
        if p1_peak_mask.any():
            ax2.scatter(df.index[valid_max[p1_peak_mask]],
                        poc_spread[valid_max[p1_peak_mask]],
                        color="darkred", marker="x", s=100, zorder=6)
        ax2.legend(fontsize=7, loc="upper left", ncol=2)
        ax2.set_ylabel("Spread (%)")
        ax2.tick_params(axis="x", labelrotation=30, labelsize=7)

    if idx % 50 == 0 or idx == len(TICKERS):
        print(f"  … {idx}/{len(TICKERS)} done")

if skipped:
    print(f"\n[WARN] Skipped {len(skipped)} tickers: {[t for t, _ in skipped[:10]]}"
          + (" …" if len(skipped) > 10 else ""))

if do_charts:
    plt.savefig("fvg_poc_analysis.png", dpi=110, bbox_inches="tight")
    plt.close()
    print("\n[INFO] Per-ticker chart → fvg_poc_analysis.png")


# ─────────────────────────────────────────────────────────────────────────────
# Aggregated cross-ticker summary
# ─────────────────────────────────────────────────────────────────────────────
if not all_results:
    print("[ERROR] No results collected.")
else:
    df_res = pd.DataFrame(all_results)
    df_res.to_csv("fvg_poc_results_full.csv", index=False)
    n_tickers = df_res["ticker"].nunique()
    print(f"\n[INFO] Full results → fvg_poc_results_full.csv "
          f"({len(df_res)} rows, {n_tickers} tickers)")

    # ── Define all patterns in display order ──────────────────────────────────
    TROUGH_PATS = [
        ("baseline_trough",  "Baseline (all troughs)",      "gray"),
        ("P1b_ModerateLow",  "P1b: spread < mean−1σ",       "cornflowerblue"),
        ("P1_ExtremeLow",    "P1:  spread < mean−2σ",        "steelblue"),
        ("P2_BullDominant",  "P2:  bull FVG dominant",       "forestgreen"),
        ("P3_POCDivergence", "P3:  POC divergence",          "purple"),
        ("P4_P1andP2",       "P4:  P1 ∩ P2 (combined)",      "crimson"),
        ("P5_VW_ExtremeLow", "P5:  VW spread < mean−2σ",     "darkorange"),
    ]
    PEAK_PATS = [
        ("baseline_peak",    "Baseline (all peaks)",         "gray"),
        ("P1_ExtremeHigh",   "P1_peak: spread > mean+2σ",    "tomato"),
        ("P2p_BearDominant", "P2_peak: bear FVG dominant",   "saddlebrown"),
    ]

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 85)
    print(f"BUY-SIDE PATTERNS AT PRICE TROUGHS  ({n_tickers} US stocks, daily 5y)")
    print("=" * 85)

    for fwd in FORWARD_DAYS:
        sub = df_res[df_res["fwd_days"] == fwd]
        base_wr = sub["baseline_trough_winrate"].mean()
        base_rt = sub["baseline_trough_mean"].mean()
        print(f"\n  Forward {fwd:2d}d   (baseline WR={base_wr:.1f}%, mean ret={base_rt:+.2f}%)")
        print(f"  {'Pattern':<30} {'N':>6}  {'Win%':>7}  {'ΔWR':>6}  {'MeanRet':>9}  Verdict")
        print(f"  {'-' * 78}")
        for pat, lbl, _ in TROUGH_PATS:
            n_   = int(sub[f"{pat}_n"].sum())
            wr   = sub[f"{pat}_winrate"].mean()
            rt   = sub[f"{pat}_mean"].mean()
            dwr  = wr - base_wr if pat != "baseline_trough" else 0.0
            if pat == "baseline_trough":
                verdict = "baseline"
            elif wr > 65 and dwr > 5:
                verdict = "★ STRONG EDGE"
            elif wr > 58 and dwr > 3:
                verdict = "◆ MODERATE EDGE"
            elif wr > 52 and dwr > 0:
                verdict = "▷ SLIGHT EDGE"
            else:
                verdict = "– no clear edge"
            print(f"  {lbl:<30} {n_:>6,}  {wr:>6.1f}%  {dwr:>+5.1f}pp {rt:>+9.2f}%  {verdict}")

    print("\n" + "=" * 85)
    print(f"SELL-SIDE PATTERNS AT PRICE PEAKS  (win = price falls, returns flipped)")
    print("=" * 85)

    for fwd in FORWARD_DAYS:
        sub = df_res[df_res["fwd_days"] == fwd]
        base_wr = sub["baseline_peak_winrate"].mean()
        base_rt = sub["baseline_peak_mean"].mean()
        print(f"\n  Forward {fwd:2d}d   (baseline WR={base_wr:.1f}%, mean ret={base_rt:+.2f}%)")
        print(f"  {'Pattern':<30} {'N':>6}  {'Win%':>7}  {'ΔWR':>6}  {'MeanRet':>9}  Verdict")
        print(f"  {'-' * 78}")
        for pat, lbl, _ in PEAK_PATS:
            n_   = int(sub[f"{pat}_n"].sum())
            wr   = sub[f"{pat}_winrate"].mean()
            rt   = sub[f"{pat}_mean"].mean()
            dwr  = wr - base_wr if pat != "baseline_peak" else 0.0
            if pat == "baseline_peak":
                verdict = "baseline"
            elif wr > 65 and dwr > 5:
                verdict = "★ STRONG EDGE"
            elif wr > 58 and dwr > 3:
                verdict = "◆ MODERATE EDGE"
            elif wr > 52 and dwr > 0:
                verdict = "▷ SLIGHT EDGE"
            else:
                verdict = "– no clear edge"
            print(f"  {lbl:<30} {n_:>6,}  {wr:>6.1f}%  {dwr:>+5.1f}pp {rt:>+9.2f}%  {verdict}")

    # ── Asymmetry test: do extreme-spread events cluster at troughs vs peaks? ──
    print("\n" + "=" * 85)
    print("ASYMMETRY CHECK  —  Do extreme signals skew toward troughs or peaks?")
    print("=" * 85)
    sub5 = df_res[df_res["fwd_days"] == 20]  # use 20d as representative window
    n_ext_trough = int(sub5["P1_ExtremeLow_n"].sum())
    n_ext_peak   = int(sub5["P1_ExtremeHigh_n"].sum())
    n_base_t     = int(sub5["baseline_trough_n"].sum())
    n_base_p     = int(sub5["baseline_peak_n"].sum())
    rate_t = n_ext_trough / max(n_base_t, 1) * 100
    rate_p = n_ext_peak   / max(n_base_p, 1) * 100
    print(f"\n  Extreme-low  signals at troughs : {n_ext_trough:,} / {n_base_t:,} "
          f"({rate_t:.1f}% of all troughs)")
    print(f"  Extreme-high signals at peaks   : {n_ext_peak:,}   / {n_base_p:,} "
          f"({rate_p:.1f}% of all peaks)")
    if abs(rate_t - rate_p) > 5:
        dom = "troughs" if rate_t > rate_p else "peaks"
        print(f"  → Extreme events are asymmetrically concentrated at {dom}  "
              f"(Δ = {abs(rate_t - rate_p):.1f}pp)")
    else:
        print(f"  → Extreme events appear roughly symmetric between troughs and peaks.")

    # ── Summary bar chart ─────────────────────────────────────────────────────
    fig2, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig2.suptitle(
        f"FVG Profile + Rolling POC  —  Predictive Pattern Analysis\n"
        f"{n_tickers} US Stocks · daily 5y · argrelextrema order={ORDER} · "
        f"POC lookback={PERIOD} bars",
        fontsize=13, fontweight="bold"
    )

    fwd_labels = [f"{f}d" for f in FORWARD_DAYS]
    x          = np.arange(len(FORWARD_DAYS))
    n_pats     = len(TROUGH_PATS)
    width      = 0.8 / n_pats

    def pat_bar_values(pat_list, metric):
        """Return (values, labels, colors) arrays for a given metric."""
        vals, lbls, cols = [], [], []
        for pat, lbl, col in pat_list:
            v = [df_res[df_res["fwd_days"] == f][f"{pat}_{metric}"].mean()
                 for f in FORWARD_DAYS]
            vals.append(v); lbls.append(lbl); cols.append(col)
        return vals, lbls, cols

    # Top-left: trough win rates
    ax = axes[0, 0]
    vals, lbls, cols = pat_bar_values(TROUGH_PATS, "winrate")
    for i, (v, lbl, col) in enumerate(zip(vals, lbls, cols)):
        offsets = x + (i - n_pats / 2 + 0.5) * width
        bars = ax.bar(offsets, v, width, label=lbl, color=col, alpha=0.82)
    ax.axhline(50, color="black", lw=1.2, ls="--", alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(fwd_labels)
    ax.set_ylabel("Win Rate (%)"); ax.set_title("Buy-Side Win Rate at Troughs")
    ax.set_ylim(0, 105); ax.legend(fontsize=8, loc="upper right")

    # Top-right: trough mean returns
    ax = axes[0, 1]
    vals, lbls, cols = pat_bar_values(TROUGH_PATS, "mean")
    for i, (v, lbl, col) in enumerate(zip(vals, lbls, cols)):
        offsets = x + (i - n_pats / 2 + 0.5) * width
        ax.bar(offsets, v, width, label=lbl, color=col, alpha=0.82)
    ax.axhline(0, color="black", lw=1.2, ls="--", alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(fwd_labels)
    ax.set_ylabel("Avg Forward Return (%)"); ax.set_title("Buy-Side Avg Return at Troughs")
    ax.legend(fontsize=8, loc="upper left")

    # Bottom-left: peak win rates (sell-side)
    n_ppats = len(PEAK_PATS)
    ax = axes[1, 0]
    vals, lbls, cols = pat_bar_values(PEAK_PATS, "winrate")
    for i, (v, lbl, col) in enumerate(zip(vals, lbls, cols)):
        offsets = x + (i - n_ppats / 2 + 0.5) * (0.8 / n_ppats)
        ax.bar(offsets, v, 0.8 / n_ppats, label=lbl, color=col, alpha=0.82)
    ax.axhline(50, color="black", lw=1.2, ls="--", alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(fwd_labels)
    ax.set_ylabel("Win Rate (%)"); ax.set_title("Sell-Side Win Rate at Peaks")
    ax.set_ylim(0, 105); ax.legend(fontsize=8)

    # Bottom-right: win rates at troughs vs peaks (20d window)
    ax = axes[1, 1]
    sub5 = df_res[df_res["fwd_days"] == 20]
    trough_wr_by_pat = [(lbl, sub5[f"{pat}_winrate"].mean()) for pat, lbl, _ in TROUGH_PATS]
    peak_wr_by_pat   = [(lbl, sub5[f"{pat}_winrate"].mean()) for pat, lbl, _ in PEAK_PATS]

    all_pats_20d = (
        [(lbl, wr, col, "Trough") for (pat, lbl, col), (_, wr) in
         zip(TROUGH_PATS, trough_wr_by_pat)]
        + [(lbl, wr, col, "Peak")  for (pat, lbl, col), (_, wr) in
           zip(PEAK_PATS, peak_wr_by_pat)]
    )
    pat_names = [a[0] for a in all_pats_20d]
    pat_wrs   = [a[1] for a in all_pats_20d]
    pat_cols  = [a[2] for a in all_pats_20d]
    y_pos     = np.arange(len(pat_names))

    bars = ax.barh(y_pos, pat_wrs, color=pat_cols, alpha=0.85)
    ax.axvline(50, color="black", lw=1.2, ls="--", alpha=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pat_names, fontsize=8)
    ax.set_xlabel("Win Rate (%)")
    ax.set_title("20-Day Win Rates — All Patterns Ranked")
    ax.set_xlim(0, 100)
    for bar, wr in zip(bars, pat_wrs):
        if not np.isnan(wr):
            ax.text(wr + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{wr:.1f}%", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig("fvg_poc_summary_chart.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("\n[INFO] Summary chart → fvg_poc_summary_chart.png")

    # ── Best pattern highlight ────────────────────────────────────────────────
    print("\n" + "─" * 85)
    print("BEST PATTERN SUMMARY  (ranked by 60-day win rate)")
    print("─" * 85)
    sub5 = df_res[df_res["fwd_days"] == 60]
    ranked = []
    for pat, lbl, _ in TROUGH_PATS:
        wr = sub5[f"{pat}_winrate"].mean()
        rt = sub5[f"{pat}_mean"].mean()
        n_ = int(sub5[f"{pat}_n"].sum())
        ranked.append((lbl, wr, rt, n_))
    ranked.sort(key=lambda x: x[1], reverse=True)
    for lbl, wr, rt, n_ in ranked:
        print(f"  {lbl:<32}  WR={wr:.1f}%  MeanRet={rt:+.2f}%  N={n_:,}")

    print("\nInterpretation")
    print("  ★ STRONG EDGE    : win-rate >65%, ≥5pp above baseline — actionable signal")
    print("  ◆ MODERATE EDGE  : win-rate >58%, ≥3pp above baseline — worth monitoring")
    print("  ▷ SLIGHT EDGE    : win-rate >52%, positive but small — use with confluence")
    print("  – no clear edge  : indicator does not reliably improve on any trough")

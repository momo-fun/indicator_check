"""
MTF Fisher Transform + Extrema Prediction Analysis
===================================================
Tests whether the Fisher Transform indicator (ported from PineScript) has
predictive ability at price troughs detected by argrelextrema.

Hypothesis:
    When the price is at a local trough AND Fisher < (mean - 2*std),
    the forward price returns should be positive more often than not.

Tickers : GL, AJG, WFC, NOW, SMG, MSGS, CRM, UBER  (daily)
Data     : yfinance (5 years), with synthetic GBM fallback when offline
Extrema  : scipy.signal.argrelextrema, order=10
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.signal import argrelextrema
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data generator (Geometric Brownian Motion)
# Used when yfinance cannot reach the network.
# Seeds are fixed per ticker so results are reproducible.
# ─────────────────────────────────────────────────────────────────────────────
_TICKER_SEEDS = {
    "GL":   10, "AJG":  20, "WFC":  30, "NOW":  40,
    "SMG":  50, "MSGS": 60, "CRM":  70, "UBER": 80,
}

def _make_synthetic(ticker: str, n_days: int = 1260) -> pd.DataFrame:
    """
    Generate n_days of daily OHLCV data using Geometric Brownian Motion.

    Parameters roughly calibrated to typical US mid/large-cap stocks:
      mu=8% annualised drift, sigma=25% annualised vol.
    Adds occasional mean-reverting 'shock' clusters so argrelextrema finds
    a reasonable number of local extrema at order=10.
    """
    rng  = np.random.default_rng(_TICKER_SEEDS.get(ticker, 0))
    mu   = 0.08 / 252          # daily drift
    sig  = 0.25 / np.sqrt(252) # daily vol

    # GBM log-returns with periodic larger shocks to create visible cycles
    base_ret = rng.normal(mu, sig, n_days)

    # Add mean-reverting shock bursts every ~60 bars
    shock_times = np.arange(55, n_days, 60)
    for st in shock_times:
        direction = rng.choice([-1, 1])
        burst = np.linspace(direction * 0.015, 0, 12)
        end   = min(st + 12, n_days)
        base_ret[st:end] += burst[:end - st]

    price = 100.0 * np.exp(np.cumsum(base_ret))

    # Realistic OHLCV from Close
    daily_range = np.abs(rng.normal(0, sig * 1.5, n_days)) * price
    high  = price + daily_range * 0.6
    low   = price - daily_range * 0.4
    open_ = np.roll(price, 1); open_[0] = price[0]
    vol   = rng.integers(500_000, 5_000_000, n_days).astype(float)

    dates = pd.date_range(end="2026-04-10", periods=n_days, freq="B")
    return pd.DataFrame({
        "Open": open_, "High": high, "Low": low,
        "Close": price, "Volume": vol,
    }, index=dates)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
PERIOD       = "5y"          # 5 years of daily data (yfinance)
ORDER        = 10            # argrelextrema neighbourhood
LEN_PRICE    = 9             # Fisher price lookback
LEN_RMA      = 9             # RMA (Wilder MA) length
LEN_LR       = 20            # Linear-regression length
BAND         = 2.0           # ±2 Fisher static bands (same as PineScript)
FORWARD_DAYS = [1, 5, 10, 20]  # forward windows to test

# Directory that holds pre-downloaded CSVs (one file per ticker).
# Override via env var:  DATA_DIR=/path/to/csvs python fisher_extrema_analysis.py
DATA_DIR     = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "price_data"))

# Auto-detect tickers from CSV files; fall back to original 8 if folder is empty.
_csv_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv")) if os.path.isdir(DATA_DIR) else []
TICKERS     = [os.path.splitext(f)[0] for f in _csv_files] or \
              ["GL", "AJG", "WFC", "NOW", "SMG", "MSGS", "CRM", "UBER"]

# Per-ticker charts are useful for small sets; skip when > this threshold.
CHART_TICKER_LIMIT = 20


# ─────────────────────────────────────────────────────────────────────────────
# Fisher Transform (Ehlers-style) – Python port of the PineScript indicator
# ─────────────────────────────────────────────────────────────────────────────
def fisher_transform(close: np.ndarray,
                     len_price: int = 9,
                     len_rma: int   = 9,
                     len_lr: int    = 20):
    """
    Returns (fisher, trigger, rma_fisher, lr_fisher) arrays aligned to `close`.

    Porting notes
    -------------
    * The recursive `_v` smoothing uses 0.33/0.67 weights (matches PineScript).
    * ta.rma  → Wilder MA  : alpha = 1/len_rma
    * ta.linreg(x, len, 0) → value at the most-recent bar of an OLS fit
    """
    n = len(close)
    s = pd.Series(close)

    max_h = s.rolling(len_price).max().values   # ta.highest
    min_l = s.rolling(len_price).min().values   # ta.lowest

    # ── Recursive normalised value (v) ──────────────────────────────────────
    v = np.zeros(n)
    for i in range(1, n):
        rng = max(max_h[i] - min_l[i], 1e-10)
        raw = 0.33 * 2.0 * ((close[i] - min_l[i]) / rng - 0.5) + 0.67 * v[i - 1]
        v[i] = max(-0.999, min(0.999, raw))

    # ── Fisher & trigger ────────────────────────────────────────────────────
    fish    = 0.5 * np.log((1.0 + v) / np.maximum(1.0 - v, 1e-10))
    trigger = np.empty(n)
    trigger[0] = 0.0
    trigger[1:] = fish[:-1]          # nz(fish[1]) in PineScript

    # ── RMA  (Wilder's smoothed MA) of Fisher ───────────────────────────────
    alpha = 1.0 / len_rma
    rma = np.zeros(n)
    rma[0] = fish[0]
    for i in range(1, n):
        rma[i] = alpha * fish[i] + (1.0 - alpha) * rma[i - 1]

    # ── Linear Regression of Fisher  (ta.linreg offset=0) ───────────────────
    lr = np.full(n, np.nan)
    for i in range(len_lr - 1, n):
        y = fish[i - len_lr + 1 : i + 1]
        x = np.arange(len_lr, dtype=float)
        # OLS coefficients
        xm, ym = x.mean(), y.mean()
        slope   = np.dot(x - xm, y - ym) / np.dot(x - xm, x - xm)
        intercept = ym - slope * xm
        lr[i]   = intercept + slope * (len_lr - 1)   # value at the last x

    return fish, trigger, rma, lr


# ─────────────────────────────────────────────────────────────────────────────
# Forward-return helpers
# ─────────────────────────────────────────────────────────────────────────────
def forward_returns(close: np.ndarray, locs: np.ndarray, fwd: int) -> np.ndarray:
    """% return from close[loc] to close[loc+fwd] for each loc."""
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
        win_rate = float(np.mean(returns > 0) * 100),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis loop
# ─────────────────────────────────────────────────────────────────────────────
all_results  = []    # one row per (ticker, fwd_days)
skipped      = []
do_charts    = len(TICKERS) <= CHART_TICKER_LIMIT

if do_charts:
    fig = plt.figure(figsize=(22, len(TICKERS) * 7))
    gs  = gridspec.GridSpec(len(TICKERS), 2, figure=fig, hspace=0.55, wspace=0.3)

print(f"[INFO] Processing {len(TICKERS)} tickers "
      f"({'with' if do_charts else 'without'} per-ticker charts) …\n")

for idx, ticker in enumerate(TICKERS, 1):
    # ── Data loading: local CSV → yfinance → synthetic GBM ──────────────────
    csv_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if os.path.isfile(csv_path):
        try:
            # yfinance CSVs have 3 header rows: Price / Ticker / Date label.
            # Skip rows 1 & 2, use row 0 as column names, col 0 as index.
            df = pd.read_csv(csv_path, header=0, skiprows=[1, 2],
                             index_col=0, parse_dates=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna(subset=["Close"])
            if df.empty:
                raise ValueError("empty after dropna")
            data_source = "CSV"
        except Exception as e:
            skipped.append((ticker, str(e)))
            continue
    else:
        try:
            df = yf.download(ticker, period=PERIOD, interval="1d",
                             auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty or "Close" not in df.columns:
                raise ValueError("empty download")
            df = df.dropna(subset=["Close"])
            data_source = "yfinance"
        except Exception:
            df = _make_synthetic(ticker)
            data_source = "synthetic"

    close = df["Close"].values.astype(float)

    # ── Fisher indicator ─────────────────────────────────────────────────────
    fish, trigger, rma_fish, lr_fish = fisher_transform(
        close, LEN_PRICE, LEN_RMA, LEN_LR)

    # ── Price extrema ────────────────────────────────────────────────────────
    ilocs_max_all = argrelextrema(close, np.greater_equal, order=ORDER)[0]
    ilocs_min_all = argrelextrema(close, np.less_equal,   order=ORDER)[0]

    warm    = LEN_PRICE + LEN_LR
    max_fwd = max(FORWARD_DAYS)
    valid_min = ilocs_min_all[(ilocs_min_all >= warm) &
                              (ilocs_min_all <= len(close) - max_fwd - 1)]
    valid_max = ilocs_max_all[(ilocs_max_all >= warm) &
                              (ilocs_max_all <= len(close) - max_fwd - 1)]

    # ── Fisher statistics ─────────────────────────────────────────────────────
    fish_series = fish[warm:]
    f_mean  = np.nanmean(fish_series)
    f_std   = np.nanstd(fish_series)
    thresh  = f_mean - 2.0 * f_std

    # ── Classify troughs ─────────────────────────────────────────────────────
    fisher_at_troughs = fish[valid_min]
    extreme_mask      = fisher_at_troughs < thresh
    extreme_locs      = valid_min[extreme_mask]
    normal_locs       = valid_min[~extreme_mask]

    # ── Forward returns ──────────────────────────────────────────────────────
    for fwd in FORWARD_DAYS:
        ext_ret = forward_returns(close, extreme_locs, fwd)
        nor_ret = forward_returns(close, normal_locs,  fwd)
        es = stats_block(ext_ret)
        ns = stats_block(nor_ret)
        all_results.append(dict(
            ticker       = ticker,
            data_source  = data_source,
            bars         = len(close),
            fwd_days     = fwd,
            f_mean       = round(f_mean, 4),
            f_std        = round(f_std,  4),
            thresh       = round(thresh, 4),
            n_troughs    = len(valid_min),
            ext_n        = es["n"],
            ext_mean_ret = es["mean"],
            ext_median   = es["median"],
            ext_win_rate = es["win_rate"],
            nor_n        = ns["n"],
            nor_mean_ret = ns["mean"],
            nor_median   = ns["median"],
            nor_win_rate = ns["win_rate"],
        ))

    # Progress heartbeat every 50 tickers
    if idx % 50 == 0 or idx == len(TICKERS):
        print(f"  … {idx}/{len(TICKERS)} done")

    # ── Per-ticker charts (only when ticker count is small) ──────────────────
    if do_charts:
        chart_idx = idx - 1
        ax1 = fig.add_subplot(gs[chart_idx, 0])
        ax1.set_title(f"{ticker}  –  Price with Extrema", fontsize=10, fontweight="bold")
        ax1.plot(df.index, close, color="black", linewidth=0.9, label="Close")
        if len(valid_min) > 0:
            ax1.scatter(df.index[valid_min], close[valid_min],
                        color="green", marker="^", s=50, zorder=5, label="Trough")
        if len(valid_max) > 0:
            ax1.scatter(df.index[valid_max], close[valid_max],
                        color="red", marker="v", s=50, zorder=5, label="Peak")
        if len(extreme_locs) > 0:
            ax1.scatter(df.index[extreme_locs], close[extreme_locs],
                        color="blue", marker="*", s=130, zorder=6,
                        label=f"Extreme (Fisher<{thresh:+.2f})")
        ax1.legend(fontsize=7, loc="upper left")
        ax1.set_ylabel("Price")
        ax1.tick_params(axis="x", labelrotation=30, labelsize=7)

        ax2 = fig.add_subplot(gs[chart_idx, 1])
        ax2.set_title(f"{ticker}  –  Fisher Transform", fontsize=10, fontweight="bold")
        ax2.plot(df.index, fish,     color="steelblue",  linewidth=1.4, label="Fisher")
        ax2.plot(df.index, trigger,  color="gray",       linewidth=0.8, alpha=0.6, label="Trigger")
        ax2.plot(df.index, rma_fish, color="royalblue",  linewidth=1.2, linestyle="--", label="RMA")
        ax2.plot(df.index, lr_fish,  color="darkorchid", linewidth=1.2, linestyle="--", label="LR")
        ax2.axhline(0,      color="gray",       linewidth=0.8)
        ax2.axhline( BAND,  color="lightgray",  linewidth=1.0, linestyle="--")
        ax2.axhline(-BAND,  color="lightgray",  linewidth=1.0, linestyle="--")
        ax2.axhline(thresh, color="darkorange", linewidth=1.5, linestyle=":",
                    label=f"mean−2σ={thresh:+.2f}")
        if len(valid_min) > 0:
            ax2.scatter(df.index[valid_min], fish[valid_min],
                        color="green", marker="^", s=50, zorder=5)
        if len(extreme_locs) > 0:
            ax2.scatter(df.index[extreme_locs], fish[extreme_locs],
                        color="blue", marker="*", s=130, zorder=6)
        ax2.legend(fontsize=7, loc="upper left", ncol=2)
        ax2.set_ylabel("Fisher Value")
        ax2.tick_params(axis="x", labelrotation=30, labelsize=7)

if skipped:
    print(f"\n[WARN] Skipped {len(skipped)} tickers: {[t for t,_ in skipped]}")

if do_charts:
    plt.savefig("fisher_analysis.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("\n[INFO] Per-ticker chart saved → fisher_analysis.png")


# ─────────────────────────────────────────────────────────────────────────────
# Cross-ticker aggregated summary + summary chart
# ─────────────────────────────────────────────────────────────────────────────
if not all_results:
    print("[ERROR] No results collected.")
else:
    df_res = pd.DataFrame(all_results)

    # Save full per-ticker results to CSV
    df_res.to_csv("fisher_results_full.csv", index=False)
    print(f"[INFO] Full results saved → fisher_results_full.csv  "
          f"({len(df_res)} rows, {df_res['ticker'].nunique()} tickers)")

    # Aggregate across all tickers
    agg = df_res.groupby("fwd_days").agg(
        tickers_with_data     = ("ticker",       "nunique"),
        total_extreme_signals = ("ext_n",        "sum"),
        total_normal_signals  = ("nor_n",        "sum"),
        ext_mean_ret          = ("ext_mean_ret", "mean"),
        ext_median_ret        = ("ext_median",   "mean"),
        ext_win_rate          = ("ext_win_rate", "mean"),
        nor_mean_ret          = ("nor_mean_ret", "mean"),
        nor_median_ret        = ("nor_median",   "mean"),
        nor_win_rate          = ("nor_win_rate", "mean"),
    ).round(2)

    print("\n" + "=" * 75)
    print(f"CROSS-TICKER SUMMARY  ({df_res['ticker'].nunique()} tickers, "
          f"Fisher < mean−2σ  vs  normal troughs)")
    print("=" * 75)
    print(agg.to_string())

    print("\nPrediction Verdict  (Extreme Trough vs Normal Trough)")
    print("-" * 75)
    verdicts = {}
    for fwd, row in agg.iterrows():
        ext_wr  = row["ext_win_rate"]
        nor_wr  = row["nor_win_rate"]
        ext_rt  = row["ext_mean_ret"]
        nor_rt  = row["nor_mean_ret"]
        dwr     = ext_wr - nor_wr
        drt     = ext_rt - nor_rt
        verdict = ("STRONG EDGE"   if ext_wr > 65 and dwr > 5
                   else "MODERATE EDGE" if ext_wr > 55 and dwr > 0
                   else "NO CLEAR EDGE")
        verdicts[fwd] = verdict
        n_ext = int(row["total_extreme_signals"])
        print(f"  {fwd:>2d}d  ExtWR={ext_wr:.1f}%  NorWR={nor_wr:.1f}%  "
              f"ΔWR={dwr:+.1f}pp  ΔReturn={drt:+.2f}%  "
              f"N={n_ext:,}  → {verdict}")

    print("\nInterpretation Guide")
    print("  STRONG EDGE    : Fisher<mean−2σ at troughs wins >65%, ≥5pp above normal.")
    print("  MODERATE EDGE  : Win-rate >55%, positive edge but weaker.")
    print("  NO CLEAR EDGE  : Indicator does not reliably outperform any trough.")

    # ── Summary chart ────────────────────────────────────────────────────────
    fwd_labels  = [f"{f}d" for f in agg.index]
    x           = np.arange(len(fwd_labels))
    width       = 0.35

    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle(
        f"Fisher Transform Extrema Analysis  –  {df_res['ticker'].nunique()} US Stocks  "
        f"(daily, 5y)\nExtreme trough = Fisher < mean−2σ at argrelextrema trough (order=10)",
        fontsize=11, fontweight="bold"
    )

    # Left: win rate comparison
    ax = axes[0]
    bars1 = ax.bar(x - width/2, agg["ext_win_rate"], width,
                   label="Extreme trough", color="steelblue", alpha=0.85)
    bars2 = ax.bar(x + width/2, agg["nor_win_rate"], width,
                   label="Normal trough",  color="lightcoral", alpha=0.85)
    ax.axhline(50, color="gray", linewidth=1, linestyle="--", label="50% baseline")
    ax.set_xticks(x); ax.set_xticklabels(fwd_labels)
    ax.set_ylabel("Win Rate (%)"); ax.set_title("Forward Win Rate")
    ax.set_ylim(0, 110); ax.legend(fontsize=9)
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8)

    # Right: mean return comparison
    ax = axes[1]
    bars3 = ax.bar(x - width/2, agg["ext_mean_ret"], width,
                   label="Extreme trough", color="steelblue", alpha=0.85)
    bars4 = ax.bar(x + width/2, agg["nor_mean_ret"], width,
                   label="Normal trough",  color="lightcoral", alpha=0.85)
    ax.axhline(0, color="gray", linewidth=1, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(fwd_labels)
    ax.set_ylabel("Avg Return (%)"); ax.set_title("Avg Forward Return")
    ax.legend(fontsize=9)
    for bar in list(bars3) + list(bars4):
        ypos = bar.get_height() + (0.1 if bar.get_height() >= 0 else -0.4)
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f"{bar.get_height():+.2f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig("fisher_summary_chart.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("\n[INFO] Summary chart saved → fisher_summary_chart.png")

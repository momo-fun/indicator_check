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
TICKERS      = ["GL", "AJG", "WFC", "NOW", "SMG", "MSGS", "CRM", "UBER"]
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
all_results = []   # rows for cross-ticker summary

n_tickers = len(TICKERS)
fig = plt.figure(figsize=(22, n_tickers * 7))
gs  = gridspec.GridSpec(n_tickers, 2, figure=fig, hspace=0.55, wspace=0.3)

for idx, ticker in enumerate(TICKERS):
    sep = "=" * 65
    print(f"\n{sep}\n  {ticker}\n{sep}")

    # ── Data loading: local CSV → yfinance → synthetic GBM ──────────────────
    #
    # Priority 1: local CSV  (price_data/<TICKER>.csv)
    #   Produce these on any internet-connected machine with:
    #     import yfinance as yf, os
    #     os.makedirs("price_data", exist_ok=True)
    #     for t in ["GL","AJG","WFC","NOW","SMG","MSGS","CRM","UBER"]:
    #         yf.download(t, period="5y", interval="1d",
    #                     auto_adjust=True, progress=False).to_csv(f"price_data/{t}.csv")
    #   then copy the price_data/ folder into this directory.
    #
    # Priority 2: yfinance live download (requires internet access)
    #
    # Priority 3: synthetic GBM — fully offline, reproducible, for smoke-testing

    csv_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if os.path.isfile(csv_path):
        # yfinance saves CSVs with 3 header rows:
        #   row 0: Price, Close, High, Low, Open, Volume
        #   row 1: Ticker, GL, GL, ...
        #   row 2: Date, , , ...
        # Skip rows 1 & 2, use row 0 as column names, col 0 as DatetimeIndex.
        df = pd.read_csv(csv_path, header=0, skiprows=[1, 2],
                         index_col=0, parse_dates=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Close"])
        data_source = f"local CSV ({csv_path})"
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
        except Exception as exc:
            print(f"  [WARN] yfinance failed ({exc.__class__.__name__}). "
                  f"Using synthetic GBM data.\n"
                  f"  Tip : place real CSVs in {DATA_DIR}/<TICKER>.csv to use real data.")
            df = _make_synthetic(ticker)
            data_source = "synthetic GBM"

    print(f"  Data source : {data_source}  |  bars={len(df)}")
    close = df["Close"].values.astype(float)

    # ── Fisher indicator ─────────────────────────────────────────────────────
    fish, trigger, rma_fish, lr_fish = fisher_transform(
        close, LEN_PRICE, LEN_RMA, LEN_LR)

    # ── Price extrema ────────────────────────────────────────────────────────
    ilocs_max_all = argrelextrema(close, np.greater_equal, order=ORDER)[0]
    ilocs_min_all = argrelextrema(close, np.less_equal,   order=ORDER)[0]

    # Require enough warm-up bars and at least max(FORWARD_DAYS) left
    warm    = LEN_PRICE + LEN_LR
    max_fwd = max(FORWARD_DAYS)
    valid_min = ilocs_min_all[(ilocs_min_all >= warm) &
                              (ilocs_min_all <= len(close) - max_fwd - 1)]
    valid_max = ilocs_max_all[(ilocs_max_all >= warm) &
                              (ilocs_max_all <= len(close) - max_fwd - 1)]

    # ── Fisher statistics (full series, post warm-up) ────────────────────────
    fish_series = fish[warm:]
    f_mean  = np.nanmean(fish_series)
    f_std   = np.nanstd(fish_series)
    thresh  = f_mean - 2.0 * f_std    # "extreme low" threshold

    print(f"  Fisher  mean={f_mean:+.3f}  std={f_std:.3f}  "
          f"threshold(mean-2σ)={thresh:+.3f}")
    print(f"  Troughs (valid): {len(valid_min)}  |  "
          f"Peaks (valid): {len(valid_max)}")

    # ── Classify troughs ─────────────────────────────────────────────────────
    fisher_at_troughs = fish[valid_min]
    extreme_mask      = fisher_at_troughs < thresh
    extreme_locs      = valid_min[extreme_mask]
    normal_locs       = valid_min[~extreme_mask]

    print(f"  Extreme troughs (Fisher < {thresh:+.3f}): {len(extreme_locs)}")
    print(f"  Normal  troughs                         : {len(normal_locs)}")

    # ── Forward returns ──────────────────────────────────────────────────────
    hdr = (f"  {'Days':>5s}  |  {'Extreme':>7s}  {'Ext-WR%':>8s}  "
           f"{'N-ext':>5s}  |  {'Normal':>7s}  {'Nor-WR%':>8s}  {'N-nor':>5s}")
    print(f"\n{hdr}")
    print("  " + "-" * (len(hdr) - 2))

    for fwd in FORWARD_DAYS:
        ext_ret  = forward_returns(close, extreme_locs, fwd)
        nor_ret  = forward_returns(close, normal_locs,  fwd)
        es = stats_block(ext_ret)
        ns = stats_block(nor_ret)

        print(f"  {fwd:>5d}d  |  "
              f"{es['mean']:>+7.2f}%  {es['win_rate']:>7.1f}%  {es['n']:>5d}  |  "
              f"{ns['mean']:>+7.2f}%  {ns['win_rate']:>7.1f}%  {ns['n']:>5d}")

        all_results.append(dict(
            ticker          = ticker,
            fwd_days        = fwd,
            # extreme-trough stats
            ext_n           = es["n"],
            ext_mean_ret    = es["mean"],
            ext_win_rate    = es["win_rate"],
            # normal-trough stats
            nor_n           = ns["n"],
            nor_mean_ret    = ns["mean"],
            nor_win_rate    = ns["win_rate"],
        ))

    # ── Plot: Price panel ────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[idx, 0])
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
                    label=f"Extreme trough\n(Fisher<{thresh:+.2f})")

    ax1.legend(fontsize=7, loc="upper left")
    ax1.set_ylabel("Price")
    ax1.tick_params(axis="x", labelrotation=30, labelsize=7)

    # ── Plot: Fisher panel ───────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[idx, 1])
    ax2.set_title(f"{ticker}  –  Fisher Transform", fontsize=10, fontweight="bold")

    ax2.plot(df.index, fish,     color="steelblue",   linewidth=1.4, label="Fisher")
    ax2.plot(df.index, trigger,  color="gray",        linewidth=0.8, alpha=0.6, label="Trigger")
    ax2.plot(df.index, rma_fish, color="royalblue",   linewidth=1.2, linestyle="--", label="RMA(Fisher)")
    ax2.plot(df.index, lr_fish,  color="darkorchid",  linewidth=1.2, linestyle="--", label="LR(Fisher)")

    ax2.axhline(0,      color="gray",      linewidth=0.8, linestyle="-")
    ax2.axhline( BAND,  color="lightgray", linewidth=1.0, linestyle="--", label=f"±{BAND}")
    ax2.axhline(-BAND,  color="lightgray", linewidth=1.0, linestyle="--")
    ax2.axhline(thresh, color="darkorange",linewidth=1.5, linestyle=":", label=f"mean−2σ={thresh:+.2f}")

    if len(valid_min) > 0:
        ax2.scatter(df.index[valid_min], fish[valid_min],
                    color="green", marker="^", s=50, zorder=5)
    if len(extreme_locs) > 0:
        ax2.scatter(df.index[extreme_locs], fish[extreme_locs],
                    color="blue", marker="*", s=130, zorder=6)

    ax2.legend(fontsize=7, loc="upper left", ncol=2)
    ax2.set_ylabel("Fisher Value")
    ax2.tick_params(axis="x", labelrotation=30, labelsize=7)

plt.savefig("fisher_analysis.png", dpi=120, bbox_inches="tight")
plt.close()
print("\n[INFO] Chart saved → fisher_analysis.png")


# ─────────────────────────────────────────────────────────────────────────────
# Cross-ticker aggregated summary
# ─────────────────────────────────────────────────────────────────────────────
if all_results:
    df_res = pd.DataFrame(all_results)

    print("\n" + "=" * 65)
    print("CROSS-TICKER SUMMARY  –  Extreme Troughs (Fisher < mean−2σ)")
    print("=" * 65)

    agg = df_res.groupby("fwd_days").agg(
        total_extreme_signals = ("ext_n",        "sum"),
        avg_return_pct        = ("ext_mean_ret", "mean"),
        avg_win_rate_pct      = ("ext_win_rate", "mean"),
        avg_normal_return_pct = ("nor_mean_ret", "mean"),
        avg_normal_win_rate   = ("nor_win_rate", "mean"),
    )

    print(agg.round(2).to_string())

    print("\nPrediction Verdict  (Extreme Trough vs Normal Trough)")
    print("-" * 65)
    for fwd, row in agg.iterrows():
        ext_wr = row["avg_win_rate_pct"]
        ext_rt = row["avg_return_pct"]
        nor_wr = row["avg_normal_win_rate"]
        nor_rt = row["avg_normal_return_pct"]

        edge_wr = ext_wr - nor_wr
        edge_rt = ext_rt - nor_rt

        verdict = ("STRONG EDGE" if ext_wr > 65 and edge_wr > 5
                   else "MODERATE EDGE" if ext_wr > 55 and edge_wr > 0
                   else "NO CLEAR EDGE")

        print(f"  {fwd:>2d}d  ExtWR={ext_wr:.1f}%  NorWR={nor_wr:.1f}%  "
              f"ΔWR={edge_wr:+.1f}pp  ΔReturn={edge_rt:+.2f}%  → {verdict}")

    print("\nInterpretation Guide")
    print("  STRONG EDGE    : Fisher<mean−2σ at troughs wins >65% of the time,")
    print("                   at least 5pp better win-rate than ordinary troughs.")
    print("  MODERATE EDGE  : Win-rate > 55%, positive edge but weaker.")
    print("  NO CLEAR EDGE  : Indicator does not reliably predict upward moves.")

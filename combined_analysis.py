"""
Multi-Indicator Combination Analysis
=====================================
Tests whether combining multiple indicators improves prediction over any
single indicator alone. All conditions are evaluated at the price TROUGH bar
(argrelextrema min, order=10) for a clean apples-to-apples comparison.

Indicators ported
-----------------
  Supertrend    : ATR(10) × 3.0  (PineScript v4 port)
  Fisher        : Ehlers-style recursive, len=9  (same as fisher_extrema_analysis.py)
  CVD           : bar-delta approximation from daily OHLCV
  Order Block   : LuxAlgo bull OB detector, length=5  (orderblock_analysis.py)
  FVG POC       : BigBeluga FVG Profile rolling POC, period=100  (fvg_poc_analysis.py)

Signals (all evaluated at the trough bar index t)
--------------------------------------------------
  E    : ST buy flip within ±5 bars of t              (Pattern E  — ST best)
  R    : CVD acceleration > 0 at t                    (Pattern R  — CVD best)
  Q    : CVD delta       > 0 at t
  Fext : Fisher < mean − 2σ at t                      (Fisher best)
  OB   : Bull OB formed in [t−15, t]                  (OB combo best)
  FVG1 : FVG poc_spread  < mean − 2σ at t             (FVG P1/P5 best)
  FVG5 : FVG vw_spread   < mean − 2σ at t             (volume-weighted variant)

Combination patterns tested (all at troughs)
---------------------------------------------
  baseline   : every trough
  E          : Supertrend confluence (previous champion)
  E_R        : E ∩ R  (best from CVD analysis)
  E_FVG1     : E ∩ FVG1            ← NEW
  E_FVG5     : E ∩ FVG5            ← NEW
  E_Fext     : E ∩ Fext
  E_OB       : E ∩ OB
  E_R_FVG1   : E ∩ R ∩ FVG1       ← NEW three-way
  E_R_FVG5   : E ∩ R ∩ FVG5       ← NEW three-way
  E_Q_FVG1   : E ∩ Q ∩ FVG1
  E_FVG1_Fext: E ∩ FVG1 ∩ Fext
  E_R_FVG1_Fx: E ∩ R ∩ FVG1 ∩ Fext  (four-way maximum filter)
  FVG1       : FVG1 standalone (no E requirement)
  FVG1_R     : FVG1 ∩ R  (without E)
  FVG1_Fext  : FVG1 ∩ Fext (without E)

Forward windows
---------------
  20 / 30 / 60 trading days  (windows > ORDER=10 so not trivially biased)

Dataset
-------
  price_data/ — 503+ US stocks, daily, 5 years
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
CONF_W       = 5      # ±bars for ST↔trough confluence (Pattern E)
OB_LENGTH    = 5      # LuxAlgo order-block pivot length
OB_COMBO_W   = 15     # look-back for OB near trough
FVG_PERIOD   = 100    # rolling FVG POC lookback
FVG_SMOOTH   = 10     # SMA smoothing on FVG POC
FISHER_LEN   = 9      # Fisher price lookback
FORWARD_DAYS = [20, 30, 60]

_csv_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv")) \
             if os.path.isdir(DATA_DIR) else []
TICKERS = [os.path.splitext(f)[0] for f in _csv_files]
if not TICKERS:
    raise SystemExit(f"No CSV files in {DATA_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_csv(path):
    df = pd.read_csv(path, header=0, skiprows=[1, 2],
                     index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=["Close"]).sort_index()


# ─────────────────────────────────────────────────────────────────────────────
# Supertrend
# ─────────────────────────────────────────────────────────────────────────────
def supertrend(high, low, close, period=10, mult=3.0):
    N   = len(close)
    hl2 = (high + low) / 2.0
    tr  = np.empty(N); tr[0] = high[0] - low[0]
    for i in range(1, N):
        tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    atr = np.zeros(N); atr[0] = tr[0]; alpha = 1.0 / period
    for i in range(1, N):
        atr[i] = alpha * tr[i] + (1-alpha) * atr[i-1]
    bu  = hl2 - mult * atr
    bd  = hl2 + mult * atr
    up  = np.empty(N); dn = np.empty(N); trend = np.ones(N, dtype=int)
    up[0] = bu[0]; dn[0] = bd[0]
    for i in range(1, N):
        up[i] = bu[i] if close[i-1] <= up[i-1] else max(bu[i], up[i-1])
        dn[i] = bd[i] if close[i-1] >= dn[i-1] else min(bd[i], dn[i-1])
        if   trend[i-1] == -1 and close[i] > dn[i-1]: trend[i] =  1
        elif trend[i-1] ==  1 and close[i] < up[i-1]: trend[i] = -1
        else:                                           trend[i] = trend[i-1]
    buy_sig = np.zeros(N, dtype=bool)
    buy_sig[1:] = (trend[1:] == 1) & (trend[:-1] == -1)
    return buy_sig


# ─────────────────────────────────────────────────────────────────────────────
# Fisher Transform  (Ehlers recursive, matching fisher_extrema_analysis.py)
# ─────────────────────────────────────────────────────────────────────────────
def fisher_transform(close, len_price=9):
    n   = len(close)
    s   = pd.Series(close)
    mxh = s.rolling(len_price).max().values
    mnl = s.rolling(len_price).min().values
    v   = np.zeros(n)
    for i in range(1, n):
        rng = max(mxh[i] - mnl[i], 1e-10)
        raw = 0.33 * 2.0 * ((close[i] - mnl[i]) / rng - 0.5) + 0.67 * v[i-1]
        v[i] = max(-0.999, min(0.999, raw))
    return 0.5 * np.log((1.0 + v) / np.maximum(1.0 - v, 1e-10))


# ─────────────────────────────────────────────────────────────────────────────
# CVD (bar-delta approximation from daily OHLCV)
# ─────────────────────────────────────────────────────────────────────────────
def compute_cvd(high, low, close, volume):
    hl    = np.where(high - low < 1e-10, 1e-10, high - low)
    delta = volume * (2.0 * close - high - low) / hl
    accel = np.empty(len(delta)); accel[0] = 0.0; accel[1:] = delta[1:] - delta[:-1]
    return delta, accel


# ─────────────────────────────────────────────────────────────────────────────
# Order Block detector  (LuxAlgo bull OB, matching orderblock_analysis.py)
# ─────────────────────────────────────────────────────────────────────────────
def bull_ob_formed(high, low, close, volume, length=5):
    """Return bool array (N,): True on bars where a bull OB was confirmed."""
    N   = len(close)
    hl2 = (high + low) / 2.0

    upper = pd.Series(high).rolling(length).max().values
    lower = pd.Series(low).rolling(length).min().values
    target_bull = lower   # Wick mitigation

    high_lag = np.full(N, np.nan); high_lag[length:] = high[:N-length]
    low_lag  = np.full(N, np.nan); low_lag[length:]  = low[:N-length]

    raw_os = np.where(high_lag > upper, 0.0,
             np.where(low_lag  < lower, 1.0, np.nan))
    os_arr = pd.Series(raw_os).ffill().fillna(0).astype(int).values

    # Volume pivot high
    phv = np.full(N, np.nan)
    for i in range(length, N - length):
        v = volume[i]
        if v > volume[i-length:i].max() and v > volume[i+1:i+length+1].max():
            phv[i] = v

    ob_formed = np.zeros(N, dtype=bool)
    for i in range(2*length, N):
        pivot = i - length
        if not np.isnan(phv[pivot]) and os_arr[i] == 1:
            ob_formed[i] = True

    return ob_formed


# ─────────────────────────────────────────────────────────────────────────────
# FVG Profile + Rolling POC  (BigBeluga, matching fvg_poc_analysis.py)
# ─────────────────────────────────────────────────────────────────────────────
def compute_fvg_poc(df, period=100, sma_smooth=10):
    close  = df["Close"].values.astype(float)
    high   = df["High"].values.astype(float)
    low    = df["Low"].values.astype(float)
    volume = df["Volume"].values.astype(float)
    n      = len(close)

    bull_mask = np.zeros(n, bool); bear_mask = np.zeros(n, bool)
    bull_mask[2:] = high[:-2] < low[2:]
    bear_mask[2:] = low[:-2]  > high[2:]

    bull_mid = np.zeros(n); bear_mid = np.zeros(n)
    bull_mid[2:] = (high[:-2] + low[2:])  / 2.0
    bear_mid[2:] = (low[:-2]  + high[2:]) / 2.0

    s = pd.Series
    ew_bv = s(np.where(bull_mask, bull_mid, 0.0)).rolling(period).sum().values
    ew_dv = s(np.where(bear_mask, bear_mid, 0.0)).rolling(period).sum().values
    ew_bc = s(bull_mask.astype(float)).rolling(period).sum().values
    ew_dc = s(bear_mask.astype(float)).rolling(period).sum().values
    ew_tot= ew_bc + ew_dc

    raw_poc  = np.where(ew_tot > 0, (ew_bv+ew_dv)/ew_tot, np.nan)
    poc      = s(raw_poc).rolling(sma_smooth).mean().values
    poc_spread = np.where(~np.isnan(poc), (close-poc)/close*100.0, np.nan)

    # Volume-weighted variant
    vw_bv = s(np.where(bull_mask, bull_mid*volume, 0.0)).rolling(period).sum().values
    vw_dv = s(np.where(bear_mask, bear_mid*volume, 0.0)).rolling(period).sum().values
    vw_bw = s(np.where(bull_mask, volume, 0.0)).rolling(period).sum().values
    vw_dw = s(np.where(bear_mask, volume, 0.0)).rolling(period).sum().values
    vw_tot= vw_bw + vw_dw
    raw_vw  = np.where(vw_tot > 0, (vw_bv+vw_dv)/vw_tot, np.nan)
    vw_poc  = s(raw_vw).rolling(sma_smooth).mean().values
    vw_spread = np.where(~np.isnan(vw_poc), (close-vw_poc)/close*100.0, np.nan)

    return poc_spread, vw_spread


# ─────────────────────────────────────────────────────────────────────────────
# Forward return helpers
# ─────────────────────────────────────────────────────────────────────────────
def fwd_ret(close, locs, fwd):
    valid = locs[locs + fwd < len(close)]
    if not len(valid):
        return np.array([])
    return (close[valid+fwd] - close[valid]) / close[valid] * 100.0

def stats(rets):
    if not len(rets):
        return dict(n=0, mean=np.nan, median=np.nan, win_rate=np.nan)
    return dict(n=len(rets), mean=float(np.mean(rets)),
                median=float(np.median(rets)),
                win_rate=float(np.mean(rets > 0)*100.0))


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis loop
# ─────────────────────────────────────────────────────────────────────────────
WARM    = FVG_PERIOD + FVG_SMOOTH + ORDER + CONF_W + 5
MAX_FWD = max(FORWARD_DAYS)
MIN_BARS= WARM + MAX_FWD + 20

all_rows = []
skipped  = []

print(f"[INFO] Combined multi-indicator analysis — {len(TICKERS)} tickers")
print(f"[INFO] Warm-up={WARM} bars, forward windows={FORWARD_DAYS}\n")

for idx, ticker in enumerate(TICKERS, 1):
    csv_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if not os.path.isfile(csv_path):
        skipped.append(ticker); continue
    try:
        df = load_csv(csv_path)
    except Exception:
        skipped.append(ticker); continue

    if len(df) < MIN_BARS:
        skipped.append(ticker); continue

    close  = df["Close"].values.astype(float)
    high   = df["High"].values.astype(float)
    low    = df["Low"].values.astype(float)
    volume = df["Volume"].values.astype(float)
    N      = len(close)

    # ── Compute all indicators ───────────────────────────────────────────────
    buy_sig  = supertrend(high, low, close, ST_PERIOD, ST_MULT)
    fish     = fisher_transform(close, FISHER_LEN)
    delta, accel = compute_cvd(high, low, close, volume)
    ob_form  = bull_ob_formed(high, low, close, volume, OB_LENGTH)
    poc_sp, vw_sp = compute_fvg_poc(df, FVG_PERIOD, FVG_SMOOTH)

    # ── Price troughs (argrelextrema) ────────────────────────────────────────
    all_min = argrelextrema(close, np.less_equal, order=ORDER)[0]
    troughs = all_min[(all_min >= WARM) & (all_min <= N - MAX_FWD - 1)]

    if len(troughs) < 3:
        skipped.append(ticker); continue

    # ── Per-indicator thresholds (computed on warm-up-excluded series) ────────
    warm_sl = slice(WARM, None)

    # Fisher
    fish_vals = fish[warm_sl]
    fish_mean = np.nanmean(fish_vals); fish_std = np.nanstd(fish_vals)
    fish_thresh = fish_mean - 2.0 * fish_std

    # FVG spread
    sp_vals = poc_sp[warm_sl]
    sp_mean = np.nanmean(sp_vals); sp_std = np.nanstd(sp_vals)
    fvg_thresh = sp_mean - 2.0 * sp_std

    vw_vals = vw_sp[warm_sl]
    vw_mean = np.nanmean(vw_vals); vw_std = np.nanstd(vw_vals)
    vw_thresh = vw_mean - 2.0 * vw_std

    # ── Boolean masks at each trough ─────────────────────────────────────────
    buy_locs  = np.where(buy_sig)[0]

    # E: any ST buy flip within ±CONF_W bars of trough
    def has_flip_near(t):
        return np.any(np.abs(buy_locs - t) <= CONF_W)

    m_E    = np.array([has_flip_near(t) for t in troughs])
    m_R    = accel[troughs] > 0                            # CVD accel positive
    m_Q    = delta[troughs] > 0                            # CVD delta positive
    m_Fext = fish[troughs]   < fish_thresh                 # Fisher extreme low
    m_FVG1 = poc_sp[troughs] < fvg_thresh                  # FVG P1
    m_FVG5 = vw_sp[troughs]  < vw_thresh                   # FVG P5 (VW)

    # OB: a bull OB was confirmed in the past OB_COMBO_W bars
    ob_locs = np.where(ob_form)[0]
    def ob_near_before(t):
        return np.any((ob_locs >= t - OB_COMBO_W) & (ob_locs <= t))
    m_OB = np.array([ob_near_before(t) for t in troughs])

    # ── Define all combination patterns ──────────────────────────────────────
    patterns = {
        "baseline"   : np.ones(len(troughs), bool),        # every trough
        "E"          : m_E,
        "E_R"        : m_E & m_R,
        "E_Q"        : m_E & m_Q,
        "E_Fext"     : m_E & m_Fext,
        "E_OB"       : m_E & m_OB,
        "E_FVG1"     : m_E & m_FVG1,
        "E_FVG5"     : m_E & m_FVG5,
        "E_R_FVG1"   : m_E & m_R & m_FVG1,
        "E_R_FVG5"   : m_E & m_R & m_FVG5,
        "E_Q_FVG1"   : m_E & m_Q & m_FVG1,
        "E_FVG1_Fext": m_E & m_FVG1 & m_Fext,
        "E_R_FVG1_Fx": m_E & m_R & m_FVG1 & m_Fext,
        "FVG1"       : m_FVG1,                             # standalone FVG
        "FVG1_R"     : m_FVG1 & m_R,
        "FVG1_Fext"  : m_FVG1 & m_Fext,
    }

    # ── Collect forward returns ───────────────────────────────────────────────
    for fwd in FORWARD_DAYS:
        for pat, mask in patterns.items():
            locs = troughs[mask]
            s    = stats(fwd_ret(close, locs, fwd))
            all_rows.append(dict(
                ticker=ticker, pattern=pat, fwd_days=fwd,
                n=s["n"], mean_ret=s["mean"],
                median=s["median"], win_rate=s["win_rate"],
            ))

    if idx % 50 == 0 or idx == len(TICKERS):
        print(f"  … {idx}/{len(TICKERS)} done")

if skipped:
    print(f"\n[WARN] Skipped {len(skipped)} tickers: {skipped[:10]}"
          + (" …" if len(skipped) > 10 else ""))


# ─────────────────────────────────────────────────────────────────────────────
# Aggregated results
# ─────────────────────────────────────────────────────────────────────────────
df_res = pd.DataFrame(all_rows)
df_res.to_csv("combined_results_full.csv", index=False)
n_tickers = df_res["ticker"].nunique()
print(f"\n[INFO] Results → combined_results_full.csv "
      f"({len(df_res)} rows, {n_tickers} tickers)\n")

# Pattern display order with descriptions and colours
PAT_META = [
    ("baseline",    "Baseline (all troughs)",                 "silver"),
    ("E",           "E — Supertrend+Trough (prev champion)",  "gray"),
    ("E_R",         "E ∩ R (CVD accel>0)",                    "darkgray"),
    ("FVG1",        "FVG1 (poc_spread<−2σ, standalone)",      "dodgerblue"),
    ("E_FVG1",      "E ∩ FVG1",                               "steelblue"),
    ("E_FVG5",      "E ∩ FVG5 (vol-weighted)",                "cornflowerblue"),
    ("E_Fext",      "E ∩ Fisher<−2σ",                         "mediumorchid"),
    ("E_OB",        "E ∩ OB (order block)",                   "olive"),
    ("E_Q",         "E ∩ Q (CVD delta>0)",                    "peru"),
    ("E_R_FVG1",    "E ∩ R ∩ FVG1",                           "tomato"),
    ("E_R_FVG5",    "E ∩ R ∩ FVG5",                           "orangered"),
    ("E_Q_FVG1",    "E ∩ Q ∩ FVG1",                           "darkorange"),
    ("E_FVG1_Fext", "E ∩ FVG1 ∩ Fisher<−2σ",                 "purple"),
    ("E_R_FVG1_Fx", "E ∩ R ∩ FVG1 ∩ Fisher<−2σ  (4-way)",   "crimson"),
    ("FVG1_R",      "FVG1 ∩ R (no E)",                        "teal"),
    ("FVG1_Fext",   "FVG1 ∩ Fisher<−2σ (no E)",              "darkcyan"),
]
ALL_PATS = [p for p, _, _ in PAT_META]

# ── Print tables ─────────────────────────────────────────────────────────────
print("=" * 95)
print(f"COMBINED MULTI-INDICATOR PREDICTION  ({n_tickers} US stocks, daily 5y, "
      f"argrelextrema order={ORDER})")
print("(all conditions measured at the price TROUGH bar)")
print("=" * 95)

for fwd in FORWARD_DAYS:
    sub = df_res[df_res["fwd_days"] == fwd]
    base_wr = sub.loc[sub.pattern=="baseline", "win_rate"].mean()
    base_rt = sub.loc[sub.pattern=="baseline", "mean_ret"].mean()
    print(f"\n  Forward {fwd:2d}d  (baseline: WR={base_wr:.1f}%, "
          f"mean ret={base_rt:+.2f}%)")
    print(f"  {'Pattern':<35} {'N':>6}  {'WR':>7}  {'ΔWR':>7}  {'MeanRet':>9}  Verdict")
    print(f"  {'-' * 80}")
    for pat, lbl, _ in PAT_META:
        row   = sub[sub.pattern == pat]
        if row.empty: continue
        n_    = int(row["n"].sum())
        wr    = row["win_rate"].mean()
        rt    = row["mean_ret"].mean()
        dwr   = wr - base_wr if pat != "baseline" else 0.0
        if pat == "baseline":
            verdict = "baseline"
        elif wr > 93 and dwr > 10:
            verdict = "★★ ELITE"
        elif wr > 90 and dwr > 7:
            verdict = "★ STRONG"
        elif wr > 85 and dwr > 3:
            verdict = "◆ MODERATE"
        elif wr > 80 and dwr > 0:
            verdict = "▷ SLIGHT"
        else:
            verdict = "–"
        print(f"  {lbl:<35} {n_:>6,}  {wr:>6.1f}%  {dwr:>+6.1f}pp {rt:>+9.2f}%  {verdict}")

# ── Champion pattern summary ──────────────────────────────────────────────────
print("\n" + "─" * 95)
print("CHAMPION PATTERNS  (ranked by 60d win rate, all combined patterns)")
print("─" * 95)
sub60 = df_res[df_res["fwd_days"] == 60]
base60_wr = sub60.loc[sub60.pattern=="baseline","win_rate"].mean()
ranked = []
for pat, lbl, _ in PAT_META:
    row = sub60[sub60.pattern==pat]
    if row.empty: continue
    wr = row["win_rate"].mean()
    rt = row["mean_ret"].mean()
    n_ = int(row["n"].sum())
    ranked.append((lbl, wr, rt, n_, wr - base60_wr))
ranked.sort(key=lambda x: x[1], reverse=True)
for lbl, wr, rt, n_, dwr in ranked:
    print(f"  {lbl:<38}  WR={wr:.1f}%  ΔWR={dwr:+.1f}pp  "
          f"MeanRet={rt:+.2f}%  N={n_:,}")

# ── Summary chart ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(22, 9))
fig.suptitle(
    f"Multi-Indicator Combination Analysis — {n_tickers} US Stocks · daily 5y\n"
    f"All conditions at price trough (argrelextrema order={ORDER})  |  "
    f"FVG POC period={FVG_PERIOD} · ST({ST_PERIOD},{ST_MULT}) · Fisher len={FISHER_LEN}",
    fontsize=12, fontweight="bold"
)

lbls   = [lbl for _, lbl, _ in PAT_META]
cols   = [col for _, _, col in PAT_META]
y_pos  = np.arange(len(PAT_META))

for ax_i, fwd in enumerate(FORWARD_DAYS):
    ax  = axes[ax_i]
    sub = df_res[df_res["fwd_days"] == fwd]
    wrs = []
    ns  = []
    for pat, _, _ in PAT_META:
        row = sub[sub.pattern==pat]
        wrs.append(row["win_rate"].mean() if not row.empty else np.nan)
        ns.append(int(row["n"].sum()) if not row.empty else 0)

    bars = ax.barh(y_pos, wrs, color=cols, alpha=0.85, edgecolor="white")
    ax.axvline(50, color="black", lw=1, ls="--", alpha=0.5)
    # Mark 90% line as a reference
    ax.axvline(90, color="red", lw=0.8, ls=":", alpha=0.5, label="90% line")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(lbls, fontsize=8.5)
    ax.set_xlabel("Win Rate (%)")
    ax.set_title(f"{fwd}-Day Forward Win Rate", fontsize=11)
    ax.set_xlim(0, 105)

    # Annotate bars with WR% and N
    for bar, wr, n_ in zip(bars, wrs, ns):
        if not np.isnan(wr) and n_ > 0:
            ax.text(wr + 0.4, bar.get_y() + bar.get_height()/2,
                    f"{wr:.1f}% (n={n_:,})",
                    va="center", fontsize=7.5)

    if ax_i == 0:
        ax.legend(fontsize=8, loc="lower right")

plt.tight_layout()
plt.savefig("combined_summary_chart.png", dpi=130, bbox_inches="tight")
plt.close()
print(f"\n[INFO] Summary chart → combined_summary_chart.png")

print("\nVerdict guide")
print("  ★★ ELITE    : WR >93%, ΔWR >10pp above baseline")
print("  ★  STRONG   : WR >90%, ΔWR >7pp")
print("  ◆  MODERATE : WR >85%, ΔWR >3pp")
print("  ▷  SLIGHT   : WR >80%, ΔWR >0pp")

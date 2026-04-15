"""
Multi-Indicator Combination Analysis  —  Buy-Side AND Sell-Side
================================================================
Tests buy signals at price TROUGHS and sell signals at price PEAKS using
identical logic (argrelextrema) and mirrored indicator conditions.

A strategy is considered robust when BOTH the entry (trough buy) and the
exit (peak sell) signals show strong predictive win-rates.

Indicators ported
-----------------
  Supertrend : ATR(10) × 3.0  (PineScript v4 port)
  Fisher     : Ehlers recursive, len=9
  CVD        : bar-delta approximation from daily OHLCV
  Order Block: LuxAlgo bull/bear OB detector, length=5
  FVG POC    : BigBeluga rolling POC, period=100, sma=10

BUY-SIDE signals (at troughs — argrelextrema np.less_equal, order=10)
----------------------------------------------------------------------
  E    : ST buy  flip within ±5 bars of trough
  R    : CVD acceleration > 0 at trough
  Q    : CVD delta        > 0 at trough
  Fext : Fisher  < mean − 2σ at trough   (oversold extreme)
  OB   : Bull OB formed  in [t−15, t]
  FVG1 : poc_spread       < mean − 2σ    (price well below FVG activity centre)
  FVG5 : vw_spread        < mean − 2σ    (volume-weighted variant)

SELL-SIDE signals (at peaks — argrelextrema np.greater_equal, order=10)
-----------------------------------------------------------------------
  Es   : ST sell flip within ±5 bars of peak
  Rs   : CVD acceleration < 0 at peak
  Qs   : CVD delta        < 0 at peak
  Fexts: Fisher  > mean + 2σ at peak     (overbought extreme)
  OBs  : Bear OB formed  in [t−15, t]
  FVGs1: poc_spread       > mean + 2σ    (price well above FVG activity centre)
  FVGs5: vw_spread        > mean + 2σ

Win definitions
---------------
  Buy-side  : price rises   after trough → forward_return > 0 = win
  Sell-side : price FALLS   after peak   → forward_return < 0 = win
              (stored with flipped sign so win_rate formula stays identical)

Forward windows: 20 / 30 / 60 trading days
Dataset        : price_data/ — 503 US stocks, daily, 5 years
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
ORDER        = 10
ST_PERIOD    = 10
ST_MULT      = 3.0
CONF_W       = 5
OB_LENGTH    = 5
OB_COMBO_W   = 15
FVG_PERIOD   = 100
FVG_SMOOTH   = 10
FISHER_LEN   = 9
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
# Supertrend — returns both buy_sig AND sell_sig
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
    bu = hl2 - mult*atr; bd = hl2 + mult*atr
    up = np.empty(N); dn = np.empty(N); trend = np.ones(N, dtype=int)
    up[0] = bu[0]; dn[0] = bd[0]
    for i in range(1, N):
        up[i] = bu[i] if close[i-1] <= up[i-1] else max(bu[i], up[i-1])
        dn[i] = bd[i] if close[i-1] >= dn[i-1] else min(bd[i], dn[i-1])
        if   trend[i-1]==-1 and close[i]>dn[i-1]: trend[i]=  1
        elif trend[i-1]== 1 and close[i]<up[i-1]: trend[i]= -1
        else:                                       trend[i]=trend[i-1]
    buy_sig  = np.zeros(N, bool); buy_sig[1:]  = (trend[1:]== 1)&(trend[:-1]==-1)
    sell_sig = np.zeros(N, bool); sell_sig[1:] = (trend[1:]==-1)&(trend[:-1]== 1)
    return buy_sig, sell_sig


# ─────────────────────────────────────────────────────────────────────────────
# Fisher Transform
# ─────────────────────────────────────────────────────────────────────────────
def fisher_transform(close, len_price=9):
    n=len(close); s=pd.Series(close)
    mxh=s.rolling(len_price).max().values; mnl=s.rolling(len_price).min().values
    v=np.zeros(n)
    for i in range(1, n):
        rng=max(mxh[i]-mnl[i], 1e-10)
        raw=0.33*2.0*((close[i]-mnl[i])/rng-0.5)+0.67*v[i-1]
        v[i]=max(-0.999, min(0.999, raw))
    return 0.5*np.log((1.0+v)/np.maximum(1.0-v, 1e-10))


# ─────────────────────────────────────────────────────────────────────────────
# CVD
# ─────────────────────────────────────────────────────────────────────────────
def compute_cvd(high, low, close, volume):
    hl=np.where(high-low<1e-10, 1e-10, high-low)
    delta=volume*(2.0*close-high-low)/hl
    accel=np.empty(len(delta)); accel[0]=0.0; accel[1:]=delta[1:]-delta[:-1]
    return delta, accel


# ─────────────────────────────────────────────────────────────────────────────
# Order Block detector — returns both bull and bear OB formation arrays
# ─────────────────────────────────────────────────────────────────────────────
def ob_formed_arrays(high, low, close, volume, length=5):
    """
    Returns (bull_ob, bear_ob) — bool arrays (N,), True on confirmation bar.
    Bull OB: os==1 (bullish structure) at volume pivot bar.
    Bear OB: os==0 (bearish structure) at volume pivot bar.
    """
    N   = len(close)
    hl2 = (high+low)/2.0
    upper = pd.Series(high).rolling(length).max().values
    lower = pd.Series(low).rolling(length).min().values

    high_lag = np.full(N, np.nan); high_lag[length:] = high[:N-length]
    low_lag  = np.full(N, np.nan); low_lag[length:]  = low[:N-length]

    raw_os = np.where(high_lag>upper, 0.0, np.where(low_lag<lower, 1.0, np.nan))
    os_arr = pd.Series(raw_os).ffill().fillna(0).astype(int).values

    phv = np.full(N, np.nan)
    for i in range(length, N-length):
        v = volume[i]
        if v > volume[i-length:i].max() and v > volume[i+1:i+length+1].max():
            phv[i] = v

    bull_ob = np.zeros(N, bool)
    bear_ob = np.zeros(N, bool)
    for i in range(2*length, N):
        pivot = i - length
        if np.isnan(phv[pivot]):
            continue
        if os_arr[i] == 1:
            bull_ob[i] = True
        else:
            bear_ob[i] = True

    return bull_ob, bear_ob


# ─────────────────────────────────────────────────────────────────────────────
# FVG Profile + Rolling POC
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
    bull_mid[2:] = (high[:-2]+low[2:]) /2.0
    bear_mid[2:] = (low[:-2]+high[2:]) /2.0

    S = pd.Series
    ew_bv = S(np.where(bull_mask,bull_mid,0.0)).rolling(period).sum().values
    ew_dv = S(np.where(bear_mask,bear_mid,0.0)).rolling(period).sum().values
    ew_bc = S(bull_mask.astype(float)).rolling(period).sum().values
    ew_dc = S(bear_mask.astype(float)).rolling(period).sum().values
    ew_tot= ew_bc+ew_dc

    raw_poc = np.where(ew_tot>0,(ew_bv+ew_dv)/ew_tot, np.nan)
    poc     = S(raw_poc).rolling(sma_smooth).mean().values
    poc_spread = np.where(~np.isnan(poc),(close-poc)/close*100.0, np.nan)

    vw_bv = S(np.where(bull_mask,bull_mid*volume,0.0)).rolling(period).sum().values
    vw_dv = S(np.where(bear_mask,bear_mid*volume,0.0)).rolling(period).sum().values
    vw_bw = S(np.where(bull_mask,volume,0.0)).rolling(period).sum().values
    vw_dw = S(np.where(bear_mask,volume,0.0)).rolling(period).sum().values
    vw_tot= vw_bw+vw_dw
    raw_vw = np.where(vw_tot>0,(vw_bv+vw_dv)/vw_tot, np.nan)
    vw_poc  = S(raw_vw).rolling(sma_smooth).mean().values
    vw_spread = np.where(~np.isnan(vw_poc),(close-vw_poc)/close*100.0, np.nan)

    return poc_spread, vw_spread


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def fwd_ret(close, locs, fwd):
    valid = locs[locs+fwd < len(close)]
    if not len(valid): return np.array([])
    return (close[valid+fwd]-close[valid])/close[valid]*100.0

def stats(rets):
    if not len(rets):
        return dict(n=0, mean=np.nan, median=np.nan, win_rate=np.nan)
    return dict(n=len(rets), mean=float(np.mean(rets)),
                median=float(np.median(rets)),
                win_rate=float(np.mean(rets>0)*100.0))


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
WARM    = FVG_PERIOD + FVG_SMOOTH + ORDER + CONF_W + 5
MAX_FWD = max(FORWARD_DAYS)
MIN_BARS= WARM + MAX_FWD + 20

all_rows = []
skipped  = []

print(f"[INFO] Combined buy+sell analysis — {len(TICKERS)} tickers")
print(f"[INFO] Warm={WARM} bars | forward={FORWARD_DAYS}\n")

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
    buy_sig, sell_sig    = supertrend(high, low, close, ST_PERIOD, ST_MULT)
    fish                 = fisher_transform(close, FISHER_LEN)
    delta, accel         = compute_cvd(high, low, close, volume)
    bull_ob, bear_ob     = ob_formed_arrays(high, low, close, volume, OB_LENGTH)
    poc_sp, vw_sp        = compute_fvg_poc(df, FVG_PERIOD, FVG_SMOOTH)

    # ── Price extrema ────────────────────────────────────────────────────────
    all_min = argrelextrema(close, np.less_equal,    order=ORDER)[0]
    all_max = argrelextrema(close, np.greater_equal, order=ORDER)[0]

    troughs = all_min[(all_min >= WARM) & (all_min <= N-MAX_FWD-1)]
    peaks   = all_max[(all_max >= WARM) & (all_max <= N-MAX_FWD-1)]

    if len(troughs) < 3 or len(peaks) < 3:
        skipped.append(ticker); continue

    # ── Indicator thresholds (warm-up excluded) ──────────────────────────────
    warm_sl = slice(WARM, None)
    fish_mean = np.nanmean(fish[warm_sl]);   fish_std = np.nanstd(fish[warm_sl])
    sp_mean   = np.nanmean(poc_sp[warm_sl]); sp_std   = np.nanstd(poc_sp[warm_sl])
    vw_mean   = np.nanmean(vw_sp[warm_sl]);  vw_std   = np.nanstd(vw_sp[warm_sl])

    fish_lo = fish_mean - 2.0*fish_std   # oversold
    fish_hi = fish_mean + 2.0*fish_std   # overbought
    fvg_lo  = sp_mean   - 2.0*sp_std    # price well below POC
    fvg_hi  = sp_mean   + 2.0*sp_std    # price well above POC
    vw_lo   = vw_mean   - 2.0*vw_std
    vw_hi   = vw_mean   + 2.0*vw_std

    buy_locs  = np.where(buy_sig)[0]
    sell_locs = np.where(sell_sig)[0]

    def flip_near(ref_locs, t):
        return np.any(np.abs(ref_locs - t) <= CONF_W)

    def ob_near_before(ob_arr, t):
        near = np.where(ob_arr)[0]
        return np.any((near >= t-OB_COMBO_W) & (near <= t))

    # ── BUY-SIDE masks at troughs ────────────────────────────────────────────
    m_E    = np.array([flip_near(buy_locs,  t) for t in troughs])
    m_R    = accel[troughs]   >  0
    m_Q    = delta[troughs]   >  0
    m_Fext = fish[troughs]    <  fish_lo
    m_FVG1 = poc_sp[troughs]  <  fvg_lo
    m_FVG5 = vw_sp[troughs]   <  vw_lo
    m_OB   = np.array([ob_near_before(bull_ob, t) for t in troughs])

    buy_patterns = {
        "baseline" : np.ones(len(troughs), bool),
        "E"        : m_E,
        "E_R"      : m_E & m_R,
        "FVG1"     : m_FVG1,
        "E_FVG1"   : m_E & m_FVG1,
        "E_FVG5"   : m_E & m_FVG5,
        "E_Fext"   : m_E & m_Fext,
        "E_OB"     : m_E & m_OB,
        "E_R_FVG1" : m_E & m_R & m_FVG1,
        "E_R_FVG5" : m_E & m_R & m_FVG5,
        "E_Q_FVG1" : m_E & m_Q & m_FVG1,
        "E_FVG1_Fx": m_E & m_FVG1 & m_Fext,
        "FVG1_R"   : m_FVG1 & m_R,
    }

    # ── SELL-SIDE masks at peaks (mirror conditions) ─────────────────────────
    ms_E    = np.array([flip_near(sell_locs, t) for t in peaks])
    ms_R    = accel[peaks]   <  0              # selling accelerating
    ms_Q    = delta[peaks]   <  0              # net selling pressure
    ms_Fext = fish[peaks]    >  fish_hi        # overbought extreme
    ms_FVG1 = poc_sp[peaks]  >  fvg_hi        # price well above POC
    ms_FVG5 = vw_sp[peaks]   >  vw_hi
    ms_OB   = np.array([ob_near_before(bear_ob, t) for t in peaks])

    sell_patterns = {
        "baseline" : np.ones(len(peaks), bool),
        "E"        : ms_E,
        "E_R"      : ms_E & ms_R,
        "FVG1"     : ms_FVG1,
        "E_FVG1"   : ms_E & ms_FVG1,
        "E_FVG5"   : ms_E & ms_FVG5,
        "E_Fext"   : ms_E & ms_Fext,
        "E_OB"     : ms_E & ms_OB,
        "E_R_FVG1" : ms_E & ms_R & ms_FVG1,
        "E_R_FVG5" : ms_E & ms_R & ms_FVG5,
        "E_Q_FVG1" : ms_E & ms_Q & ms_FVG1,
        "E_FVG1_Fx": ms_E & ms_FVG1 & ms_Fext,
        "FVG1_R"   : ms_FVG1 & ms_R,
    }

    # ── Collect forward returns ───────────────────────────────────────────────
    for fwd in FORWARD_DAYS:
        # Buy-side: win = price rises (return > 0)
        for pat, mask in buy_patterns.items():
            locs = troughs[mask]
            s    = stats(fwd_ret(close, locs, fwd))
            all_rows.append(dict(side="buy", ticker=ticker, pattern=pat,
                                 fwd_days=fwd, n=s["n"], mean_ret=s["mean"],
                                 median=s["median"], win_rate=s["win_rate"]))
        # Sell-side: win = price falls (return < 0) → flip sign
        for pat, mask in sell_patterns.items():
            locs = peaks[mask]
            s    = stats(-fwd_ret(close, locs, fwd))   # negated: down = win
            all_rows.append(dict(side="sell", ticker=ticker, pattern=pat,
                                 fwd_days=fwd, n=s["n"], mean_ret=s["mean"],
                                 median=s["median"], win_rate=s["win_rate"]))

    if idx % 50 == 0 or idx == len(TICKERS):
        print(f"  … {idx}/{len(TICKERS)} done")

if skipped:
    print(f"\n[WARN] Skipped {len(skipped)} tickers: {skipped[:10]}"
          + (" …" if len(skipped)>10 else ""))


# ─────────────────────────────────────────────────────────────────────────────
# Aggregated results
# ─────────────────────────────────────────────────────────────────────────────
df_res = pd.DataFrame(all_rows)
df_res.to_csv("combined_results_full.csv", index=False)
n_tk = df_res["ticker"].nunique()
print(f"\n[INFO] Results → combined_results_full.csv ({len(df_res)} rows, {n_tk} tickers)\n")

# Pattern metadata: (key, label, buy_colour, sell_colour)
PAT_META = [
    ("baseline",  "Baseline (all extrema)",             "silver",        "silver"),
    ("E",         "E — ST flip near extremum",           "gray",          "dimgray"),
    ("E_R",       "E ∩ R  (CVD accel)",                  "darkgray",      "darkslategray"),
    ("FVG1",      "FVG1 standalone (poc<±2σ)",           "dodgerblue",    "salmon"),
    ("E_FVG1",    "E ∩ FVG1",                            "steelblue",     "tomato"),
    ("E_FVG5",    "E ∩ FVG5 (vol-weighted)",             "cornflowerblue","orangered"),
    ("E_Fext",    "E ∩ Fisher ±2σ",                      "mediumorchid",  "darkorchid"),
    ("E_OB",      "E ∩ OB",                              "olive",         "darkolivegreen"),
    ("E_R_FVG1",  "E ∩ R ∩ FVG1",                        "tomato",        "firebrick"),
    ("E_R_FVG5",  "E ∩ R ∩ FVG5",                        "orangered",     "darkred"),
    ("E_Q_FVG1",  "E ∩ Q ∩ FVG1",                        "darkorange",    "saddlebrown"),
    ("E_FVG1_Fx", "E ∩ FVG1 ∩ Fisher",                   "purple",        "indigo"),
    ("FVG1_R",    "FVG1 ∩ R  (no E)",                    "teal",          "darkcyan"),
]
ALL_PATS = [p for p,*_ in PAT_META]

def verdict(wr, dwr):
    if   wr>93 and dwr>10: return "★★ ELITE"
    elif wr>90 and dwr>7:  return "★  STRONG"
    elif wr>85 and dwr>3:  return "◆  MODERATE"
    elif wr>80 and dwr>0:  return "▷  SLIGHT"
    else:                  return "–"

# ── Print tables ─────────────────────────────────────────────────────────────
for side_label, side_key, side_desc in [
        ("BUY",  "buy",  "win = price RISES after trough"),
        ("SELL", "sell", "win = price FALLS after peak  (returns flipped)")]:

    print("=" * 100)
    print(f"{side_label}-SIDE PATTERNS  ({n_tk} US stocks, 5y daily)  |  {side_desc}")
    print("=" * 100)

    sub_side = df_res[df_res["side"] == side_key]

    for fwd in FORWARD_DAYS:
        sub = sub_side[sub_side["fwd_days"] == fwd]
        base_wr = sub.loc[sub.pattern=="baseline","win_rate"].mean()
        base_rt = sub.loc[sub.pattern=="baseline","mean_ret"].mean()
        print(f"\n  Forward {fwd:2d}d  (baseline: WR={base_wr:.1f}%,  "
              f"mean ret={base_rt:+.2f}%)")
        print(f"  {'Pattern':<35} {'N':>6}  {'WR':>7}  {'ΔWR':>7}  "
              f"{'MeanRet':>9}  Verdict")
        print(f"  {'-' * 82}")
        for pat, lbl, *_ in PAT_META:
            row = sub[sub.pattern==pat]
            if row.empty: continue
            n_  = int(row["n"].sum())
            wr  = row["win_rate"].mean()
            rt  = row["mean_ret"].mean()
            dwr = wr - base_wr if pat!="baseline" else 0.0
            v   = "baseline" if pat=="baseline" else verdict(wr, dwr)
            print(f"  {lbl:<35} {n_:>6,}  {wr:>6.1f}%  {dwr:>+6.1f}pp "
                  f"{rt:>+9.2f}%  {v}")
    print()

# ── Side-by-side comparison at 20d ───────────────────────────────────────────
print("=" * 100)
print("STRATEGY SCORECARD  —  Buy vs Sell side-by-side @ 20d / 30d / 60d")
print("A good strategy needs BOTH sides to be strong.")
print("=" * 100)
print(f"\n  {'Pattern':<35} "
      f"{'Buy 20d':>8} {'Sell 20d':>9} "
      f"{'Buy 30d':>8} {'Sell 30d':>9} "
      f"{'Buy 60d':>8} {'Sell 60d':>9}  Assessment")
print(f"  {'-' * 100}")

for pat, lbl, *_ in PAT_META:
    row_str = f"  {lbl:<35}"
    scores  = []
    for fwd in FORWARD_DAYS:
        for side in ("buy", "sell"):
            sub = df_res[(df_res.side==side) & (df_res.fwd_days==fwd)
                         & (df_res.pattern==pat)]
            wr = sub["win_rate"].mean() if not sub.empty else np.nan
            row_str += f" {wr:>8.1f}%"
            scores.append(wr)

    # Assessment: both sides > 85% at any forward window = strong strategy
    buy_scores  = scores[0::2]   # indices 0,2,4
    sell_scores = scores[1::2]   # indices 1,3,5
    both_strong = any(b>85 and s>75 for b,s in zip(buy_scores, sell_scores)
                      if not np.isnan(b) and not np.isnan(s))
    both_elite  = any(b>90 and s>80 for b,s in zip(buy_scores, sell_scores)
                      if not np.isnan(b) and not np.isnan(s))
    if both_elite:
        assess = "★★ ELITE STRATEGY"
    elif both_strong:
        assess = "★  STRONG STRATEGY"
    elif pat == "baseline":
        assess = "baseline"
    else:
        assess = "–  one-sided only"
    print(row_str + f"  {assess}")


# ── Summary chart ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(24, 14))
fig.suptitle(
    f"Multi-Indicator Combination — Buy-Side (Troughs) vs Sell-Side (Peaks)\n"
    f"{n_tk} US Stocks · daily 5y · argrelextrema order={ORDER}  |  "
    f"Win = price rises after trough / falls after peak",
    fontsize=13, fontweight="bold"
)

lbls = [lbl for _, lbl, *_ in PAT_META]
y    = np.arange(len(PAT_META))

for col_i, fwd in enumerate(FORWARD_DAYS):
    for row_i, side in enumerate(("buy", "sell")):
        ax  = axes[row_i, col_i]
        sub = df_res[(df_res.side==side) & (df_res.fwd_days==fwd)]
        wrs = []
        ns  = []
        clrs= []
        for pat, _, bc, sc in PAT_META:
            row = sub[sub.pattern==pat]
            wrs.append(row["win_rate"].mean() if not row.empty else np.nan)
            ns.append(int(row["n"].sum()) if not row.empty else 0)
            clrs.append(bc if side=="buy" else sc)

        bars = ax.barh(y, wrs, color=clrs, alpha=0.85, edgecolor="white")
        ax.axvline(50, color="black", lw=1, ls="--", alpha=0.4)
        ax.axvline(85, color="green", lw=0.8, ls=":", alpha=0.5, label="85%")
        ax.axvline(90, color="red",   lw=0.8, ls=":", alpha=0.5, label="90%")
        ax.set_yticks(y)
        ax.set_yticklabels(lbls, fontsize=8)
        ax.set_xlabel("Win Rate (%)")
        side_title = "BUY  (trough → rise)" if side=="buy" else "SELL (peak → fall)"
        ax.set_title(f"{side_title}  |  {fwd}d forward", fontsize=10, fontweight="bold")
        ax.set_xlim(0, 110)
        if col_i == 0:
            ax.legend(fontsize=7, loc="lower right")

        for bar, wr, n_ in zip(bars, wrs, ns):
            if not np.isnan(wr) and n_ > 10:
                ax.text(wr+0.3, bar.get_y()+bar.get_height()/2,
                        f"{wr:.0f}% n={n_}", va="center", fontsize=7)

plt.tight_layout()
plt.savefig("combined_summary_chart.png", dpi=130, bbox_inches="tight")
plt.close()
print(f"\n[INFO] Chart → combined_summary_chart.png")

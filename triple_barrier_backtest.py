#!/usr/bin/env python3
"""
=============================================================================
TRIPLE-BARRIER BACKTEST  (path-aware, MAE/MFE, time-to-TP/SL, cost-aware)
=============================================================================
Replaces the fixed-horizon "ret_5d_oc_pct sign" analysis with a realistic
trade simulation:

  ENTRY  : signal at day T close -> BUY at T+1 OPEN (no look-ahead).
  EXITS  : whichever of the three barriers is hit FIRST, scanning bars
           T+1 .. T+MAX_HOLD intrabar via daily high/low:
             - TAKE PROFIT  (TP)
             - STOP LOSS    (SL)
             - TIME         (close of the last held bar)
           On a bar where BOTH TP and SL are touched, we assume SL first
           (conservative / worst-case).
  GRIDS  : TP in {2,3,5,10}% and {2,3,5,10}xATR
           SL in {2,3,5}%   and {2,3,5}xATR
  PER TRADE: exit_reason, holding_bars, time_to_tp, time_to_sl,
             MAE (max adverse excursion), MFE (max favorable excursion),
             gross & net return (NET = gross - COST_BPS round-trip).
  REPORTS : every metric broken out by 5%-wide PROBABILITY BUCKET
            (e.g. [0.45,0.50), [0.50,0.55), ...), by regime, Full + OOS.

This module is MODEL-AGNOSTIC: point BASE_DIR at a model's output folder and
it scores that panel with that model's router. Run it once for the binary
model and once for the rank model, then compare the two output folders.

ATR for the ATR-multiple barriers is D_atr14 at the SIGNAL day (known at the
decision point), expressed as a fraction of the entry price.

NOTE ON REALISM / KNOWN ASSUMPTIONS (read these before trusting numbers):
  * Barrier fills are assumed to occur exactly at the TP/SL price. Intrabar
    gaps THROUGH a stop would fill worse in reality (this is mildly optimistic
    on SL); the SL-first tie rule partly offsets that.
  * Per-trade rows OVERLAP (a 5-day hold, a signal every day) and same-day
    picks are cross-sectionally correlated, so per-trade std / Sharpe understate
    true risk. Treat the per-trade Sharpe as a relative ranking, not a live
    Sharpe -- a portfolio layer (daily top-K, max concurrent positions) is the
    next step once you specify sizing.
  * OOS defaults to the last 10% of trading days to approximate the model's
    held-out TEST window (the model splits 70/20/10 train/cal/test). For a
    strictly leak-free read set OOS_START_DATE to the test-start date printed
    in the training log.
  * No circuit-filter / liquidity-fill modelling. Universe gate is the same
    close>=2 & avg20_vol>=200k used live.
=============================================================================
"""

import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================
BASE_DIR = Path(r"C:\Users\karanvsi\Desktop\Kite Connect\v3_2_output_full")
PANEL_PATH = BASE_DIR / "panel_cache.parquet"
FEATURES_PATH = BASE_DIR / "features_train.json"
MODELS_DIR = BASE_DIR / "models"
ROUTER_PATH = MODELS_DIR / "m5_regime_router.joblib"
OUT_DIR = BASE_DIR / "backtest_triple_barrier"

IST = "Asia/Kolkata"

# Universe filters (match the live watchlist gate)
MIN_CLOSE = 2.0
MIN_AVG20_VOL = 200_000

# Trade simulation
MAX_HOLD = 5                 # trading bars held (T+1 .. T+5)
COST_BPS = 30                # round-trip cost in bps, subtracted from gross return
TP_PCTS = [0.02, 0.03, 0.05, 0.10]      # fixed % take-profits
SL_PCTS = [0.02, 0.03, 0.05]            # fixed % stop-losses
TP_ATR_MULTS = [2.0, 3.0, 5.0, 10.0]    # ATR-multiple take-profits
SL_ATR_MULTS = [2.0, 3.0, 5.0]          # ATR-multiple stop-losses

# Probability buckets (5%-wide). Range covers what the models actually emit.
PROB_BUCKET_WIDTH = 0.05
PROB_MIN = 0.30
PROB_MAX = 0.80

# OOS window: last fraction of unique trading days (approx model TEST set).
OOS_FRACTION = 0.10
OOS_START_DATE = None        # e.g. "2024-09-01" for a strictly leak-free cut

# Per-trade detail dump uses this "primary" fixed combo so you can inspect the
# MAE/MFE/timing distributions of a representative strategy.
PRIMARY_TP_PCT = 0.05
PRIMARY_SL_PCT = 0.03

ALLOWED_REGIMES = None       # None = all regimes; or e.g. ["bull_trend","bear_trend"]


# =============================================================================
# CORE ENGINE  (pure-numpy, vectorized, unit-testable, no I/O)
# =============================================================================
def run_barrier(hr: np.ndarray, lr: np.ndarray, cr: np.ndarray,
                tp_ratio: np.ndarray, sl_ratio: np.ndarray) -> dict:
    """
    Vectorized triple-barrier simulation.

    Inputs (all per-ENTRY, level-independent so they're computed once):
      hr : (n, H) high[T+k] / entry_open          for k=1..H
      lr : (n, H) low[T+k]  / entry_open
      cr : (n, H) close[T+k]/ entry_open
      tp_ratio : (n,) take-profit price / entry   (e.g. 1.05 for +5%)
      sl_ratio : (n,) stop-loss   price / entry   (e.g. 0.97 for -3%)

    Returns dict of (n,) arrays:
      reason     : +1 TP, -1 SL, 0 TIME
      hold_bars  : 1..H bars held until exit
      gross      : realized gross return fraction (TP=+tp%, SL=-sl%, TIME=close)
      mae        : max adverse excursion fraction up to exit (<=0 typically)
      mfe        : max favorable excursion fraction up to exit (>=0 typically)
      time_to_tp : hold_bars if TP else NaN
      time_to_sl : hold_bars if SL else NaN
    """
    n, H = hr.shape
    tp_r = np.asarray(tp_ratio, dtype="float64").reshape(-1, 1)
    sl_r = np.asarray(sl_ratio, dtype="float64").reshape(-1, 1)

    tp_hit = hr >= tp_r                      # (n,H)
    sl_hit = lr <= sl_r
    # Same-bar tie -> SL first (conservative). Encode SL=2, TP=1, none=0.
    code = np.where(sl_hit, 2, np.where(tp_hit, 1, 0))
    any_hit = code > 0
    has = any_hit.any(axis=1)
    first = np.argmax(any_hit, axis=1)       # 0 when no hit -> fixed up by `has`
    exit_idx = np.where(has, first, H - 1)   # 0-indexed exit bar
    code_at = code[np.arange(n), exit_idx]
    reason = np.where(~has, 0, np.where(code_at == 2, -1, 1))

    # MAE/MFE over bars 0..exit_idx inclusive
    cummax_hr = np.maximum.accumulate(hr, axis=1)
    cummin_lr = np.minimum.accumulate(lr, axis=1)
    mfe = cummax_hr[np.arange(n), exit_idx] - 1.0
    mae = cummin_lr[np.arange(n), exit_idx] - 1.0

    gross = np.empty(n, dtype="float64")
    is_tp = reason == 1
    is_sl = reason == -1
    is_time = reason == 0
    gross[is_tp] = tp_r[is_tp, 0] - 1.0
    gross[is_sl] = sl_r[is_sl, 0] - 1.0
    gross[is_time] = cr[is_time, H - 1] - 1.0

    hold_bars = exit_idx + 1
    time_to_tp = np.where(is_tp, hold_bars, np.nan)
    time_to_sl = np.where(is_sl, hold_bars, np.nan)
    return {
        "reason": reason, "hold_bars": hold_bars, "gross": gross,
        "mae": mae, "mfe": mfe, "time_to_tp": time_to_tp, "time_to_sl": time_to_sl,
    }


def bucket_label(p: float) -> str:
    lo = np.floor(p / PROB_BUCKET_WIDTH) * PROB_BUCKET_WIDTH
    return f"[{lo:.2f}, {lo + PROB_BUCKET_WIDTH:.2f})"


def _summary(net: np.ndarray, gross: np.ndarray, reason: np.ndarray,
             mae: np.ndarray, mfe: np.ndarray, hold: np.ndarray,
             ttp: np.ndarray, tsl: np.ndarray) -> dict:
    n = len(net)
    if n == 0:
        return {"n": 0}
    tp = reason == 1
    sl = reason == -1
    tm = reason == 0
    wins = net > 0
    std = net.std()
    return {
        "n": n,
        "tp_rate_pct": round(100 * tp.mean(), 2),
        "sl_rate_pct": round(100 * sl.mean(), 2),
        "time_rate_pct": round(100 * tm.mean(), 2),
        "win_rate_pct": round(100 * wins.mean(), 2),
        "avg_net_pct": round(100 * net.mean(), 3),
        "expectancy_pct": round(100 * net.mean(), 3),
        "median_net_pct": round(100 * np.median(net), 3),
        "avg_gross_pct": round(100 * gross.mean(), 3),
        "avg_mae_pct": round(100 * np.nanmean(mae), 3),
        "avg_mfe_pct": round(100 * np.nanmean(mfe), 3),
        "worst_mae_pct": round(100 * np.nanmin(mae), 2),
        "best_mfe_pct": round(100 * np.nanmax(mfe), 2),
        "avg_hold_bars": round(float(np.mean(hold)), 2),
        "avg_time_to_tp": round(float(np.nanmean(ttp)), 2) if tp.any() else np.nan,
        "avg_time_to_sl": round(float(np.nanmean(tsl)), 2) if sl.any() else np.nan,
        "sharpe_per_trade": round(net.mean() / (std + 1e-12), 3),
    }


# =============================================================================
# PIPELINE
# =============================================================================
def _load_scored_panel() -> pd.DataFrame:
    print("=" * 70)
    print("TRIPLE-BARRIER BACKTEST")
    print("=" * 70)
    print(f"\n[1/5] Loading panel + model from {BASE_DIR} ...")
    panel = pd.read_parquet(PANEL_PATH)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    if panel["timestamp"].dt.tz is None:
        panel["timestamp"] = panel["timestamp"].dt.tz_localize(IST)
    else:
        panel["timestamp"] = panel["timestamp"].dt.tz_convert(IST)
    panel = panel.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    print(f"  Panel: {len(panel):,} rows")
    for need in ("stock_regime", "open", "high", "low", "close", "D_atr14"):
        if need not in panel.columns:
            raise SystemExit(f"FATAL: panel missing '{need}'")

    schema = json.loads(FEATURES_PATH.read_text())
    feats = schema["features"]
    impute = {k: float(v) for k, v in schema["impute"].items()}
    print(f"  Features: {len(feats)}")
    if not ROUTER_PATH.exists():
        raise SystemExit("FATAL: regime router not found")
    router = load(ROUTER_PATH)

    print("\n[2/5] Scoring panel ...")
    t0 = time.perf_counter()
    X = panel.reindex(columns=feats).copy()
    for c in feats:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(impute.get(c, 0.0))
    if not hasattr(router, "predict_proba_by_regime"):
        raise SystemExit("FATAL: router missing predict_proba_by_regime")
    panel["prob"] = router.predict_proba_by_regime(X, panel["stock_regime"])
    print(f"  scored in {time.perf_counter()-t0:.1f}s; "
          f"prob range [{panel['prob'].min():.3f}, {panel['prob'].max():.3f}]")
    return panel


def _build_forward_paths(panel: pd.DataFrame):
    """Compute entry + forward (high/low/close)/entry matrices on the FULL,
    per-symbol-contiguous panel BEFORE any universe filtering, so the barrier
    path uses real consecutive calendar bars."""
    print("\n[3/5] Building forward paths + universe/eligibility mask ...")
    g = panel.groupby("symbol", sort=False)
    o = pd.to_numeric(panel["open"], errors="coerce")
    h = pd.to_numeric(panel["high"], errors="coerce")
    low = pd.to_numeric(panel["low"], errors="coerce")
    c = pd.to_numeric(panel["close"], errors="coerce")
    entry = g["open"].shift(-1)                      # BUY at T+1 open
    H = MAX_HOLD
    hr = np.empty((len(panel), H)); lr = np.empty((len(panel), H)); cr = np.empty((len(panel), H))
    valid_path = entry.notna().to_numpy()
    ent = entry.to_numpy()
    for k in range(1, H + 1):
        hk = g["high"].shift(-k).to_numpy()
        lk = g["low"].shift(-k).to_numpy()
        ck = g["close"].shift(-k).to_numpy()
        valid_path &= np.isfinite(hk) & np.isfinite(lk) & np.isfinite(ck)
        with np.errstate(invalid="ignore", divide="ignore"):
            hr[:, k - 1] = hk / ent
            lr[:, k - 1] = lk / ent
            cr[:, k - 1] = ck / ent

    avg20 = g["volume"].transform(lambda s: s.rolling(20, min_periods=1).mean()).to_numpy()
    atr = pd.to_numeric(panel["D_atr14"], errors="coerce").to_numpy()
    atr_frac = np.where(ent > 0, atr / ent, np.nan)   # ATR as fraction of entry

    eligible = (
        valid_path
        & np.isfinite(panel["prob"].to_numpy())
        & (pd.to_numeric(panel["close"], errors="coerce").to_numpy() >= MIN_CLOSE)
        & (avg20 >= MIN_AVG20_VOL)
    )
    if ALLOWED_REGIMES is not None:
        eligible &= panel["stock_regime"].isin(ALLOWED_REGIMES).to_numpy()

    sel = np.where(eligible)[0]
    print(f"  Eligible signals: {len(sel):,} (of {len(panel):,})")
    meta = pd.DataFrame({
        "timestamp": panel["timestamp"].to_numpy()[sel],
        "symbol": panel["symbol"].to_numpy()[sel],
        "regime": panel["stock_regime"].to_numpy()[sel],
        "prob": panel["prob"].to_numpy()[sel],
        "atr_frac": atr_frac[sel],
    })
    return meta.reset_index(drop=True), hr[sel].astype("float64"), lr[sel].astype("float64"), cr[sel].astype("float64")


def _oos_mask(ts: pd.Series) -> np.ndarray:
    days = ts.dt.normalize()
    if OOS_START_DATE is not None:
        cut = pd.Timestamp(OOS_START_DATE, tz=IST)
    else:
        ud = np.sort(days.unique())
        cut = ud[int(len(ud) * (1 - OOS_FRACTION))]
    return (days >= cut).to_numpy(), pd.Timestamp(cut)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = _load_scored_panel()
    meta, hr, lr, cr = _build_forward_paths(panel)

    meta["bucket"] = meta["prob"].apply(bucket_label)
    in_range = (meta["prob"] >= PROB_MIN) & (meta["prob"] < PROB_MAX)
    is_oos, cut = _oos_mask(meta["timestamp"])
    meta["is_oos"] = is_oos
    meta["year"] = pd.to_datetime(meta["timestamp"]).dt.year
    print(f"  OOS cut: {cut.date()}  | OOS signals: {int(is_oos.sum()):,}")

    cost = COST_BPS / 10000.0
    af = meta["atr_frac"].to_numpy()

    print("\n[4/5] Running TP/SL grid (fixed % + ATR multiples) ...")
    combos = []
    for tp in TP_PCTS:
        for sl in SL_PCTS:
            combos.append(("pct", tp, sl,
                           np.full(len(meta), 1.0 + tp),
                           np.full(len(meta), 1.0 - sl)))
    for tpm in TP_ATR_MULTS:
        for slm in SL_ATR_MULTS:
            combos.append(("atr", tpm, slm,
                           1.0 + tpm * af,
                           1.0 - slm * af))

    grid_rows, bucket_rows, regime_rows = [], [], []
    primary_detail = None
    for mode, tp, sl, tp_ratio, sl_ratio in combos:
        res = run_barrier(hr, lr, cr, tp_ratio, sl_ratio)
        net = res["gross"] - cost
        tag = (f"TP{tp:.0%}/SL{sl:.0%}" if mode == "pct"
               else f"TP{tp:g}xATR/SL{sl:g}xATR")

        for scope, m in [("Full", np.ones(len(meta), bool)), ("OOS", is_oos)]:
            mm = m & in_range.to_numpy()
            s = _summary(net[mm], res["gross"][mm], res["reason"][mm], res["mae"][mm],
                         res["mfe"][mm], res["hold_bars"][mm], res["time_to_tp"][mm],
                         res["time_to_sl"][mm])
            s.update({"mode": mode, "tp": tp, "sl": sl, "combo": tag, "scope": scope})
            grid_rows.append(s)

        # per-bucket and per-regime breakdown for every combo (OOS + Full)
        for scope, base in [("Full", np.ones(len(meta), bool)), ("OOS", is_oos)]:
            for bkt in sorted(meta["bucket"].unique()):
                mm = base & (meta["bucket"].to_numpy() == bkt) & in_range.to_numpy()
                if mm.sum() == 0:
                    continue
                s = _summary(net[mm], res["gross"][mm], res["reason"][mm], res["mae"][mm],
                             res["mfe"][mm], res["hold_bars"][mm], res["time_to_tp"][mm],
                             res["time_to_sl"][mm])
                s.update({"combo": tag, "mode": mode, "tp": tp, "sl": sl,
                          "scope": scope, "bucket": bkt})
                bucket_rows.append(s)
            for rg in sorted(meta["regime"].unique()):
                mm = base & (meta["regime"].to_numpy() == rg) & in_range.to_numpy()
                if mm.sum() == 0:
                    continue
                s = _summary(net[mm], res["gross"][mm], res["reason"][mm], res["mae"][mm],
                             res["mfe"][mm], res["hold_bars"][mm], res["time_to_tp"][mm],
                             res["time_to_sl"][mm])
                s.update({"combo": tag, "mode": mode, "scope": scope, "regime": rg})
                regime_rows.append(s)

        # per-trade detail for the primary fixed combo
        if mode == "pct" and abs(tp - PRIMARY_TP_PCT) < 1e-9 and abs(sl - PRIMARY_SL_PCT) < 1e-9:
            primary_detail = meta.copy()
            primary_detail["exit_reason"] = np.where(res["reason"] == 1, "TP",
                                              np.where(res["reason"] == -1, "SL", "TIME"))
            primary_detail["hold_bars"] = res["hold_bars"]
            primary_detail["time_to_tp"] = res["time_to_tp"]
            primary_detail["time_to_sl"] = res["time_to_sl"]
            primary_detail["mae_pct"] = (res["mae"] * 100).round(3)
            primary_detail["mfe_pct"] = (res["mfe"] * 100).round(3)
            primary_detail["gross_pct"] = (res["gross"] * 100).round(3)
            primary_detail["net_pct"] = (net * 100).round(3)

    grid_df = pd.DataFrame(grid_rows)
    bucket_df = pd.DataFrame(bucket_rows)
    regime_df = pd.DataFrame(regime_rows)

    print("\n[5/5] Saving outputs ...")
    grid_df.to_csv(OUT_DIR / "01_grid_overall.csv", index=False)
    bucket_df.to_csv(OUT_DIR / "02_grid_x_probbucket.csv", index=False)
    regime_df.to_csv(OUT_DIR / "03_grid_x_regime.csv", index=False)
    if primary_detail is not None:
        primary_detail.to_csv(OUT_DIR / "04_trades_primary_TP5_SL3.csv", index=False)

    # ---- console: OOS leaderboard by expectancy (fixed-% combos) ----
    oos = grid_df[grid_df["scope"] == "OOS"].copy().sort_values("expectancy_pct", ascending=False)
    print("\n" + "=" * 104)
    print("OOS LEADERBOARD by net expectancy/trade  (30 bps round-trip; first-hit; SL-first ties)")
    print("=" * 104)
    hdr = (f"  {'combo':<18}{'n':>8}{'TP%':>7}{'SL%':>7}{'Time%':>7}{'Win%':>7}"
           f"{'Exp%':>8}{'MAE%':>8}{'MFE%':>8}{'tTP':>6}{'tSL':>6}{'Shrp':>7}")
    print(hdr)
    for _, r in oos.head(25).iterrows():
        print(f"  {r['combo']:<18}{r['n']:>8,}{r['tp_rate_pct']:>7.1f}{r['sl_rate_pct']:>7.1f}"
              f"{r['time_rate_pct']:>7.1f}{r['win_rate_pct']:>7.1f}{r['expectancy_pct']:>+8.3f}"
              f"{r['avg_mae_pct']:>+8.2f}{r['avg_mfe_pct']:>+8.2f}"
              f"{(r['avg_time_to_tp'] if pd.notna(r['avg_time_to_tp']) else 0):>6.2f}"
              f"{(r['avg_time_to_sl'] if pd.notna(r['avg_time_to_sl']) else 0):>6.2f}"
              f"{r['sharpe_per_trade']:>7.3f}")

    # ---- console: does expectancy rise with probability? (primary combo) ----
    pr = bucket_df[(bucket_df["scope"] == "OOS") & (bucket_df["mode"] == "pct")
                   & (np.isclose(bucket_df["tp"], PRIMARY_TP_PCT))
                   & (np.isclose(bucket_df["sl"], PRIMARY_SL_PCT))].sort_values("bucket")
    print("\n" + "=" * 104)
    print(f"PROB-BUCKET MONOTONICITY  (OOS, primary combo TP{PRIMARY_TP_PCT:.0%}/SL{PRIMARY_SL_PCT:.0%})")
    print("=" * 104)
    print(f"  {'bucket':<16}{'n':>8}{'Win%':>8}{'TP%':>8}{'SL%':>8}{'Exp%':>9}{'MAE%':>8}{'MFE%':>8}")
    for _, r in pr.iterrows():
        print(f"  {r['bucket']:<16}{r['n']:>8,}{r['win_rate_pct']:>8.1f}{r['tp_rate_pct']:>8.1f}"
              f"{r['sl_rate_pct']:>8.1f}{r['expectancy_pct']:>+9.3f}{r['avg_mae_pct']:>+8.2f}"
              f"{r['avg_mfe_pct']:>+8.2f}")

    print("\n" + "=" * 104)
    print(f"OUTPUTS in {OUT_DIR}")
    print("  01_grid_overall.csv         - every TP/SL combo, Full + OOS")
    print("  02_grid_x_probbucket.csv    - every combo x 5% prob bucket")
    print("  03_grid_x_regime.csv        - every combo x regime")
    print("  04_trades_primary_TP5_SL3.csv - per-trade detail (MAE/MFE/timing)")
    print("=" * 104)
    print("\nTo COMPARE binary vs rank: run this twice with BASE_DIR pointed at each")
    print("model's output folder, then diff 01_grid_overall.csv (scope=OOS).")


if __name__ == "__main__":
    main()

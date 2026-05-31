#!/usr/bin/env python3
"""
=============================================================================
FILTERED BACKTEST: high-probability buckets in trend regimes
=============================================================================

Restricts the analysis to:
  - stock_regime in {bull_trend, bear_trend}
  - probability >= 0.70

Then splits by 5%-wide probability buckets:
  [0.70, 0.75), [0.75, 0.80), [0.80, 0.85), [0.85, 0.90), [0.90, 0.95), [0.95, 1.00]

For each bucket, computes:
  - n_trades (full period + OOS)
  - win_rate
  - avg_ret, avg_win, avg_loss, expectancy
  - sharpe
  - calibration (predicted vs actual)
  - cost sensitivity (0, 10, 25, 50 bps)
  - per-regime split (bull_trend vs bear_trend)
  - timeline stability (yearly breakdown)

Uses the same model, panel, and forward return convention as backtest.py
(T+1 open to T+6 close via ret_5d_oc_pct).
=============================================================================
"""

import json
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
FALLBACK_PATH = MODELS_DIR / "m5_ensemble.joblib"

OUT_DIR = BASE_DIR / "backtest_results_filtered"
OUT_DIR.mkdir(exist_ok=True)

IST = "Asia/Kolkata"

# Universe filters (match backtest.py)
MIN_CLOSE = 2.0
MIN_AVG20_VOL = 200_000

# Backtest config
RET_COL = "ret_5d_oc_pct"  # T+1 open to T+6 close
OOS_FRACTION = 0.20

# Filter config
ALLOWED_REGIMES = ["bull_trend", "bear_trend"]
PROB_BUCKETS = [
    (0.70, 0.75),
    (0.75, 0.80),
    (0.80, 0.85),
    (0.85, 0.90),
    (0.90, 0.95),
    (0.95, 1.00),
]

# Cost sensitivity
COST_BPS_LIST = [0, 10, 25, 50]

# =============================================================================
# LOAD
# =============================================================================

print("=" * 70)
print("FILTERED BACKTEST: prob >= 0.70, trend regimes only, 5% buckets")
print("=" * 70)

print("\n[1/5] Loading panel and models...")
panel = pd.read_parquet(PANEL_PATH)
panel["timestamp"] = pd.to_datetime(panel["timestamp"])
if panel["timestamp"].dt.tz is None:
    panel["timestamp"] = panel["timestamp"].dt.tz_localize(IST)
else:
    panel["timestamp"] = panel["timestamp"].dt.tz_convert(IST)
panel = panel.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
print(f"  Panel: {len(panel):,} rows")

if "stock_regime" not in panel.columns:
    raise SystemExit("FATAL: stock_regime missing. Run enrich_panel.py first.")
if RET_COL not in panel.columns:
    raise SystemExit(f"FATAL: panel missing '{RET_COL}'")

# Universe filter
panel["avg20_vol"] = (
    panel.groupby("symbol")["volume"]
         .transform(lambda s: s.rolling(20, min_periods=1).mean())
)
panel = panel[(panel["close"] >= MIN_CLOSE) & (panel["avg20_vol"] >= MIN_AVG20_VOL)].reset_index(drop=True)
print(f"  After universe filter: {len(panel):,} rows")

# Schema
schema = json.loads(FEATURES_PATH.read_text())
FEATURES = schema["features"]
IMPUTE = {k: float(v) for k, v in schema["impute"].items()}
print(f"  Features: {len(FEATURES)}")

# Load model
if ROUTER_PATH.exists():
    router = load(ROUTER_PATH)
    print(f"  Loaded regime router")
else:
    raise SystemExit("FATAL: regime router not found")

# =============================================================================
# PREDICT
# =============================================================================

print("\n[2/5] Generating predictions...")
import time
t0 = time.perf_counter()

X = panel.reindex(columns=FEATURES).copy()
for c in FEATURES:
    X[c] = pd.to_numeric(X[c], errors="coerce").fillna(IMPUTE.get(c, 0.0))

if hasattr(router, "predict_proba_by_regime"):
    probs = router.predict_proba_by_regime(X, panel["stock_regime"])
else:
    raise SystemExit("FATAL: router missing predict_proba_by_regime")

panel["prob"] = probs
print(f"  Done in {time.perf_counter()-t0:.1f}s")
print(f"  Probability range: [{probs.min():.4f}, {probs.max():.4f}]")

# Forward return
panel["fwd_ret_pct"] = pd.to_numeric(panel[RET_COL], errors="coerce")

# =============================================================================
# APPLY FILTERS
# =============================================================================

print("\n[3/5] Applying filters...")
trades = panel[panel["fwd_ret_pct"].notna() & panel["prob"].notna()].copy()
print(f"  Total valid trades: {len(trades):,}")

# Filter 1: trend regimes only
trades = trades[trades["stock_regime"].isin(ALLOWED_REGIMES)].copy()
print(f"  After trend regime filter: {len(trades):,}")

# Filter 2: prob >= 0.70
trades = trades[trades["prob"] >= 0.70].copy()
print(f"  After prob >= 0.70 filter: {len(trades):,}")

# Assign 5% probability bucket
def assign_prob_bucket(p):
    for low, high in PROB_BUCKETS:
        if low <= p < high or (high == 1.00 and p == 1.00):
            return f"[{low:.2f}, {high:.2f})"
    return None

trades["prob_bucket"] = trades["prob"].apply(assign_prob_bucket)

# OOS split
trades = trades.sort_values("timestamp").reset_index(drop=True)
# Use the SAME OOS cutoff as the main backtest (last 20% of all dates in full panel)
all_dates = panel["timestamp"].dt.normalize().drop_duplicates().sort_values()
oos_cutoff = all_dates.iloc[int(len(all_dates) * (1 - OOS_FRACTION))]
trades["is_oos"] = trades["timestamp"] >= oos_cutoff
trades["is_win"] = (trades["fwd_ret_pct"] > 0).astype(int)
trades["year"] = trades["timestamp"].dt.year

print(f"  OOS cutoff: {oos_cutoff.date()}")
print(f"  OOS trades: {trades['is_oos'].sum():,}")

# =============================================================================
# COMPUTE STATS
# =============================================================================

def stats(sub: pd.DataFrame) -> dict:
    if len(sub) == 0:
        return {"n": 0}
    r = sub["fwd_ret_pct"].values
    wins = (r > 0)
    losses = ~wins
    return {
        "n": len(sub),
        "win_rate_pct": round(100 * wins.mean(), 2),
        "avg_ret_pct": round(r.mean(), 3),
        "avg_win_pct": round(r[wins].mean() if wins.any() else 0, 3),
        "avg_loss_pct": round(r[losses].mean() if losses.any() else 0, 3),
        "median_ret_pct": round(np.median(r), 3),
        "std_pct": round(r.std(), 3),
        "sharpe_per_trade": round(r.mean() / (r.std() + 1e-9), 3),
        "sharpe_annual": round(r.mean() / (r.std() + 1e-9) * np.sqrt(50), 2),
        "min_ret": round(r.min(), 2),
        "max_ret": round(r.max(), 2),
        "predicted_avg_prob": round(sub["prob"].mean(), 4),
        "calib_gap": round(sub["prob"].mean() - wins.mean(), 4),
    }


print("\n[4/5] Computing reports...")

# --- Report 1: Per probability bucket (full + OOS) ---
bucket_rows = []
for low, high in PROB_BUCKETS:
    label = f"[{low:.2f}, {high:.2f})"
    for scope, df in [("Full", trades), ("OOS", trades[trades["is_oos"]])]:
        sub = df[df["prob_bucket"] == label]
        s = stats(sub)
        s["prob_bucket"] = label
        s["scope"] = scope
        bucket_rows.append(s)
bucket_df = pd.DataFrame(bucket_rows)
bucket_df = bucket_df[["prob_bucket", "scope", "n", "win_rate_pct", "predicted_avg_prob",
                       "calib_gap", "avg_ret_pct", "avg_win_pct", "avg_loss_pct",
                       "median_ret_pct", "std_pct", "sharpe_per_trade", "sharpe_annual",
                       "min_ret", "max_ret"]]

# --- Report 2: Per bucket + per regime (full period and OOS) ---
regime_rows = []
for low, high in PROB_BUCKETS:
    label = f"[{low:.2f}, {high:.2f})"
    for regime in ALLOWED_REGIMES:
        for scope, df in [("Full", trades), ("OOS", trades[trades["is_oos"]])]:
            sub = df[(df["prob_bucket"] == label) & (df["stock_regime"] == regime)]
            s = stats(sub)
            s["prob_bucket"] = label
            s["regime"] = regime
            s["scope"] = scope
            regime_rows.append(s)
regime_df = pd.DataFrame(regime_rows)
regime_df = regime_df[["prob_bucket", "regime", "scope", "n", "win_rate_pct",
                       "avg_ret_pct", "median_ret_pct", "sharpe_annual"]]

# --- Report 3: Cost sensitivity per bucket (OOS only) ---
cost_rows = []
for low, high in PROB_BUCKETS:
    label = f"[{low:.2f}, {high:.2f})"
    sub = trades[(trades["prob_bucket"] == label) & trades["is_oos"]]
    if len(sub) == 0:
        continue
    gross = sub["fwd_ret_pct"].mean()
    std = sub["fwd_ret_pct"].std()
    for bps in COST_BPS_LIST:
        cost = bps / 100
        net = gross - cost
        cost_rows.append({
            "prob_bucket": label,
            "n_trades_oos": len(sub),
            "bps": bps,
            "gross_avg_pct": round(gross, 3),
            "net_avg_pct": round(net, 3),
            "net_win_rate_pct": round(100 * ((sub["fwd_ret_pct"] - cost) > 0).mean(), 2),
            "net_sharpe_annual": round(net / (std + 1e-9) * np.sqrt(50), 2),
        })
cost_df = pd.DataFrame(cost_rows)

# --- Report 4: Yearly stability per bucket ---
year_rows = []
for low, high in PROB_BUCKETS:
    label = f"[{low:.2f}, {high:.2f})"
    for year in sorted(trades["year"].unique()):
        sub = trades[(trades["prob_bucket"] == label) & (trades["year"] == year)]
        s = stats(sub)
        s["prob_bucket"] = label
        s["year"] = year
        year_rows.append(s)
year_df = pd.DataFrame(year_rows)
year_df = year_df[["prob_bucket", "year", "n", "win_rate_pct", "avg_ret_pct", "sharpe_annual"]]

# --- Report 5: Combined "prob >= threshold" cumulative analysis ---
threshold_rows = []
for thresh in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
    for scope, df in [("Full", trades), ("OOS", trades[trades["is_oos"]])]:
        sub = df[df["prob"] >= thresh]
        s = stats(sub)
        s["prob_threshold"] = f">= {thresh:.2f}"
        s["scope"] = scope
        threshold_rows.append(s)
threshold_df = pd.DataFrame(threshold_rows)
threshold_df = threshold_df[["prob_threshold", "scope", "n", "win_rate_pct",
                              "predicted_avg_prob", "calib_gap", "avg_ret_pct",
                              "median_ret_pct", "sharpe_annual", "min_ret", "max_ret"]]

# --- Report 6: Daily trade frequency per bucket (OOS) ---
freq_rows = []
oos_dates = trades[trades["is_oos"]]["timestamp"].dt.normalize().drop_duplicates()
n_oos_days = len(oos_dates)
for low, high in PROB_BUCKETS:
    label = f"[{low:.2f}, {high:.2f})"
    sub = trades[(trades["prob_bucket"] == label) & trades["is_oos"]]
    freq_rows.append({
        "prob_bucket": label,
        "n_trades_oos": len(sub),
        "oos_trading_days": n_oos_days,
        "avg_trades_per_day": round(len(sub) / max(n_oos_days, 1), 2),
        "days_with_signals": sub["timestamp"].dt.normalize().nunique(),
        "pct_days_with_signal": round(100 * sub["timestamp"].dt.normalize().nunique() / max(n_oos_days, 1), 1),
    })
freq_df = pd.DataFrame(freq_rows)

# =============================================================================
# SAVE
# =============================================================================

print("\n[5/5] Saving outputs...")

csv_outputs = {
    "01_bucket_overview.csv": bucket_df,
    "02_bucket_x_regime.csv": regime_df,
    "03_cost_sensitivity.csv": cost_df,
    "04_yearly_stability.csv": year_df,
    "05_threshold_cumulative.csv": threshold_df,
    "06_frequency.csv": freq_df,
}
for fname, df in csv_outputs.items():
    df.to_csv(OUT_DIR / fname, index=False)
print(f"  Saved {len(csv_outputs)} CSVs to {OUT_DIR}")

# Excel
def strip_tz(df):
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    for c in out.columns:
        try:
            if pd.api.types.is_datetime64_any_dtype(out[c]):
                if hasattr(out[c].dt, "tz") and out[c].dt.tz is not None:
                    out[c] = out[c].dt.tz_localize(None)
        except Exception:
            pass
    return out

excel_path = OUT_DIR / "filtered_backtest.xlsx"
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    strip_tz(bucket_df).to_excel(writer, sheet_name="01 Bucket Overview", index=False)
    strip_tz(regime_df).to_excel(writer, sheet_name="02 Bucket x Regime", index=False)
    strip_tz(cost_df).to_excel(writer, sheet_name="03 Cost Sensitivity OOS", index=False)
    strip_tz(year_df).to_excel(writer, sheet_name="04 Yearly Stability", index=False)
    strip_tz(threshold_df).to_excel(writer, sheet_name="05 Threshold Cumulative", index=False)
    strip_tz(freq_df).to_excel(writer, sheet_name="06 Daily Frequency", index=False)
print(f"  Saved Excel: {excel_path}")

# =============================================================================
# CONSOLE SUMMARY
# =============================================================================

print("\n" + "=" * 90)
print("BUCKET OVERVIEW: trend regimes only, by probability bucket (OOS)")
print("=" * 90)
print(f"  {'Bucket':<18} {'N':>7} {'Win%':>7} {'Pred':>7} {'Gap':>7} {'AvgRet%':>9} {'Med%':>8} {'Sharpe':>8}")
for _, row in bucket_df[bucket_df["scope"] == "OOS"].iterrows():
    if row["n"] == 0:
        continue
    print(f"  {row['prob_bucket']:<18} {row['n']:>7,} "
          f"{row['win_rate_pct']:>7.1f} {row['predicted_avg_prob']:>7.4f} "
          f"{row['calib_gap']:>+7.4f} {row['avg_ret_pct']:>+9.3f} "
          f"{row['median_ret_pct']:>+8.3f} {row['sharpe_annual']:>8.2f}")

print("\n" + "=" * 90)
print("CUMULATIVE BY THRESHOLD (OOS - everything at or above each threshold)")
print("=" * 90)
print(f"  {'Threshold':<12} {'N':>7} {'Win%':>7} {'AvgRet%':>9} {'Sharpe':>8} {'Min':>7} {'Max':>8}")
for _, row in threshold_df[threshold_df["scope"] == "OOS"].iterrows():
    if row["n"] == 0:
        continue
    print(f"  {row['prob_threshold']:<12} {row['n']:>7,} "
          f"{row['win_rate_pct']:>7.1f} {row['avg_ret_pct']:>+9.3f} "
          f"{row['sharpe_annual']:>8.2f} {row['min_ret']:>+7.2f} {row['max_ret']:>+8.2f}")

print("\n" + "=" * 90)
print("PER-REGIME (OOS)")
print("=" * 90)
print(f"  {'Bucket':<18} {'Regime':<12} {'N':>7} {'Win%':>7} {'AvgRet%':>9} {'Sharpe':>8}")
for _, row in regime_df[regime_df["scope"] == "OOS"].iterrows():
    if row["n"] == 0:
        continue
    print(f"  {row['prob_bucket']:<18} {row['regime']:<12} {row['n']:>7,} "
          f"{row['win_rate_pct']:>7.1f} {row['avg_ret_pct']:>+9.3f} "
          f"{row['sharpe_annual']:>8.2f}")

print("\n" + "=" * 90)
print("COST SENSITIVITY (OOS, top buckets only)")
print("=" * 90)
print(f"  {'Bucket':<18} {'N':>7} {'BPS':>5} {'Gross%':>9} {'Net%':>9} {'NetWin%':>9} {'Sharpe':>8}")
for _, row in cost_df.iterrows():
    print(f"  {row['prob_bucket']:<18} {row['n_trades_oos']:>7,} {row['bps']:>5} "
          f"{row['gross_avg_pct']:>+9.3f} {row['net_avg_pct']:>+9.3f} "
          f"{row['net_win_rate_pct']:>9.1f} {row['net_sharpe_annual']:>8.2f}")

print("\n" + "=" * 90)
print("OOS DAILY FREQUENCY")
print("=" * 90)
print(f"  {'Bucket':<18} {'N OOS':>8} {'AvgPerDay':>12} {'Days w/sig':>12} {'%days':>8}")
for _, row in freq_df.iterrows():
    print(f"  {row['prob_bucket']:<18} {row['n_trades_oos']:>8,} "
          f"{row['avg_trades_per_day']:>12.2f} {row['days_with_signals']:>12} "
          f"{row['pct_days_with_signal']:>8.1f}")

print("\n" + "=" * 90)
print(f"OUTPUTS: {OUT_DIR}")
print(f"Excel: {excel_path}")
print("=" * 90)
print("\nKey questions to answer from the output:")
print("  1. Does avg_ret_pct rise monotonically with prob_bucket in OOS?")
print("  2. Is OOS realized win rate close to predicted prob, or does the gap widen?")
print("  3. At what threshold does avg_trades_per_day become tradeable for you?")
print("  4. Is bull_trend or bear_trend stronger at each prob level?")
print("  5. Does net return stay positive at realistic cost levels (20-30 bps)?")
print("=" * 90)

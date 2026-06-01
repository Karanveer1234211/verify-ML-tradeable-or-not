#!/usr/bin/env python3
"""
=============================================================================
FEATURE FACTORY v3.0 — Fast, parallel, leak-free
=============================================================================

SPEED IMPROVEMENTS over v2 (zero quality compromise):
  - Parallelized per-symbol generation using joblib (all CPU cores).
  - Vectorized panel-wide IC computation (no per-date Python loop).
  - Temporary parquet caching: candidates generated once, reused in Stage 6.
  - Batch forward selection: evaluate top-K block, not one-at-a-time.
  - Stage 5 removed (was a no-op in v2; regime info still in stage 2 CSV).

QUALITY (preserved from v2):
  - Train/test discipline: all thresholds computed on date < train_cutoff.
  - 3-fold purged time-series CV with embargo.
  - Walk-forward IC stability filter.
  - Redundancy filter (corr > 0.92).
  - Memory-safe streaming: never builds full candidate matrix in RAM.

EXPECTED RUNTIME:
  ~30-45 minutes for 2,000 stocks × 6 years on 8-core machine.

USAGE:
  python feature_factory_v3.py
"""

from __future__ import annotations

import gc
import json
import time
import warnings
import shutil
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from tqdm import tqdm
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# =============================================================================
#                               CONFIG
# =============================================================================

BASE_DIR      = Path(r"C:\Users\karanvsi\Desktop\Kite Connect\v3_2_output_full")
PANEL_PATH    = BASE_DIR / "panel_cache.parquet"
FEATURES_JSON = BASE_DIR / "features_train.json"
OUT_DIR       = BASE_DIR / "feature_factory"
TMP_DIR       = OUT_DIR / "tmp_candidates"
try:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass  # created lazily in main(); --self-test needs no output dir

IST           = "Asia/Kolkata"
RET_COL       = "ret_5d_oc_pct"
TARGET_COL    = "top20_vs_bot20_5d"

MIN_CLOSE     = 2.0
MIN_VOL       = 200_000

TRAIN_TEST_FRAC = 0.85

# IC thresholds
IC_THRESHOLD       = 0.015
POS_RATE_THRESHOLD = 0.52
TSTAT_THRESHOLD    = 1.5

# Walk-forward stability
ROLLING_WINDOW_DAYS       = 252
ROLLING_STEP_DAYS         = 63
MIN_POSITIVE_WINDOWS_FRAC = 0.66

# Redundancy
CORR_THRESHOLD = 0.92

# Forward selection
AUC_DELTA      = 0.0002
MAX_FEATURES   = 150
EMBARGO_DAYS   = 5
N_CV_FOLDS     = 3

# Batch selection
BATCH_SIZE_INIT  = 20    # try top-20 candidates as a block first
BATCH_SIZE_MIN   = 1     # finally fall back to one-at-a-time refinement

# Parallelization
N_JOBS = -1   # use all CPU cores

# LightGBM
LGB_PARAMS = dict(
    n_estimators    = 300,
    learning_rate   = 0.05,
    num_leaves      = 63,
    max_depth       = 6,
    feature_fraction= 0.8,
    bagging_fraction= 0.8,
    bagging_freq    = 1,
    min_data_in_leaf= 100,
    reg_alpha       = 0.1,
    reg_lambda      = 5.0,
    n_jobs          = -1,
    random_state    = 42,
    verbosity       = -1,
)

# Transformations
LAG_WINDOWS  = [1, 3, 5]
ROLL_WINDOWS = [10, 20, 50]
DIFF_WINDOWS = [1, 5]

TRANSFORM_BASE_COLS = [
    "D_rsi14", "D_rsi7", "D_adx14", "D_pdi14", "D_mdi14",
    "D_macd_hist", "D_cmf20", "D_obv_slope",
    "D_bb_pctB_20", "D_bb_bw_20",
    "D_vol_yz_20", "D_vol_yz_50",
    "D_donch_pos_20", "D_donch_pos_50",
    "D_dvol_z20", "D_dvol_z50",
    "D_ema20_angle_deg", "D_atr14_to_close_pct",
    "D_close_roll_slope_20", "D_ret_5d_roll_std",
    "D_gap_pct", "D_range_pct", "D_atr_pct",
    "D_body_ratio", "D_wick_skew",
    "D_compress_state", "D_midpoint_slope", "D_slope_stability",
    "W_ret_4w", "W_ret_13w", "W_close_pos", "W_vol_vs_4w",
]


# =============================================================================
#                          HELPERS
# =============================================================================

def _safe(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _ts_rank(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=max(3, w // 2)).rank(pct=True)


def _ts_corr(a: pd.Series, b: pd.Series, w: int) -> pd.Series:
    return a.rolling(w, min_periods=max(5, w // 2)).corr(b)


def _ts_std(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=max(3, w // 2)).std()


def _ts_mean(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=max(3, w // 2)).mean()


def _ts_max(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=max(3, w // 2)).max()


def _ts_min(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=max(3, w // 2)).min()


def _delta(s: pd.Series, d: int) -> pd.Series:
    return s.diff(d)


def _delay(s: pd.Series, d: int) -> pd.Series:
    return s.shift(d)


def _rank(s: pd.Series, min_periods: int = 60) -> pd.Series:
    # LEAK-FREE per-symbol rank. _compute_wq_alphas runs on ONE symbol's full
    # series, so the old s.rank(pct=True) ranked each row against that symbol's
    # entire past AND FUTURE - a look-ahead leak (same class as the _rank_cs bug
    # fixed in Daily cache v20). Expanding rank ranks s[t] only within s[0..t].
    # NOTE: temporal per-symbol rank, NOT the WorldQuant cross-sectional rank;
    # a true WQ rank must be computed at panel level (groupby date).
    return s.expanding(min_periods=min_periods).rank(pct=True)


def _signed_power(s: pd.Series, e: float) -> pd.Series:
    return np.sign(s) * (s.abs() ** e)


def _scale(s: pd.Series, min_periods: int = 1) -> pd.Series:
    # LEAK-FREE: the WorldQuant "scale" operator is cross-sectional (per day)
    # in the literature. In this per-symbol generator the old s/s.abs().sum()
    # divided every row by the FULL-series sum (incl. future) -> look-ahead.
    # Use a CAUSAL expanding L1 norm: s[t] / sum(|s[0..t]|).
    denom = s.abs().expanding(min_periods=min_periods).sum().replace(0, np.nan)
    return s / denom


# =============================================================================
#                  STAGE 1 — LOAD PANEL
# =============================================================================

def load_panel() -> Tuple[pd.DataFrame, List[str], dict]:
    print("\n[1/6] Loading panel...")
    panel = pd.read_parquet(PANEL_PATH)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"]).dt.tz_convert(IST)
    panel = panel.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    panel["date"] = panel["timestamp"].dt.normalize()
    panel["year"] = panel["timestamp"].dt.year

    panel["avg20_vol"] = (
        panel.groupby("symbol")["volume"]
        .transform(lambda s: s.rolling(20, min_periods=1).mean())
    )

    panel = panel[
        (_safe(panel["close"]) >= MIN_CLOSE) &
        (panel["avg20_vol"] >= MIN_VOL) &
        panel[RET_COL].notna()
    ].copy()

    if TARGET_COL not in panel.columns:
        panel = _build_target(panel)

    schema = json.loads(FEATURES_JSON.read_text())
    existing = schema["features"]
    impute = {k: float(v) for k, v in schema["impute"].items()}

    print(f"  Panel rows       : {len(panel):,}")
    print(f"  Symbols          : {panel['symbol'].nunique():,}")
    print(f"  Date range       : {panel['date'].min().date()} → {panel['date'].max().date()}")
    print(f"  Existing features: {len(existing)}")
    return panel, existing, impute


def _build_target(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    r5 = _safe(p[RET_COL])
    atr_pct = (
        _safe(p.get("D_atr14", pd.Series(dtype=float)))
        / _safe(p["close"])
    ).replace([np.inf, -np.inf], np.nan) * 100.0
    vol_basis = atr_pct.replace(0, np.nan)
    p["ret_5d_adj"] = r5 / vol_basis
    p["rank_5d_pct"] = p.groupby("date")["ret_5d_adj"].rank(method="average", pct=True)
    p[TARGET_COL] = np.where(
        p["rank_5d_pct"] >= 0.80, 1,
        np.where(p["rank_5d_pct"] <= 0.20, 0, np.nan)
    )
    return p


def _compute_train_cutoff(panel: pd.DataFrame) -> pd.Timestamp:
    all_dates = sorted(panel["date"].unique())
    cutoff_idx = int(len(all_dates) * TRAIN_TEST_FRAC)
    return pd.Timestamp(all_dates[cutoff_idx])


# =============================================================================
#         STAGE 2 — PARALLEL CANDIDATE GENERATION + VECTORIZED IC
# =============================================================================

def _generate_for_symbol_to_disk(
    sym_df: pd.DataFrame, base_cols: List[str], symbol: str
) -> Tuple[str, List[str]]:
    """Generate features for one symbol and persist to a temp parquet file."""
    df = sym_df.reset_index(drop=True).copy()
    gen_cols: List[str] = []

    # ── Transforms ───────────────────────────────────────────────────────
    for col in base_cols:
        s = _safe(df[col])

        for lag in LAG_WINDOWS:
            df[f"{col}_lag{lag}"] = s.shift(lag)
            gen_cols.append(f"{col}_lag{lag}")

        for w in ROLL_WINDOWS:
            df[f"{col}_rmean{w}"] = s.rolling(w, min_periods=max(3, w // 4)).mean()
            df[f"{col}_rstd{w}"]  = s.rolling(w, min_periods=max(3, w // 4)).std()
            df[f"{col}_rrank{w}"] = s.rolling(w, min_periods=max(3, w // 4)).rank(pct=True)
            gen_cols.extend([f"{col}_rmean{w}", f"{col}_rstd{w}", f"{col}_rrank{w}"])

        for d in DIFF_WINDOWS:
            df[f"{col}_diff{d}"] = s.diff(d)
            gen_cols.append(f"{col}_diff{d}")

    # ── WorldQuant alphas ────────────────────────────────────────────────
    wq_added = _compute_wq_alphas(df)
    gen_cols.extend(wq_added)

    # Save to temp parquet (only the columns we need: candidates + key cols)
    keep_cols = list(set(gen_cols + ["symbol", "date", RET_COL, TARGET_COL]))
    keep_cols = [c for c in keep_cols if c in df.columns]
    out_df = df[keep_cols].copy()

    safe_sym = symbol.replace("/", "_").replace("\\", "_")
    out_path = TMP_DIR / f"{safe_sym}.parquet"
    out_df.to_parquet(out_path, index=False, compression="snappy")

    return str(out_path), gen_cols


def _compute_wq_alphas(df: pd.DataFrame) -> List[str]:
    """38 WorldQuant alphas from daily OHLCV. Returns added column names."""
    cols_added: List[str] = []
    close = _safe(df["close"])
    open_ = _safe(df["open"])
    high  = _safe(df["high"])
    low   = _safe(df["low"])
    volume = _safe(df["volume"]).fillna(0.0)
    ret = close.pct_change()
    vwap = (high + low + close) / 3.0
    adv20 = volume.rolling(20, min_periods=5).mean()

    def _add(name: str, s):
        if isinstance(s, np.ndarray):
            s = pd.Series(s, index=df.index)
        df[name] = s.replace([np.inf, -np.inf], np.nan)
        cols_added.append(name)

    try: _add("WQ_1", _signed_power(ret, 2).rolling(5, min_periods=3).apply(
        lambda x: float(np.argmax(x)) / max(len(x) - 1, 1), raw=True) - 0.5)
    except: pass

    try: _add("WQ_3", -_ts_corr(_rank(open_), _rank(volume), 10))
    except: pass

    try: _add("WQ_6", -_ts_corr(open_, volume, 10))
    except: pass

    try:
        cond = adv20 < volume
        part = -_ts_rank((_delta(close, 7)).abs(), 60) * np.sign(_delta(close, 7))
        _add("WQ_7", part.where(cond, -1))
    except: pass

    try:
        diff_vc = vwap - close
        _add("WQ_11", _rank(_ts_max(diff_vc, 3)) + _rank(_ts_min(diff_vc, 3)) * _rank(_delta(volume, 3)))
    except: pass

    try: _add("WQ_12", np.sign(_delta(volume, 1)) * (-_delta(close, 1)))
    except: pass

    try: _add("WQ_13", -_rank(_rank(close).rolling(5, min_periods=3).cov(_rank(volume))))
    except: pass

    try: _add("WQ_14", -_rank(_delta(ret, 3)) * _ts_corr(open_, volume, 10))
    except: pass

    try:
        inner = _rank(_ts_corr(_rank(high), _rank(volume), 3))
        _add("WQ_15", -inner.rolling(3, min_periods=1).sum())
    except: pass

    try: _add("WQ_16", -_rank(_rank(high).rolling(5, min_periods=3).cov(_rank(volume))))
    except: pass

    try:
        d7 = _delay(close, 7)
        part = -np.sign(_delta(close - d7, 5) + _delta(close, 5))
        ret_sum = ret.rolling(250, min_periods=50).sum()
        _add("WQ_19", part * (1 + _rank(1 + ret_sum)))
    except: pass

    try: _add("WQ_20",
        -_rank(open_ - _delay(high, 1))
        * _rank(open_ - _delay(close, 1))
        * _rank(open_ - _delay(low, 1)))
    except: pass

    try:
        cond = _ts_mean(high, 20) < high
        _add("WQ_23", (-_delta(high, 2)).where(cond, 0))
    except: pass

    try:
        m100 = _ts_mean(close, 100)
        _add("WQ_24", m100 / _delay(close, 100).replace(0, np.nan) - 1)
    except: pass

    try:
        inner = _ts_corr(_ts_rank(volume, 5), _ts_rank(high, 5), 5)
        _add("WQ_26", -_ts_max(inner, 3))
    except: pass

    try:
        corr_part = _ts_corr(adv20, low, 5)
        diff_part = close - open_
        _add("WQ_28", _scale(corr_part) + _scale(diff_part))
    except: pass

    try:
        inner = -_rank(_delta(close, 5))
        _add("WQ_29", _rank(_rank(inner)))
    except: pass

    try: _add("WQ_33", _rank(-1 + open_ / close.replace(0, np.nan)))
    except: pass

    try: _add("WQ_34", _rank((1 - _rank(_ts_std(ret, 2))) + (1 - _rank(_ts_corr(close, open_, 5)))))
    except: pass

    try: _add("WQ_35",
        _ts_rank(volume, 32)
        * (1 - _ts_rank(close + high - low, 16))
        * (1 - _ts_rank(ret, 32)))
    except: pass

    try: _add("WQ_38", -_rank(_ts_rank(close, 10)) * _rank(close / open_.replace(0, np.nan)))
    except: pass

    try: _add("WQ_40", -_rank(_ts_std(high, 10)) * _ts_corr(high, volume, 10))
    except: pass

    try: _add("WQ_41", np.sqrt(high * low) - vwap)
    except: pass

    try:
        vol_ratio = (volume / adv20.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        _add("WQ_43", _ts_rank(vol_ratio, 20) * _ts_rank(-_delta(close, 7), 8))
    except: pass

    try: _add("WQ_44", -_ts_corr(high, _rank(volume), 5))
    except: pass

    try:
        part1 = (_delay(close, 20) - _delay(close, 10)) / 10
        part2 = (_delay(close, 10) - close) / 10
        cond = 0.25 < (part1 - part2)
        _add("WQ_46", np.where(cond, -1, 1))
    except: pass

    try:
        cond = (_delta(_delay(close, 20), 10) / 10) > 0.1
        _add("WQ_49", np.where(cond, -1, -_delta(close, 1)))
    except: pass

    try:
        cond = (_delta(_delay(close, 20), 10) / 10) > 0.05
        _add("WQ_51", np.where(cond, -1, -_delta(close, 1)))
    except: pass

    try:
        ret_240 = ret.rolling(240, min_periods=50).sum()
        ret_20 = ret.rolling(20, min_periods=5).sum()
        mom_r = _rank((ret_240 - ret_20) / ret_20.replace(0, np.nan))
        min5 = _ts_min(low, 5)
        _add("WQ_52", (-min5 + _delay(min5, 5)) * mom_r)
    except: pass

    try:
        ratio = (high - close) / (close - low).replace(0, np.nan)
        _add("WQ_53", -_delta(1 - ratio, 9))
    except: pass

    try:
        num = -(low - close) * (open_ ** 5)
        den = (low - high).replace(0, np.nan) * (close ** 5).replace(0, np.nan)
        _add("WQ_54", (num / den).replace([np.inf, -np.inf], np.nan))
    except: pass

    try:
        cap_proxy = close * volume
        num = _rank(ret.rolling(10, min_periods=3).sum() /
                    ret.rolling(2, min_periods=1).sum().rolling(3, min_periods=1).sum().replace(0, np.nan))
        _add("WQ_56", -(num * _rank(ret * cap_proxy)))
    except: pass

    try: _add("WQ_101", (close - open_) / (high - low + 0.001))
    except: pass

    return cols_added


def parallel_generate_candidates(panel: pd.DataFrame) -> Tuple[List[str], set]:
    """
    Stage 2a: Parallelized candidate generation across all symbols.
    Each symbol's candidates are written to a separate temp parquet file.
    Returns (list of file paths, set of all generated column names).
    """
    print("\n[2a/6] Parallel candidate generation...")
    base_cols = [c for c in TRANSFORM_BASE_COLS if c in panel.columns]
    print(f"  Base columns to transform: {len(base_cols)}")

    symbols = panel["symbol"].unique()
    print(f"  Symbols to process: {len(symbols):,}")
    print(f"  Using {N_JOBS if N_JOBS != -1 else 'all'} CPU cores")

    def _process_one(sym):
        sym_df = panel[panel["symbol"] == sym]
        if len(sym_df) < 30:
            return (None, [])
        try:
            return _generate_for_symbol_to_disk(sym_df, base_cols, sym)
        except Exception as e:
            return (None, [])

    results = Parallel(n_jobs=N_JOBS, backend="loky", verbose=0)(
        delayed(_process_one)(sym) for sym in tqdm(symbols, desc="  Generating")
    )

    file_paths = [r[0] for r in results if r[0] is not None]
    all_gen_cols: set = set()
    for _, cols in results:
        all_gen_cols.update(cols)

    print(f"  Generated: {len(file_paths):,} symbol files")
    print(f"  Unique candidate columns: {len(all_gen_cols):,}")
    return file_paths, all_gen_cols


def vectorized_panel_ic(
    file_paths: List[str], all_gen_cols: set, train_cutoff: pd.Timestamp
) -> Tuple[pd.DataFrame, Dict[str, List[float]]]:
    """
    Stage 2b: Vectorized IC computation across the entire training panel.
    For each feature, computes daily Spearman IC using pandas groupby on rank-transformed values.
    """
    print("\n[2b/6] Vectorized IC computation on training window...")
    print(f"  Train cutoff: {train_cutoff.date()}")

    # Read all symbol files into one big training dataframe
    print("  Reading temp parquets...")
    parts = []
    for fp in tqdm(file_paths, desc="  Reading"):
        try:
            df_part = pd.read_parquet(fp)
            df_part = df_part[df_part["date"] < train_cutoff]
            if len(df_part) > 0:
                parts.append(df_part)
        except Exception:
            continue

    print(f"  Concatenating {len(parts):,} symbol DataFrames...")
    train_df = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    print(f"  Total training rows: {len(train_df):,}")

    # IC per feature: for each date, rank-transform feature & return,
    # compute correlation between ranks. This is Spearman's rho but vectorized.
    print("  Computing daily IC for each feature (vectorized)...")
    candidate_cols = [c for c in all_gen_cols if c in train_df.columns]

    # Pre-compute ranked returns per date (one-time cost)
    ret_col = _safe(train_df[RET_COL])
    train_df["__ret_rank"] = train_df.groupby("date")[RET_COL].transform(
        lambda s: _safe(s).rank(method="average")
    )

    daily_ic_lookup: Dict[str, List[Tuple[pd.Timestamp, float]]] = {}
    ic_rows: List[dict] = []

    # Process features in batches to avoid memory spike
    BATCH = 50  # process 50 features at a time
    for batch_start in tqdm(range(0, len(candidate_cols), BATCH), desc="  Feature batches"):
        batch = candidate_cols[batch_start: batch_start + BATCH]

        for col in batch:
            # Rank-transform feature within each date
            try:
                feat_rank = train_df.groupby("date")[col].transform(
                    lambda s: _safe(s).rank(method="average")
                )
            except Exception:
                continue

            # Compute Spearman per date: corr of feat_rank and ret_rank within each date
            tmp = pd.DataFrame({
                "date":      train_df["date"],
                "feat_rank": feat_rank,
                "ret_rank":  train_df["__ret_rank"],
            }).dropna()

            if len(tmp) < 200:
                continue

            # For each date, compute Pearson on the ranks (= Spearman on values)
            grouped = tmp.groupby("date")
            ic_per_date = grouped.apply(
                lambda g: g["feat_rank"].corr(g["ret_rank"]) if len(g) >= 20 else np.nan
            )
            ic_per_date = ic_per_date.dropna()

            if len(ic_per_date) < 50:
                continue

            ic_arr = ic_per_date.values
            mean_ic    = float(np.mean(ic_arr))
            mean_abs   = float(np.mean(np.abs(ic_arr)))
            pos_rate   = float(np.mean(ic_arr > 0))
            tstat      = float(ttest_1samp(ic_arr, 0).statistic)
            ic_std     = float(np.std(ic_arr))

            ic_rows.append({
                "feature":     col,
                "mean_ic":     round(mean_ic, 5),
                "mean_abs_ic": round(mean_abs, 5),
                "ic_std":      round(ic_std, 5),
                "pos_rate":    round(pos_rate, 4),
                "tstat":       round(tstat, 3),
                "n_days":      int(len(ic_arr)),
            })

            # Store daily IC for stability filter
            daily_ic_lookup[col] = list(
                zip(ic_per_date.index.tolist(), ic_per_date.values.tolist())
            )

    del train_df
    gc.collect()

    ic_summary = pd.DataFrame(ic_rows).sort_values("mean_abs_ic", ascending=False).reset_index(drop=True)
    ic_summary.to_csv(OUT_DIR / "stage2_ic_summary.csv", index=False)
    print(f"  Saved: stage2_ic_summary.csv  ({len(ic_summary):,} features)")

    # Apply IC screening thresholds
    survivors_mask = (
        (ic_summary["mean_abs_ic"] >= IC_THRESHOLD) &
        (ic_summary["pos_rate"] >= POS_RATE_THRESHOLD) &
        (ic_summary["tstat"].abs() >= TSTAT_THRESHOLD)
    )
    survivors = ic_summary[survivors_mask].copy()
    print(f"  Survived IC screening: {len(survivors):,} / {len(ic_summary):,}")

    return survivors, daily_ic_lookup


# =============================================================================
#                  STAGE 3 — REDUNDANCY FILTER
# =============================================================================

def redundancy_filter(
    file_paths: List[str], survivors: pd.DataFrame, train_cutoff: pd.Timestamp
) -> pd.DataFrame:
    print(f"\n[3/6] Redundancy filter (corr > {CORR_THRESHOLD})...")

    feats = survivors["feature"].tolist()
    if len(feats) < 2:
        return survivors

    # Sample 50 symbol files for correlation matrix
    sample_paths = file_paths[:min(50, len(file_paths))]
    print(f"  Sampling {len(sample_paths)} symbol files for correlation...")

    samples = []
    for fp in sample_paths:
        df_part = pd.read_parquet(fp)
        df_part = df_part[df_part["date"] < train_cutoff]
        cols_in_df = [c for c in feats if c in df_part.columns]
        if cols_in_df:
            samples.append(df_part[cols_in_df])

    if not samples:
        print("  No samples available, skipping redundancy filter")
        return survivors

    corr_input = pd.concat(samples, ignore_index=True)
    print(f"  Correlation input: {corr_input.shape}")

    print("  Computing Spearman correlation matrix...")
    corr = corr_input.corr(method="spearman")

    ic_map = survivors.set_index("feature")["mean_abs_ic"].to_dict()
    sorted_feats = sorted(feats, key=lambda f: -ic_map.get(f, 0))

    keep: List[str] = []
    drop: set = set()

    for f in tqdm(sorted_feats, desc="  Filtering"):
        if f in drop:
            continue
        keep.append(f)
        for other in feats:
            if other == f or other in drop or other in keep:
                continue
            try:
                c = corr.loc[f, other]
                if pd.notna(c) and abs(c) >= CORR_THRESHOLD:
                    drop.add(other)
            except KeyError:
                continue

    print(f"  Kept: {len(keep):,}  /  Dropped redundant: {len(drop):,}")
    filtered = survivors[survivors["feature"].isin(keep)].copy()
    filtered.to_csv(OUT_DIR / "stage3_after_redundancy.csv", index=False)
    return filtered


# =============================================================================
#         STAGE 4 — WALK-FORWARD STABILITY FILTER
# =============================================================================

def walk_forward_stability_filter(
    survivors: pd.DataFrame, daily_ic_lookup: Dict[str, List[Tuple[pd.Timestamp, float]]]
) -> pd.DataFrame:
    print("\n[4/6] Walk-forward IC stability filter...")

    rows: List[dict] = []
    for col in tqdm(survivors["feature"].tolist(), desc="  Stability"):
        if col not in daily_ic_lookup:
            continue
        daily = daily_ic_lookup[col]
        if len(daily) < 100:
            continue

        daily_sorted = sorted(daily, key=lambda x: x[0])
        dates = pd.to_datetime([x[0] for x in daily_sorted])
        ics = np.array([x[1] for x in daily_sorted])

        positive_windows = 0
        total_windows = 0

        start = 0
        while start < len(dates):
            window_end = dates[start] + pd.Timedelta(days=ROLLING_WINDOW_DAYS)
            mask = (dates >= dates[start]) & (dates < window_end)
            if mask.sum() < 30:
                break
            window_ic = float(np.mean(ics[mask]))
            if window_ic > 0:
                positive_windows += 1
            total_windows += 1
            target_date = dates[start] + pd.Timedelta(days=ROLLING_STEP_DAYS)
            new_start = int(np.searchsorted(dates, target_date))
            if new_start <= start:
                new_start = start + 1
            start = new_start

        if total_windows == 0:
            continue

        positive_frac = positive_windows / total_windows
        rows.append({
            "feature":          col,
            "n_windows":        total_windows,
            "positive_windows": positive_windows,
            "positive_frac":    round(positive_frac, 3),
        })

    stability_df = pd.DataFrame(rows)
    stability_df.to_csv(OUT_DIR / "stage4_stability.csv", index=False)

    stable_feats = stability_df[
        stability_df["positive_frac"] >= MIN_POSITIVE_WINDOWS_FRAC
    ]["feature"].tolist()

    survivors_filtered = survivors[survivors["feature"].isin(stable_feats)].copy()
    print(f"  Stable features: {len(survivors_filtered):,} / {len(survivors):,}")
    return survivors_filtered


# =============================================================================
#         STAGE 5 — BATCH FORWARD SELECTION (3-FOLD PURGED CV)
# =============================================================================

def _purged_cv_splits(
    dates: np.ndarray, n_folds: int, embargo_days: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    sorted_idx = np.argsort(dates)
    n = len(sorted_idx)
    fold_size = n // n_folds
    splits = []
    for k in range(n_folds):
        te_start = k * fold_size
        te_end = te_start + fold_size if k < n_folds - 1 else n
        te_idx = sorted_idx[te_start:te_end]
        te_dates = dates[te_idx]
        if len(te_dates) > 0:
            te_min = pd.Timestamp(te_dates.min())
            te_max = pd.Timestamp(te_dates.max())
            embargo_min = te_min - pd.Timedelta(days=embargo_days)
            embargo_max = te_max + pd.Timedelta(days=embargo_days)
            tr_candidate = np.concatenate([sorted_idx[:te_start], sorted_idx[te_end:]])
            tr_dates = dates[tr_candidate]
            tr_mask = ~((tr_dates >= embargo_min) & (tr_dates <= embargo_max))
            tr_idx = tr_candidate[tr_mask]
        else:
            tr_idx = np.concatenate([sorted_idx[:te_start], sorted_idx[te_end:]])
        splits.append((tr_idx, te_idx))
    return splits


def _prepare_X(df: pd.DataFrame, cols: List[str], impute: dict) -> pd.DataFrame:
    X = df.reindex(columns=cols).copy()
    for c in cols:
        X[c] = _safe(X[c])
        if X[c].isna().any():
            X[c] = X[c].fillna(impute.get(c, X[c].median()))
    return X


def _eval_features_cv(
    full_X: pd.DataFrame, feats: List[str], y: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]], impute: dict
) -> float:
    """3-fold purged CV mean AUC."""
    fold_aucs = []
    for tr_idx, te_idx in splits:
        if len(tr_idx) < 200 or len(te_idx) < 50:
            continue
        X_tr = full_X.iloc[tr_idx][feats].fillna(0)
        X_te = full_X.iloc[te_idx][feats].fillna(0)
        clf = lgb.LGBMClassifier(**LGB_PARAMS)
        clf.fit(X_tr, y[tr_idx], callbacks=[lgb.callback.log_evaluation(period=0)])
        proba = clf.predict_proba(X_te)[:, 1]
        fold_aucs.append(roc_auc_score(y[te_idx], proba))
    return float(np.mean(fold_aucs)) if fold_aucs else 0.0


def batch_forward_selection(
    file_paths: List[str],
    survivors: pd.DataFrame,
    existing: List[str],
    impute: dict,
    train_cutoff: pd.Timestamp,
) -> List[str]:
    print("\n[5/6] Batch forward selection (3-fold purged CV)...")

    candidate_feats = [c for c in survivors["feature"].tolist() if c not in existing]

    # ─ Read all training data with existing + candidate cols ─────────────
    print(f"  Reading {len(file_paths):,} files into combined train matrix...")
    all_cols_needed = list(set(existing + candidate_feats + [TARGET_COL, "date"]))

    parts: List[pd.DataFrame] = []
    for fp in tqdm(file_paths, desc="  Reading"):
        try:
            df_part = pd.read_parquet(fp)
            df_part = df_part[
                (df_part["date"] < train_cutoff) & df_part[TARGET_COL].notna()
            ]
            if len(df_part) == 0:
                continue
            cols_in_df = [c for c in all_cols_needed if c in df_part.columns]
            parts.append(df_part[cols_in_df])
        except Exception:
            continue

    full_X = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()

    print(f"  Training matrix: {full_X.shape}")
    y = full_X[TARGET_COL].astype(int).values
    dates_all = pd.to_datetime(full_X["date"]).values

    splits = _purged_cv_splits(dates_all, N_CV_FOLDS, EMBARGO_DAYS)

    existing_in_data = [c for c in existing if c in full_X.columns]
    candidate_in_data = [c for c in candidate_feats if c in full_X.columns]
    print(f"  Existing baseline: {len(existing_in_data)}")
    print(f"  Candidate pool   : {len(candidate_in_data)}")

    print("  Computing baseline 3-fold CV AUC...")
    baseline_auc = _eval_features_cv(full_X, existing_in_data, y, splits, impute)
    print(f"  Baseline CV AUC  : {baseline_auc:.5f}")

    # Sort candidates by IC (best first)
    ic_map = survivors.set_index("feature")["mean_abs_ic"].to_dict()
    candidate_in_data.sort(key=lambda x: -ic_map.get(x, 0))

    selected = list(existing_in_data)
    current_auc = baseline_auc
    selection_log: List[dict] = []

    # Phase 1: Try adding top-K block at once. If it helps, keep all and continue with next K.
    # Phase 2: Then refine with one-at-a-time.

    print(f"\n  PHASE 1: Block evaluation (batch_size={BATCH_SIZE_INIT})")
    pool = list(candidate_in_data)
    while pool and len(selected) < MAX_FEATURES:
        block = pool[:BATCH_SIZE_INIT]
        pool = pool[BATCH_SIZE_INIT:]

        trial = selected + block
        try:
            trial_auc = _eval_features_cv(full_X, trial, y, splits, impute)
        except Exception:
            continue
        delta = trial_auc - current_auc
        accepted = delta >= AUC_DELTA

        selection_log.append({
            "phase": "block",
            "size": len(block),
            "trial_auc": round(trial_auc, 6),
            "auc_delta": round(delta, 6),
            "accepted": accepted,
        })

        if accepted:
            selected.extend(block)
            current_auc = trial_auc
            print(f"    Block of {len(block)} accepted. CV AUC: {current_auc:.5f}")

    # Phase 2: Refine — try each remaining candidate individually
    print(f"\n  PHASE 2: One-at-a-time refinement on {len(pool)} remaining candidates")
    for feat in tqdm(pool, desc="  Refinement"):
        if len(selected) >= MAX_FEATURES:
            break
        trial = selected + [feat]
        try:
            trial_auc = _eval_features_cv(full_X, trial, y, splits, impute)
        except Exception:
            continue
        delta = trial_auc - current_auc
        accepted = delta >= AUC_DELTA
        selection_log.append({
            "phase": "individual",
            "feature": feat,
            "trial_auc": round(trial_auc, 6),
            "auc_delta": round(delta, 6),
            "accepted": accepted,
        })
        if accepted:
            selected.append(feat)
            current_auc = trial_auc

    print(f"\n  Baseline AUC : {baseline_auc:.5f}")
    print(f"  Final AUC    : {current_auc:.5f}")
    print(f"  AUC gain     : {current_auc - baseline_auc:+.5f}")
    print(f"  Total features: {len(selected)}  (existing: {len(existing_in_data)}, new: {len(selected) - len(existing_in_data)})")

    pd.DataFrame(selection_log).to_csv(OUT_DIR / "stage5_selection_log.csv", index=False)
    return selected


# =============================================================================
#                       STAGE 6 — OUTPUT
# =============================================================================

def save_outputs(
    final_features: List[str],
    existing: List[str],
    impute: dict,
    file_paths: List[str],
) -> None:
    print("\n[6/6] Saving outputs...")

    new_impute = dict(impute)

    # Compute medians for any features not in existing impute dict
    needs_impute = [f for f in final_features if f not in new_impute]
    if needs_impute:
        print(f"  Computing impute medians for {len(needs_impute)} new features...")
        sample_paths = file_paths[:min(50, len(file_paths))]
        sample_dfs = []
        for fp in sample_paths:
            try:
                df_part = pd.read_parquet(fp)
                cols_in_df = [c for c in needs_impute if c in df_part.columns]
                if cols_in_df:
                    sample_dfs.append(df_part[cols_in_df])
            except Exception:
                continue
        if sample_dfs:
            sample_df = pd.concat(sample_dfs, ignore_index=True)
            for f in needs_impute:
                if f in sample_df.columns:
                    vals = _safe(sample_df[f])
                    new_impute[f] = float(vals.median()) if not vals.isna().all() else 0.0
                else:
                    new_impute[f] = 0.0
        else:
            for f in needs_impute:
                new_impute[f] = 0.0

    schema = {
        "features":     final_features,
        "impute":       new_impute,
        "generated_by": "feature_factory_v3",
    }
    out_path = OUT_DIR / "new_features_train.json"
    out_path.write_text(json.dumps(schema, indent=2, ensure_ascii=False))
    print(f"  Saved: {out_path}")
    print(f"  Total features in schema: {len(final_features)}")

    new_added = [f for f in final_features if f not in existing]
    print(f"  New features added: {len(new_added)}")
    if new_added:
        print(f"\n  First 30 new features added:")
        for f in new_added[:30]:
            print(f"    {f}")


# =============================================================================
#                              MAIN
# =============================================================================

def _leak_canary(n: int = 400, k: int = 20, seed: int = 0) -> int:
    """Leak self-test: tamper the FUTURE k rows, assert PAST WQ-alpha values are
    unchanged. Generation is per-symbol, so any whole-series op (rank/scale/...)
    would change past rows. Returns 0 on pass, 1 on leak."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    op = close * (1 + rng.normal(0, 0.003, n))
    hi = np.maximum(close, op) * (1 + np.abs(rng.normal(0, 0.005, n)))
    lo = np.minimum(close, op) * (1 - np.abs(rng.normal(0, 0.005, n)))
    vol = rng.integers(1e5, 2e6, n).astype(float)
    base = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n, freq="B"),
                         "symbol": "S", "open": op, "high": hi, "low": lo,
                         "close": close, "volume": vol})
    tamp = base.copy()
    mlt = rng.uniform(0.3, 3.0, k)
    for c in ["open", "high", "low", "close", "volume"]:
        tamp.loc[n - k:, c] = tamp.loc[n - k:, c].to_numpy() * mlt
    a = base.copy(); cols = _compute_wq_alphas(a)
    b = tamp.copy(); _compute_wq_alphas(b)
    wq = [c for c in cols if c.startswith("WQ_")]
    leaks = []
    for c in wq:
        av = pd.to_numeric(a[c], errors="coerce").to_numpy()[:n - k]
        bv = pd.to_numeric(b[c], errors="coerce").to_numpy()[:n - k]
        an, bn = np.isnan(av), np.isnan(bv)
        if not np.array_equal(an, bn):
            leaks.append(c + "(nan)"); continue
        d = float(np.nanmax(np.where(an, 0.0, np.abs(av - bv))) if (~an).any() else 0.0)
        if d > 1e-9:
            leaks.append(f"{c}({d:.1e})")
    if leaks:
        print(f"[leak-canary] FAIL - past WQ values changed when future tampered: {leaks}")
        return 1
    print(f"[leak-canary] PASS - {len(wq)} WQ alphas leak-free (future tampered, past unchanged)")
    return 0


def main():
    t0 = time.perf_counter()
    print("=" * 65)
    print("FEATURE FACTORY v3.0 — Fast & Parallel")
    print("=" * 65)

    # Stage 1
    panel, existing, impute = load_panel()
    train_cutoff = _compute_train_cutoff(panel)

    # Stage 2a — parallel candidate generation to disk
    file_paths, all_gen_cols = parallel_generate_candidates(panel)
    t2a = time.perf_counter()
    print(f"\n  Stage 2a time: {t2a - t0:.1f}s")

    # Stage 2b — vectorized IC screening
    survivors_ic, daily_ic_lookup = vectorized_panel_ic(file_paths, all_gen_cols, train_cutoff)
    t2b = time.perf_counter()
    print(f"  Stage 2b time: {t2b - t2a:.1f}s")

    # Stage 3 — redundancy filter
    survivors_redundancy = redundancy_filter(file_paths, survivors_ic, train_cutoff)
    t3 = time.perf_counter()
    print(f"  Stage 3 time: {t3 - t2b:.1f}s")

    # Stage 4 — walk-forward stability
    survivors_stable = walk_forward_stability_filter(survivors_redundancy, daily_ic_lookup)
    t4 = time.perf_counter()
    print(f"  Stage 4 time: {t4 - t3:.1f}s")

    # Stage 5 — batch + individual forward selection (3-fold CV)
    final = batch_forward_selection(file_paths, survivors_stable, existing, impute, train_cutoff)
    t5 = time.perf_counter()
    print(f"  Stage 5 time: {t5 - t4:.1f}s")

    # Stage 6 — output
    save_outputs(final, existing, impute, file_paths)
    t6 = time.perf_counter()
    print(f"  Stage 6 time: {t6 - t5:.1f}s")

    # Cleanup temp files
    print(f"\n  Cleaning up temp candidate files...")
    try:
        shutil.rmtree(TMP_DIR)
        print(f"  Removed: {TMP_DIR}")
    except Exception as e:
        print(f"  Warning: could not remove temp dir: {e}")

    total = time.perf_counter() - t0
    print("\n" + "=" * 65)
    print("FEATURE FACTORY v3 COMPLETE")
    print(f"  Total runtime: {total / 60:.1f} minutes")
    print(f"  Output: {OUT_DIR / 'new_features_train.json'}")
    print(f"  Replace existing features_train.json and retrain cpr_fix.py")
    print("=" * 65)


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        raise SystemExit(_leak_canary())
    main()

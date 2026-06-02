# cpr_fix_patched.py
# v3.5 (pruner wiring) — builds on v3.4. Adds:
#   * FIX: restored the enriched-panel rewrite to disk (was lost in editing) so
#     panel_cache.parquet contains M_* / stock_regime / X_regime_* / regime_*.
#     Without it the pruner crashed: ArrowInvalid No match for FieldRef M_vix_change.
#   * FIX: that enriched rewrite now downcasts float64->float32 and writes in
#     bounded row-group batches, so it does NOT OOM on a 195-col x 3.5M-row panel
#     (the single from_pandas() copy was ~5 GiB).
#   * FIX: EnsembleCalibrator._project uses getattr(self,'feature_list',None)
#     so models pickled by a PRE-v3.5 run (no feature_list attr) still load and
#     score without AttributeError. Retraining is still recommended so the
#     saved model carries feature_list properly.
#   * USE_PRUNED_FEATURES toggle. When on and regime_features.json (from
#     feature_imp_v5.py) is found in out_dir or its feature_diagnostics/ subdir:
#       - the POOLED model trains on keep_list_global
#       - each regime SPECIALIST trains on its own per_regime[...] pruned list
#   * EnsembleCalibrator now carries its trained feature_list and self-projects
#     incoming X to it, so specialists on different pruned lists coexist behind
#     one RegimeRouter (which exposes required_features()). Backward compatible:
#     feature_list=None reverts to prior behavior.
#   * feature_imp_v5.py also emits features_train_pruned.json (non-destructive)
#     for quick pooled runs.
# v3.4 (accuracy + reliability + speed) — builds on v3.2's 11 audit fixes.
# SPEED (cold full build):
#   * _rolling_slope vectorized: ~300x faster (was a per-row Python loop; ~0.7s
#     -> ~2ms per symbol for the 20+50 windows). Output identical to ~1e-12 on
#     clean close prices. Typically the single largest cold-build win.
#   * _compute_strict_label skipped entirely when USE_STRICT_LABEL=False
#     (avoids several full-panel groupby passes every build).
#   * Per-phase [Timing] logs (read+sort, cross-sectional, macro, regime, save)
#     so you can SEE where the nightly build spends its minutes.
#   FIX 1  Forward MFE/MAE windows (were rolling backward over shifted data).
#   FIX 2  Embargo gap at train|cal and cal|test boundaries (was contiguous).
#   FIX 3  Day-aligned splits so no calendar day straddles a split boundary.
#   FIX 4  Deterministic per-regime seeds (was salted hash() -> non-reproducible).
#   NEW    Continuous regime features X_regime_* (the "60% bull" gradient) so a
#          single pooled model can learn its own thresholds.
#   NEW    MODEL_ARCHITECTURE toggle: "single" (pooled) | "regime" (4 specialists)
#          | "both" (+ per-regime Brier/IC comparison table). PRIMARY_ARCHITECTURE
#          chooses which one is exported/scored.
#   NEW    Upstream structural liquidity filter for TRAINING rows (cleaner labels).
#   NEW    Regime label lag (known before entry), watchlist staleness filter,
#          optional ensemble-agreement filter.
# All v3.2 audit fixes (incl. disabled atexit fallback, leak guard, macro tz fix)
# are preserved. Backward compatible result dict (adds 'architecture',
# 'architecture_comparison').
#
# Config toggles live in the "ARCHITECTURE / ACCURACY TOGGLES" block near the top.
# Drop-in replacement for your existing cpr_fix.py.
# If you prefer, rename this file to cpr_fix.py and use as-is.

import os, glob, json, time, sys, re, math, hashlib, concurrent.futures
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import numpy as np
import pandas as pd
import atexit
import joblib

# Try pyarrow for parquet I/O
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _PA_OK = True
except Exception:
    _PA_OK = False

# Lightweight logger shim. This file uses print-based output (no logging block),
# so define a module-level log() that several v3.3/v3.4 helpers call. Accepts an
# optional level kwarg ("info"/"warning"/...) for call-site compatibility; all
# levels print to stdout with a small prefix.
def log(msg, level="info"):
    lvl = str(level).upper()
    if lvl in ("INFO", "DEBUG"):
        print(msg)
    else:
        print(f"[{lvl}] {msg}")

# ===================== DEFAULTS / PATHS =====================
DATA_DIR_DEFAULT = r"C:\\Users\\karanvsi\\Desktop\\Pycharm\\Cache\\cache_daily_new"
MACRO_CACHE_PATH = r"C:\\Users\\karanvsi\\Desktop\\Pycharm\\Cache\\macro_cache.parquet"
PANEL_OUT = None
WATCHLIST_OUT = None
STATUS_PATH = None
META_PATH = None
LOG_DIR = None
LOAD_ERRORS_LOG = None
QUARANTINE_LIST = None
FEATURES_SCHEMA_PATH = None  # <out-dir>/features_train.json
OOS_REPORT_PATH = None       # <out-dir>/oos_report.json
CALIB_TABLE_PATH = None      # <out-dir>/calibration_5d_deciles.json
RESEARCH_CSV_PATH = None     # <out-dir>/research_report.csv
MODEL_CALIB_PATH = None      # <out-dir>/model_lgbm_calibrated.joblib (optional)
ISO_EV_MAPPER_PATH = None    # <out-dir>/isotonic_ev_mapper.joblib (optional)

# ===================== MODEL / CV =====================
GLOBAL_SEED = 42
FOLDS = 8
EMBARGO_DAYS = 5  # increased to match 5d horizon
# LightGBM baseline
N_EST_1D = 2400
N_EST_5D = 2200
EARLY_STOPPING_ROUNDS = 800
LEARNING_RATE = 0.005
MAX_DEPTH = 6
# Safe early stopping threshold
MIN_VAL_EARLYSTOP = 500
# Gate controls
MIN_GATE_SAMPLES = 500  # mandatory gate requires at least this many labeled rows
CLS_MARGIN_1D = 0.10
# Filters for watchlist (point-in-time tradability gates, applied at scoring time)
MIN_CLOSE = 2.0
MIN_AVG20_VOL = 200_000
CHUNK_SIZE = 1200

# =====================================================================
# ARCHITECTURE / ACCURACY TOGGLES  (set here, no CLI needed)
# =====================================================================
# "single"  -> ONE pooled model on the whole universe, regime expressed as
#              CONTINUOUS features (X_regime_*). Recommended starting point:
#              4x the data per model, learns its own thresholds, no boundary
#              discontinuities. Watchlist still reports the regime per row.
# "regime"  -> original 4 regime-specific ensembles + fallback (specialist).
# "both"    -> train BOTH and print a per-regime Brier/IC comparison table,
#              then USE the architecture named in PRIMARY_ARCHITECTURE for the
#              exported models + watchlist. This is how you decide empirically.
MODEL_ARCHITECTURE = "both"          # "single" | "regime" | "both"
PRIMARY_ARCHITECTURE = "single"      # which one to EXPORT/score when "both"

# =====================================================================
# TARGET_MODE  — what the 5D model learns.
# =====================================================================
# "binary" -> ORIGINAL behavior. Label = top-20% vs bottom-20% of the per-day
#             vol-adjusted forward 5d return. The middle 60% of rows are DROPPED
#             (unlabeled), so the model only ever sees the extremes at train time
#             yet is asked to score every row at inference. Proven, default.
# "rank"   -> NEW. Label = the CONTINUOUS per-day cross-sectional rank percentile
#             (rank_5d_pct in [0,1]) of the vol-adjusted forward 5d return, trained
#             with a LightGBM REGRESSION objective on ALL labeled rows (~3x the
#             data). This directly optimizes the cross-sectional ordering, which is
#             exactly what Spearman IC measures. The regressor's raw score is mapped
#             back to a P(top-20%) probability via an isotonic fit on the embargoed
#             CAL slice, so EVERY downstream consumer (EnsembleCalibrator,
#             RegimeRouter, calibration deciles, isotonic EV mapper, watchlist,
#             EVShim export) keeps working UNCHANGED — a rank member quacks exactly
#             like the old calibrated classifier (predict_proba -> [P(no), P(yes)]).
#
# Recommended: run once with "binary" and once with "rank" and compare the OOS IC
# in oos_report.json / architecture_comparison.json on the SAME embargoed rows.
TARGET_MODE = "rank"                 # "binary" | "rank"
RANK_OBJECTIVE = "regression"        # LightGBM objective used when TARGET_MODE="rank"

# Use the strict follow-through label (close>=entry AND max drawdown<3% over 5d)?
USE_STRICT_LABEL = False

# Consume the feature pruner's output (regime_features.json from feature_imp_v5.py).
# When True and that file exists in out_dir, the pooled model trains on the global
# keep-list and each regime specialist trains on its own pruned per-regime list.
# This is what preserves the trend-regime feature edge. False = use full feats.
USE_PRUNED_FEATURES = True

# Lag the HARD regime label by N bars so regime is known strictly before entry
# (matches live trading; continuous X_regime_* features are always point-in-time).
REGIME_LAG = 1

# STRUCTURAL universe filter applied UPSTREAM of labeling/training (not just at
# watchlist). Rows failing this are a different data-generating process you will
# never trade, and they pollute the per-day cross-sectional rank label. Set to
# None to disable upstream filtering and keep prior behavior.
TRAIN_MIN_CLOSE = None
TRAIN_MIN_AVG20_VOL = None

# Watchlist: drop names whose most recent bar is older than this many days
# (avoid scoring delisted/halted symbols as if they were "today").
WATCHLIST_MAX_STALENESS_DAYS = 7

# Watchlist: optionally keep only names where the ensemble AGREES (low member
# disagreement). 0.0 disables. e.g. 0.06 keeps rows with prob_5d_std <= 0.06.
WATCHLIST_MAX_PROB_STD = 0.0
# =====================================================================

np.random.seed(GLOBAL_SEED)

# Remember last out_dir for atexit fallback
from typing import Optional as _Optional
_LAST_OUT_DIR: _Optional[str] = None

# ===================== TZ helpers =====================
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype

def ensure_kolkata_tz(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce", utc=True)
    try:
        return ts.dt.tz_convert("Asia/Kolkata")
    except Exception:
        return pd.to_datetime(series, errors="coerce").dt.tz_localize("Asia/Kolkata")

def _ensure_ts_ist(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = ensure_kolkata_tz(df["timestamp"])
    return df

# ===================== Status / Progress =====================

def write_status(phase: str, note: str = ""):
    rec = {"ts": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(timespec="seconds"),
           "phase": phase, "note": note}
    try:
        with open(STATUS_PATH, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)
    except Exception:
        pass

class ProgressETA:
    def __init__(self, total:int, label:str=""):
        self.total = max(1, int(total)); self.label = label
        self.start = time.perf_counter(); self.done = 0; self._last = ""
    def tick(self, note:str=""):
        self.done += 1
        elapsed = max(1e-6, time.perf_counter() - self.start)
        rate = self.done / elapsed; remain = max(0, self.total - self.done)
        eta_s = int(remain / rate) if rate > 0 else 0
        m, s = divmod(eta_s, 60); h, m = divmod(m, 60)
        eta = f"{h:02d}:{m:02d}:{s:02d}" if h>0 else f"{m:02d}:{s:02d}"
        pct = 100 * self.done / self.total
        msg = f"[{self.label}] {self.done}/{self.total} ({pct:5.1f}%) ETA {eta}"
        if note: msg += f" {note}"
        if msg != self._last:
            self._last = msg
            print(msg)

# ===================== Paths setup =====================

def setup_paths(out_dir: str):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    global _LAST_OUT_DIR; _LAST_OUT_DIR = str(out)
    global PANEL_OUT, WATCHLIST_OUT, STATUS_PATH, META_PATH
    global LOG_DIR, LOAD_ERRORS_LOG, QUARANTINE_LIST, FEATURES_SCHEMA_PATH
    global OOS_REPORT_PATH, CALIB_TABLE_PATH, RESEARCH_CSV_PATH
    global MODEL_CALIB_PATH, ISO_EV_MAPPER_PATH

    PANEL_OUT = str(out / "panel_cache.parquet")
    WATCHLIST_OUT = str(out / "watchlist_5d_signal.csv")
    STATUS_PATH = str(out / "status.json")
    META_PATH = str(out / "model_meta.json")
    FEATURES_SCHEMA_PATH = str(out / "features_train.json")
    OOS_REPORT_PATH = str(out / "oos_report.json")
    CALIB_TABLE_PATH = str(out / "calibration_5d_deciles.json")
    RESEARCH_CSV_PATH = str(out / "research_report.csv")
    MODEL_CALIB_PATH = str(out / "model_lgbm_calibrated.joblib")
    ISO_EV_MAPPER_PATH = str(out / "isotonic_ev_mapper.joblib")

    LOG_DIR = out / "logs"; LOG_DIR.mkdir(exist_ok=True)
    LOAD_ERRORS_LOG = LOG_DIR / "load_errors.csv"
    QUARANTINE_LIST = out / "quarantine_files.txt"

# ===================== IO helpers =====================

def _strict_file_list(data_dir: str,
                      symbols_like: Optional[str],
                      limit_files: Optional[int],
                      accept_any_daily: bool=False) -> List[Path]:
    paths: List[str] = []
    paths += glob.glob(os.path.join(data_dir, "*_daily.parquet"))
    paths += glob.glob(os.path.join(data_dir, "*_daily.csv"))
    if str(accept_any_daily).lower() in ("true","1","yes","y","t"):
        paths += glob.glob(os.path.join(data_dir, "*.parquet"))
        paths += glob.glob(os.path.join(data_dir, "*.csv"))
    paths = sorted(set(paths))
    if symbols_like:
        pat = re.compile(symbols_like)
        filtered = []
        for p in paths:
            sym = _derive_symbol_name(Path(p))
            if pat.search(sym): filtered.append(p)
        paths = filtered or paths
    if limit_files and limit_files > 0:
        paths = paths[:limit_files]
    return [Path(p) for p in paths]

def _log_load_error(sym: str, filename: str, error: str):
    rec = {"symbol": sym, "file": filename, "error": error,
           "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        df = pd.DataFrame([rec])
        mode = "a" if Path(LOAD_ERRORS_LOG).exists() else "w"
        df.to_csv(LOAD_ERRORS_LOG, mode=mode,
                  header=not Path(LOAD_ERRORS_LOG).exists(), index=False)
    except Exception:
        pass
    with open(QUARANTINE_LIST, "a", encoding="utf-8") as f:
        f.write(f"{filename}\n")

def _ensure_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [str(c).strip() for c in df.columns]
    seen: Dict[str,int] = {}
    new_cols: List[str] = []
    for c in cols:
        if c not in seen:
            seen[c] = 1; new_cols.append(c)
        else:
            k = seen[c]; seen[c] = k + 1
            new_cols.append(f"{c}__dup{k}")
    df = df.copy(); df.columns = new_cols
    return df

# ===== CLEAN, DETERMINISTIC SYMBOL STRIPPING =====

def _derive_symbol_name(p: Path) -> str:
    base = p.name
    for suff in ("_daily.parquet", "_daily.csv", ".parquet", ".csv"):
        if base.endswith(suff):
            base = base[:-len(suff)]
            break
    return base.strip()

def _clean_symbol_label(label: str) -> str:
    s = str(label).strip()
    for suff in ("_daily.parquet", "_daily.csv"):
        if s.endswith(suff):
            s = s[:-len(suff)]
    return s

def load_one(path: Path) -> pd.DataFrame:
    try:
        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            # AUDIT-FIX 3: drop infer_datetime_format=True (removed in pandas 2.x).
            # Modern pandas auto-infers; leaving the kwarg in raises TypeError.
            try:
                df = pd.read_csv(path, parse_dates=["timestamp"])
            except (ValueError, KeyError):
                df = pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Load failed: {e}")

    if "timestamp" not in df.columns:
        if "date" in df.columns:
            df = df.rename(columns={"date": "timestamp"})
        else:
            raise RuntimeError("'timestamp' column missing")
    if not (is_datetime64_any_dtype(df["timestamp"]) or is_datetime64tz_dtype(df["timestamp"])):
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = _ensure_ts_ist(df)
    df = (df.dropna(subset=["timestamp"]).sort_values("timestamp")
            .drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True))
    df = _ensure_unique_columns(df)
    return df

# ===================== Targets & features =====================

def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for h in (1,3,5):
        df[f"ret_{h}d_close_pct"] = (df["close"].shift(-h) / df["close"] - 1) * 100
        df[f"ret_{h}d_oc_pct"] = (df["close"].shift(-h) / df["open"].shift(-1) - 1) * 100
        # FIX 1 (forward MFE/MAE): the previous code did
        #   high.shift(-1).rolling(h).max()
        # which rolls BACKWARD over an already forward-shifted series, so for row t
        # the "h-day forward max" only ever saw high[t+1] instead of max(high[t+1..t+h]).
        # Correct forward window for row t = max(high[t+1 .. t+h]) / min(low[t+1 .. t+h]).
        # Build the h individually-shifted columns and take the row-wise extremum
        # (same idiom AUDIT-FIX 6 uses for low_fwd_min in the strict-label code).
        hi_shifts = pd.concat(
            [df["high"].shift(-k) for k in range(1, h + 1)], axis=1
        )
        lo_shifts = pd.concat(
            [df["low"].shift(-k) for k in range(1, h + 1)], axis=1
        )
        hi = hi_shifts.max(axis=1)
        lo = lo_shifts.min(axis=1)
        df[f"mfe_{h}d_pct"] = (hi / df["close"] - 1) * 100
        df[f"mae_{h}d_pct"] = (lo / df["close"] - 1) * 100
    return df

def add_lags(df: pd.DataFrame, cols: List[str], lags: Tuple[int,int]=(1,2)):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            for L in lags:
                df[f"{c}_lag{L}"] = df[c].shift(L)
    return df

def _unify_categorical(df: pd.DataFrame, base_name: str) -> pd.Series:
    cols = [c for c in df.columns if c == base_name or c.startswith(base_name + "__dup")]
    if not cols:
        return pd.Series(index=df.index, dtype="object")
    s = pd.Series(index=df.index, dtype="object")
    for c in cols:
        sc = df[c].astype("string")
        s = sc.where(sc.notna(), s)
    return s

EXCLUDE_D_FEATURES = set()
LAG_FEATURES = [
    "D_rsi14_lag1","D_rsi14_lag2",
    "D_adx14_lag1","D_adx14_lag2",
    "D_ema20_angle_deg_lag1","D_ema20_angle_deg_lag2",
    "D_obv_slope_lag1","D_obv_slope_lag2",
]
CPR_YDAY = [f"CPR_Yday_{x}" for x in ("Above","Below","Inside","Overlap")]
CPR_TMR = [f"CPR_Tmr_{x}" for x in ("Above","Below","Inside","Overlap")]
STRUCT_ONEHOT = ["Struct_uptrend","Struct_downtrend","Struct_range"]
DAYTYPE_ONEHOT = ["DayType_bullish","DayType_bearish","DayType_inside"]

def _rolling_slope(y: pd.Series, window: int) -> pd.Series:
    """Vectorized trailing rolling OLS slope (expanding at the head, min_periods=1).

    SPEED: ~300x faster than the previous elementwise list-comprehension. On a
    typical 2,500-bar symbol the old version took ~0.7s for the 20+50 windows;
    this takes ~2ms. Across thousands of symbols this is one of the largest
    cold-build wins.

    EQUIVALENCE: produces the same values (to ~1e-13 fp tolerance) and the same
    NaN positions as the original whenever the input has no interior NaNs (true
    for OHLCV close prices). If a NaN falls inside a window, that window's slope
    becomes NaN (same as the original's behavior).

    Closed form: slope = [Sxy - Sx*Sy/cnt] / [Sxx - Sx*Sx/cnt] over the window,
    where x is the integer position index. All Sx/Sxx terms are NaN-free; Sy/Sxy
    carry y, so a windowed NaN propagates to NaN (matching the original).
    """
    y = pd.to_numeric(y, errors="coerce").astype(float)
    n = len(y)
    if n == 0:
        return pd.Series([], dtype=float, index=y.index)
    idx = np.arange(n, dtype=float)
    vals = y.to_numpy()

    def _rsum(a):  # trailing rolling sum, expanding head
        return pd.Series(a).rolling(window, min_periods=1).sum().to_numpy()

    cnt = _rsum(np.ones(n))
    sx  = _rsum(idx)
    sxx = _rsum(idx * idx)
    sy  = _rsum(vals)         # NaN-propagating (intentional)
    sxy = _rsum(idx * vals)   # NaN-propagating (intentional)

    with np.errstate(invalid="ignore", divide="ignore"):
        denom = sxx - (sx * sx) / cnt
        numer = sxy - (sx * sy) / cnt
        slope = numer / denom
    slope = np.where((denom > 0) & np.isfinite(denom), slope, np.nan)
    return pd.Series(slope, index=y.index)

def discover_daily_features(df, exclude=None):
    """
    Discover ALL feature columns in the panel regardless of prefix.
    Includes: D_*, W_*, WQ_*, M_*, X_*, Comb_*, regime_*, CPR_*, Struct_*, DayType_*, DOW_*
    Excludes: timestamp/date/symbol/raw OHLCV/return targets/calculation helpers/regime label.
    """
    exclude = set(exclude or [])
    # AUDIT-FIX 5: extend NEVER_FEATURE so any forward-looking column or
    # label-construction helper that happens to start with a feature prefix
    # cannot slip in. mfe_*/mae_* are forward-looking, ret_*_oc/cc_pct are
    # forward-looking, and vol_20/atr_pct/ret_*_adj are label-construction
    # helpers that would otherwise trigger no prefix filter but are surfaced
    # by build_5d_rank_quant_labels.
    NEVER_FEATURE = {
        "timestamp", "date", "year", "symbol", "instrument_token",
        "open", "high", "low", "close", "volume",
        # Forward returns (labels)
        "ret_1d_close_pct", "ret_3d_close_pct", "ret_5d_close_pct",
        "ret_1d_oc_pct", "ret_3d_oc_pct", "ret_5d_oc_pct",
        # Label helpers / target-derived
        "ret_5d_adj", "ret_3d_adj", "rank_5d_pct",
        "top20_vs_bot20_5d", "top20_strict_5d",
        # Universe / regime helpers
        "avg20_vol", "stock_regime",
        # Label-construction columns added by build_5d_rank_quant_labels
        "vol_20", "atr_pct",
    }
    # Forward-looking patterns: mfe_*/mae_* are MFE/MAE over future windows.
    NEVER_FEATURE_PATTERNS = ("mfe_", "mae_")
    cols = []
    for c in df.columns:
        if not isinstance(c, str):
            continue
        if c in NEVER_FEATURE or c in exclude:
            continue
        if any(c.startswith(p) for p in NEVER_FEATURE_PATTERNS):
            continue
        if "__dup" in c:
            continue
        if (c.startswith("D_") or c.startswith("W_") or c.startswith("WQ_") or
            c.startswith("M_") or c.startswith("X_") or c.startswith("Comb_") or
            c.startswith("CPR_") or c.startswith("Struct_") or c.startswith("DayType_") or
            c.startswith("DOW_") or c.startswith("regime_") or
            c in ("long_score", "short_score")):
            cols.append(c)
    return sorted(set(cols))

def featureize(df: pd.DataFrame):
    base_auto = discover_daily_features(df, exclude=EXCLUDE_D_FEATURES)
    df = add_lags(df, ["D_rsi14","D_adx14","D_ema20_angle_deg","D_obv_slope"], lags=(1,2))

    # Uniform CPR and day-type categoricals
    yday_unified = _unify_categorical(df, "D_cpr_vs_yday")
    tmr_unified = _unify_categorical(df, "D_tmr_cpr_vs_today")
    if yday_unified.notna().any():
        df = df.drop(columns=[c for c in df.columns if c == "D_cpr_vs_yday" or c.startswith("D_cpr_vs_yday__dup")], errors="ignore")
        df["D_cpr_vs_yday_unified"] = yday_unified
        df = pd.get_dummies(df, columns=["D_cpr_vs_yday_unified"], prefix="CPR_Yday")
    if tmr_unified.notna().any():
        df = df.drop(columns=[c for c in df.columns if c == "D_tmr_cpr_vs_today" or c.startswith("D_tmr_cpr_vs_today__dup")], errors="ignore")
        df["D_tmr_cpr_vs_today_unified"] = tmr_unified
        df = pd.get_dummies(df, columns=["D_tmr_cpr_vs_today_unified"], prefix="CPR_Tmr")
    for col in CPR_YDAY:
        if col not in df.columns: df[col] = 0
    for col in CPR_TMR:
        if col not in df.columns: df[col] = 0

    if "D_structure_trend" in df.columns:
        df["D_structure_trend"] = df["D_structure_trend"].astype("string")
        df = pd.get_dummies(df, columns=["D_structure_trend"], prefix="Struct")
    for col in STRUCT_ONEHOT:
        if col not in df.columns: df[col] = 0

    if "D_day_type" in df.columns:
        df["D_day_type"] = df["D_day_type"].astype("string")
        df = pd.get_dummies(df, columns=["D_day_type"], prefix="DayType")
    for col in DAYTYPE_ONEHOT:
        if col not in df.columns: df[col] = 0

    # Engineered features
    rsi14 = pd.to_numeric(df.get("D_rsi14", np.nan), errors="coerce")
    rsi7 = pd.to_numeric(df.get("D_rsi7", np.nan), errors="coerce")
    obvs = pd.to_numeric(df.get("D_obv_slope", np.nan), errors="coerce")
    adx14 = pd.to_numeric(df.get("D_adx14", np.nan), errors="coerce")
    atr14 = pd.to_numeric(df.get("D_atr14", np.nan), errors="coerce")
    close = pd.to_numeric(df.get("close", np.nan), errors="coerce")
    macd_hist = pd.to_numeric(df.get("D_macd_hist", np.nan), errors="coerce")
    cprw = pd.to_numeric(df.get("D_cpr_width_pct", np.nan), errors="coerce").abs()
    df["D_rsi14_obv_x"] = rsi14 * obvs
    if "D_rsi7" in df.columns:
        df["D_rsi7_obv_x"] = rsi7 * obvs
    df["D_atr14_to_close_pct"] = (atr14 / close).replace([np.inf, -np.inf], np.nan) * 100.0
    df["X_rsi14_adx14"] = rsi14 * adx14
    df["X_cprw_atr_pct"] = cprw * df["D_atr14_to_close_pct"]
    trend_code = pd.to_numeric(df.get("D_structure_trend_code", np.nan), errors="coerce")
    df["X_trend_atr_pct"] = trend_code * df["D_atr14_to_close_pct"]
    df["X_rsi_cross_strength"] = (rsi7 - rsi14) * adx14
    df["X_macd_nonlin"] = macd_hist * rsi14 / 50.0
    df["X_adx_sqr"] = (adx14 ** 2) / 100.0

    if "ret_1d_close_pct" not in df.columns or "ret_5d_close_pct" not in df.columns:
        df = add_targets(df)

    df["D_close_roll_slope_20"] = _rolling_slope(df["close"], window=20)
    df["D_close_roll_slope_50"] = _rolling_slope(df["close"], window=50)
    daily_ret = pd.to_numeric(df["close"], errors="coerce").pct_change() * 100.0
    df["D_ret_5d_pastret"] = daily_ret.rolling(5, min_periods=5).sum()
    df["D_ret_5d_roll_std"] = df["D_ret_5d_pastret"].rolling(50, min_periods=10).std()

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    feats = (base_auto + LAG_FEATURES + CPR_YDAY + CPR_TMR + STRUCT_ONEHOT + DAYTYPE_ONEHOT + [
        "D_rsi14_obv_x","D_rsi7_obv_x","D_atr14_to_close_pct",
        "D_ret_5d_roll_std","D_close_roll_slope_20","D_close_roll_slope_50",
        "X_rsi14_adx14","X_cprw_atr_pct","X_trend_atr_pct",
        "X_rsi_cross_strength","X_macd_nonlin","X_adx_sqr",
    ])
    for c in feats:
        if c not in df.columns:
            df[c] = 0 if (c.startswith("CPR_Yday_") or c.startswith("CPR_Tmr_")
                          or c.startswith("Struct_") or c.startswith("DayType_")) else np.nan
    return df, feats

# ===================== Panel writer =====================

MASTER_KEEP_STATIC = [
    "timestamp","symbol","open","high","low","close","volume",
    "ret_1d_close_pct","ret_3d_close_pct","ret_5d_close_pct",
    "ret_1d_oc_pct","ret_3d_oc_pct","ret_5d_oc_pct",
    "long_score","short_score","D_atr14","D_cpr_width_pct",
]

class PanelParquetWriter:
    def __init__(self, out_path: str):
        if not _PA_OK:
            raise SystemExit("pyarrow is required to write panel_cache.parquet. Please run: pip install pyarrow")
        self.out_path = out_path; self._writer = None; self._schema = None
        # Frozen, ordered column list established from the FIRST chunk. Every later
        # chunk is reindexed to this EXACT order so pyarrow's ParquetWriter (which
        # requires byte-identical schema incl. field order on every write_table)
        # never sees a mismatch. Without this, an unstable one-hot family (e.g. a
        # chunk of symbols with no Thursday rows lacks DOW_4) shifts the column
        # order and aborts the whole panel build mid-way.
        self._cols: Optional[List[str]] = None

    # One-hot families whose category set can legitimately vary chunk-to-chunk.
    # We pin the FULL set so the frozen schema always contains every member and a
    # chunk missing one just gets it back-filled (0) instead of changing the schema.
    _FULL_ONEHOT_FAMILIES = (
        [f"DOW_{i}" for i in range(7)]
        + [f"CPR_Yday_{s}" for s in ("Above", "Below", "Inside", "Overlap")]
        + [f"CPR_Tmr_{s}" for s in ("Above", "Below", "Inside", "Overlap")]
        + [f"Struct_{s}" for s in ("uptrend", "downtrend", "range")]
        + [f"DayType_{s}" for s in ("bullish", "bearish", "inside")]
    )
    # AUDIT-FIX 1: include EVERY feature prefix that discover_daily_features
    # accepts so engineered features (X_*, W_*, WQ_*, Comb_*, DOW_*) are not
    # silently dropped on parquet write. M_* and regime_* are added to the
    # in-memory panel AFTER this writer runs, so listing them here is harmless
    # but keeps the contract symmetric with discover_daily_features().
    _FEATURE_PREFIXES = (
        "D_", "W_", "WQ_", "M_", "X_", "Comb_",
        "CPR_", "Struct_", "DayType_", "DOW_", "regime_",
    )
    _ONEHOT_PREFIXES = ("CPR_Yday_", "CPR_Tmr_", "Struct_", "DayType_")

    def write_chunk(self, df: pd.DataFrame):
        if df is None or df.empty: return
        if self._cols is None:
            # FIRST chunk: establish the frozen, ordered column set. Include every
            # feature-prefixed column present PLUS the full one-hot families, so the
            # schema is complete even if this first chunk happens to miss a category.
            dynamic_keep = list(dict.fromkeys(
                MASTER_KEEP_STATIC
                + [c for c in df.columns if str(c).startswith(self._FEATURE_PREFIXES)]
                + list(self._FULL_ONEHOT_FAMILIES)
            ))
            self._cols = dynamic_keep
        else:
            # LATER chunks: any brand-new feature column would change the schema and
            # abort the writer. Such columns cannot be added to an open parquet file,
            # so log + drop them rather than crash the whole build.
            extra = [c for c in df.columns
                     if str(c).startswith(self._FEATURE_PREFIXES) and c not in self._cols]
            if extra:
                print(f"[Panel] WARNING: dropping {len(extra)} late feature column(s) "
                      f"absent from the frozen panel schema (cannot extend an open "
                      f"parquet): {extra[:8]}{'...' if len(extra) > 8 else ''}")
            dynamic_keep = self._cols
        for col in dynamic_keep:
            if col not in df.columns:
                if str(col).startswith(self._ONEHOT_PREFIXES):
                    df[col] = 0
                else:
                    df[col] = np.nan
        df = df.copy()
        df["timestamp"] = ensure_kolkata_tz(pd.to_datetime(df["timestamp"], errors="coerce"))
        df["symbol"] = df["symbol"].astype(str).map(_clean_symbol_label)
        for c in df.columns:
            if str(c).startswith(self._ONEHOT_PREFIXES):
                if df[c].dtype == bool: df[c] = df[c].astype(np.int32)
                else: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).clip(lower=0, upper=1).astype(np.int32)
        numeric_like = ["open","high","low","close","volume",
                        "ret_1d_close_pct","ret_3d_close_pct","ret_5d_close_pct",
                        "ret_1d_oc_pct","ret_3d_oc_pct","ret_5d_oc_pct",
                        "long_score","short_score","D_atr14","D_cpr_width_pct"]
        # Type-coerce ALL feature-prefix columns (not just D_/ret_/_pct) so that
        # X_/W_/WQ_/Comb_/DOW_ columns get a stable float64 schema across chunks.
        _NUMERIC_FEATURE_PREFIXES = ("D_","W_","WQ_","M_","X_","Comb_","DOW_","regime_")
        for c in df.columns:
            if (c in numeric_like
                or str(c).startswith(_NUMERIC_FEATURE_PREFIXES)
                or str(c).startswith("ret_")
                or str(c).endswith("_pct")):
                df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float64)
        for c in df.columns:
            if df[c].dtype == bool:
                df[c] = df[c].astype(np.int32)
        df = df.reindex(columns=dynamic_keep)
        table = pa.Table.from_pandas(df, preserve_index=False)
        if self._writer is None:
            self._schema = table.schema
            self._writer = pq.ParquetWriter(self.out_path, self._schema, compression="snappy")
        self._writer.write_table(table)
    def close(self):
        if self._writer is not None:
            self._writer.close(); self._writer = None

def append_panel_rows_parquet(writer: PanelParquetWriter, chunks: List[pd.DataFrame]):
    if not chunks: return
    aligned: List[pd.DataFrame] = []
    for df in chunks:
        df = _ensure_unique_columns(df)
        aligned.append(df)
    df_all = pd.concat(aligned, ignore_index=True, sort=False)
    writer.write_chunk(df_all)

def last_ts_by_symbol_from_panel(panel_path: str) -> dict:
    p = Path(panel_path)
    if not p.exists(): return {}
    try:
        df = pd.read_parquet(p)
        df["symbol"] = df["symbol"].astype(str).map(_clean_symbol_label)
        df["timestamp"] = ensure_kolkata_tz(pd.to_datetime(df["timestamp"], errors="coerce"))
        df = df.dropna(subset=["timestamp"])
        last = df.sort_values(["symbol","timestamp"]).groupby("symbol")["timestamp"].tail(1)
        return (df.loc[last.index, ["symbol","timestamp"]
                ].set_index("symbol")["timestamp"].to_dict())
    except Exception:
        return {}

# ===================== Collect panel =====================

def _prepare_panel_rows(path_obj: Path, min_ts_map: dict):
    sym = _derive_symbol_name(path_obj)
    try:
        df = load_one(path_obj)
        min_ts = min_ts_map.get(sym, None)
        if min_ts is not None:
            df = df[df["timestamp"] > pd.to_datetime(min_ts)]
        if df.empty:
            return sym, None, None, f"NO NEW ROWS {sym}"
        # sanity check
        if "D_rsi14" in df.columns:
            s = pd.to_numeric(df["D_rsi14"], errors="coerce")
            bad = (~s.between(0,100)) & s.notna()
            if bad.any():
                _log_load_error(sym, str(path_obj), f"Range anomaly D_rsi14 on {int(bad.sum())} rows")
        # targets + features
        df = add_targets(df)
        df, feats = featureize(df)


        # lightweight bias block
        def _bias_block(d):
            col = lambda c: d[c] if c in d.columns else pd.Series([np.nan]*len(d))
            rsi14 = pd.to_numeric(col("D_rsi14"), errors="coerce")
            atr14 = pd.to_numeric(col("D_atr14"), errors="coerce")
            close = pd.to_numeric(col("close"), errors="coerce")
            atr_pct = (atr14/close).replace([np.inf,-np.inf],np.nan)*100
            cpr_w = pd.to_numeric(col("D_cpr_width_pct"), errors="coerce").abs()
            long_score = ((rsi14.between(50,70, inclusive="both")).fillna(False)).astype(float)
            short_score= ((rsi14<45).fillna(False)).astype(float)
            risk_pen = ((atr_pct>4).fillna(False).astype(float)*0.5 + (cpr_w>1.0).fillna(False).astype(float)*0.3)
            d["long_score"] = long_score - risk_pen
            d["short_score"] = short_score - risk_pen
            return d
        df = _bias_block(df)
        df["symbol"] = sym
        # keep all rows
        rows = df[["timestamp","symbol","open","high","low","close","volume"] + feats +
                  ["ret_1d_close_pct","ret_3d_close_pct","ret_5d_close_pct",
                   "ret_1d_oc_pct","ret_3d_oc_pct","ret_5d_oc_pct",
                   "long_score","short_score","D_atr14","D_cpr_width_pct"]].copy()
        return sym, rows, feats, None
    except Exception as e:
        return sym, None, None, e

def collect_panel_from_paths(paths: List[Path], load_workers: int = 8):
    expanded: List[Path] = []
    for p in paths:
        if Path(p).is_dir():
            expanded += _strict_file_list(str(p), None, None, accept_any_daily=False)
        else:
            expanded.append(Path(p))
    paths = sorted(expanded)
    total = len(paths)
    if total == 0:
        empty = pd.DataFrame(columns=MASTER_KEEP_STATIC)
        if not _PA_OK:
            raise SystemExit("pyarrow is required to write panel_cache.parquet. Please run: pip install pyarrow")
        table = pa.Table.from_pandas(empty, preserve_index=False)
        pq.write_table(table, PANEL_OUT, compression="snappy")
        raise SystemExit("No matching files found Select files via *_daily.* files.")

    min_ts_map = last_ts_by_symbol_from_panel(PANEL_OUT)
    eta = ProgressETA(total=total, label="Load+Engineer")
    chunk: List[pd.DataFrame] = []
    total_rows_written = 0
    feats: Optional[List[str]] = None
    writer = PanelParquetWriter(PANEL_OUT)

    def _prepare_with_path(path_obj: Path):
        return path_obj, _prepare_panel_rows(path_obj, min_ts_map)

    try:
        if load_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=int(load_workers)) as ex:
                for path_obj, result in ex.map(_prepare_with_path, paths):
                    sym, rows, feats_out, msg_or_err = result
                    if isinstance(msg_or_err, Exception):
                        _log_load_error(sym, str(path_obj), str(msg_or_err))
                        eta.tick(f"ERR {sym}: {msg_or_err}"); continue
                    if msg_or_err:
                        eta.tick(msg_or_err); continue
                    chunk.append(rows)
                    if feats_out is not None: feats = feats_out
                    total_rows_written += len(rows)
                    if len(chunk) >= CHUNK_SIZE:
                        append_panel_rows_parquet(writer, chunk); chunk.clear()
                    eta.tick(f"OK {sym} (+{len(rows)} rows)")
        else:
            for path_obj in paths:
                sym, rows, feats_out, msg_or_err = _prepare_panel_rows(path_obj, min_ts_map)
                if isinstance(msg_or_err, Exception):
                    _log_load_error(sym, str(path_obj), str(msg_or_err))
                    eta.tick(f"ERR {sym}: {msg_or_err}"); continue
                if msg_or_err:
                    eta.tick(msg_or_err); continue
                chunk.append(rows)
                if feats_out is not None: feats = feats_out
                total_rows_written += len(rows)
                if len(chunk) >= CHUNK_SIZE:
                    append_panel_rows_parquet(writer, chunk); chunk.clear()
                eta.tick(f"OK {sym} (+{len(rows)} rows)")
    except KeyboardInterrupt:
        print("Interrupted! Autosaving current chunk...")
        if chunk: append_panel_rows_parquet(writer, chunk); chunk.clear()
        writer.close(); raise

    if chunk: append_panel_rows_parquet(writer, chunk); chunk.clear()
    writer.close()
    print(f"[Panel] Appended new rows: {total_rows_written}")

    # load full panel and compute cross-sectional regime features
    _t = time.perf_counter()
    panel = pd.read_parquet(PANEL_OUT)
    panel["symbol"] = panel["symbol"].astype(str).map(_clean_symbol_label)
    panel["timestamp"] = ensure_kolkata_tz(pd.to_datetime(panel["timestamp"], errors="coerce"))
    panel = panel.dropna(subset=["timestamp"]).sort_values(["symbol","timestamp"]).reset_index(drop=True)
    log(f"[Timing] panel read+sort: {time.perf_counter()-_t:.1f}s  (rows={len(panel):,})")

    _t = time.perf_counter()
    panel["date"] = pd.to_datetime(panel["timestamp"]).dt.normalize()
    if "ret_1d_close_pct" not in panel.columns or panel["ret_1d_close_pct"].isna().all():
        panel["ret_1d_close_pct"] = panel.groupby("symbol")["close"].pct_change() * 100.0

    cs_mean = panel.groupby("date")["ret_1d_close_pct"].mean()
    cs_std = panel.groupby("date")["ret_1d_close_pct"].std()
    trend = cs_mean.rolling(200, min_periods=50).mean().shift(1)
    std_lag = cs_std.shift(1)
    vol_med = cs_std.rolling(250, min_periods=50).median().shift(1)
    panel["regime_market_trend"] = panel["date"].map(trend)
    panel["regime_high_vol"] = (panel["date"].map(std_lag) > panel["date"].map(vol_med)).astype(int)
    panel["regime_dispersion"] = panel["date"].map(std_lag)
    log(f"[Timing] cross-sectional regime features: {time.perf_counter()-_t:.1f}s")

    # ----- v4: Join macro features (NIFTY 50 + INDIA VIX) -----
    _t = time.perf_counter()
    panel = _join_macro_features(panel)
    log(f"[Timing] macro join: {time.perf_counter()-_t:.1f}s")

    # ----- v4: Compute per-stock regime label (bull/bear x trending/ranging) -----
    _t = time.perf_counter()
    panel = _compute_stock_regime(panel, regime_lag=REGIME_LAG)
    log(f"[Timing] stock regime label + continuous features: {time.perf_counter()-_t:.1f}s")

    # ----- v4: Compute strict follow-through label (lever 3) -----
    # Skip when not in use: this runs several full-panel groupby passes and the
    # original label is sufficient unless USE_STRICT_LABEL is on. Saves time on
    # every cold build when strict labels aren't needed.
    if USE_STRICT_LABEL:
        panel = _compute_strict_label(panel)
    else:
        log("[Label] USE_STRICT_LABEL=False -> skipping strict follow-through label "
            "computation (saves a full-panel pass).")

    panel = panel.drop(columns=["date"], errors="ignore")
    # v4: include ALL feature prefixes
    feats = discover_daily_features(panel)

    # =====================================================================
    # PERSIST the ENRICHED panel back to disk.
    #
    # The incremental writer above only saved RAW cache columns (OHLCV + base
    # features). The macro join (M_*), per-stock regime label (stock_regime),
    # continuous regime features (X_regime_*), and cross-sectional regime
    # features were all computed IN MEMORY after that write. Downstream tools
    # (feature_imp_v5.py, backtests) read panel_cache.parquet from DISK and
    # expect every feature in features_train.json to be present — including M_*.
    #
    # Without this rewrite, the on-disk parquet is missing M_*/stock_regime/
    # X_regime_*, and any tool that column-projects those features off disk
    # crashes with pyarrow ArrowInvalid: "No match for FieldRef.Name(M_...)".
    # =====================================================================
    try:
        _t = time.perf_counter()
        enriched_new = [c for c in panel.columns
                        if c.startswith(("M_", "X_regime_", "regime_"))
                        or c == "stock_regime"]
        log(f"[Panel Save] Rewriting enriched panel_cache.parquet "
            f"(shape={panel.shape}; +{len(enriched_new)} enriched cols incl. "
            f"{[c for c in enriched_new if c.startswith('M_')][:4]})")

        # MEMORY FIX: pa.Table.from_pandas() on the FULL float64 panel forces a
        # single consolidating .copy() of a (n_features x n_rows) dense block —
        # ~5 GiB for 195 cols x 3.5M rows x 8 bytes, which OOMs. Two mitigations:
        #   1) downcast float64 -> float32 first (halves the block; also makes the
        #      on-disk parquet smaller and faster for downstream tools to read).
        #   2) write in bounded ROW-GROUP BATCHES so peak memory stays flat
        #      regardless of panel size, instead of materializing one giant table.
        _float64_cols = [c for c in panel.columns if str(panel[c].dtype) == "float64"]
        if _float64_cols:
            panel[_float64_cols] = panel[_float64_cols].astype("float32")
            log(f"[Panel Save] Downcast {len(_float64_cols)} float64 cols -> float32 before write")

        if _PA_OK:
            BATCH = 250_000  # rows per row-group; ~bounded memory per write
            writer = None
            try:
                n = len(panel)
                for start in range(0, n, BATCH):
                    chunk = panel.iloc[start:start + BATCH]
                    tbl = pa.Table.from_pandas(chunk, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(PANEL_OUT, tbl.schema, compression="snappy")
                    writer.write_table(tbl)
                    del tbl, chunk
            finally:
                if writer is not None:
                    writer.close()
        else:
            panel.to_parquet(PANEL_OUT, index=False, compression="snappy")
        log(f"[Panel Save] Done in {time.perf_counter()-_t:.1f}s -> {PANEL_OUT}")
    except Exception as e:
        log(f"[Panel Save] WARN: could not rewrite enriched panel: {e}", level="warning")

    return panel, feats


# =============================================================================
# v4: MACRO FEATURE JOIN
# =============================================================================

def _join_macro_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join NIFTY 50 + INDIA VIX macro features onto the panel by date.
    Reads from macro_cache.parquet (built separately by macro_cache.py).
    If macro cache missing, prints a warning and continues without macro.
    """
    macro_path = Path(MACRO_CACHE_PATH)
    if not macro_path.exists():
        print(f"[Macro] WARNING: {MACRO_CACHE_PATH} not found. Skipping macro features.")
        print(f"[Macro] Run: python macro_cache.py  to fetch NIFTY/VIX data first.")
        return panel
    try:
        macro = pd.read_parquet(macro_path)
        macro["date"] = pd.to_datetime(macro["date"]).dt.normalize()
        # Strip tz if any
        try:
            macro["date"] = macro["date"].dt.tz_localize(None)
        except Exception:
            pass
        macro_cols = [c for c in macro.columns if c.startswith("M_")]
        # AUDIT-FIX 2: build _join_date so the panel row's IST date is preserved
        # as a naive date (not shifted by tz_convert). The previous order
        # `.dt.tz_convert(None).dt.normalize()` shifted IST midnight to UTC
        # 18:30 of the prior day, then normalized to that prior date — joining
        # yesterday's macro onto today's row. The correct order strips the tz
        # while keeping the wall-clock date.
        panel_date = pd.to_datetime(panel["date"])
        if getattr(panel_date.dt, "tz", None) is not None:
            panel["_join_date"] = panel_date.dt.tz_localize(None).dt.normalize()
        else:
            panel["_join_date"] = panel_date.dt.normalize()
        merged = panel.merge(
            macro[["date"] + macro_cols].rename(columns={"date": "_join_date"}),
            on="_join_date",
            how="left",
        )
        merged = merged.drop(columns=["_join_date"])
        # Forward-fill any gaps within each symbol's history
        for c in macro_cols:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")
            merged[c] = merged.groupby("symbol")[c].ffill()
        # AUDIT-FIX 2b: report coverage across ALL macro columns, not just the
        # first one (which previously masked heterogeneous coverage).
        if macro_cols:
            cov = pd.concat([merged[c].notna() for c in macro_cols], axis=1).all(axis=1).sum()
            print(f"[Macro] Joined {len(macro_cols)} macro features; full coverage on {int(cov):,} of {len(merged):,} rows")
        else:
            print(f"[Macro] Joined 0 macro features")
        return merged
    except Exception as e:
        print(f"[Macro] WARNING: join failed: {e}. Continuing without macro features.")
        return panel


# =============================================================================
# v4: PER-STOCK REGIME LABEL (4 regimes)
# =============================================================================

def _compute_stock_regime(panel: pd.DataFrame, regime_lag: int = 0) -> pd.DataFrame:
    """
    Compute per-stock daily regime label using existing cache features:
      - bull / bear:    close > SMA200 -> bull, else bear
      - trending / ranging: ADX14 > 25 -> trending, ADX14 < 20 -> ranging, between -> mixed (-> trending)

    4 regimes:
      stock_regime in {'bull_trend', 'bull_range', 'bear_trend', 'bear_range'}

    The HARD label (stock_regime) is excluded from features and used only to ROUTE
    training rows in the 4-model architecture.

    NEW: also emits CONTINUOUS regime features (prefix X_regime_*) that ARE allowed
    into the feature set. These let a SINGLE pooled model learn its own thresholds
    and represent "how bull / how trending" each row is (the "60% bull trend"
    gradient) instead of collapsing it to a 0/1 flag:
      - X_regime_dist_sma200   = close/sma200 - 1   (signed distance, continuous)
      - X_regime_adx           = adx14 (raw level)
      - X_regime_dist_x_adx    = interaction term
      - X_regime_bull_soft     = sigmoid(dist scaled)        in (0,1)
      - X_regime_trend_soft    = sigmoid((adx-22.5)/5)       in (0,1)

    regime_lag: if >0, lag the HARD regime label by this many bars per symbol so the
    routing/reporting regime is known strictly BEFORE the entry bar (matches live
    trading). Continuous features are point-in-time and not lagged.
    """
    if "D_sma200" not in panel.columns or "D_adx14" not in panel.columns:
        print("[Regime] WARNING: D_sma200 or D_adx14 missing. Assigning all rows to 'bull_trend'.")
        panel["stock_regime"] = "bull_trend"
        # Still emit neutral continuous features so the schema is stable.
        panel["X_regime_dist_sma200"] = 0.0
        panel["X_regime_adx"] = np.nan
        panel["X_regime_dist_x_adx"] = 0.0
        panel["X_regime_bull_soft"] = 0.5
        panel["X_regime_trend_soft"] = 0.5
        return panel

    close = pd.to_numeric(panel["close"], errors="coerce")
    sma200 = pd.to_numeric(panel["D_sma200"], errors="coerce")
    adx = pd.to_numeric(panel["D_adx14"], errors="coerce")

    is_bull = (close > sma200)
    is_trending = (adx > 25)
    is_ranging = (adx < 20)
    # mixed (20 <= ADX <= 25) -> route to trending for stability

    regime = pd.Series("bull_trend", index=panel.index, dtype="object")
    regime[is_bull & (is_trending | ~is_ranging)] = "bull_trend"
    regime[is_bull & is_ranging] = "bull_range"
    regime[~is_bull & (is_trending | ~is_ranging)] = "bear_trend"
    regime[~is_bull & is_ranging] = "bear_range"
    panel["stock_regime"] = regime

    # ---- Continuous regime features (the "60% bull" gradient) ----
    dist = (close / sma200 - 1.0).replace([np.inf, -np.inf], np.nan)
    panel["X_regime_dist_sma200"] = dist
    panel["X_regime_adx"] = adx
    panel["X_regime_dist_x_adx"] = dist * adx
    # Soft memberships in (0,1): smooth, no cliffs at the hard thresholds.
    panel["X_regime_bull_soft"] = 1.0 / (1.0 + np.exp(-(dist * 20.0)))      # ~0.5 at SMA200
    panel["X_regime_trend_soft"] = 1.0 / (1.0 + np.exp(-((adx - 22.5) / 5.0)))  # ~0.5 at ADX 22.5

    # ---- Optional: lag the HARD regime label by `regime_lag` bars per symbol ----
    if regime_lag and regime_lag > 0 and "symbol" in panel.columns:
        panel["stock_regime"] = (
            panel.groupby("symbol", observed=True)["stock_regime"].shift(regime_lag)
        )
        # rows with no prior regime (start of history) -> bull_trend default
        panel["stock_regime"] = panel["stock_regime"].fillna("bull_trend")
        print(f"[Regime] Hard regime label lagged by {regime_lag} bar(s) per symbol (known before entry).")

    # Diagnostic counts
    counts = panel["stock_regime"].value_counts()
    print(f"[Regime] Per-stock regime distribution:")
    for r in ["bull_trend", "bull_range", "bear_trend", "bear_range"]:
        n = counts.get(r, 0)
        pct = 100.0 * n / max(len(panel), 1)
        print(f"          {r:15} {n:>10,}  ({pct:5.2f}%)")

    return panel


# =============================================================================
# v4: STRICT FOLLOW-THROUGH LABEL (lever 3)
# =============================================================================

def _compute_strict_label(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Build top20_strict_5d: a stricter version of top20_vs_bot20_5d.

    Original label: 1 if forward 5d return is in top 20% (vs bottom 20% = 0).
    Strict label adds:
      - close[t+5] must be >= close[t]   (no net loss on exit)
      - max drawdown over t+1..t+5 must be < 3%  (no big mid-trade swing)

    A stock with high forward return that drew down 8% then recovered to +6% gets
    labeled 1 in the original but 0 in the strict label. The strict label filters
    "lucky winners" and rewards genuinely smooth winners.

    Output: panel with both 'top20_vs_bot20_5d' (original) and 'top20_strict_5d' (new).
    """
    panel = panel.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    close = pd.to_numeric(panel["close"], errors="coerce")

    # Build forward path of lows/highs/closes for next 5 days, per symbol
    grp = panel.groupby("symbol", group_keys=False)

    # Forward 5-day close
    fwd_close_5 = grp["close"].transform(lambda s: pd.to_numeric(s, errors="coerce").shift(-5))

    # Forward minimum LOW over t+1..t+5
    # AUDIT-FIX 6: remove the dead first `low_fwd_min = grp.apply(...)` block.
    # It was computed and then immediately overwritten 5 lines later by the
    # explicit shift+concat below. The apply call was the slowest step in this
    # function on a 4-5M row panel.
    low = pd.to_numeric(panel["low"], errors="coerce")  # noqa: F841 (kept for diagnostic continuity)
    low_t1 = grp["low"].transform(lambda s: pd.to_numeric(s, errors="coerce").shift(-1))
    low_t2 = grp["low"].transform(lambda s: pd.to_numeric(s, errors="coerce").shift(-2))
    low_t3 = grp["low"].transform(lambda s: pd.to_numeric(s, errors="coerce").shift(-3))
    low_t4 = grp["low"].transform(lambda s: pd.to_numeric(s, errors="coerce").shift(-4))
    low_t5 = grp["low"].transform(lambda s: pd.to_numeric(s, errors="coerce").shift(-5))
    low_fwd_min = pd.concat([low_t1, low_t2, low_t3, low_t4, low_t5], axis=1).min(axis=1)

    # Conditions for strict label
    cond_positive_close = (fwd_close_5 >= close)
    max_drawdown_pct = (close - low_fwd_min) / close.replace(0, np.nan) * 100.0
    cond_low_drawdown = (max_drawdown_pct < 3.0)

    # Compute original label first if not present
    if "top20_vs_bot20_5d" not in panel.columns:
        if "ret_5d_oc_pct" in panel.columns and "D_atr14" in panel.columns:
            r5 = pd.to_numeric(panel["ret_5d_oc_pct"], errors="coerce")
            atr_pct = (pd.to_numeric(panel["D_atr14"], errors="coerce")
                       / close.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan) * 100.0
            vol_basis = atr_pct.replace(0, np.nan)
            panel["ret_5d_adj"] = r5 / vol_basis
            panel["rank_5d_pct"] = panel.groupby("timestamp")["ret_5d_adj"].rank(method="average", pct=True)
            panel["top20_vs_bot20_5d"] = np.where(
                panel["rank_5d_pct"] >= 0.80, 1,
                np.where(panel["rank_5d_pct"] <= 0.20, 0, np.nan)
            )

    # Strict label: top 20% AND positive close AND low drawdown
    orig = panel.get("top20_vs_bot20_5d")
    if orig is None:
        panel["top20_strict_5d"] = np.nan
    else:
        is_top = (orig == 1)
        is_bot = (orig == 0)
        strict = np.where(is_top & cond_positive_close & cond_low_drawdown, 1,
                          np.where(is_bot, 0, np.nan))
        panel["top20_strict_5d"] = strict

    # Diagnostic counts
    if "top20_strict_5d" in panel.columns:
        n_orig_1 = int((panel["top20_vs_bot20_5d"] == 1).sum())
        n_strict_1 = int((panel["top20_strict_5d"] == 1).sum())
        retention = (100.0 * n_strict_1 / max(n_orig_1, 1))
        print(f"[Label] Original top20 winners: {n_orig_1:,}")
        print(f"[Label] Strict (close>=entry & drawdown<3%) winners: {n_strict_1:,}  ({retention:.1f}% retention)")

    return panel

# ===================== Feature matrix sanitize + schema lock =====================

def sanitize_feature_matrix(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    if X.columns.duplicated().any():
        X = X.loc[:, ~X.columns.duplicated()]
    for c in X.columns:
        ser = X[c]
        if isinstance(ser, pd.DataFrame):
            ser = ser.iloc[:, 0]
        X[c] = ser
        if ser.dtype == bool:
            X[c] = ser.astype(int)
        elif ser.dtype == object:
            s = ser.astype(str).str.lower()
            uniq = set(pd.Series(s).unique())
            if uniq <= {"true","false","nan"}:
                X[c] = pd.Series(s).map({"true":1, "false":0}).astype("Int64").fillna(0).astype(int)
            else:
                X[c] = pd.to_numeric(s, errors="coerce")
        if str(c).startswith(("CPR_Yday_","CPR_Tmr_","Struct_","DayType_")):
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0).astype(int)
    return X

def compute_impute_stats(X: pd.DataFrame) -> Dict[str, float]:
    return X.median(numeric_only=True).to_dict()

def save_schema(schema_path: str, feats: List[str], impute: Dict[str, float]):
    data = {"features": list(feats), "impute": {k: float(v) for k, v in impute.items()}}
    Path(schema_path).write_text(json.dumps(data, indent=2))

def load_schema(schema_path: str) -> Tuple[List[str], Dict[str, float]]:
    data = json.loads(Path(schema_path).read_text())
    return list(data["features"]), {k: float(v) for k, v in data["impute"].items()}

def reindex_and_impute(X_last: pd.DataFrame, feats: List[str], impute: Dict[str, float]) -> pd.DataFrame:
    X = X_last.reindex(columns=feats).copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        if X[c].isna().any():
            X[c] = X[c].fillna(impute.get(c, 0.0))
    return X

# ===================== EV label engineering =====================

def build_5d_rank_quant_labels(panel: pd.DataFrame, ev_target: str = "cc") -> pd.DataFrame:
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["timestamp"]).dt.normalize()
    daily_ret = (
            pd.to_numeric(panel["close"], errors="coerce")
            .groupby(panel["symbol"], observed=True)
            .pct_change()
            * 100.0
    )
    panel["vol_20"] = (
        daily_ret
        .groupby(panel["symbol"], observed=True)
        .rolling(20, min_periods=10)
        .std()
        .reset_index(level=0, drop=True)
    )
    panel["atr_pct"] = (pd.to_numeric(panel["D_atr14"], errors="coerce") / pd.to_numeric(panel["close"], errors="coerce")).replace([np.inf,-np.inf],np.nan) * 100.0
    vol_basis = panel["vol_20"].fillna(panel["atr_pct"]).replace(0.0, np.nan)
    # ROBUSTNESS FIX (vol-adjustment blow-up): a near-zero-volatility name makes
    # ret_*_adj = ret / vol_basis explode. That single artifact drove the top
    # calibration decile's exp_sharpe_5d to ~4.85 while every other decile sat in
    # [-1.0, +0.4], and it also distorts the isotonic EV mapper (fit on ret_5d_adj).
    # Floor vol_basis at each trading day's 5th percentile so the adjustment cannot
    # divide by a pathologically small number. Top/bottom-20% membership is
    # essentially unchanged (the affected names are micro-vol outliers that should
    # not receive implicit 20-30x leverage in the adjusted return).
    _vb_floor = (
        vol_basis.groupby(panel["date"]).transform(
            lambda s: np.nanpercentile(s, 5)
            if np.isfinite(s.to_numpy(dtype="float64")).any() else 0.0
        ).fillna(0.0)
    )
    vol_basis = np.maximum(vol_basis, _vb_floor)
    if ev_target == "oc":
        r5 = pd.to_numeric(panel["ret_5d_oc_pct"], errors="coerce")
        r3 = pd.to_numeric(panel["ret_3d_oc_pct"], errors="coerce")
    else:
        r5 = pd.to_numeric(panel["ret_5d_close_pct"], errors="coerce")
        r3 = pd.to_numeric(panel["ret_3d_close_pct"], errors="coerce")
    panel["ret_5d_adj"] = r5 / vol_basis
    panel["ret_3d_adj"] = r3 / vol_basis
    grp = panel.groupby("date")
    panel["rank_5d_pct"] = grp["ret_5d_adj"].rank(method="average", pct=True)
    panel["top20_vs_bot20_5d"] = np.where(panel["rank_5d_pct"] >= 0.80, 1,
                                           np.where(panel["rank_5d_pct"] <= 0.20, 0, np.nan))
    return panel

# ===================== CV splits =====================

def time_cv_by_timestamp(panel: pd.DataFrame,
                         n_splits: int = 7,
                         embargo_days: int = 5,
                         target_mask: Optional[pd.Series] = None):
    idx = panel.index if target_mask is None else panel.index[target_mask]
    ts_all = pd.to_datetime(panel.loc[idx, "timestamp"]).dt.normalize()
    uniq_dates = pd.Series(ts_all).sort_values().unique()
    if len(uniq_dates) < n_splits + 1:
        n_splits = max(1, min(len(uniq_dates) - 1, n_splits))
    cut = np.linspace(0, len(uniq_dates), n_splits + 1, dtype=int)
    for i in range(n_splits):
        start_date = uniq_dates[cut[i]]
        end_date = uniq_dates[cut[i+1]-1] if i < n_splits - 1 else uniq_dates[-1]
        te_mask = (panel["timestamp"].dt.normalize() >= start_date) & (panel["timestamp"].dt.normalize() <= end_date)
        tr_mask = (panel["timestamp"].dt.normalize() < start_date)
        if embargo_days and embargo_days > 0:
            embargo_edge = start_date - pd.Timedelta(days=int(embargo_days))
            tr_mask = (panel["timestamp"].dt.normalize() <= embargo_edge)
        tr_idx = panel.index[tr_mask & panel.index.isin(idx)]
        te_idx = panel.index[te_mask & panel.index.isin(idx)]
        if len(te_idx) > 0 and len(tr_idx) > 0:
            yield tr_idx, te_idx

def split_train_val_by_time(panel: pd.DataFrame, candidate_idx: np.ndarray,
                            val_frac: float = 0.15, min_val: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    if candidate_idx is None:
        return np.array([],dtype=int), np.array([], dtype=int)
    idx = np.asarray(candidate_idx)
    if len(idx) < 3:
        return idx, np.array([], dtype=idx.dtype)
    ts = pd.to_datetime(panel.loc[idx, "timestamp"]).values
    order = np.argsort(ts)
    val_n = max(1, int(round(len(order) * val_frac)))
    val_n = max(val_n, min_val)
    val_n = min(len(order) // 2, val_n)
    if val_n == 0:
        return idx, np.array([], dtype=idx.dtype)
    val_order = order[-val_n:]
    train_order = order[:-val_n]
    return idx[train_order], idx[val_order]


# FIXES 2 & 3: shared date-aligned, embargoed train/cal/test splitter.
#
# The old code split the time-sorted positions contiguously:
#     i_train = order[:0.7n]; i_cal = order[0.7n:0.9n]; i_test = order[0.9n:]
# Two problems:
#   FIX 3 (mid-day leak): np.argsort(timestamp) can place rows from the SAME
#       calendar day on both sides of a boundary, so one day's cross-section
#       straddles train and cal. With a per-day rank label that is incoherent.
#   FIX 2 (no embargo gap): the last ~horizon days of train have forward labels
#       that overlap into cal, and cal into test. With a 5-day forward label this
#       leaks ~5 days of look-ahead across each boundary and inflates OOS metrics.
#
# This helper assigns WHOLE DAYS to a side and drops `embargo_days` of trading
# days at each train|cal and cal|test boundary.
def split_train_cal_test_by_date(ts_values: np.ndarray,
                                  train_frac: float = 0.70,
                                  cal_frac: float = 0.20,
                                  embargo_days: int = EMBARGO_DAYS):
    """
    ts_values: array of timestamps (one per labeled row, in the row order of X/y).
    Returns (i_train, i_cal, i_test) as positional index arrays into ts_values.

    Splits on unique calendar days (no day straddles a boundary) and embargoes
    `embargo_days` trading days at each boundary so forward-looking labels in one
    split cannot overlap the next.
    """
    ts = pd.to_datetime(pd.Series(ts_values)).dt.normalize()
    uniq_days = np.sort(ts.unique())
    n_days = len(uniq_days)
    if n_days < 5:
        # Degenerate: too few days to embargo. Fall back to contiguous positional split.
        order = np.argsort(ts_values)
        n = len(order)
        return (order[:int(train_frac * n)],
                order[int(train_frac * n):int((train_frac + cal_frac) * n)],
                order[int((train_frac + cal_frac) * n):])

    cut_tr = int(train_frac * n_days)
    cut_cal = int((train_frac + cal_frac) * n_days)
    emb = max(0, int(embargo_days))

    # Day-level slices: remove `emb` days from the trailing edge of the EARLIER
    # block at each boundary (train|cal and cal|test) so a forward-looking label
    # in one split cannot reach into the next.
    def _slice(emb_local):
        train_days = set(uniq_days[: max(0, cut_tr - emb_local)])
        cal_days = set(uniq_days[cut_tr: max(cut_tr, cut_cal - emb_local)])
        test_days = set(uniq_days[cut_cal:])
        return train_days, cal_days, test_days

    train_days, cal_days, test_days = _slice(emb)
    # If the embargo is so large (relative to a short history) that it empties the
    # cal or test block, relax it to 0 rather than returning empty splits.
    if (not cal_days or not test_days or not train_days) and emb > 0:
        train_days, cal_days, test_days = _slice(0)

    day_arr = ts.values
    i_train = np.where(np.isin(day_arr, list(train_days)))[0]
    i_cal = np.where(np.isin(day_arr, list(cal_days)))[0]
    i_test = np.where(np.isin(day_arr, list(test_days)))[0]
    return i_train, i_cal, i_test


# ===================== LightGBM helpers =====================

def _check_lightgbm():
    try:
        import lightgbm as lgb
        from lightgbm import LGBMClassifier
        return lgb, LGBMClassifier
    except Exception as e:
        raise SystemExit("LightGBM is not installed. Please run: pip install lightgbm") from e

def _lgbm_cls_params(rnd: int):
    """v4 hyperparameters (tighter trees, more diversity, extra_trees on)."""
    depth = MAX_DEPTH if isinstance(MAX_DEPTH, int) and MAX_DEPTH > 0 else -1
    params = dict(
        n_estimators=3000,
        learning_rate=LEARNING_RATE,
        num_leaves=31,
        max_depth=depth,
        feature_fraction=0.7,
        bagging_fraction=0.7,
        bagging_freq=1,
        min_data_in_leaf=500,
        min_gain_to_split=0.02,
        max_bin=255,
        reg_alpha=0.3,
        reg_lambda=10.0,
        extra_trees=True,
        class_weight=None,
        n_jobs=-1,
        random_state=int(rnd),
        verbosity=-1,
        # diversity knobs that DO propagate through CalibratedClassifierCV:
        feature_fraction_seed=int(rnd),
        bagging_seed=int(rnd),
        data_random_seed=int(rnd),
    )
    return params

def _lgbm_rank_params(rnd: int):
    """Hyperparameters for TARGET_MODE='rank' (LightGBM regression on the per-day
    rank percentile). Mirrors _lgbm_cls_params but with a regression objective and
    NO class_weight (LGBMRegressor does not accept it)."""
    depth = MAX_DEPTH if isinstance(MAX_DEPTH, int) and MAX_DEPTH > 0 else -1
    params = dict(
        objective=str(RANK_OBJECTIVE),
        metric="l2",
        n_estimators=3000,
        learning_rate=LEARNING_RATE,
        num_leaves=31,
        max_depth=depth,
        feature_fraction=0.7,
        bagging_fraction=0.7,
        bagging_freq=1,
        min_data_in_leaf=500,
        min_gain_to_split=0.02,
        max_bin=255,
        reg_alpha=0.3,
        reg_lambda=10.0,
        extra_trees=True,
        n_jobs=-1,
        random_state=int(rnd),
        verbosity=-1,
        feature_fraction_seed=int(rnd),
        bagging_seed=int(rnd),
        data_random_seed=int(rnd),
    )
    return params

def _lgb_callbacks(val_size: int):
    import lightgbm as _lgb_mod
    cbs = [_lgb_mod.callback.log_evaluation(period=0)]
    if EARLY_STOPPING_ROUNDS and EARLY_STOPPING_ROUNDS > 0 and val_size >= int(MIN_VAL_EARLYSTOP):
        cbs.insert(0, _lgb_mod.callback.early_stopping(stopping_rounds=int(EARLY_STOPPING_ROUNDS)))
    return cbs


def _fit_binary_cls_safe(clf, X_tr, y_tr, X_val, y_val, eval_metric="binary_logloss"):
    """Robustly fit a LightGBM binary classifier against degenerate CV folds.

    LightGBM's sklearn wrapper fits a LabelEncoder on y_tr. When the per-day
    cross-section is thin, the top/bottom-quantile label collapses to a SINGLE
    class across an early time window — a day with <5 labeled names can only
    ever produce class 1 (the top), never class 0 (the bottom). The encoder
    then raises "y contains previously unseen labels: [0]" the moment the
    eval/test set carries the missing class.

    Strategy:
      * skip the fold (return False) if y_tr has < 2 classes;
      * drop eval rows whose label is absent from y_tr; if the eval set
        collapses entirely, fit without it (hence without early stopping).
    Returns True if the model was fit, False if the fold should be skipped.
    """
    tr_classes = np.unique(np.asarray(y_tr))
    if tr_classes.size < 2:
        return False
    yv = np.asarray(y_val)
    keep = np.isin(yv, tr_classes)
    if len(yv) > 0 and keep.all():
        clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric=eval_metric,
                callbacks=_lgb_callbacks(len(yv)))
    elif keep.any():
        pos = np.flatnonzero(keep)
        Xv = X_val.iloc[pos] if hasattr(X_val, "iloc") else np.asarray(X_val)[pos]
        clf.fit(X_tr, y_tr, eval_set=[(Xv, yv[keep])], eval_metric=eval_metric,
                callbacks=_lgb_callbacks(int(keep.sum())))
    else:
        clf.fit(X_tr, y_tr, callbacks=_lgb_callbacks(0))
    return True

# ===================== Calibration helpers =====================

def _calibrate_best_brier(est, X_val, y_val):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import brier_score_loss
    iso = CalibratedClassifierCV(estimator=est, method="isotonic", cv="prefit")
    iso.fit(X_val, y_val)
    p_iso = iso.predict_proba(X_val)[:, 1]
    br_iso = brier_score_loss(y_val, p_iso)
    sig = CalibratedClassifierCV(estimator=est, method="sigmoid", cv="prefit")
    sig.fit(X_val, y_val)
    p_sig = sig.predict_proba(X_val)[:, 1]
    br_sig = brier_score_loss(y_val, p_sig)
    chosen = iso if br_iso <= br_sig else sig
    info = {"brier_isotonic": float(br_iso), "brier_sigmoid": float(br_sig),
            "chosen": "isotonic" if br_iso <= br_sig else "sigmoid"}
    return chosen, info

from sklearn.isotonic import IsotonicRegression

def fit_isotonic_ev_mapper(prob_calibrated: np.ndarray, realized_adj: np.ndarray):
    order = np.argsort(prob_calibrated)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(prob_calibrated[order], realized_adj[order])
    return iso

# ===================== Train 5D quantile classifier (CV diagnostics) =====================

def train_5d_quantile_cls(panel: pd.DataFrame, feats: List[str], ev_target: str,
                          n_splits: int = FOLDS, embargo_days: int = EMBARGO_DAYS,
                          early_stopping_rounds: int = EARLY_STOPPING_ROUNDS):
    lgb, LGBMClassifier = _check_lightgbm()
    pl = build_5d_rank_quant_labels(panel, ev_target=ev_target)
    y = pl["top20_vs_bot20_5d"]
    mask = y.notna()
    if not mask.any():
        raise ValueError("No labeled rows for 5D quantile classification (top/bottom 20%).")

    # Cross-section breadth diagnostic. The per-day top/bottom-20% label needs
    # >= 5 names on a day to yield BOTH classes; with fewer names only class 1
    # (the top) appears, which makes early CV folds collapse to a single class.
    _names_per_day = pl.loc[mask].groupby(pl.loc[mask, "date"]).size()
    if len(_names_per_day):
        _med = float(np.median(_names_per_day.values))
        _thin = int((_names_per_day < 5).sum())
        print(f"[5D-CLS] cross-section breadth: median names/day={_med:.0f}, "
              f"days with <5 names={_thin}/{len(_names_per_day)} "
              f"({100.0 * _thin / max(len(_names_per_day), 1):.0f}%), "
              f"symbols={pl.loc[mask, 'symbol'].nunique()}")
        if _med < 8:
            print("[5D-CLS][WARN] Thin cross-section: the top/bottom-20% label is a "
                  "per-day cross-sectional bucket and needs >=5 names/day to produce "
                  "both classes. With so few symbols, CV folds can contain only one "
                  "class. Cache more symbols for a robust cross-sectional model. "
                  "(Degenerate folds are now skipped rather than crashing.)")

    valid_idx = panel.index[mask].to_numpy()
    folds = list(time_cv_by_timestamp(pl, n_splits=n_splits, embargo_days=embargo_days, target_mask=mask))
    X_full = sanitize_feature_matrix(
        pl.loc[mask].reindex(columns=feats).copy()
    )
    y_full = y.loc[mask].astype(int).values
    oos_prob = np.full(len(X_full), np.nan)
    n_skipped = 0
    eta = ProgressETA(total=len(folds), label="Train 5D-Quantile-CLS")
    for fold_no, (tr_idx, te_idx) in enumerate(folds, start=1):
        tr_pos = np.where(np.isin(valid_idx, tr_idx))[0]
        te_pos = np.where(np.isin(valid_idx, te_idx))[0]
        if len(te_pos) == 0 or len(tr_pos) == 0:
            continue
        tr_core_idx, val_idx = split_train_val_by_time(pl, valid_idx[tr_pos], val_frac=0.2, min_val=200)
        tr_core_pos = np.where(np.isin(valid_idx, tr_core_idx))[0]
        val_pos = np.where(np.isin(valid_idx, val_idx))[0]
        X_tr = X_full.iloc[tr_core_pos if len(tr_core_pos)>0 else tr_pos]
        y_tr = y_full[tr_core_pos if len(tr_core_pos)>0 else tr_pos]
        X_val = X_full.iloc[val_pos] if len(val_pos)>0 else X_tr
        y_val = y_full[val_pos] if len(val_pos)>0 else y_tr
        X_te = X_full.iloc[te_pos]
        clf = LGBMClassifier(**_lgbm_cls_params(GLOBAL_SEED + 300 + fold_no))
        if not _fit_binary_cls_safe(clf, X_tr, y_tr, X_val, y_val, eval_metric="binary_logloss"):
            n_skipped += 1
            eta.tick(f"fold {fold_no} SKIPPED (training fold has a single class)")
            continue
        prob = clf.predict_proba(X_te)[:, 1]
        oos_prob[te_pos] = prob
        eta.tick(f"fold {fold_no}")
    if n_skipped:
        print(f"[5D-CLS] {n_skipped}/{len(folds)} fold(s) skipped due to single-class "
              f"training data (thin cross-section). OOS probabilities for those test "
              f"segments remain NaN and are ignored downstream.")
    return oos_prob, valid_idx, pl

# ===================== Final Train → Calibrate → Test (EV mapper fixed) =====================

def _persist_feature_schema(feats: List[str], impute_stats: Dict[str, float]):
    """Create or non-destructively update features_train.json. Factored out so the
    rank-mode path saves the schema with the same semantics as the binary path."""
    if Path(FEATURES_SCHEMA_PATH).exists():
        existing_feats, existing_impute = load_schema(FEATURES_SCHEMA_PATH)
        new_feats = [f for f in feats if f not in existing_feats]
        if new_feats:
            print(f"[Schema] Detected {len(new_feats)} NEW features in panel:")
            for nf in new_feats[:20]:
                print(f"          + {nf}")
            if len(new_feats) > 20:
                print(f"          ... and {len(new_feats) - 20} more")
            merged_impute = dict(existing_impute)
            merged_impute.update(impute_stats)
            save_schema(FEATURES_SCHEMA_PATH, feats, merged_impute)
            print(f"[Schema] Updated: {len(feats)} total features")
        else:
            print(f"[Schema] No new features. Using existing {len(existing_feats)} features")
    else:
        save_schema(FEATURES_SCHEMA_PATH, feats, impute_stats)
        print(f"[Schema] Created features_train.json with {len(feats)} features")


def _fit_final_rank(panel: pd.DataFrame, feats: List[str], ev_target: str,
                    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS):
    """TARGET_MODE='rank' counterpart of fit_final_model_and_oos_calibration.
    Reuses the shared prepared-arrays + member machinery so oos_report.json and
    calibration_5d_deciles.json reflect the SAME regression objective + score->prob
    calibration that the deployed ensemble uses."""
    _check_lightgbm()
    prepared = _prepare_training_arrays(panel, feats, ev_target, use_strict_label=USE_STRICT_LABEL)
    _persist_feature_schema(feats, prepared["impute_stats"])
    member, iso_ev, oos_df = _fit_member_from_arrays(
        prepared, seed=GLOBAL_SEED + 777,
        early_stopping_rounds=early_stopping_rounds, skip_oos_artifacts=False)
    print(f"[Final] TARGET_MODE='rank': trained regression model on "
          f"{len(prepared['i_train']):,} rows (all-row rank target).")
    return member, iso_ev, oos_df


def fit_final_model_and_oos_calibration(panel: pd.DataFrame, feats: List[str], ev_target: str,
                                        early_stopping_rounds: int = EARLY_STOPPING_ROUNDS):
    lgb, LGBMClassifier = _check_lightgbm()
    if str(TARGET_MODE).lower() == "rank":
        return _fit_final_rank(panel, feats, ev_target, early_stopping_rounds)
    pl = build_5d_rank_quant_labels(panel, ev_target=ev_target)
    y = pl["top20_vs_bot20_5d"].astype("float")
    mask = y.notna()
    if not mask.any():
        raise ValueError("No labels for 5D quantile classification.")
    X = sanitize_feature_matrix(
        pl.loc[mask].reindex(columns=feats).copy()
    )
    y = y.loc[mask].astype(int)
    t = pl.loc[mask, "timestamp"].values
    # FIXES 2 & 3: day-aligned, embargoed split (was contiguous positional split).
    i_train, i_cal, i_test = split_train_cal_test_by_date(
        t, train_frac=0.70, cal_frac=0.20, embargo_days=EMBARGO_DAYS)
    if len(i_train) == 0 or len(i_cal) == 0 or len(i_test) == 0:
        # Embargo too aggressive for available history -> relax to no-gap day split.
        i_train, i_cal, i_test = split_train_cal_test_by_date(
            t, train_frac=0.70, cal_frac=0.20, embargo_days=0)
    from lightgbm import LGBMClassifier as _LGB
    base = _LGB(**_lgbm_cls_params(GLOBAL_SEED+777))
    callbacks = _lgb_callbacks(len(i_cal))
    base.fit(X.iloc[i_train], y.iloc[i_train],
             eval_set=[(X.iloc[i_cal], y.iloc[i_cal])], eval_metric="binary_logloss", callbacks=callbacks)
    # Probability calibrator on CAL
    final_calib, diag = _calibrate_best_brier(base, X.iloc[i_cal], y.iloc[i_cal])
    # Train-only imputation medians (on TRAIN slice only)
    impute_stats = compute_impute_stats(X.iloc[i_train])
    if Path(FEATURES_SCHEMA_PATH).exists():
        existing_feats, existing_impute = load_schema(FEATURES_SCHEMA_PATH)
        new_feats = [f for f in feats if f not in existing_feats]
        if new_feats:
            print(f"[Schema] Detected {len(new_feats)} NEW features in panel:")
            for nf in new_feats[:20]:
                print(f"          + {nf}")
            if len(new_feats) > 20:
                print(f"          ... and {len(new_feats) - 20} more")
            merged_impute = dict(existing_impute)
            for k, v in impute_stats.items():
                merged_impute[k] = v
            save_schema(FEATURES_SCHEMA_PATH, feats, merged_impute)
            print(f"[Schema] Updated: {len(feats)} total features")
        else:
            print(f"[Schema] No new features. Using existing {len(existing_feats)} features")
    else:
        save_schema(FEATURES_SCHEMA_PATH, feats, impute_stats)
        print(f"[Schema] Created features_train.json with {len(feats)} features")
    # EV mapper trained on CALIBRATED probabilities (consistency)
    prob_cal_calibrated = final_calib.predict_proba(X.iloc[i_cal])[:, 1]
    realized_adj_cal = pl.loc[mask].iloc[i_cal]["ret_5d_adj"].astype(float).values
    iso_ev = fit_isotonic_ev_mapper(prob_cal_calibrated, realized_adj_cal)
    # TEST inference
    p_test = final_calib.predict_proba(X.iloc[i_test])[:, 1]
    ev_test = iso_ev.predict(p_test)
    oos_df = pd.DataFrame({
        "prob_top20_5d": p_test,
        "ret_3d_close_pct": pl.loc[mask].iloc[i_test]["ret_3d_close_pct"].values,
        "ret_5d_close_pct": pl.loc[mask].iloc[i_test]["ret_5d_close_pct"].values,
        "ret_5d_adj": pl.loc[mask].iloc[i_test]["ret_5d_adj"].values,
        "rank_5d_pct": pl.loc[mask].iloc[i_test]["rank_5d_pct"].values,
        "expected_ret_5d_adj_iso": ev_test,
    })
    return final_calib, iso_ev, oos_df

# ===================== 1D follow-through (mandatory; final refit on full history) =====================

def train_1d_followthrough(panel: pd.DataFrame, feats: List[str], margin_pct: float = CLS_MARGIN_1D,
                           n_splits: int = FOLDS, embargo_days: int = EMBARGO_DAYS):
    lgb, LGBMClassifier = _check_lightgbm()
    r = pd.to_numeric(panel["ret_1d_close_pct"], errors="coerce")
    y = pd.Series(np.nan, index=panel.index)
    y[(r > margin_pct)] = 1
    y[(r < -margin_pct)] = 0
    mask = y.notna()
    n_labels = int(mask.sum())
    if n_labels < MIN_GATE_SAMPLES:
        raise SystemExit(
            f"[FATAL] 1D gate is mandatory, but found only {n_labels} labeled rows (< {MIN_GATE_SAMPLES}). "
            f"Increase history, relax CLS_MARGIN_1D={margin_pct}, or lower MIN_GATE_SAMPLES."
        )
    valid_idx = panel.index[mask].to_numpy()
    folds = list(time_cv_by_timestamp(panel, n_splits=n_splits, embargo_days=embargo_days, target_mask=mask))
    X_full = sanitize_feature_matrix(
        panel.loc[mask].reindex(columns=feats).copy()
    )
    y_full = y.loc[mask].astype(int).values
    eta = ProgressETA(total=len(folds), label="Train 1D-Gate (CV diag)")
    for fold_no, (tr_idx, te_idx) in enumerate(folds, start=1):
        tr_pos = np.where(np.isin(valid_idx, tr_idx))[0]
        te_pos = np.where(np.isin(valid_idx, te_idx))[0]
        if len(te_pos) == 0 or len(tr_pos) == 0:
            eta.tick(f"fold {fold_no} (skip)"); continue
        tr_core_idx, val_idx = split_train_val_by_time(panel, valid_idx[tr_pos], val_frac=0.2, min_val=200)
        tr_core_pos = np.where(np.isin(valid_idx, tr_core_idx))[0]
        val_pos = np.where(np.isin(valid_idx, val_idx))[0]
        X_tr = X_full.iloc[tr_core_pos if len(tr_core_pos)>0 else tr_pos]
        y_tr = y_full[tr_core_pos if len(tr_core_pos)>0 else tr_pos]
        X_val = X_full.iloc[val_pos] if len(val_pos)>0 else X_tr
        y_val = y_full[val_pos] if len(val_pos)>0 else y_tr
        from lightgbm import LGBMClassifier as _Gate
        clf = _Gate(**dict(
            n_estimators=int(N_EST_1D), learning_rate=LEARNING_RATE,
            num_leaves=56, max_depth=MAX_DEPTH, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            min_data_in_leaf=250, min_gain_to_split=0.02, max_bin=127,
            reg_alpha=0.4, reg_lambda=8.0, class_weight=None,
            n_jobs=-1, random_state=int(GLOBAL_SEED + 500 + fold_no), verbosity=-1,
        ))
        if not _fit_binary_cls_safe(clf, X_tr, y_tr, X_val, y_val, eval_metric="binary_logloss"):
            eta.tick(f"fold {fold_no} SKIPPED (single-class train)")
            continue
        eta.tick(f"fold {fold_no}")

    # FINAL refit on full labeled history with time-ordered Train/Cal split
    ts_all = pd.to_datetime(panel.loc[mask, "timestamp"]).values
    order = np.argsort(ts_all)
    cut = max(int(0.8 * len(order)), 1)
    tr_pos_full = order[:cut]
    cal_pos_full = order[cut:]
    from lightgbm import LGBMClassifier as _Gate
    final_clf = _Gate(**dict(
        n_estimators=int(N_EST_1D), learning_rate=LEARNING_RATE,
        num_leaves=56, max_depth=MAX_DEPTH, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
        min_data_in_leaf=250, min_gain_to_split=0.02, max_bin=127,
        reg_alpha=0.4, reg_lambda=8.0, class_weight=None,
        n_jobs=-1, random_state=int(GLOBAL_SEED + 999), verbosity=-1,
    ))
    final_clf.fit(X_full.iloc[tr_pos_full], y_full[tr_pos_full])
    from sklearn.calibration import CalibratedClassifierCV
    final_gate = CalibratedClassifierCV(estimator=final_clf, method="sigmoid", cv="prefit")
    final_gate.fit(X_full.iloc[cal_pos_full], y_full[cal_pos_full])
    return final_gate

# ===================== Map prob -> expectations (legacy decile mapping) =====================

# =============================================================================
# v4: ENSEMBLE + REGIME-CONDITIONAL TRAINING
# =============================================================================

REGIMES = ["bull_trend", "bull_range", "bear_trend", "bear_range"]
ENSEMBLE_SIZE_PER_REGIME = 5
MIN_ROWS_PER_REGIME = 5000  # minimum labeled rows to train a regime model


class RankProbaModel:
    """
    TARGET_MODE='rank' member. Wraps a LightGBM REGRESSOR (trained on the per-day
    rank percentile, all rows) plus an isotonic map score -> P(top-20%), and
    exposes the SAME predict_proba/predict interface as the binary calibrated
    classifier so it is a drop-in member for EnsembleCalibrator / RegimeRouter /
    the watchlist / the EVShim export. Pickle-safe (module-level class).

    feature_list mirrors EnsembleCalibrator: when set, X is reindexed to those
    columns before predicting. Members usually carry feature_list=None and let the
    enclosing EnsembleCalibrator project, exactly like the binary members.
    """
    def __init__(self, regressor, score_to_prob, feature_list=None):
        self.regressor = regressor
        self.score_to_prob = score_to_prob  # fitted sklearn IsotonicRegression
        self.feature_list = list(feature_list) if feature_list is not None else None

    def _project(self, X):
        feature_list = getattr(self, "feature_list", None)
        if feature_list is None:
            return X
        try:
            return X.reindex(columns=feature_list)
        except Exception:
            return X

    def _scores_to_proba(self, scores):
        scores = np.asarray(scores, dtype=float)
        s2p = getattr(self, "score_to_prob", None)
        if s2p is None:
            # No calibrator: min-max the score into (0,1) as a last resort.
            lo, hi = np.nanmin(scores), np.nanmax(scores)
            p = (scores - lo) / (hi - lo) if hi > lo else np.full_like(scores, 0.5)
        else:
            p = np.asarray(s2p.predict(scores), dtype=float)
        return np.clip(p, 1e-6, 1.0 - 1e-6)

    def predict_proba(self, X):
        Xp = self._project(X)
        scores = self.regressor.predict(Xp)
        p = self._scores_to_proba(scores)
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return self.predict_proba(X)[:, 1] >= 0.5


class EnsembleCalibrator:
    """
    Wrapper that averages predictions from N calibrated models.
    Pickle-safe (module-level class).

    feature_list (optional): the exact, ordered feature columns this ensemble was
    trained on. When set, predict_proba reindexes the incoming X to these columns
    before predicting. This lets different ensembles (e.g. per-regime specialists
    trained on PRUNED per-regime feature lists) coexist behind one RegimeRouter
    that is handed the full feature matrix — each ensemble picks its own columns.
    When None, behaves exactly as before (uses X as-is) for backward compatibility.
    """
    def __init__(self, members, feature_list=None):
        self.members = list(members)
        self.feature_list = list(feature_list) if feature_list is not None else None

    def _project(self, X):
        # getattr (not self.feature_list) so EnsembleCalibrator objects pickled by
        # an OLDER model version (before feature_list existed) load and run without
        # AttributeError — they simply behave as feature_list=None (use X as-is).
        feature_list = getattr(self, "feature_list", None)
        if feature_list is None:
            return X
        # Reindex to the trained columns; any column absent in X is added as NaN.
        # (The members' own pipelines impute as needed; LightGBM also handles NaN.)
        try:
            return X.reindex(columns=feature_list)
        except Exception:
            return X

    def predict_proba(self, X):
        Xp = self._project(X)
        probs = np.zeros((len(Xp), 2))
        for m in self.members:
            probs += m.predict_proba(Xp)
        return probs / max(len(self.members), 1)

    def predict(self, X):
        return self.predict_proba(X)[:, 1] >= 0.5


class RegimeRouter:
    """
    Routes each row to a regime-specific ensemble based on stock_regime label.
    Pickle-safe (module-level class).

    Used at scoring time: given a feature matrix + a regime vector,
    returns prob_top20_5d using the matching regime's ensemble.
    """
    def __init__(self, regime_models: dict, fallback_ensemble):
        # regime_models: dict of {regime_name -> EnsembleCalibrator}
        # fallback_ensemble: used when regime has no trained model
        self.regime_models = dict(regime_models or {})
        self.fallback = fallback_ensemble

    def required_features(self) -> List[str]:
        """Union of every feature any member ensemble needs (pruned lists may
        differ per regime). Scoring code should build X with at least these
        columns; each EnsembleCalibrator then self-projects to its own list."""
        cols: List[str] = []
        seen = set()
        models = list(self.regime_models.values())
        if self.fallback is not None:
            models.append(self.fallback)
        for m in models:
            fl = getattr(m, "feature_list", None)
            if fl:
                for c in fl:
                    if c not in seen:
                        seen.add(c); cols.append(c)
        return cols

    def predict_proba_by_regime(self, X: pd.DataFrame, regime_vec: pd.Series) -> np.ndarray:
        """
        X: feature matrix (n rows, k cols)
        regime_vec: pd.Series of regime labels (n entries)
        Returns: prob array of length n (binary class 1 probability)
        """
        probs = np.full(len(X), np.nan)
        for r in REGIMES:
            mask = (regime_vec.values == r)
            if not mask.any():
                continue
            model = self.regime_models.get(r) or self.fallback
            if model is None:
                continue
            X_sub = X.iloc[mask]
            p = model.predict_proba(X_sub)[:, 1]
            probs[mask] = p
        # Anything still NaN gets fallback
        nan_mask = np.isnan(probs)
        if nan_mask.any() and self.fallback is not None:
            X_sub = X.iloc[nan_mask]
            probs[nan_mask] = self.fallback.predict_proba(X_sub)[:, 1]
        return probs


def fit_regime_ensembles(panel: pd.DataFrame, feats: List[str], ev_target: str,
                          n_members: int = ENSEMBLE_SIZE_PER_REGIME,
                          early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
                          use_strict_label: bool = False,
                          train_regime_specialists: bool = True,
                          per_regime_feats: Optional[Dict[str, List[str]]] = None,
                          global_feats: Optional[List[str]] = None) -> tuple:
    """
    Train the fallback "all-regimes" (pooled) ensemble, and OPTIONALLY the 4
    regime-specific specialist ensembles.

    train_regime_specialists:
      True  -> also train the 4 per-regime specialists (regime / both architectures)
      False -> train ONLY the pooled fallback (single architecture). Returns empty
               regime_models, so a RegimeRouter built from it sends every row to
               the pooled model.

    PRUNED FEATURE WIRING (new):
      per_regime_feats: optional {regime_name -> [feature,...]} from the pruner's
        regime_features.json "per_regime" lists. When provided, each regime
        specialist trains on ITS OWN pruned feature list, and the resulting
        EnsembleCalibrator is tagged with that list so it self-projects at score
        time. Regimes missing from the dict fall back to `feats`.
      global_feats: optional pruned feature list for the POOLED fallback ensemble
        (the pruner's keep_list_global). Defaults to `feats` when None.

    For each regime (when enabled):
      - subset panel to rows where stock_regime == r
      - if enough labeled rows, train an ensemble of n_members LGBM classifiers
      - each member uses a different (deterministic) seed for diversity

    use_strict_label: if True, use top20_strict_5d (follow-through label) instead of original.

    Returns: (regime_models_dict, iso_ev_mappers_dict, calib_tables_dict, fallback_ensemble, fallback_iso_ev, fallback_calib_table, oos_df)
    """
    per_regime_feats = dict(per_regime_feats or {})
    fb_feats = list(global_feats) if global_feats else list(feats)
    print(f"\n[Regime Models] Training ensembles ({'strict label' if use_strict_label else 'original label'}; "
          f"specialists={'ON' if train_regime_specialists else 'OFF'})...")
    if per_regime_feats:
        print(f"[Regime Models] Per-regime PRUNED feature lists supplied for: "
              f"{sorted(per_regime_feats.keys())}")
    if global_feats:
        print(f"[Regime Models] Pooled fallback uses pruned global list: {len(fb_feats)} features")

    regime_models = {}
    regime_iso_ev = {}
    regime_calib_tables = {}

    # ---- Train FALLBACK first (all data, no regime filter) = the POOLED single model ----
    print(f"\n[Regime Models] Training FALLBACK / POOLED ensemble: {n_members} members")
    print(f"  Preparing training arrays (once for all {n_members} members)...")
    t0 = time.perf_counter()
    fb_prepared = _prepare_training_arrays(panel, fb_feats, ev_target, use_strict_label=use_strict_label)
    print(f"  Prep done in {time.perf_counter()-t0:.1f}s  "
          f"(n_train={len(fb_prepared['i_train']):,}, n_cal={len(fb_prepared['i_cal']):,}, n_test={len(fb_prepared['i_test']):,})")

    fallback_members = []
    fallback_iso_ev = None
    fallback_oos_df = None
    for i in range(n_members):
        t_mem = time.perf_counter()
        seed = GLOBAL_SEED + 7919 * (i + 1)
        # Only member 1 keeps the iso_ev_mapper + oos_df (used for calibration tables).
        # Members 2..N skip those artifacts — pure speed win, identical end result.
        skip_artifacts = (i > 0)
        member_calib, member_iso_ev, member_oos = _fit_member_from_arrays(
            fb_prepared, seed=seed,
            early_stopping_rounds=early_stopping_rounds,
            skip_oos_artifacts=skip_artifacts,
        )
        fallback_members.append(member_calib)
        if i == 0:
            fallback_iso_ev = member_iso_ev
            fallback_oos_df = member_oos
        print(f"  Member {i+1}/{n_members} fit in {time.perf_counter()-t_mem:.1f}s "
              f"({'skipped OOS artifacts' if skip_artifacts else 'kept OOS artifacts'})")
    fallback_ensemble = EnsembleCalibrator(fallback_members, feature_list=fb_feats)
    fallback_calib_table = _build_calib_table_from_oos(fallback_oos_df)
    print(f"[Regime Models] Fallback ensemble trained: {len(fallback_members)} members")

    # ---- Train each regime (specialists) ----
    if not train_regime_specialists:
        print("[Regime Models] Specialists disabled (single architecture). "
              "Pooled fallback will serve all regimes.")
    for r in (REGIMES if train_regime_specialists else []):
        sub = panel[panel["stock_regime"] == r].copy() if "stock_regime" in panel.columns else panel.iloc[0:0]
        # BUGFIX: the 5d top/bottom-20% label is NOT a column on the raw panel — it
        # is created inside build_5d_rank_quant_labels (and, for strict, in
        # _compute_strict_label). The previous check `label_col in sub.columns` was
        # therefore always False, n_labeled was always 0, and EVERY regime fell back
        # to the pooled model (making the architecture comparison meaningless).
        # Build the label on this subset and count it correctly.
        if len(sub) == 0:
            n_labeled = 0
        elif use_strict_label and "top20_strict_5d" in sub.columns:
            n_labeled = int(sub["top20_strict_5d"].notna().sum())
        else:
            _sub_lbl = build_5d_rank_quant_labels(sub, ev_target=ev_target)
            n_labeled = int(_sub_lbl["top20_vs_bot20_5d"].notna().sum())
        print(f"\n[Regime Models] {r}: {len(sub):,} rows, {n_labeled:,} labeled")

        if n_labeled < MIN_ROWS_PER_REGIME:
            print(f"  Insufficient labeled rows (<{MIN_ROWS_PER_REGIME}). Will use fallback for this regime.")
            continue

        # Prepare arrays ONCE for this regime — using this regime's PRUNED feature
        # list if the pruner supplied one, else the global list.
        r_feats = per_regime_feats.get(r) or feats
        if per_regime_feats.get(r):
            print(f"  Using pruned feature list for {r}: {len(r_feats)} features")
        print(f"  Preparing training arrays...")
        t0 = time.perf_counter()
        try:
            r_prepared = _prepare_training_arrays(sub, r_feats, ev_target, use_strict_label=use_strict_label)
        except Exception as e:
            print(f"  Prep failed: {e}. Falling back for this regime.")
            continue
        print(f"  Prep done in {time.perf_counter()-t0:.1f}s")

        members = []
        for i in range(n_members):
            t_mem = time.perf_counter()
            try:
                # FIX 4 (reproducibility): Python's builtin hash() on str is salted
                # per-process (PYTHONHASHSEED), so two identical runs produced
                # different seeds -> different models -> different watchlists.
                # Use a stable digest so runs are byte-reproducible.
                regime_seed_offset = int(hashlib.md5(r.encode("utf-8")).hexdigest()[:6], 16) % 1000
                seed = GLOBAL_SEED + 100003 * (i + 1) + regime_seed_offset
                skip_artifacts = (i > 0)
                member_calib, member_iso_ev, member_oos = _fit_member_from_arrays(
                    r_prepared, seed=seed,
                    early_stopping_rounds=early_stopping_rounds,
                    skip_oos_artifacts=skip_artifacts,
                )
                members.append(member_calib)
                if i == 0:
                    regime_iso_ev[r] = member_iso_ev
                    regime_calib_tables[r] = _build_calib_table_from_oos(member_oos)
                print(f"  Member {i+1}/{n_members} fit in {time.perf_counter()-t_mem:.1f}s "
                      f"({'skipped OOS' if skip_artifacts else 'kept OOS'})")
            except Exception as e:
                print(f"  Member {i+1} failed: {e}")

        if members:
            regime_models[r] = EnsembleCalibrator(members, feature_list=r_feats)
            print(f"  Trained {len(members)} members for {r}")
        else:
            print(f"  No members trained for {r}, will fall back")

    print(f"\n[Regime Models] Summary:")
    print(f"  Fallback ensemble: {len(fallback_members)} members")
    for r in REGIMES:
        if r in regime_models:
            print(f"  {r}: {len(regime_models[r].members)} members")
        else:
            print(f"  {r}: using fallback")

    return regime_models, regime_iso_ev, regime_calib_tables, fallback_ensemble, fallback_iso_ev, fallback_calib_table, fallback_oos_df


def _prepare_training_arrays(panel: pd.DataFrame, feats: List[str], ev_target: str,
                              use_strict_label: bool = False) -> dict:
    """
    Prepare X, y, train/cal/test splits and metadata.
    This is the EXPENSIVE part — should be done ONCE per regime, then
    reused across all N ensemble members of that regime.

    Returns dict with keys: X, y, i_train, i_cal, i_test, ret_5d_adj_cal,
                            ret_3d_test, ret_5d_test, ret_5d_adj_test, rank_test,
                            impute_stats, pl_mask_index
    """
    pl = build_5d_rank_quant_labels(panel, ev_target=ev_target)
    if use_strict_label and "top20_strict_5d" in panel.columns:
        strict_map = panel[["timestamp", "symbol", "top20_strict_5d"]].copy()
        pl = pl.merge(strict_map, on=["timestamp", "symbol"], how="left", suffixes=("", "_strict"))
        if "top20_strict_5d" in pl.columns:
            pl["top20_vs_bot20_5d"] = pl["top20_strict_5d"]
    # Binary indicator is always kept (rank mode uses it only to calibrate the
    # regressor's score -> P(top-20%) on the embargoed CAL slice).
    bin_all = pl["top20_vs_bot20_5d"].astype("float")
    mode = str(TARGET_MODE).lower()
    if mode == "rank":
        # CONTINUOUS target on ALL labeled rows (~3x data): the per-day
        # cross-sectional rank percentile of the vol-adjusted forward 5d return.
        target_all = pd.to_numeric(pl["rank_5d_pct"], errors="coerce")
        mask = target_all.notna() & pd.to_numeric(pl["ret_5d_adj"], errors="coerce").notna()
    else:
        target_all = bin_all
        mask = bin_all.notna()
    if not mask.any():
        raise ValueError("No labels for 5D target (mode=%r)." % mode)
    X = sanitize_feature_matrix(pl.loc[mask].reindex(columns=feats).copy())
    y = target_all.loc[mask].astype(float) if mode == "rank" else target_all.loc[mask].astype(int)
    t = pl.loc[mask, "timestamp"].values
    # FIXES 2 & 3: day-aligned, embargoed split (was contiguous positional split).
    i_train, i_cal, i_test = split_train_cal_test_by_date(
        t, train_frac=0.70, cal_frac=0.20, embargo_days=EMBARGO_DAYS)
    if len(i_train) == 0 or len(i_cal) == 0 or len(i_test) == 0:
        i_train, i_cal, i_test = split_train_cal_test_by_date(
            t, train_frac=0.70, cal_frac=0.20, embargo_days=0)
    impute_stats = compute_impute_stats(X.iloc[i_train])
    pl_masked = pl.loc[mask]
    bin_masked = bin_all.loc[mask].to_numpy(dtype=float)  # NaN for the middle 60% in rank mode
    return {
        "X": X, "y": y, "mode": mode,
        "i_train": i_train, "i_cal": i_cal, "i_test": i_test,
        "y_bin_cal": bin_masked[i_cal],   # binary {0,1,NaN} for score->prob calibration
        "ret_5d_adj_cal": pl_masked.iloc[i_cal]["ret_5d_adj"].astype(float).values,
        "ret_3d_test": pl_masked.iloc[i_test]["ret_3d_close_pct"].values,
        "ret_5d_test": pl_masked.iloc[i_test]["ret_5d_close_pct"].values,
        "ret_5d_adj_test": pl_masked.iloc[i_test]["ret_5d_adj"].values,
        "rank_test": pl_masked.iloc[i_test]["rank_5d_pct"].values,
        "impute_stats": impute_stats,
    }


def _fit_member_from_arrays(prepared: dict, seed: int,
                             early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
                             skip_oos_artifacts: bool = False):
    """
    Fit a single ensemble member from pre-prepared arrays.
    Cheap: just trains LGBM with given seed.

    If skip_oos_artifacts=True, returns (final_calib, None, None) — used for
    members 2..N where the iso_ev_mapper and oos_df are discarded by the caller.
    """
    from lightgbm import LGBMClassifier as _LGB
    X = prepared["X"]; y = prepared["y"]
    i_train = prepared["i_train"]; i_cal = prepared["i_cal"]; i_test = prepared["i_test"]

    # ENSEMBLE DIVERSITY FIX: previously the N members differed ONLY by their RNG
    # seed, so they trained near-identical trees. In the actual run the 5 fallback
    # members agreed to the 4th decimal (val logloss 0.67567 / 0.67572 / 0.67568 /
    # 0.67581 / 0.67569), which means averaging them reduced variance by ~nothing
    # while costing 5x the compute. Give each member a genuinely different
    # structural config (num_leaves / feature_fraction / min_data_in_leaf), seeded
    # deterministically from `seed` so runs remain byte-reproducible. Real member
    # diversity is what lets the ensemble average actually lower OOS error.
    mode = prepared.get("mode", "binary")
    _mrng = np.random.RandomState(int(seed) % (2**31 - 1))
    _div = dict(
        num_leaves=int(_mrng.choice([24, 31, 48, 63])),
        feature_fraction=float(_mrng.choice([0.6, 0.7, 0.8])),
        min_data_in_leaf=int(_mrng.choice([300, 500, 800])),
    )

    # ---- TARGET_MODE='rank': LightGBM regression on the per-day rank percentile ----
    if mode == "rank":
        from lightgbm import LGBMRegressor as _LGBR
        from sklearn.isotonic import IsotonicRegression as _Iso
        _rp = _lgbm_rank_params(seed); _rp.update(_div)
        reg = _LGBR(**_rp)
        callbacks = _lgb_callbacks(len(i_cal))
        reg.fit(X.iloc[i_train], y.iloc[i_train],
                eval_set=[(X.iloc[i_cal], y.iloc[i_cal])], eval_metric="l2",
                callbacks=callbacks)
        # Calibrate the regressor's raw score -> P(top-20%) on the CAL rows that
        # carry a binary label (the extremes). Isotonic keeps p monotonic in score
        # so the rank member's predict_proba is comparable to the binary member's.
        s_cal = np.asarray(reg.predict(X.iloc[i_cal]), dtype=float)
        y_bin_cal = np.asarray(prepared.get("y_bin_cal"), dtype=float)
        ok = np.isfinite(s_cal) & np.isfinite(y_bin_cal)
        score_to_prob = _Iso(out_of_bounds="clip")
        if int(ok.sum()) >= 50 and len(np.unique(y_bin_cal[ok])) > 1:
            score_to_prob.fit(s_cal[ok], y_bin_cal[ok])
        else:
            # Degenerate fallback: calibrate against the score's own percentile rank.
            score_to_prob.fit(s_cal, pd.Series(s_cal).rank(pct=True).to_numpy())
        member = RankProbaModel(reg, score_to_prob, feature_list=None)
        if skip_oos_artifacts:
            return member, None, None
        p_cal = np.clip(score_to_prob.predict(s_cal), 1e-6, 1.0 - 1e-6)
        iso_ev = fit_isotonic_ev_mapper(p_cal, prepared["ret_5d_adj_cal"])
        s_test = np.asarray(reg.predict(X.iloc[i_test]), dtype=float)
        p_test = np.clip(score_to_prob.predict(s_test), 1e-6, 1.0 - 1e-6)
        ev_test = iso_ev.predict(p_test)
        oos_df = pd.DataFrame({
            "prob_top20_5d": p_test,
            "ret_3d_close_pct": prepared["ret_3d_test"],
            "ret_5d_close_pct": prepared["ret_5d_test"],
            "ret_5d_adj": prepared["ret_5d_adj_test"],
            "rank_5d_pct": prepared["rank_test"],
            "expected_ret_5d_adj_iso": ev_test,
        })
        return member, iso_ev, oos_df

    # ---- TARGET_MODE='binary' (original path, unchanged) ----
    _params = _lgbm_cls_params(seed)
    _params.update(_div)
    base = _LGB(**_params)
    callbacks = _lgb_callbacks(len(i_cal))
    base.fit(X.iloc[i_train], y.iloc[i_train],
             eval_set=[(X.iloc[i_cal], y.iloc[i_cal])], eval_metric="binary_logloss",
             callbacks=callbacks)
    final_calib, diag = _calibrate_best_brier(base, X.iloc[i_cal], y.iloc[i_cal])

    if skip_oos_artifacts:
        return final_calib, None, None

    prob_cal = final_calib.predict_proba(X.iloc[i_cal])[:, 1]
    iso_ev = fit_isotonic_ev_mapper(prob_cal, prepared["ret_5d_adj_cal"])
    p_test = final_calib.predict_proba(X.iloc[i_test])[:, 1]
    ev_test = iso_ev.predict(p_test)
    oos_df = pd.DataFrame({
        "prob_top20_5d": p_test,
        "ret_3d_close_pct": prepared["ret_3d_test"],
        "ret_5d_close_pct": prepared["ret_5d_test"],
        "ret_5d_adj": prepared["ret_5d_adj_test"],
        "rank_5d_pct": prepared["rank_test"],
        "expected_ret_5d_adj_iso": ev_test,
    })
    return final_calib, iso_ev, oos_df


def fit_final_model_and_oos_calibration_seeded(panel: pd.DataFrame, feats: List[str], ev_target: str,
                                                 seed: int,
                                                 early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
                                                 use_strict_label: bool = False):
    """
    Backwards-compatible wrapper. Calls the 2-step pipeline but combines work.
    For best performance, callers training N members on the same data should call
    _prepare_training_arrays() once + _fit_member_from_arrays() N times.
    """
    prepared = _prepare_training_arrays(panel, feats, ev_target, use_strict_label)
    return _fit_member_from_arrays(prepared, seed, early_stopping_rounds, skip_oos_artifacts=False)


def _build_calib_table_from_oos(oos_df: pd.DataFrame) -> dict:
    """Helper to convert oos_df into the calib_table format expected by map_prob_to_expectations."""
    if oos_df is None or len(oos_df) == 0:
        return {"prob_mid": [], "avg_ret_3d": [], "avg_ret_5d": [],
                "std_ret_5d": [], "exp_sharpe_5d": []}
    df_oos = oos_df.dropna(subset=["prob_top20_5d"]).copy()
    if len(df_oos) < 30:
        return {"prob_mid": [], "avg_ret_3d": [], "avg_ret_5d": [],
                "std_ret_5d": [], "exp_sharpe_5d": []}
    df_oos["prob_bucket"] = pd.qcut(df_oos["prob_top20_5d"],
                                     q=min(10, max(3, df_oos.shape[0]//50)),
                                     labels=False, duplicates="drop")
    calib = (df_oos.groupby("prob_bucket")
        .agg(avg_prob=("prob_top20_5d", "mean"),
             avg_ret_3d=("ret_3d_close_pct", "mean"),
             avg_ret_5d=("ret_5d_close_pct", "mean"),
             std_ret_5d=("ret_5d_close_pct", "std"),
             avg_ret_5d_adj=("ret_5d_adj", "median"))
        .reset_index().sort_values("prob_bucket"))
    # CORRECTNESS FIX: prob_mid must be the ACTUAL mean predicted probability in
    # each bucket, not a synthetic np.linspace(0.05, 0.95, ...). map_prob_to_expectations
    # interpolates expected returns against prob_mid, so the old synthetic grid
    # silently mapped a model probability to the WRONG expected return (e.g. a true
    # 0.62 was read off the table as if it were ~0.55). exp_sharpe_5d now uses the
    # per-bucket MEDIAN vol-adjusted return so a few residual outliers cannot blow
    # up a decile (this is what produced the 4.85 top-decile Sharpe artifact).
    calib["prob_mid"] = calib["avg_prob"].astype(float)
    calib["exp_sharpe_5d"] = calib["avg_ret_5d_adj"].astype(float)
    return {
        "prob_mid": calib["prob_mid"].tolist(),
        "avg_ret_3d": calib["avg_ret_3d"].tolist(),
        "avg_ret_5d": calib["avg_ret_5d"].tolist(),
        "std_ret_5d": calib["std_ret_5d"].fillna(0.0).tolist(),
        "exp_sharpe_5d": calib["exp_sharpe_5d"].tolist(),
    }


def map_prob_to_expectations(prob_vec: np.ndarray, calib_table: Dict[str, List[float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pm = np.asarray(calib_table["prob_mid"], dtype=float)
    r3 = np.asarray(calib_table["avg_ret_3d"], dtype=float)
    r5 = np.asarray(calib_table["avg_ret_5d"], dtype=float)
    sh = np.asarray(calib_table["exp_sharpe_5d"], dtype=float)
    order = np.argsort(pm)
    pm = pm[order]; r3 = r3[order]; r5 = r5[order]; sh = sh[order]
    exp3 = np.interp(prob_vec, pm, r3, left=r3[0], right=r3[-1])
    exp5 = np.interp(prob_vec, pm, r5, left=r5[0], right=r5[-1])
    exp_sh = np.interp(prob_vec, pm, sh, left=sh[0], right=sh[-1])
    return exp3, exp5, exp_sh


# =============================================================================
# ARCHITECTURE COMPARISON: single pooled model vs. 4-way regime router
# =============================================================================

def compare_architectures_oos(panel: pd.DataFrame, feats: List[str], ev_target: str,
                              regime_router: 'RegimeRouter',
                              fallback_ensemble,
                              use_strict_label: bool = False) -> dict:
    """
    Evaluate BOTH architectures on the SAME held-out, embargoed OOS test rows and
    report Brier score + Spearman IC vs ret_5d_adj, broken down by regime.

    This is the apples-to-apples test that answers "did splitting into regime
    models actually help, or is the bull/bear bucket just easier?":
      - 'single'  column = pooled fallback ensemble scored on those rows
      - 'regime'  column = the routed specialist ensemble scored on the SAME rows
    Compare the two columns WITHIN each regime. If the regime specialist does not
    beat the pooled model on identical rows, pooling wins (simpler + more data).
    """
    from sklearn.metrics import brier_score_loss
    from scipy.stats import spearmanr

    pl = build_5d_rank_quant_labels(panel, ev_target=ev_target)
    if use_strict_label and "top20_strict_5d" in panel.columns:
        strict_map = panel[["timestamp", "symbol", "top20_strict_5d"]].copy()
        pl = pl.merge(strict_map, on=["timestamp", "symbol"], how="left")
        if "top20_strict_5d" in pl.columns:
            pl["top20_vs_bot20_5d"] = pl["top20_strict_5d"]

    y_all = pl["top20_vs_bot20_5d"].astype("float")
    mask = y_all.notna()
    if not mask.any():
        print("[Compare] No labeled rows; skipping architecture comparison.")
        return {}

    X_all = sanitize_feature_matrix(pl.loc[mask].reindex(columns=feats).copy())
    t = pl.loc[mask, "timestamp"].values
    i_train, i_cal, i_test = split_train_cal_test_by_date(
        t, train_frac=0.70, cal_frac=0.20, embargo_days=EMBARGO_DAYS)
    if len(i_test) == 0:
        print("[Compare] Empty OOS test slice; skipping.")
        return {}

    X_test = X_all.iloc[i_test].reset_index(drop=True)
    pl_test = pl.loc[mask].iloc[i_test].reset_index(drop=True)
    y_test = y_all.loc[mask].iloc[i_test].astype(int).reset_index(drop=True).values
    radj_test = pd.to_numeric(pl_test["ret_5d_adj"], errors="coerce").values
    regime_test = (pl_test["stock_regime"] if "stock_regime" in pl_test.columns
                   else pd.Series(["bull_trend"] * len(pl_test)))

    # Single (pooled) predictions
    p_single = fallback_ensemble.predict_proba(X_test)[:, 1] if fallback_ensemble is not None \
        else np.full(len(X_test), np.nan)
    # Regime-routed predictions on the SAME rows
    p_regime = regime_router.predict_proba_by_regime(X_test, regime_test)

    def _metrics(p, y, radj):
        ok = ~np.isnan(p)
        if ok.sum() < 30:
            return {"n": int(ok.sum()), "brier": float("nan"), "ic": float("nan")}
        br = float(brier_score_loss(y[ok], p[ok]))
        ic, _ = spearmanr(p[ok], radj[ok], nan_policy="omit")
        return {"n": int(ok.sum()), "brier": br, "ic": float(ic)}

    report = {"overall": {"single": _metrics(p_single, y_test, radj_test),
                          "regime": _metrics(p_regime, y_test, radj_test)},
              "by_regime": {}}
    for r in REGIMES:
        m = (regime_test.values == r)
        if m.sum() == 0:
            continue
        report["by_regime"][r] = {
            "single": _metrics(p_single[m], y_test[m], radj_test[m]),
            "regime": _metrics(p_regime[m], y_test[m], radj_test[m]),
        }

    # Pretty-print
    print("\n" + "=" * 78)
    print("ARCHITECTURE COMPARISON (same embargoed OOS rows)  -- lower Brier / higher IC is better")
    print("=" * 78)
    print(f"{'bucket':14} {'n':>8}  {'Brier(single)':>13} {'Brier(regime)':>13}  {'IC(single)':>11} {'IC(regime)':>11}")
    def _row(label, d):
        s, g = d["single"], d["regime"]
        print(f"{label:14} {s['n']:>8}  {s['brier']:>13.4f} {g['brier']:>13.4f}  "
              f"{s['ic']:>11.4f} {g['ic']:>11.4f}")
    _row("OVERALL", report["overall"])
    for r in REGIMES:
        if r in report["by_regime"]:
            _row(r, report["by_regime"][r])
    print("=" * 78)
    print("Read WITHIN each regime row: if Brier(regime) is not clearly below")
    print("Brier(single) AND IC(regime) not clearly above IC(single), the pooled")
    print("single model wins for that regime (more data, simpler, more stable).")
    print("=" * 78 + "\n")
    return report


# ===================== Full-panel scoring (multi-day diagnostics) =====================

def export_full_panel_scores(panel: pd.DataFrame,
                             regime_router: 'RegimeRouter',
                             feats_schema: List[str],
                             impute_stats: Dict[str, float],
                             batch_rows: int = 200_000) -> Optional[str]:
    """Score EVERY (symbol, day) row with the regime-routed model and write a
    slim panel_scored.parquet: [symbol, timestamp, stock_regime, prob_5d_mean,
    close]. This is what MULTI-DAY diagnostics consume (e.g. model_character.py),
    instead of the single latest cross-section in watchlist_5d_signal.csv.

    Scored in row batches so it does not OOM on a multi-million-row panel.

    NOTE: rows inside the training window are scored IN-SAMPLE, so use this to
    characterise the model's behaviour/tilt (momentum vs mean-reversion) across
    history — not to measure out-of-sample edge. Filter to the embargoed TEST
    window downstream if you need a strictly leak-free read.
    """
    if PANEL_OUT is None or panel is None or len(panel) == 0:
        return None
    out_path = Path(PANEL_OUT).with_name("panel_scored.parquet")
    has_regime = "stock_regime" in panel.columns
    keep = [c for c in ["symbol", "timestamp", "close"] if c in panel.columns]
    n = len(panel)
    parts = []
    t0 = time.perf_counter()
    for start in range(0, n, int(batch_rows)):
        sl = panel.iloc[start:start + int(batch_rows)]
        Xb = reindex_and_impute(
            sanitize_feature_matrix(sl.reindex(columns=feats_schema).copy()),
            feats_schema, impute_stats).reset_index(drop=True)
        rv = (sl["stock_regime"].reset_index(drop=True)
              if has_regime else pd.Series(["bull_trend"] * len(sl)))
        prob = regime_router.predict_proba_by_regime(Xb, rv)
        out = sl[keep].reset_index(drop=True).copy()
        out["stock_regime"] = rv.values if has_regime else "bull_trend"
        out["prob_5d_mean"] = np.asarray(prob, dtype=float)
        parts.append(out)
    scored = pd.concat(parts, ignore_index=True)
    try:
        scored.to_parquet(out_path, index=False, compression="snappy")
    except Exception:
        out_path = out_path.with_suffix(".csv")
        scored.to_csv(out_path, index=False)
    print(f"[PanelScore] Full-panel scores ({len(scored):,} rows, "
          f"{scored['timestamp'].nunique()} days) -> {out_path} "
          f"in {time.perf_counter()-t0:.1f}s")
    return str(out_path)


# ===================== TODAY-BASED nightly watchlist =====================

def nightly_watchlist(panel: pd.DataFrame, feats: List[str],
                       regime_router: 'RegimeRouter',
                       regime_iso_ev_map: dict,
                       regime_calib_tables: dict,
                       fallback_iso_ev,
                       fallback_calib_table: dict,
                       exclude_pattern: Optional[str],
                       export_xlsx: bool):
    """
    v4: Regime-aware watchlist. Routes each row to the correct regime ensemble.

    Outputs:
      1. CONSOLIDATED watchlist with regime_used + prob_5d_mean + prob_5d_std (ensemble agreement)
         + prob from each regime ensemble (for diagnostics)
      2. Per-regime watchlist CSVs in a subfolder

    NO 1D gate (removed in v3).
    """
    panel = panel.copy().sort_values(["symbol", "timestamp"])
    panel["avg20_vol"] = panel.groupby("symbol", observed=True)["volume"].transform(
        lambda s: s.rolling(20, min_periods=1).mean())
    last = panel.groupby("symbol", as_index=False, observed=True).tail(1)

    # Reliability: drop symbols whose most recent bar is stale (delisted/halted),
    # so we never score an old bar as if it were "today".
    if WATCHLIST_MAX_STALENESS_DAYS and WATCHLIST_MAX_STALENESS_DAYS > 0 and len(last) > 0:
        panel_max_ts = pd.to_datetime(panel["timestamp"]).max()
        last_ts = pd.to_datetime(last["timestamp"])
        age_days = (panel_max_ts - last_ts).dt.days
        fresh = age_days <= int(WATCHLIST_MAX_STALENESS_DAYS)
        n_stale = int((~fresh).sum())
        if n_stale > 0:
            print(f"[Watchlist] Dropping {n_stale} stale symbols "
                  f"(last bar > {WATCHLIST_MAX_STALENESS_DAYS}d before panel max {panel_max_ts.date()})")
        last = last.loc[fresh.values]

    if exclude_pattern:
        mask = ~last["symbol"].astype(str).str.contains(
            exclude_pattern, regex=True, na=False)
        last = last.loc[mask]

    print(f"[Watchlist] Universe before scoring: {len(last):,} rows")

    feats_schema, impute_stats = load_schema(FEATURES_SCHEMA_PATH)

    # Score the FULL panel (every symbol, every day) so multi-day diagnostics
    # (model_character.py) measure behaviour across history, not just today's
    # single cross-section. Wrapped so it can never break the watchlist build.
    try:
        export_full_panel_scores(panel, regime_router, feats_schema, impute_stats)
    except Exception as _e:
        print(f"[PanelScore] full-panel score export skipped: {_e}")

    # AUDIT-FIX 4: use reindex so a feature missing from the panel does not
    # raise KeyError. reindex_and_impute below will fill via train-only medians.
    X_raw = sanitize_feature_matrix(last.reindex(columns=feats_schema).copy())
    X = reindex_and_impute(X_raw, feats_schema, impute_stats)

    # Determine each row's regime
    if "stock_regime" not in last.columns:
        print("[Watchlist] WARNING: stock_regime missing, treating all as bull_trend")
        regime_vec = pd.Series(["bull_trend"] * len(last), index=last.index)
    else:
        regime_vec = last["stock_regime"].reset_index(drop=True)

    X = X.reset_index(drop=True)
    last_reset = last.reset_index(drop=True)

    # Get the main prediction (using regime-routed model)
    prob_main = regime_router.predict_proba_by_regime(X, regime_vec)

    # Also score with each regime ensemble (for diagnostic diversity check)
    regime_scores = {}
    for r in REGIMES:
        if r in regime_router.regime_models:
            m = regime_router.regime_models[r]
            regime_scores[r] = m.predict_proba(X)[:, 1]
        else:
            regime_scores[r] = np.full(len(X), np.nan)

    # Per-member std (computed using the ROUTED model's members for each row)
    prob_std = np.zeros(len(X))
    for r in REGIMES:
        mask = (regime_vec.values == r)
        if not mask.any():
            continue
        model = regime_router.regime_models.get(r) or regime_router.fallback
        if model is None or not hasattr(model, "members"):
            continue
        # BUGFIX: individual ensemble MEMBERS (CalibratedClassifierCV) do NOT
        # self-project to the trained feature_list — only the EnsembleCalibrator
        # wrapper does. With pruned features, members were trained on the
        # ensemble's feature_list (e.g. 137 cols) but X here has all 191, so
        # calling member.predict_proba(X) directly raised a LightGBM shape error.
        # Project X to the ensemble's columns first (mirrors EnsembleCalibrator._project).
        X_sub = X.iloc[mask]
        fl = getattr(model, "feature_list", None)
        if fl is not None:
            try:
                X_sub = X_sub.reindex(columns=fl)
            except Exception:
                pass
        member_probs = np.array([m.predict_proba(X_sub)[:, 1] for m in model.members])
        prob_std[mask] = member_probs.std(axis=0)

    # Expected return per row based on which regime model scored it
    expected_adj = np.full(len(X), np.nan)
    exp3 = np.full(len(X), np.nan)
    exp5 = np.full(len(X), np.nan)
    exp_sh = np.full(len(X), np.nan)
    for r in REGIMES:
        mask = (regime_vec.values == r)
        if not mask.any():
            continue
        iso_ev = regime_iso_ev_map.get(r) or fallback_iso_ev
        calib = regime_calib_tables.get(r) or fallback_calib_table
        if iso_ev is not None:
            expected_adj[mask] = iso_ev.predict(prob_main[mask])
        e3, e5, esh = map_prob_to_expectations(prob_main[mask], calib)
        exp3[mask] = e3
        exp5[mask] = e5
        exp_sh[mask] = esh

    wl = last_reset.copy()
    wl["regime_used"] = regime_vec
    wl["prob_5d_mean"] = prob_main
    wl["prob_5d_std"] = prob_std
    for r in REGIMES:
        wl[f"prob_5d_{r}"] = regime_scores[r]
    wl["expected_ret_5d_adj"] = expected_adj
    wl["expected_ret_5d"] = exp5
    wl["expected_ret_3d"] = exp3
    wl["expected_sharpe_5d"] = exp_sh

    pre_universe = len(wl)
    wl = wl[(wl["close"] >= MIN_CLOSE) & (wl["avg20_vol"] >= MIN_AVG20_VOL)].copy()
    print(f"[Watchlist] After universe filter (close>={MIN_CLOSE}, vol>={MIN_AVG20_VOL}): "
          f"{len(wl):,} rows ({pre_universe-len(wl)} removed)")

    # Optional: keep only names the ensemble AGREES on (low member disagreement).
    if WATCHLIST_MAX_PROB_STD and WATCHLIST_MAX_PROB_STD > 0 and "prob_5d_std" in wl.columns:
        pre_agree = len(wl)
        wl = wl[wl["prob_5d_std"] <= float(WATCHLIST_MAX_PROB_STD)].copy()
        print(f"[Watchlist] After agreement filter (prob_5d_std<={WATCHLIST_MAX_PROB_STD}): "
              f"{len(wl):,} rows ({pre_agree-len(wl)} removed)")

    wl = wl.sort_values(["prob_5d_mean", "expected_ret_5d_adj"], ascending=[False, False])

    # Build consolidated CSV
    output_cols = [
        "symbol", "timestamp", "close", "avg20_vol",
        "regime_used", "prob_5d_mean", "prob_5d_std",
        "prob_5d_bull_trend", "prob_5d_bull_range", "prob_5d_bear_trend", "prob_5d_bear_range",
        "expected_ret_5d_adj", "expected_ret_5d", "expected_ret_3d", "expected_sharpe_5d",
        "regime_market_trend", "regime_high_vol", "regime_dispersion",
        "D_atr14", "D_cpr_width_pct", "long_score", "short_score"
    ]
    # Add macro features if present
    for mc in ["M_nifty_ret", "M_nifty_ret_5d", "M_vix", "M_vix_change", "M_vix_level_z60"]:
        if mc in wl.columns:
            output_cols.append(mc)
    output_cols = [c for c in output_cols if c in wl.columns]
    wl_out = wl[output_cols].copy()

    ts = pd.Timestamp.now(tz="Asia/Kolkata").strftime("%Y%m%d_%H%M%S")
    base = Path(WATCHLIST_OUT)
    ts_csv = base.with_name(base.stem + f"_{ts}").with_suffix(".csv")
    wl_out.to_csv(ts_csv, index=False)
    print(f"[Watchlist] Saved consolidated: {ts_csv} ({len(wl_out)} rows)")

    ts_xlsx = None
    if export_xlsx:
        try:
            ts_xlsx = base.with_name(base.stem + f"_{ts}").with_suffix(".xlsx")
            wl_out.to_excel(ts_xlsx, index=False, engine="openpyxl")
            print(f"[Watchlist] Saved Excel: {ts_xlsx}")
        except Exception as e:
            print(f"[Watchlist] Excel export failed: {e}")

    # Save per-regime watchlists
    indiv_dir = base.parent / f"watchlist_per_regime_{ts}"
    indiv_dir.mkdir(exist_ok=True)
    for r in REGIMES:
        sub = wl[wl["regime_used"] == r].copy()
        sub = sub.sort_values("prob_5d_mean", ascending=False)
        sub.to_csv(indiv_dir / f"{r}.csv", index=False)
        print(f"[Watchlist] Saved {r}: {len(sub)} rows")
    print(f"[Watchlist] Per-regime watchlists in: {indiv_dir}")

    return wl_out, str(ts_csv), (str(ts_xlsx) if ts_xlsx else None)

# ===================== Quick Portfolio (optional smoke) =====================

def quick_portfolio_backtest(panel: pd.DataFrame, watchlist: pd.DataFrame, horizon: int = 5, top_k: int = 20, cost_bps: float = 20/10000):
    panel = panel.copy(); watchlist = watchlist.copy()
    panel["date"] = pd.to_datetime(panel["timestamp"]).dt.normalize()
    watchlist["date"] = pd.to_datetime(watchlist["timestamp"]).dt.normalize()
    equity = [1.0]
    prev_hold = set()
    for d, wl in watchlist.groupby("date"):
        wl = wl.sort_values("expected_ret_5d_adj", ascending=False).head(top_k)
        syms = set(wl["symbol"])
        added = syms - prev_hold
        turnover = len(added) / max(1, len(prev_hold) or 1)
        rets = []
        for s in syms:
            hist = panel[panel["symbol"] == s]
            row = hist[hist["date"] == d]
            if row.empty: continue
            i = row.index[0]
            if i+1 >= len(hist) or i+1+horizon >= len(hist): continue
            entry = hist.iloc[i+1]["open"]
            exitp = hist.iloc[i+1+horizon]["close"]
            rets.append((exitp-entry)/entry)
        pnl = np.mean(rets) if rets else 0.0
        pnl -= turnover * cost_bps
        equity.append(equity[-1]*(1+pnl))
        prev_hold = syms
    return pd.Series(equity)

# ===================== GUARANTEED JOBLIB EXPORT (FALLBACKS) =====================

class _ConstantProbClassifier:
    """Minimal classifier with predict_proba; returns a constant probability."""
    def __init__(self, p: float = 0.5):
        self.p = float(max(0.0, min(1.0, p)))
    def predict_proba(self, X):
        import numpy as _np
        n = len(X) if hasattr(X, "__len__") else 1
        p = _np.full(n, self.p, dtype=float)
        return _np.stack([1.0 - p, p], axis=1)

class _ConstantIsoEVMapper:
    """Minimal 'iso' mapper with predict(prob)->constant EV (percentage)."""
    def __init__(self, constant_ev: float = 0.0):
        self.constant_ev = float(constant_ev)
    def predict(self, prob_vec):
        import numpy as _np
        n = len(prob_vec) if hasattr(prob_vec, "__len__") else 1
        return _np.full(n, self.constant_ev, dtype=float)

class _EVShimRegressorFallback:
    """Drop-in EV shim: .predict(X)->expected 5D adjusted return (pct)."""
    def __init__(self, calibrator, iso_ev, features, impute):
        self.calibrator = calibrator
        self.iso_ev = iso_ev
        self.features = list(features or [])
        self.impute = dict(impute or {})
    def predict(self, X):
        try:
            from cpr_fix_patched import sanitize_feature_matrix, reindex_and_impute  # this module
        except Exception:
            # Inline safe versions
            def sanitize_feature_matrix(df):
                df = df.copy()
                if df.columns.duplicated().any():
                    df = df.loc[:, ~df.columns.duplicated()]
                for c in df.columns:
                    s = df[c]
                    if s.dtype == bool:
                        df[c] = s.astype(int)
                    else:
                        df[c] = pd.to_numeric(s, errors="coerce")
                return df
            def reindex_and_impute(df, feats, imp):
                df2 = df.reindex(columns=feats)
                for c in df2.columns:
                    df2[c] = pd.to_numeric(df2[c], errors="coerce").fillna(imp.get(c, 0.0))
                return df2
        Xp = sanitize_feature_matrix(X)
        Xp = reindex_and_impute(Xp, self.features, self.impute)
        prob = self.calibrator.predict_proba(Xp)[:, 1]
        ev = self.iso_ev.predict(prob)
        return ev.astype(float)

def _models_dir_for(out_dir: str) -> Path:
    md = Path(out_dir) / "models"
    md.mkdir(parents=True, exist_ok=True)
    return md

def _expected_model_paths(out_dir: str) -> dict:
    md = _models_dir_for(out_dir)
    return {
        "ev_shim": md / "m5_reg_shim.joblib",
        "m5_cls": md / "m5_classifier.joblib",
        "m1_gate": md / "m1_gate.joblib",
        "iso_ev": md / "iso_ev_mapper.joblib",
    }

def _features_schema_or_empty() -> tuple[list, dict]:
    try:
        feats, impute = load_schema(FEATURES_SCHEMA_PATH)
        return feats, impute
    except Exception:
        return [], {}

def _write_fallback_models(out_dir: str):
    """Create stub-compatible joblibs ONLY if missing. Does not overwrite existing real models."""
    paths = _expected_model_paths(out_dir)
    if all(Path(p).exists() for p in paths.values()):
        return
    feats, impute = _features_schema_or_empty()
    m5_cls = _ConstantProbClassifier(p=0.5)
    m1_gate = _ConstantProbClassifier(p=0.6)
    iso_ev = _ConstantIsoEVMapper(constant_ev=0.0)
    ev_shim = _EVShimRegressorFallback(m5_cls, iso_ev, feats, impute)
    if not paths["ev_shim"].exists():
        joblib.dump(ev_shim, paths["ev_shim"])
        print(f"[Models:FALLBACK] Saved: {paths['ev_shim']}")
    if not paths["m5_cls"].exists():
        joblib.dump(m5_cls, paths["m5_cls"])
        print(f"[Models:FALLBACK] Saved: {paths['m5_cls']}")
    if not paths["m1_gate"].exists():
        joblib.dump(m1_gate, paths["m1_gate"])
        print(f"[Models:FALLBACK] Saved: {paths['m1_gate']}")
    if not paths["iso_ev"].exists():
        joblib.dump(iso_ev, paths["iso_ev"])
        print(f"[Models:FALLBACK] Saved: {paths['iso_ev']}")

def _atexit_ensure_joblibs():
    # AUDIT-FIX 7: previously this was atexit-registered. That meant a crash
    # mid-training would silently fill the models/ dir with constant-probability
    # stubs (m5_classifier→0.5, m1_gate→0.6, iso_ev→0), and any subsequent
    # backtest/watchlist load would succeed against meaningless models. We now
    # leave this as a callable helper but DO NOT auto-register it. The explicit
    # call inside run_pipeline at successful completion is preserved and only
    # fills genuinely-missing files (it never overwrites real artifacts).
    try:
        if _LAST_OUT_DIR:
            _write_fallback_models(_LAST_OUT_DIR)
    except Exception as e:
        print(f"[Models:FALLBACK] Failed to write fallback joblibs: {e!r}")

# AUDIT-FIX 7: do NOT atexit-register the fallback writer. A failed training
# run should leave missing files so downstream loads fail loudly instead of
# returning constant-probability garbage.
# atexit.register(_atexit_ensure_joblibs)  # intentionally disabled

# ===================== PICKLE-SAFE EV SHIM (TOP LEVEL) =====================
class EVShimRegressor:
    """Fast 5D expected-return predictor using calibrated classifier + isotonic EV map.
    Defined at module top level so joblib can pickle it.
    """
    def __init__(self, calibrator, iso_ev, features, impute):
        self.calibrator = calibrator
        self.iso_ev = iso_ev
        self.features = list(features or [])
        self.impute = dict(impute or {})
    def predict(self, X):
        Xp = sanitize_feature_matrix(X)
        Xp = reindex_and_impute(Xp, self.features, self.impute)
        prob = self.calibrator.predict_proba(Xp)[:, 1]
        ev = self.iso_ev.predict(prob)
        return ev.astype(float)

# ===================== PUBLIC ENTRY POINT =====================

def run_pipeline(*,
                 data_dir: str,
                 out_dir: str,
                 symbols_like: Optional[str]=None,
                 limit_files: Optional[int]=None,
                 accept_any_daily: bool=False,
                 load_workers: int=8,
                 cv_splits: int=FOLDS,
                 embargo_days: int=EMBARGO_DAYS,
                 early_stopping_rounds: int=EARLY_STOPPING_ROUNDS,
                 # Clean, single-line, case-insensitive alternation. Matches these at the END of the symbol.
                 exclude_pattern: Optional[str] = (
                     r"(?i)(?:LIQUIDPLUS|GROWWLIQID|GLOBUSSPR|MAHKTECH|LIQUID1|"
                     r"GROWWNIFTY|LIQUID|GOLD|BEES|IETF|ETF|CASE|ADD)$"
                 ),
                 export_xlsx: bool=False,
                 quick_portfolio: bool=False,
                 portfolio_topk: int=20,
                 ev_target: str = "oc"  # 'cc' or 'oc'
                 ) -> Dict[str, object]:
    """
    End-to-end pipeline with NO CLI. Import and call from PyCharm / your script.
    Returns a dict with references to panel, features, models, watchlist, and report paths.
    """
    setup_paths(out_dir)
    # 0) Collect panel (keeps ALL rows)
    paths = _strict_file_list(data_dir, symbols_like, limit_files, accept_any_daily=accept_any_daily)
    panel, feats = collect_panel_from_paths(paths, load_workers=load_workers)
    feats = [f for f in feats if "__dup" not in f]
    assert not any("__dup" in f for f in feats)
    # AUDIT-FIX 9: hard invariants — fail loudly if a forward-return / future-
    # looking column ever ends up in the feature list. discover_daily_features
    # already excludes them by name pattern; this is a tripwire for refactors.
    _LEAK_PREFIXES = ("ret_", "mfe_", "mae_")
    _LEAK_NAMES = {"ret_5d_adj", "ret_3d_adj", "rank_5d_pct",
                   "top20_vs_bot20_5d", "top20_strict_5d"}
    leaky = [f for f in feats
             if any(f.startswith(p) for p in _LEAK_PREFIXES) or f in _LEAK_NAMES]
    assert not leaky, f"[LEAK GUARD] Forward-looking columns reached feats: {leaky}"
    # No duplicate column names in panel.
    panel_cols = list(panel.columns)
    assert len(panel_cols) == len(set(panel_cols)), \
        f"[LEAK GUARD] panel has duplicate columns: " \
        f"{[c for c in panel_cols if panel_cols.count(c) > 1][:10]}"
    # ---- Explicit model feature contract (OPTION A) ----
    from pathlib import Path

    if FEATURES_SCHEMA_PATH and Path(FEATURES_SCHEMA_PATH).exists():
        feats_schema, _ = load_schema(FEATURES_SCHEMA_PATH)
        # AUDIT-FIX 8: surface schema-vs-panel drift. The previous comparison
        # in fit_final_model_and_oos_calibration was feats-vs-itself (because
        # we override `feats = feats_schema` immediately afterwards) and could
        # never report new features. Compare BEFORE the override so users can
        # see what the panel produced that the schema is currently ignoring.
        new_in_panel = [f for f in feats if f not in set(feats_schema)]
        missing_from_panel = [f for f in feats_schema if f not in set(feats)]
        if new_in_panel:
            print(f"[Schema] Panel has {len(new_in_panel)} feature(s) NOT in schema (will be IGNORED):")
            for f in new_in_panel[:15]:
                print(f"          - {f}")
            if len(new_in_panel) > 15:
                print(f"          ... and {len(new_in_panel)-15} more")
            print(f"[Schema] To pick them up, delete features_train.json and re-run.")
        if missing_from_panel:
            print(f"[Schema] Schema lists {len(missing_from_panel)} feature(s) NOT in panel "
                  f"(will be filled with train-only impute medians at scoring time):")
            for f in missing_from_panel[:15]:
                print(f"          - {f}")
        print(f"[Schema] Using {len(feats_schema)} curated features from features_train.json")
        feats = feats_schema
    else:
        print("[Schema] No features_train.json found — model will use auto-discovered features")
    print(f"[Panel] final rows={len(panel)} cols={len(panel.columns)} feats={len(feats)}")
    # ================= MEMORY HARDENING (CRITICAL) =================
    # Prevent pandas block consolidation OOM on wide panels

    panel["symbol"] = panel["symbol"].astype("category")

    float_cols = panel.select_dtypes(include=["float64"]).columns
    panel[float_cols] = panel[float_cols].astype("float32")
    # ===============================================================

    # 1) Early excludes for MODELING
    if exclude_pattern:
        mask = ~panel["symbol"].astype(str).str.contains(
            exclude_pattern, regex=True, na=False
        )
        panel_train = panel.loc[mask]  # ✅ NO .copy()
    else:
        panel_train = panel.copy()

    # 1b) UPSTREAM structural liquidity filter (applied to TRAINING rows only).
    # These rows are a different data-generating process we will never trade and
    # they pollute the per-day cross-sectional rank label. Filtering here (not just
    # at the watchlist) keeps the labels clean. The full `panel` is left intact so
    # the watchlist can still consider everything and apply its own point-in-time gates.
    if TRAIN_MIN_CLOSE is not None or TRAIN_MIN_AVG20_VOL is not None:
        pt = panel_train.copy()
        pt["_avg20_vol"] = pt.groupby("symbol", observed=True)["volume"].transform(
            lambda s: s.rolling(20, min_periods=1).mean())
        keep = pd.Series(True, index=pt.index)
        if TRAIN_MIN_CLOSE is not None:
            keep &= pd.to_numeric(pt["close"], errors="coerce") >= float(TRAIN_MIN_CLOSE)
        if TRAIN_MIN_AVG20_VOL is not None:
            keep &= pd.to_numeric(pt["_avg20_vol"], errors="coerce") >= float(TRAIN_MIN_AVG20_VOL)
        n_before = len(pt)
        panel_train = pt.loc[keep].drop(columns=["_avg20_vol"], errors="ignore")
        print(f"[Train Filter] Upstream liquidity filter "
              f"(close>={TRAIN_MIN_CLOSE}, avg20_vol>={TRAIN_MIN_AVG20_VOL}): "
              f"{len(panel_train):,} of {n_before:,} training rows kept")

    # 2) CV diagnostics for the 5d classifier
    _oos_prob_cv, _valid_idx_cv, _pl_cv = train_5d_quantile_cls(panel_train, feats, ev_target,
        n_splits=cv_splits, embargo_days=embargo_days, early_stopping_rounds=early_stopping_rounds)

    # 3) Final model + OOS-only calibrations + TRAIN-only imputation schema
    final_calib, iso_ev, oos_df = fit_final_model_and_oos_calibration(panel_train, feats, ev_target,
        early_stopping_rounds=early_stopping_rounds)

    # 4) Build a simple decile table from TEST
    df_oos = oos_df.dropna(subset=["prob_top20_5d"]).copy()
    df_oos["prob_bucket"] = pd.qcut(df_oos["prob_top20_5d"], q=min(10, max(3, df_oos.shape[0]//50)), labels=False, duplicates="drop")
    calib = (df_oos.groupby("prob_bucket")
        .agg(avg_prob=("prob_top20_5d","mean"),
             avg_ret_3d=("ret_3d_close_pct","mean"),
             avg_ret_5d=("ret_5d_close_pct","mean"),
             std_ret_5d=("ret_5d_close_pct","std"),
             avg_ret_5d_adj=("ret_5d_adj","median"))
        .reset_index().sort_values("prob_bucket"))
    # CORRECTNESS FIX: use the real mean predicted probability per bucket for
    # prob_mid (was a synthetic np.linspace grid) and a robust MEDIAN for the
    # vol-adjusted return. See _build_calib_table_from_oos for the full rationale.
    calib["prob_mid"] = calib["avg_prob"].astype(float)
    calib["exp_sharpe_5d"] = calib["avg_ret_5d_adj"].astype(float)
    calib_table = {
        "prob_mid": calib["prob_mid"].tolist(),
        "avg_ret_3d": calib["avg_ret_3d"].tolist(),
        "avg_ret_5d": calib["avg_ret_5d"].tolist(),
        "std_ret_5d": calib["std_ret_5d"].fillna(0.0).tolist(),
        "exp_sharpe_5d": calib["exp_sharpe_5d"].tolist(),
    }
    Path(CALIB_TABLE_PATH).write_text(json.dumps(calib_table, indent=2))

    # 5) OOS report
    from sklearn.metrics import brier_score_loss
    from scipy.stats import spearmanr
    target_proxy = np.where(df_oos["rank_5d_pct"] >= 0.8, 1, np.where(df_oos["rank_5d_pct"] <= 0.2, 0, np.nan))
    msk = ~np.isnan(target_proxy)
    brier = float(brier_score_loss(pd.Series(target_proxy)[msk], df_oos.loc[msk, "prob_top20_5d"]))
    ic5, ic5_p = spearmanr(df_oos["prob_top20_5d"], df_oos["ret_5d_adj"], nan_policy="omit")
    rep = {"5d_cls": {"n_oos": int(len(df_oos)), "brier_pseudo": brier,
                       "spearman_ic_5d_adj": float(ic5), "spearman_ic_pvalue": float(ic5_p),
                       "ev_target": ev_target, "embargo_days": int(embargo_days)}}
    Path(OOS_REPORT_PATH).write_text(json.dumps(rep, indent=2))
    print(f"[OOS] Saved report: {OOS_REPORT_PATH}")
    print(f"[Calibration] Saved table: {CALIB_TABLE_PATH}")

    # 6) Train model(s) per MODEL_ARCHITECTURE.
    #    fit_regime_ensembles ALWAYS trains the fallback (= pooled single model).
    #    It additionally trains the 4 regime specialists unless we tell it not to.
    #    We pass train_regime_specialists=False for the "single" architecture so we
    #    don't waste time training specialists we won't use.
    train_specialists = (MODEL_ARCHITECTURE in ("regime", "both"))
    print(f"\n[Architecture] MODEL_ARCHITECTURE={MODEL_ARCHITECTURE!r}  "
          f"PRIMARY_ARCHITECTURE={PRIMARY_ARCHITECTURE!r}  "
          f"(train specialists: {train_specialists})")

    # ---- PRUNER WIRING: load regime_features.json if present ----
    # Produced by feature_imp_v5.py. Gives:
    #   keep_list_global -> pruned feature list for the POOLED model
    #   per_regime[r]    -> pruned feature list for each regime specialist
    # When USE_PRUNED_FEATURES is on and the file exists, the pooled model and each
    # specialist train on their own curated lists (this is how trend-regime feature
    # power is preserved). Any feature not present in the panel is simply skipped.
    per_regime_feats = None
    pruned_global_feats = None
    if USE_PRUNED_FEATURES:
        # Look in out_dir first, then the pruner's default feature_diagnostics/ subdir,
        # so you don't have to copy the file by hand after running feature_imp_v5.py.
        candidates = [Path(out_dir) / "regime_features.json",
                      Path(out_dir) / "feature_diagnostics" / "regime_features.json"]
        rf_path = next((p for p in candidates if p.exists()), None)
        if rf_path is not None:
            try:
                rf = json.loads(rf_path.read_text())
                print(f"[Pruner] Using {rf_path}")
                panel_cols = set(panel_train.columns)
                def _avail(lst):
                    return [c for c in (lst or []) if c in panel_cols]
                kg = _avail(rf.get("keep_list_global"))
                if kg:
                    pruned_global_feats = kg
                pr = {}
                for r, lst in (rf.get("per_regime") or {}).items():
                    a = _avail(lst)
                    if a:
                        pr[r] = a
                if pr:
                    per_regime_feats = pr
                print(f"[Pruner] Loaded regime_features.json "
                      f"(produced_at={rf.get('produced_at','?')}): "
                      f"global keep={len(pruned_global_feats or [])}, "
                      f"per-regime={ {k: len(v) for k,v in (per_regime_feats or {}).items()} }")
                missing_g = len(rf.get("keep_list_global") or []) - len(pruned_global_feats or [])
                if missing_g > 0:
                    print(f"[Pruner] NOTE: {missing_g} global keep features not in panel; skipped.")
            except Exception as e:
                print(f"[Pruner] WARN: could not parse regime_features.json: {e}. Using full feats.")
        else:
            print(f"[Pruner] USE_PRUNED_FEATURES=True but no regime_features.json found in "
                  f"{out_dir} or its feature_diagnostics/ subdir. Run feature_imp_v5.py first; "
                  f"using full feature set for now.")

    (regime_models, regime_iso_ev, regime_calib_tables,
     fallback_ensemble, fallback_iso_ev_v2, fallback_calib_table_v2,
     fallback_oos_df) = fit_regime_ensembles(
        panel_train, feats, ev_target,
        n_members=ENSEMBLE_SIZE_PER_REGIME,
        early_stopping_rounds=early_stopping_rounds,
        use_strict_label=USE_STRICT_LABEL,
        train_regime_specialists=train_specialists,
        per_regime_feats=per_regime_feats,
        global_feats=pruned_global_feats,
    )

    # Build the two candidate routers.
    # single  -> empty regime_models => RegimeRouter sends every row to fallback (pooled).
    # regime  -> full regime_models  => specialists where available, fallback elsewhere.
    router_single = RegimeRouter(regime_models={}, fallback_ensemble=fallback_ensemble)
    router_regime = RegimeRouter(regime_models=regime_models, fallback_ensemble=fallback_ensemble)

    # 6b) If requested, compare BOTH architectures on the same embargoed OOS rows.
    arch_report = {}
    if MODEL_ARCHITECTURE == "both" and train_specialists:
        try:
            arch_report = compare_architectures_oos(
                panel_train, feats, ev_target,
                regime_router=router_regime,
                fallback_ensemble=fallback_ensemble,
                use_strict_label=USE_STRICT_LABEL,
            )
            try:
                Path(out_dir, "architecture_comparison.json").write_text(
                    json.dumps(arch_report, indent=2, default=float))
            except Exception as _e:
                log(f"[Compare] could not save architecture_comparison.json: {_e}", level="warning")
        except Exception as e:
            print(f"[Compare] Architecture comparison failed: {e}")

    # Choose which router to EXPORT / SCORE with.
    if PRIMARY_ARCHITECTURE == "regime" and train_specialists:
        regime_router = router_regime
        active_regime_models = regime_models
        print("[Architecture] Using REGIME-routed specialists for export + watchlist.")
    else:
        regime_router = router_single
        active_regime_models = {}  # pooled model handles everything
        print("[Architecture] Using SINGLE pooled model for export + watchlist.")

    # 7) Watchlist using the chosen router
    wl, wl_csv, wl_xlsx = nightly_watchlist(
        panel, feats, regime_router,
        regime_iso_ev_map=regime_iso_ev,
        regime_calib_tables=regime_calib_tables,
        fallback_iso_ev=fallback_iso_ev_v2,
        fallback_calib_table=fallback_calib_table_v2,
        exclude_pattern=exclude_pattern,
        export_xlsx=export_xlsx,
    )

    # 8) Optional quick portfolio sanity-check
    eq_path = None
    if quick_portfolio:
        eq = quick_portfolio_backtest(panel, wl, horizon=5, top_k=int(portfolio_topk))
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8,4))
            plt.plot(eq.values)
            plt.title("Quick Portfolio (Top-K by expected_ret_5d_adj)")
            plt.xlabel("Days"); plt.ylabel("Equity")
            out_png = Path(Path(WATCHLIST_OUT).parent) / "quick_portfolio_equity.png"
            plt.tight_layout(); plt.savefig(out_png, dpi=120)
            print(f"[QuickPF] Saved: {out_png}")
            eq_path = str(out_png)
        except Exception as e:
            print(f"[QuickPF] Plot failed: {e}")

    # 9) EXPORT MODELS (joblib files) — PATCHED: top-level EVShimRegressor + safe dumps
    model_dir = Path(out_dir) / "models"
    model_dir.mkdir(exist_ok=True)

    def _safe_dump(obj, path: Path, label: str) -> bool:
        try:
            joblib.dump(obj, path)
            print(f"[Models] Saved: {path}")
            return True
        except Exception:
            import traceback
            print(f"[ERROR] Saving {label} failed -> {path}\n{traceback.format_exc()}")
            return False

    feats_schema, impute_from_schema = load_schema(FEATURES_SCHEMA_PATH)

    # v4: final_calib and iso_ev for the EVShimRegressor (backwards-compat with feature_imp & backtest)
    # use the fallback (all-regime) ensemble's first member.
    if fallback_ensemble is not None and hasattr(fallback_ensemble, "members") and len(fallback_ensemble.members) > 0:
        final_calib = fallback_ensemble.members[0]
    else:
        final_calib = None
    iso_ev = fallback_iso_ev_v2

    # BUGFIX: the EVShim / standalone classifier must use the SAME feature list the
    # member was trained on. With pruning, the fallback ensemble trains on its
    # pruned feature_list (e.g. 137), NOT the full 191-feature schema. Building the
    # shim with feats_schema (191) and a 137-feature member crashes at predict time
    # ("number of features 191 != 137"). Use the ensemble's own feature_list when present.
    shim_feats = getattr(fallback_ensemble, "feature_list", None) or feats_schema
    shim_impute = {k: v for k, v in impute_from_schema.items() if k in set(shim_feats)} \
        if shim_feats else impute_from_schema
    print(f"[Models] EVShim/classifier feature list: {len(shim_feats)} features "
          f"({'pruned ensemble list' if getattr(fallback_ensemble,'feature_list',None) else 'full schema'})")

    ok_all = True
    if final_calib is not None:
        ok_all &= _safe_dump(EVShimRegressor(final_calib, iso_ev, shim_feats, shim_impute),
                             model_dir / "m5_reg_shim.joblib", "EV shim regressor")
        ok_all &= _safe_dump(final_calib, model_dir / "m5_classifier.joblib", "m5 classifier (fallback member 1)")
    if iso_ev is not None:
        ok_all &= _safe_dump(iso_ev, model_dir / "iso_ev_mapper.joblib", "iso EV mapper")

    # v4: save pooled (fallback) ensemble + the router that reflects the CHOSEN
    # architecture (router_single has empty regime_models -> pooled for all rows;
    # router_regime carries the specialists). active_regime_models is {} when the
    # single architecture is chosen, so we only persist specialists we actually use.
    ok_all &= _safe_dump(fallback_ensemble, model_dir / "m5_ensemble.joblib", "pooled (fallback) ensemble")
    for r, m in active_regime_models.items():
        ok_all &= _safe_dump(m, model_dir / f"m5_regime_{r}.joblib", f"regime ensemble: {r}")
    ok_all &= _safe_dump(regime_router, model_dir / "m5_regime_router.joblib",
                         f"regime router ({PRIMARY_ARCHITECTURE})")
    # Save iso_ev mappers per regime (only those for active specialists)
    for r, ie in regime_iso_ev.items():
        if r in active_regime_models:
            ok_all &= _safe_dump(ie, model_dir / f"iso_ev_{r}.joblib", f"iso_ev mapper: {r}")
    # Save calib tables per regime as JSON for backtest tooling
    try:
        regime_calib_json_path = model_dir / "regime_calib_tables.json"
        json_payload = {
            "fallback": fallback_calib_table_v2,
            "regimes": regime_calib_tables,
        }
        Path(regime_calib_json_path).write_text(json.dumps(json_payload, indent=2, default=float))
        print(f"[Models] Saved: {regime_calib_json_path}")
    except Exception as e:
        print(f"[Models] Could not save regime_calib_tables.json: {e}")

    # Ensure fallbacks exist for any missing ones (no-op if all saved above)
    _write_fallback_models(out_dir)

    return {
        "panel": panel,
        "features": feats,
        "final_calibrator": final_calib,
        "iso_ev_mapper": iso_ev,
        "regime_router": regime_router,
        "regime_models": regime_models,
        "fallback_ensemble": fallback_ensemble,
        "regime_iso_ev": regime_iso_ev,
        "regime_calib_tables": regime_calib_tables,
        "architecture": PRIMARY_ARCHITECTURE,
        "architecture_comparison": arch_report,
        "smoke_test_passed": True,
        "watchlist": wl,
        "oos_report_path": OOS_REPORT_PATH,
        "calibration_table_path": CALIB_TABLE_PATH,
        "panel_path": PANEL_OUT,
        "watchlist_path": WATCHLIST_OUT,     # base name
        "watchlist_csv": wl_csv,             # actual saved CSV
        "watchlist_xlsx": wl_xlsx,           # actual saved XLSX (if any)
        "equity_plot_path": eq_path,
        "models_dir": str(model_dir),
    }

if __name__ == "__main__":
    print(
        """This module exposes run_pipeline(...). Import and call it from your script.\nExample:\nfrom cpr_fix_patched import run_pipeline\nres = run_pipeline(\n  data_dir=r"C:\\path\\to\\cache",\n  out_dir=r"C:\\path\\to\\out",\n  ev_target="cc"\n)\n"""
    )
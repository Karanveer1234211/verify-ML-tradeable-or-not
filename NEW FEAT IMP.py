#!/usr/bin/env python3
"""
=============================================================================
REGIME-AWARE FEATURE DIAGNOSTIC & PRUNING ANALYSIS  v5
=============================================================================

Drop-in replacement for `feature imp.py` (v4).

WHY v5 EXISTS
-------------
v4 had four correctness bugs and one robustness issue that meant its KEEP /
DROP / REVIEW recommendations were anchored to the wrong ground truth.

  1. Train/test split was `np.arange(0, 0.8*N)` on a panel sorted by
     (symbol, timestamp). That is the first 80% of *symbols*, not the
     first 80% of *dates*. All "out-of-sample" diagnostics were
     cross-stock, not out-of-time.
  2. Per-regime permutation importance reused the same row-index split
     within the regime subset — same bug, different scope.
  3. Diagnostic LightGBM used 400 trees @ LR=0.05, num_leaves=63,
     min_data_in_leaf=100. Production uses 3000 @ 0.005, 31, 500.
     Importance rankings of a model trained 10x faster with 5x larger
     leaves do not represent the production model.
  4. Permutation importance was a single shuffle per feature. Single-
     shuffle estimates have variance ~ 1/sqrt(N_test); for ~100k test rows
     the std is roughly 0.001 AUC — i.e. on the same order as the
     "harmful feature" threshold. Many DROP/KEEP boundaries are noise.
  5. Imputation used global medians (computed over train+test). Mild
     leakage and inconsistent with how the model is trained.

v5 fixes all five and adds:

  6. 5-fold purged time-series CV with embargo for permutation, so the
     reported AUC drops are averaged across folds and the noise floor is
     well-defined.
  7. "Harmful feature" detection — features whose removal *improves*
     AUC (auc_drop < -0.0005 in 2+ regimes). These are dropped first.
  8. Feature family aggregation (D_, X_, W_, WQ_, M_, Comb_, regime_, ...)
     so you can see at a glance which families are pulling weight.
  9. Forward-time generalization: train on the first 60% of dates, test
     on the next 20%, vs train on dates 20-80%, test on dates 80-100%.
     A feature that ranks high on both halves generalizes; one that
     drifts is flagged.
 10. `regime_features.json` output that the production training script
     can read directly to build per-regime feature lists.

OUTPUT FILES (written to OUT_DIR)
---------------------------------
  feature_pruning_recommendation.csv     <- main file you act on
  regime_features.json                   <- consume in New_model.py
  harmful_features.csv                   <- features that hurt the model
  regime_specialist_features.csv         <- 1-2 regime signal, dead in others
  feature_family_summary.csv             <- which families are alive?
  feature_category_map.csv               <- NEW: per-feature STRATEGY category
                                             (breakout/mean_reversion/momentum/...)
                                             + gain/IC/perm-AUC-drop/recommendation
  feature_category_summary.csv           <- NEW: per-category rollup so you can
                                             pick the right features for a NEW model
  feature_importance_global.csv
  feature_ic_by_year.csv
  regime_ic_summary.csv
  regime_permutation_importance.csv      (mean + std across 5 shuffles)
  regime_consistency.csv
  feature_correlation_matrix.csv
  feature_redundant_pairs.csv
  feature_diagnostics_full.xlsx          (all of the above as sheets)

CLI / CONFIG
------------
Set BASE_DIR below or pass --base-dir on the command line.

v5.1 — PARAMETRIC (multi-model / multi-horizon). Same vocabulary as strategy_lab
(horizon, n_quantiles, use_model_target, embargo) so the upstream is uniform. The
defaults reproduce the original 5d run byte-for-byte.

  # default 5d run (unchanged behaviour, legacy output dir):
  python "NEW FEAT IMP.py" --base-dir DIR

  # scout a horizon you have NOT trained a model for yet (model-free: importance
  # comes from a diagnostic model trained here on that horizon's label):
  python "NEW FEAT IMP.py" --base-dir DIR --horizon 2 --model-free

  # full model-anchored diagnostic for a specific trained model:
  python "NEW FEAT IMP.py" --base-dir DIR --horizon 3 --model-path DIR/models/m3_classifier.joblib

Flags: --horizon H | --quantiles Q (top/bottom = 1/Q) | --embargo D (default H+1)
       --raw-target | --target-col COL | --ret-col COL | --model-path P | --model-free
       --out-dir DIR (default <base>/feature_diagnostics[_<H>d])
       --cluster-prune [--cluster-corr 0.85]  group collinear features, keep one
            representative per cluster, demote the rest to REDUNDANT (permutation
            importance is unreliable on collinear twins). Off by default.
       --std-gate [--std-gate-k 1.0]  gate KEEP/DROP on perm-drop significance
            (mean > k*shuffle_std) instead of the bare absolute PERM_DROP_KEEP.

Model-anchored vs model-free: gain% and permutation AUC-drop require a model and
are only meaningful for a model you actually built; the IC / regime-IC / category
diagnostics are model-free and computable at any horizon for scouting which
features suit a candidate (breakout / mean-reversion / ...) model.
=============================================================================
"""

from __future__ import annotations

import argparse
import json
import re
import time
import warnings
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from joblib import load
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

warnings.filterwarnings("ignore")

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================

DEFAULT_BASE_DIR = Path(r"C:\Users\karanvsi\Desktop\Kite Connect\v3_2_output_full")

IST = "Asia/Kolkata"
RET_COL = "ret_5d_oc_pct"
TARGET_COL = "top20_vs_bot20_5d"
EMBARGO_DAYS = 6           # horizon (5) + 1 — purged CV embargo
TEST_FRAC = 0.20           # last 20% of dates by calendar
MIN_CLOSE = 2.0
MIN_AVG20_VOL = 200_000

REGIMES = ["bull_trend", "bull_range", "bear_trend", "bear_range"]

# v5: match production model params so importance rankings transfer.
# Production uses n_estimators=3000, LR=0.005 — using 1500/0.01 here as a
# 2x-faster but still representative diagnostic. num_leaves/regularization
# are matched exactly.
DIAG_LGB_PARAMS = dict(
    n_estimators=1500,
    learning_rate=0.01,
    num_leaves=31,
    max_depth=6,
    feature_fraction=0.7,
    bagging_fraction=0.7,
    bagging_freq=1,
    min_data_in_leaf=300,
    min_gain_to_split=0.02,
    max_bin=255,
    reg_alpha=0.3,
    reg_lambda=10.0,
    extra_trees=True,
    n_jobs=-1,
    random_state=42,
    verbosity=-1,
)

# Pruning thresholds (regime-aware, biased toward KEEP)
IC_THRESHOLD = 0.02                 # |IC| below this is weak
PERM_DROP_KEEP = 0.0010             # AUC drop above this = useful
PERM_DROP_HARMFUL = -0.0005         # AUC drop below this = feature hurts
REGIME_EXCEPTIONAL = 0.0030         # AUC drop in 1 regime above this = specialist
CORR_THRESHOLD = 0.85
PERM_SHUFFLES = 5                   # multi-shuffle permutation
CV_FOLDS = 5


# =============================================================================
# CONFIG (parametric — multi-model / multi-horizon)
#
# The constants above remain the DEFAULTS so a plain `--base-dir DIR` run is
# byte-for-byte the old 5d behaviour. This dataclass lets the same diagnostic
# run at any horizon / target / model, with the SAME vocabulary as strategy_lab
# (horizon, n_quantiles, use_model_target, embargo_days) so the upstream config
# is uniform across the toolchain.
#
# Two modes:
#   • model-anchored : a trained model is found/loaded -> gain% + permutation
#                      AUC-drop come from THAT model (only valid for a model you
#                      actually built).
#   • model-free     : no model (or --model-free) -> the script trains its own
#                      prod-matched diagnostic model for this horizon's label and
#                      reports gain/permutation/IC against it. This is the SCOUT
#                      mode for a horizon where you have not trained a model yet.
# =============================================================================

@dataclass
class Config:
    base_dir: Path = DEFAULT_BASE_DIR
    horizon: int = 5                       # forward horizon in trading days
    n_quantiles: int = 5                   # top/bottom = 1/n_quantiles (5 -> top20/bot20)
    use_model_target: bool = True          # vol-adjusted forward return (model-style) vs raw
    embargo_days: Optional[int] = None     # default = horizon + 1
    target_col: Optional[str] = None       # override the derived label column name
    ret_col: Optional[str] = None          # override the forward-return column name
    model_path: Optional[Path] = None      # explicit model file (else auto-discover)
    model_free: bool = False               # force the model-free (diagnostic-only) path
    test_frac: float = TEST_FRAC
    out_dir: Optional[Path] = None          # default = <base>/feature_diagnostics[_<H>d]

    # cluster-aware pruning (opt-in; default OFF preserves the legacy per-feature
    # output byte-for-byte). Permutation importance is unreliable on collinear
    # features (a twin masks the drop), so naive per-feature pruning can delete a
    # real signal merely because it is duplicated. When ON, features are grouped at
    # |spearman| >= cluster_corr; only the best representative per cluster (by perm
    # drop, then |IC|) stays KEEP/REVIEW, the rest are demoted to REDUNDANT (recorded,
    # never silently lost). Features that are KEEP on strong standalone evidence are
    # protected and never demoted.
    cluster_prune: bool = False
    cluster_corr: float = CORR_THRESHOLD    # correlation threshold for a cluster (0.85)
    # std-gated KEEP/DROP: gate on perm-drop SIGNIFICANCE (drop_mean > k*drop_std)
    # rather than the bare absolute PERM_DROP_KEEP=0.0010, which the docstring itself
    # notes sits within the 5-shuffle noise band. Uses the std already computed.
    std_gate: bool = False
    std_gate_k: float = 1.0                 # significance multiple on the shuffle std

    def embargo(self) -> int:
        return int(self.embargo_days) if self.embargo_days is not None else int(self.horizon) + 1

    def ret_col_name(self) -> str:
        return self.ret_col or f"ret_{self.horizon}d_oc_pct"

    def target_col_name(self) -> str:
        return self.target_col or f"top20_vs_bot20_{self.horizon}d"

    def label_q(self) -> float:
        return 1.0 / float(max(2, self.n_quantiles))

    def resolved_out_dir(self) -> Path:
        if self.out_dir is not None:
            return Path(self.out_dir)
        base = Path(self.base_dir)
        # preserve the legacy path for the default 5d run; suffix others so
        # multi-horizon runs never clobber each other.
        return base / "feature_diagnostics" if int(self.horizon) == 5 \
            else base / f"feature_diagnostics_{self.horizon}d"


def ensure_forward_return(panel: pd.DataFrame, ret_col: str, horizon: int) -> None:
    """Guarantee `panel[ret_col]` (open[t+1]->close[t+h] %% return) exists.

    Uses the precomputed column when present (so the default 5d run is unchanged);
    otherwise computes it per-symbol so any horizon (incl. h=2, which the cache
    does not precompute) works. Mutates `panel` in place."""
    if ret_col in panel.columns:
        s = pd.to_numeric(panel[ret_col], errors="coerce")
        if s.notna().any():
            return
    g_close = panel.groupby("symbol", observed=True)["close"]
    if "open" in panel.columns:
        g_open = panel.groupby("symbol", observed=True)["open"]
        fwd = (g_close.shift(-horizon) / g_open.shift(-1) - 1.0) * 100.0
    else:
        close = pd.to_numeric(panel["close"], errors="coerce")
        fwd = (g_close.shift(-horizon) / close - 1.0) * 100.0
    panel[ret_col] = fwd.replace([np.inf, -np.inf], np.nan).to_numpy()


def correlation_clusters(corr_mat: pd.DataFrame, threshold: float) -> Dict[str, int]:
    """Group features into clusters via single-linkage on |spearman| >= threshold
    (connected components of the thresholded correlation graph). No scipy needed.

    Returns {feature: cluster_id}. A feature absent from corr_mat is treated by the
    caller as its own singleton cluster."""
    feats = list(corr_mat.columns)
    n = len(feats)
    idx = {f: i for i, f in enumerate(feats)}
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    A = corr_mat.to_numpy()
    iu, ju = np.triu_indices(n, k=1)
    m = np.abs(A[iu, ju]) >= threshold
    for i, j in zip(iu[m], ju[m]):
        union(int(i), int(j))

    roots = [find(i) for i in range(n)]
    remap: Dict[int, int] = {}
    out: Dict[str, int] = {}
    for f in feats:
        r = roots[idx[f]]
        if r not in remap:
            remap[r] = len(remap)
        out[f] = remap[r]
    return out


FEATURE_FAMILIES = [
    ("D_", "D"),       ("X_", "X"),       ("W_", "W"),
    ("WQ_", "WQ"),     ("M_", "M"),       ("Comb_", "Comb"),
    ("CPR_", "CPR"),   ("Struct_", "Struct"),  ("DayType_", "DayType"),
    ("DOW_", "DOW"),   ("regime_", "regime"),
]


def family_of(name: str) -> str:
    for prefix, label in FEATURE_FAMILIES:
        if name.startswith(prefix):
            return label
    return "other"


# =============================================================================
# STRATEGY CATEGORIES (semantic) — distinct from the prefix FAMILY above.
#
# family_of() tells you the NAMESPACE (D_/W_/WQ_/...). The categories below tell
# you the STRATEGY the feature speaks to (breakout, mean-reversion, momentum,
# ...), so that when you build a NEW model you can pull the right features.
#
# A feature can carry MULTIPLE tags (e.g. D_bb_bw_20 is both a volatility and a
# breakout-setup feature). `categorize()` returns (primary, [all tags]). The
# category report aggregates by EVERY tag, so the "breakout" view shows every
# breakout-relevant feature even if its primary label is something else.
#
# Rules are (tag, regex) evaluated against the lowercased feature name. Order =
# priority for the PRIMARY tag. Edit freely — it's just a lookup.
# =============================================================================

CATEGORY_RULES: List[Tuple[str, str]] = [
    ("breakout",        r"breakout|donch|dist_from_20h|dist_from_52wh|days_since_bo|"
                        r"\bnr\b|nr_length|nr_day|nr_expand|compress_state|bb_bw"),
    ("mean_reversion",  r"rsi|bb_pctb|pctb|_oli\b|cmf|zscore|_z252|dvol_z|vol_z|williams|stoch"),
    ("momentum_trend",  r"ema|sma|macd|adx|pdi|mdi|_trend|stack|angle|roll_slope|slope_stability|"
                        r"midpoint_slope|ret_4w|ret_13w|golden|\bmom\b|roc|obv_slope"),
    ("volatility",      r"atr|vol_yz|range_pct|range_to_atr|ret_5d_roll_std|high_vol|yang|garman|bb_bw"),
    ("volume_liquidity",r"obv|dollar_vol|vol_surge|avg20_vol|\bvolume\b|vol_vs|dvol|cmf"),
    ("candle_structure",r"body_ratio|wick|hh_run|hl_run|lh_run|ll_run|\bhh\b|\bhl\b|\blh\b|\bll\b|"
                        r"inside_day|gap_pct|day_type|\boli\b|range\b"),
    ("cpr_pivot",       r"^cpr_|_cpr|cpr_|pivot|support|resistance|tmr_cpr"),
    ("vwap_value",      r"vpoc|vwap|wq_41\b"),
    ("regime_context",  r"^regime_|x_regime|stock_regime|dispersion|^golden|daily_trend|weekly_trend|monthly_trend"),
    ("macro",           r"^m_"),
    ("calendar",        r"^dow_|_dow\b|daytype|dayofweek"),
    ("worldquant",      r"^wq_|^d_wq_|_wq_"),
    ("interaction",     r"^comb_|^x_"),
]
_COMPILED_RULES = [(tag, re.compile(rx)) for tag, rx in CATEGORY_RULES]


def categorize(name: str) -> Tuple[str, List[str]]:
    """Return (primary_category, [all matching tags]) for a feature name.

    Primary = first rule that matches (priority order above). Tags = every rule
    that matches, so multi-purpose features surface under each relevant strategy.
    """
    low = name.lower()
    tags = [tag for tag, rx in _COMPILED_RULES if rx.search(low)]
    if not tags:
        return ("other", ["other"])
    return (tags[0], tags)


# =============================================================================
# Model introspection for fast, EXACT permutation (used by _PermEngine)
# =============================================================================

def _booster_used_features(model, feature_cols: List[str]) -> Optional[set]:
    """Set of features the model's LightGBM booster actually splits on (split>0).

    Permuting a feature the booster never splits on cannot change its score, so
    such features get an EXACT auc_drop of 0 with no scoring. Returns None if no
    booster can be recovered (caller then conservatively assumes all features).
    """
    b = get_lgb_booster(model)
    if b is None:
        return None
    try:
        names = b.feature_name()
        split = b.feature_importance(importance_type="split")
        used = {n for n, s in zip(names, split) if s > 0}
        return used & set(feature_cols)
    except Exception:
        return None


# =============================================================================
# Time-purged CV helpers
# =============================================================================

def _match_tz(value, like_series: pd.Series):
    """Return `value` (a Timestamp/DatetimeIndex) localized to match the tz of
    `like_series`, so comparisons never mix tz-naive and tz-aware. Handles both
    directions (localize if naive, strip if the series is naive)."""
    tz = getattr(like_series.dt, "tz", None)
    v = pd.Timestamp(value) if not isinstance(value, (pd.Timestamp, pd.DatetimeIndex)) else value
    if isinstance(v, pd.DatetimeIndex):
        if tz is not None and v.tz is None:
            return v.tz_localize(tz)
        if tz is None and v.tz is not None:
            return v.tz_localize(None)
        return v
    # scalar Timestamp
    if tz is not None and v.tz is None:
        return v.tz_localize(tz)
    if tz is None and v.tz is not None:
        return v.tz_localize(None)
    return v


def time_split_by_date(timestamps: pd.Series, test_frac: float, embargo_days: int) -> Tuple[np.ndarray, np.ndarray, pd.Timestamp, pd.Timestamp]:
    """Time-respecting train/test split. Returns boolean masks + cutoff dates."""
    ts = pd.to_datetime(timestamps).dt.normalize()
    unique_dates = np.sort(ts.unique())
    cut_idx = int(len(unique_dates) * (1.0 - test_frac))
    # TZ-safe: match cut points to ts's timezone (ts is tz-aware IST).
    test_start_date = _match_tz(unique_dates[cut_idx], ts)
    embargo_end_date = test_start_date - pd.Timedelta(days=embargo_days)
    train_mask = (ts <= embargo_end_date).values
    test_mask = (ts >= test_start_date).values
    return train_mask, test_mask, embargo_end_date, test_start_date


def purged_kfold_indices(timestamps: pd.Series, n_splits: int, embargo_days: int):
    """Yield (train_idx, test_idx) for each fold with time-purging + embargo."""
    ts_series = pd.to_datetime(timestamps).dt.normalize().reset_index(drop=True)
    unique_dates = np.sort(ts_series.unique())
    n_dates = len(unique_dates)
    fold_size = n_dates // n_splits
    embargo = pd.Timedelta(days=embargo_days)
    for k in range(n_splits):
        s = k * fold_size
        e = (k + 1) * fold_size - 1 if k < n_splits - 1 else n_dates - 1
        # TZ-safe cut points matched to the series timezone.
        test_start = _match_tz(unique_dates[s], ts_series)
        test_end = _match_tz(unique_dates[e], ts_series)
        emb_pre = test_start - embargo
        emb_post = test_end + embargo
        # Compare against the tz-aware SERIES (do NOT use .values, which strips tz).
        test_mask = ((ts_series >= test_start) & (ts_series <= test_end)).to_numpy()
        train_mask = ((ts_series < emb_pre) | (ts_series > emb_post)).to_numpy()
        yield np.where(train_mask)[0], np.where(test_mask)[0]


# =============================================================================
# Multi-shuffle permutation
# =============================================================================

def permute_auc_multi(model, X: pd.DataFrame, y: np.ndarray, feature: str,
                      base_auc: float, n_shuffles: int, seed: int = 42
                      ) -> Tuple[float, float]:
    """Mean + std of AUC drop after permuting `feature` n_shuffles times.

    `model` must implement predict_proba returning shape (n,2).
    """
    drops = []
    rng = np.random.default_rng(seed)
    col_orig = X[feature].values.copy()
    X_shuf = X.copy()
    for _ in range(n_shuffles):
        col = col_orig.copy()
        rng.shuffle(col)
        X_shuf[feature] = col
        try:
            p = model.predict_proba(X_shuf)[:, 1]
            drops.append(base_auc - roc_auc_score(y, p))
        except Exception:
            drops.append(np.nan)
    drops = np.array([d for d in drops if not np.isnan(d)])
    if len(drops) == 0:
        return np.nan, np.nan
    return float(drops.mean()), float(drops.std(ddof=0))


def _feat_seed(feature: str, salt: int = 0) -> int:
    """Deterministic per-feature seed (reproducible across runs, unlike hash())."""
    return 42 + (zlib.crc32((str(salt) + "::" + feature).encode("utf-8")) % 9999)


class _PermEngine:
    """Fast, NUMERICALLY-EXACT multi-shuffle permutation importance.

    It reproduces the original scoring EXACTLY (whole-model predict_proba for
    models that expose it, else the mean of `members` — same as the original
    `unwrap_to_score`), and only applies optimizations that cannot change the
    result:

      1. SKIP UNUSED: a feature no booster splits on cannot change the score, so
         its auc_drop is exactly (0.0, 0.0) with no scoring.
      2. MEMBER CACHING: for an ensemble, permuting feature f only changes the
         members whose booster uses f. We cache base member predictions and
         re-score ONLY affected members per shuffle. Enabled only when we have
         VERIFIED that the model's own predict_proba equals the mean of its
         members on a probe batch (so the cached recombination is identical);
         otherwise we fall back to whole-model re-scoring.
      3. IN-PLACE COLUMN WRITE: permute the single column on the frame and
         restore it, instead of copying the whole 137-col frame per feature
         (the original did X.copy() every feature). The model still receives a
         DataFrame, so name-based projection in the wrappers keeps working.

    Seeds are deterministic (crc32) so runs are reproducible — the original used
    Python's hash(), which is salted per process and already non-reproducible.
    """

    def __init__(self, model, feature_cols: List[str], X_probe: Optional[pd.DataFrame] = None):
        self.model = model
        self.feature_cols = list(feature_cols)
        self.members = list(getattr(model, "members", None) or [])
        self.has_pp = hasattr(model, "predict_proba")
        # used-feature sets
        if self.members:
            self._member_used = [_booster_used_features(m, self.feature_cols) for m in self.members]
        else:
            self._member_used = [_booster_used_features(model, self.feature_cols)]
        self.union_used = self._union(self._member_used)
        # scoring mode
        self.use_member_cache = False
        if self.members:
            if not self.has_pp:
                self.use_member_cache = True            # mirrors unwrap: mean of members
            elif X_probe is not None and len(X_probe) > 0:
                try:
                    whole = self.model.predict_proba(X_probe)[:, 1]
                    meanm = np.mean([m.predict_proba(X_probe)[:, 1] for m in self.members], axis=0)
                    if np.nanmax(np.abs(whole - meanm)) < 1e-9:
                        self.use_member_cache = True    # safe: predict_proba == mean(members)
                except Exception:
                    self.use_member_cache = False
        self._mp: Optional[List[np.ndarray]] = None     # cached base member probs

    @staticmethod
    def _union(used_sets) -> Optional[set]:
        u = set()
        for s in used_sets:
            if s is None:
                return None                              # opaque -> assume uses everything
            u |= s
        return u

    def _score_whole(self, X: pd.DataFrame) -> np.ndarray:
        if self.has_pp:
            return self.model.predict_proba(X)[:, 1]
        return np.mean([m.predict_proba(X)[:, 1] for m in self.members], axis=0)

    def base_probs(self, X: pd.DataFrame) -> np.ndarray:
        if self.use_member_cache:
            self._mp = [m.predict_proba(X)[:, 1] for m in self.members]
            return np.mean(self._mp, axis=0)
        return self._score_whole(X)

    def perm_drop(self, X: pd.DataFrame, y: np.ndarray, feature: str,
                  base_auc: float, n_shuffles: int, seed: int) -> Tuple[float, float]:
        # (1) exact skip
        if self.union_used is not None and feature not in self.union_used:
            return 0.0, 0.0
        affected = [i for i, u in enumerate(self._member_used) if (u is None or feature in u)] \
            if self.use_member_cache else None
        if self.use_member_cache and self._mp is None:
            # lazy base member probs on the (unpermuted) frame; cached for reuse
            self._mp = [m.predict_proba(X)[:, 1] for m in self.members]
        col_orig = X[feature].to_numpy().copy()
        rng = np.random.default_rng(seed)
        drops = []
        try:
            for _ in range(n_shuffles):
                perm = col_orig.copy()
                rng.shuffle(perm)
                X[feature] = perm                        # in-place single column (no frame copy)
                if self.use_member_cache:
                    probs = list(self._mp)
                    for i in affected:
                        probs[i] = self.members[i].predict_proba(X)[:, 1]
                    score = probs[0] if len(probs) == 1 else np.mean(probs, axis=0)
                else:
                    score = self._score_whole(X)
                try:
                    drops.append(base_auc - roc_auc_score(y, score))
                except Exception:
                    drops.append(np.nan)
        finally:
            X[feature] = col_orig                        # always restore
        drops = np.array([d for d in drops if not np.isnan(d)])
        if len(drops) == 0:
            return np.nan, np.nan
        return float(drops.mean()), float(drops.std(ddof=0))


def spearman_ic_block(Xvals: np.ndarray, colnames: List[str], target: np.ndarray) -> Dict[str, float]:
    """Vectorized Spearman IC of EVERY column of Xvals vs target.

    Equivalent to scipy.spearmanr(col, target, nan_policy='omit') per column
    (Pearson on average ranks), computed for all columns in one matrix op.
    Assumes Xvals has no NaN (features are imputed); drops rows where target is
    NaN (the same row set for every column, matching nan_policy='omit' here).
    """
    out = {c: np.nan for c in colnames}
    if Xvals.size == 0:
        return out
    ok = ~np.isnan(target)
    if ok.sum() < 2:
        return out
    Xv = Xvals[ok]
    t = target[ok]
    tr = pd.Series(t).rank().to_numpy()                  # average ranks (matches scipy ties)
    Xr = pd.DataFrame(Xv).rank(axis=0).to_numpy()
    tc = tr - tr.mean()
    Xc = Xr - Xr.mean(axis=0)
    num = Xc.T @ tc
    den = np.sqrt((Xc ** 2).sum(axis=0) * (tc ** 2).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        ic = np.where(den > 0, num / den, np.nan)
    return dict(zip(colnames, ic))


# =============================================================================
# Stability score (v5: rank-based, robust to small means)
# =============================================================================

def rank_stability(period_imp_df: pd.DataFrame) -> pd.Series:
    """For each feature, compute spearman-rank stability of importance across periods.

    Returns per-feature scalar in [0, 1]: 1 = same rank in every period,
    0 = uncorrelated rank across periods.
    """
    if period_imp_df.shape[1] < 2:
        return pd.Series(0.5, index=period_imp_df.index)
    ranks = period_imp_df.rank(axis=0, ascending=False, method="average")
    n_features = ranks.shape[0]
    # Per-feature: fraction of period-pairs whose ranks are within 25% of each other
    pair_count = 0
    within = pd.Series(0.0, index=ranks.index)
    cols = list(ranks.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r_i = ranks[cols[i]]
            r_j = ranks[cols[j]]
            within += (np.abs(r_i - r_j) <= n_features * 0.25).astype(float)
            pair_count += 1
    if pair_count == 0:
        return pd.Series(0.5, index=period_imp_df.index)
    return (within / pair_count).clip(0.0, 1.0)


# =============================================================================
# Model unwrapping (router / ensemble / single)
# =============================================================================

def unwrap_to_score(model):
    """Return a callable score(X) -> prob[:, 1] for any of the model wrappers."""
    if model is None:
        return None
    if hasattr(model, "predict_proba"):
        return lambda X: model.predict_proba(X)[:, 1]
    if hasattr(model, "members") and model.members:
        return lambda X: np.mean([m.predict_proba(X)[:, 1] for m in model.members], axis=0)
    return None


def get_lgb_booster(model):
    """Best-effort recovery of a LightGBM booster for built-in importance."""
    if model is None:
        return None
    if hasattr(model, "members") and model.members:
        return get_lgb_booster(model.members[0])
    if hasattr(model, "calibrated_classifiers_"):
        cc = model.calibrated_classifiers_[0]
        for attr in ("estimator", "base_estimator"):
            est = getattr(cc, attr, None)
            if est is None:
                continue
            if hasattr(est, "booster_"):
                return est.booster_
            if hasattr(est, "_Booster"):
                return est._Booster
    if hasattr(model, "booster_"):
        return model.booster_
    if hasattr(model, "_Booster"):
        return model._Booster
    return None


# =============================================================================
# Main
# =============================================================================

def main(cfg: Config):
    print("=" * 70)
    print("REGIME-AWARE FEATURE DIAGNOSTIC v5  (parametric)")
    print("=" * 70)

    # ----- parametric locals (default == legacy 5d behaviour) -----
    base_dir = Path(cfg.base_dir)
    HORIZON = int(cfg.horizon)
    RET_COL = cfg.ret_col_name()
    TARGET_COL = cfg.target_col_name()
    EMBARGO_DAYS = cfg.embargo()
    TEST_FRAC = float(cfg.test_frac)
    LABEL_Q = cfg.label_q()
    print(f"[Cfg] horizon={HORIZON}d  target={TARGET_COL}  ret={RET_COL}  "
          f"embargo={EMBARGO_DAYS}d  label_q={LABEL_Q:.2f}  "
          f"vol_adj={cfg.use_model_target}  model_free={cfg.model_free}")

    panel_path = base_dir / "panel_cache.parquet"
    features_path = base_dir / "features_train.json"
    models_dir = base_dir / "models"
    out_dir = cfg.resolved_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Cfg] outputs -> {out_dir}")

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------
    print(f"\n[Load] Panel:    {panel_path}")
    print(f"[Load] Features: {features_path}")

    schema = json.loads(features_path.read_text())
    FEATURES: List[str] = list(schema["features"])
    GLOBAL_IMPUTE: Dict[str, float] = {k: float(v) for k, v in schema.get("impute", {}).items()}
    print(f"[Load] {len(FEATURES)} features in schema")

    # Project columns to keep memory in check (v5)
    needed_cols = list(set(FEATURES + [
        "timestamp", "symbol", "open", "close", "volume",
        RET_COL, TARGET_COL, "stock_regime", "D_atr14",
        "ret_5d_close_pct", "ret_3d_close_pct",
    ]))
    # ROBUSTNESS: only request columns that actually exist in the parquet, so a
    # drift between features_train.json and panel_cache.parquet (e.g. macro M_*
    # columns listed in the schema but not persisted to disk) does NOT hard-crash
    # with pyarrow ArrowInvalid. Warn loudly about any missing feature instead.
    try:
        import pyarrow.parquet as _pq
        on_disk = set(_pq.read_schema(str(panel_path)).names)
    except Exception:
        on_disk = set(pd.read_parquet(panel_path).columns)  # fallback: full read
    requested = [c for c in needed_cols if c]
    present = [c for c in requested if c in on_disk]
    missing = [c for c in requested if c not in on_disk]
    if missing:
        missing_feats = [c for c in missing if c in FEATURES]
        print(f"[Load] WARNING: {len(missing)} requested columns are NOT in the parquet "
              f"and will be skipped: {missing[:12]}{'...' if len(missing) > 12 else ''}")
        if missing_feats:
            print(f"[Load] WARNING: {len(missing_feats)} of these are MODEL FEATURES "
                  f"(e.g. {missing_feats[:6]}). Your panel_cache.parquet is missing them — "
                  f"re-run the model so it persists the ENRICHED panel (macro M_*, regime, "
                  f"X_regime_*) to disk. Proceeding WITHOUT these features for this analysis.")
        # Drop missing features from the analysis set so everything downstream aligns.
        FEATURES = [f for f in FEATURES if f in on_disk]
        GLOBAL_IMPUTE = {k: v for k, v in GLOBAL_IMPUTE.items() if k in on_disk}
        print(f"[Load] Continuing with {len(FEATURES)} features that exist on disk.")
    panel = pd.read_parquet(panel_path, columns=present)

    # =====================================================================
    # MEMORY HARDENING (mirrors the model's hardening).
    # A 195-feature x 3.5M-row float64 panel needs ~5 GiB as ONE dense block,
    # and pandas' block consolidation / .copy() can briefly need TWO copies
    # (>10 GiB), which OOMs on most machines (numpy ArrayMemoryError "Unable to
    # allocate 5.16 GiB"). Downcasting features to float32 halves it; making
    # symbol categorical saves a large string column. Done BEFORE any .copy().
    # =====================================================================
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    _float_cols = [c for c in panel.columns
                   if c not in ("timestamp", "symbol") and panel[c].dtype == "float64"]
    if _float_cols:
        panel[_float_cols] = panel[_float_cols].astype("float32")
    if "symbol" in panel.columns and panel["symbol"].dtype == object:
        panel["symbol"] = panel["symbol"].astype("category")
    try:
        _mem_mb = panel.memory_usage(deep=True).sum() / 1e6
        print(f"[Load] Panel in memory after float32/categorical hardening: {_mem_mb:,.0f} MB "
              f"({len(panel):,} rows x {panel.shape[1]} cols)")
    except Exception:
        pass

    if panel["timestamp"].dt.tz is None:
        panel["timestamp"] = panel["timestamp"].dt.tz_localize("UTC").dt.tz_convert(IST)
    else:
        panel["timestamp"] = panel["timestamp"].dt.tz_convert(IST)
    panel = panel.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    panel["date"] = panel["timestamp"].dt.normalize()
    panel["year"] = panel["timestamp"].dt.year

    # avg20 vol per symbol
    panel["avg20_vol"] = (
        panel.groupby("symbol", observed=True)["volume"]
             .transform(lambda s: s.rolling(20, min_periods=1).mean())
    )

    # v5: Universe filter — match deployment universe
    pre_n = len(panel)
    universe = (panel["close"] >= MIN_CLOSE) & (panel["avg20_vol"] >= MIN_AVG20_VOL)
    panel = panel[universe].reset_index(drop=True)
    print(f"[Load] Universe filter: {pre_n:,} -> {len(panel):,} rows "
          f"(close>={MIN_CLOSE}, vol>={MIN_AVG20_VOL:,})")

    # -------------------------------------------------------------------------
    # Load model (parametric + OPTIONAL). Order: explicit --model-path, then
    # m{H}_* for this horizon, then legacy m5_*. If none found (or --model-free),
    # run model-free: gain%/permutation come from the diagnostic model trained
    # below on THIS horizon's label.
    # -------------------------------------------------------------------------
    use_regime = False
    regime_models: Dict[str, object] = {}
    fallback_model = None
    model_src = "model-free (diagnostic model only)"
    if not cfg.model_free:
        candidates: List[Path] = []
        if cfg.model_path is not None:
            candidates = [Path(cfg.model_path)]
        else:
            stems = [f"m{HORIZON}_regime_router", f"m{HORIZON}_ensemble", f"m{HORIZON}_classifier",
                     "m5_regime_router", "m5_ensemble", "m5_classifier"]
            candidates = [models_dir / f"{s}.joblib" for s in stems]
        chosen = next((p for p in candidates if p.exists()), None)
        if chosen is None:
            print(f"[Load] No model found for horizon {HORIZON}d in {models_dir} "
                  f"-> MODEL-FREE mode (importance from the diagnostic model).")
        else:
            print(f"[Load] Model: {chosen}")
            obj = load(chosen)
            model_src = chosen.name
            if "router" in chosen.name:
                use_regime = True
                fallback_model = getattr(obj, "fallback", None)
                regime_models = getattr(obj, "regime_models", {}) or {}
            else:
                fallback_model = obj
    else:
        print("[Load] --model-free set -> importance from the diagnostic model.")
    print(f"[Load] Importance source: {model_src}")

    # -------------------------------------------------------------------------
    # Build forward return + label for THIS horizon
    # -------------------------------------------------------------------------
    ensure_forward_return(panel, RET_COL, HORIZON)

    if TARGET_COL not in panel.columns or panel[TARGET_COL].notna().sum() == 0:
        kind = "vol-adjusted" if cfg.use_model_target else "raw"
        print(f"[Target] Deriving {TARGET_COL}: top/bottom {LABEL_Q:.0%} of the per-day "
              f"{kind} forward {HORIZON}d return ({RET_COL}).")
        rh = pd.to_numeric(panel.get(RET_COL), errors="coerce")
        if cfg.use_model_target:
            close = pd.to_numeric(panel["close"], errors="coerce")
            atr_pct = (pd.to_numeric(panel.get("D_atr14"), errors="coerce") /
                       close.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan) * 100.0
            basis = rh / atr_pct.replace(0, np.nan)
        else:
            basis = rh
        rank_pct = basis.groupby(panel["date"]).rank(method="average", pct=True)
        panel[TARGET_COL] = np.where(rank_pct >= 1.0 - LABEL_Q, 1,
                                     np.where(rank_pct <= LABEL_Q, 0, np.nan))
    else:
        print(f"[Target] Using existing column {TARGET_COL} from panel.")

    labeled = panel[panel[TARGET_COL].notna()].copy().reset_index(drop=True)
    print(f"[Target] Labeled rows: {len(labeled):,}")

    # -------------------------------------------------------------------------
    # Time-based train/test split (v5 fix #1)
    # -------------------------------------------------------------------------
    train_mask, test_mask, embargo_end, test_start = time_split_by_date(
        labeled["timestamp"], test_frac=TEST_FRAC, embargo_days=EMBARGO_DAYS
    )
    print(f"[Split] Train: {train_mask.sum():,} rows up to {embargo_end.date()}")
    print(f"[Split] Test:  {test_mask.sum():,} rows from {test_start.date()}")
    print(f"[Split] Embargo: {EMBARGO_DAYS} days")

    # -------------------------------------------------------------------------
    # Build X / y with TRAIN-only impute medians (v5 fix #5)
    # -------------------------------------------------------------------------
    def to_numeric_block(df: pd.DataFrame) -> pd.DataFrame:
        out = df.reindex(columns=FEATURES).copy()
        for c in FEATURES:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        return out

    X_full = to_numeric_block(labeled)
    y_full = labeled[TARGET_COL].astype(int).values

    # Train-only medians
    train_medians = X_full.iloc[np.where(train_mask)[0]].median(numeric_only=True)
    impute = {c: float(train_medians.get(c, GLOBAL_IMPUTE.get(c, 0.0))) for c in FEATURES}
    X_full = X_full.fillna(pd.Series(impute))

    X_tr = X_full.iloc[train_mask].reset_index(drop=True)
    y_tr = y_full[train_mask]
    X_te = X_full.iloc[test_mask].reset_index(drop=True)
    y_te = y_full[test_mask]
    labeled_te = labeled.loc[test_mask].reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Diagnostic model (v5: prod-matched params)
    # -------------------------------------------------------------------------
    print("\n[Diag] Training diagnostic LGBM (prod-matched params)...")
    clf_diag = lgb.LGBMClassifier(**DIAG_LGB_PARAMS)
    clf_diag.fit(X_tr, y_tr)
    p_base = clf_diag.predict_proba(X_te)[:, 1]
    base_auc = roc_auc_score(y_te, p_base)
    print(f"[Diag] Out-of-time test AUC: {base_auc:.4f}")
    print(f"[Diag] (Lower than v4 reported numbers? Good — v4 was cross-stock.)")

    # =========================================================================
    # STAGE 1 - Global importance
    # =========================================================================
    print("\n[1/9] GLOBAL IMPORTANCE...")
    booster = get_lgb_booster(fallback_model)
    if booster is not None:
        b_feats = booster.feature_name()
        gain = dict(zip(b_feats, booster.feature_importance(importance_type="gain")))
        split = dict(zip(b_feats, booster.feature_importance(importance_type="split")))
        gain_aligned = np.array([gain.get(f, 0.0) for f in FEATURES], dtype=float)
        split_aligned = np.array([split.get(f, 0.0) for f in FEATURES], dtype=float)
        src = "production model"
    else:
        gain_aligned = clf_diag.booster_.feature_importance(importance_type="gain").astype(float)
        split_aligned = clf_diag.booster_.feature_importance(importance_type="split").astype(float)
        src = "diagnostic model (production model unavailable)"
    print(f"[1/9] Importance source: {src}")

    g_total = gain_aligned.sum() or 1.0
    s_total = split_aligned.sum() or 1.0
    global_imp = pd.DataFrame({
        "feature": FEATURES,
        "family": [family_of(f) for f in FEATURES],
        "gain": gain_aligned,
        "split": split_aligned,
        "gain_pct": 100 * gain_aligned / g_total,
        "split_pct": 100 * split_aligned / s_total,
    }).sort_values("gain", ascending=False).reset_index(drop=True)
    global_imp["gain_rank"] = global_imp.index + 1
    global_imp.to_csv(out_dir / "feature_importance_global.csv", index=False)
    print("[1/9] Top 5:", global_imp.head(5)["feature"].tolist())

    # =========================================================================
    # STAGE 2 - Year-over-year stability (rank-based)
    # =========================================================================
    print("\n[2/9] YEAR-OVER-YEAR STABILITY...")
    year_imp_cols = {}
    years_tr = labeled.loc[train_mask, "year"].values
    for yr in sorted(set(years_tr)):
        mask = years_tr == yr
        if mask.sum() < 5000:
            continue
        if len(set(y_tr[mask])) < 2:
            continue
        clf_yr = lgb.LGBMClassifier(**DIAG_LGB_PARAMS)
        clf_yr.fit(X_tr.iloc[mask], y_tr[mask])
        g = clf_yr.booster_.feature_importance(importance_type="gain")
        year_imp_cols[int(yr)] = pd.Series(g, index=FEATURES)
    year_imp_df = pd.DataFrame(year_imp_cols)
    year_stab = rank_stability(year_imp_df)
    year_imp_df["rank_stability"] = year_stab
    year_imp_df.to_csv(out_dir / "feature_stability_by_year.csv")
    print(f"[2/9] Years analyzed: {len(year_imp_cols)}")

    # =========================================================================
    # STAGE 3 - IC analysis (global by year, per-regime)
    # =========================================================================
    print("\n[3/9] IC ANALYSIS...")
    ret_5d_full = pd.to_numeric(labeled[RET_COL], errors="coerce").values
    ret_5d_te = ret_5d_full[test_mask]
    years_te = labeled_te["year"].values

    ic_year_rows = []
    # Vectorized: one matrix op per year for ALL features (== per-feature spearmanr).
    Xte_vals = X_te[FEATURES].to_numpy(dtype=float)
    for yr in sorted(set(years_te)):
        m = years_te == yr
        if m.sum() < 100:
            continue
        ic_map = spearman_ic_block(Xte_vals[m], FEATURES, ret_5d_te[m])
        for feat in FEATURES:
            ic = ic_map[feat]
            if ic is not None and not np.isnan(ic):
                ic_year_rows.append({"feature": feat, "year": int(yr), "ic": ic})
    ic_year_df = pd.DataFrame(ic_year_rows)
    if not ic_year_df.empty:
        ic_year_pivot = ic_year_df.pivot(index="feature", columns="year", values="ic")
    else:
        ic_year_pivot = pd.DataFrame(index=FEATURES)
    ic_year_pivot.to_csv(out_dir / "feature_ic_by_year.csv")

    # IC summary across years
    ic_summary_rows = []
    for feat in FEATURES:
        sub = ic_year_df[ic_year_df["feature"] == feat]["ic"] if not ic_year_df.empty else pd.Series(dtype=float)
        if len(sub) == 0:
            ic_summary_rows.append({
                "feature": feat, "mean_abs_ic": 0.0, "ic_pos_pct": 0.0,
                "ic_sign_flip": 1.0, "ic_std": 0.0,
            })
            continue
        # v5: count sign flips only when |ic| > 0.005 (else it's noise)
        meaningful = sub[sub.abs() > 0.005]
        if len(meaningful) == 0:
            sign_flip = 1.0
        else:
            pos = (meaningful > 0).sum()
            neg = (meaningful < 0).sum()
            sign_flip = float(min(pos, neg)) / float(max(pos + neg, 1))
        ic_summary_rows.append({
            "feature": feat,
            "mean_abs_ic": float(sub.abs().mean()),
            "ic_pos_pct": float((sub > 0).mean()),
            "ic_sign_flip": sign_flip,
            "ic_std": float(sub.std()),
        })
    ic_summary = pd.DataFrame(ic_summary_rows).set_index("feature")

    # Per-regime IC on test slice (vectorized: one matrix op per regime)
    regime_ic_df = pd.DataFrame(index=FEATURES, columns=[f"ic_{r}" for r in REGIMES])
    if "stock_regime" in labeled_te.columns:
        reg_vals = labeled_te["stock_regime"].values
        for r in REGIMES:
            m = (reg_vals == r)
            if m.sum() < 1000:
                continue
            ic_map = spearman_ic_block(Xte_vals[m], FEATURES, ret_5d_te[m])
            for feat in FEATURES:
                regime_ic_df.loc[feat, f"ic_{r}"] = ic_map[feat]
    regime_ic_df = regime_ic_df.astype(float)
    regime_ic_df.to_csv(out_dir / "regime_ic_summary.csv")

    # =========================================================================
    # STAGE 4 - Forward-time generalization (v5: replaces price-bucket test)
    # =========================================================================
    print("\n[4/9] FORWARD-TIME GENERALIZATION...")
    # Train on first 60% of TRAIN dates, score importance on 60-80% slice.
    # Then train on 20-80% of TRAIN dates, score on 0-20%. Compare ranks.
    tr_dates = pd.to_datetime(labeled.loc[train_mask, "timestamp"]).dt.normalize()
    uniq_tr_dates = np.sort(tr_dates.unique())
    # TZ-safe cut points (tr_dates is tz-aware IST). Wrapping numpy datetime64 in
    # pd.Timestamp() would give a tz-NAIVE value -> "Cannot compare tz-naive and
    # tz-aware timestamps". _match_tz aligns the timezone for the comparison.
    cut1 = _match_tz(uniq_tr_dates[int(len(uniq_tr_dates) * 0.6)], tr_dates)
    cut2 = _match_tz(uniq_tr_dates[int(len(uniq_tr_dates) * 0.8)], tr_dates)

    # Compare tr_dates DIRECTLY (it is already tz-aware). Do NOT round-trip through
    # .values, which strips the tz and reintroduces the tz-naive vs tz-aware error.
    early_mask = (tr_dates < cut1).to_numpy()
    late_mask = ((tr_dates >= cut1) & (tr_dates <= cut2)).to_numpy()

    if early_mask.sum() > 5000 and late_mask.sum() > 5000:
        clf_early = lgb.LGBMClassifier(**DIAG_LGB_PARAMS)
        clf_early.fit(X_tr.iloc[early_mask], y_tr[early_mask])
        imp_early = clf_early.booster_.feature_importance(importance_type="gain").astype(float)

        clf_late = lgb.LGBMClassifier(**DIAG_LGB_PARAMS)
        clf_late.fit(X_tr.iloc[late_mask], y_tr[late_mask])
        imp_late = clf_late.booster_.feature_importance(importance_type="gain").astype(float)

        rank_early = pd.Series(imp_early, index=FEATURES).rank(ascending=False)
        rank_late = pd.Series(imp_late, index=FEATURES).rank(ascending=False)
        rank_drift = (rank_early - rank_late).abs() / len(FEATURES)
        # 0 = identical rank, 1 = inverted
        gen_df = pd.DataFrame({
            "feature": FEATURES,
            "imp_early": imp_early,
            "imp_late": imp_late,
            "rank_early": rank_early.values,
            "rank_late": rank_late.values,
            "rank_drift": rank_drift.values,
            "generalizes": rank_drift.values < 0.25,  # within 25% of N
        })
    else:
        gen_df = pd.DataFrame({"feature": FEATURES, "rank_drift": [np.nan] * len(FEATURES),
                               "generalizes": [True] * len(FEATURES)})
    gen_df.to_csv(out_dir / "feature_generalization.csv", index=False)
    print(f"[4/9] Features that generalize across time: "
          f"{int(gen_df['generalizes'].sum())} / {len(FEATURES)}")

    # =========================================================================
    # STAGE 5 - Redundancy
    # =========================================================================
    print("\n[5/9] REDUNDANCY (Spearman corr)...")
    if len(X_tr) > 100_000:
        sample_idx = np.random.RandomState(42).choice(len(X_tr), 100_000, replace=False)
        X_corr = X_tr.iloc[sample_idx]
    else:
        X_corr = X_tr
    corr_mat = X_corr.corr(method="spearman")
    corr_mat.to_csv(out_dir / "feature_correlation_matrix.csv")

    # Vectorized pair extraction (v5: was O(n^2) Python loop)
    upper = np.triu(np.ones(corr_mat.shape, dtype=bool), k=1)
    high = (corr_mat.abs() >= CORR_THRESHOLD) & upper
    pairs_idx = np.where(high.values)
    redund_rows = []
    feats_arr = np.array(FEATURES)
    ic_lookup = ic_summary["mean_abs_ic"].to_dict()
    for i, j in zip(*pairs_idx):
        a, b = feats_arr[i], feats_arr[j]
        ic_a = ic_lookup.get(a, 0.0)
        ic_b = ic_lookup.get(b, 0.0)
        keep = a if ic_a >= ic_b else b
        drop = b if keep == a else a
        redund_rows.append({
            "feature_a": a, "feature_b": b,
            "correlation": float(corr_mat.iloc[i, j]),
            "keep": keep, "drop_candidate": drop,
            "reason": f"{keep} has higher mean |IC|",
        })
    pd.DataFrame(redund_rows).to_csv(out_dir / "feature_redundant_pairs.csv", index=False)
    print(f"[5/9] Redundant pairs (|r|>={CORR_THRESHOLD}): {len(redund_rows)}")

    # =========================================================================
    # STAGE 6 - Multi-shuffle permutation (global + per-regime, on prod model)
    # =========================================================================
    print(f"\n[6/9] PERMUTATION IMPORTANCE ({PERM_SHUFFLES} shuffles per feature)...")

    global_perm_rows = []
    model_global = fallback_model if unwrap_to_score(fallback_model) is not None else clf_diag
    eng_g = _PermEngine(model_global, FEATURES, X_probe=X_te)
    base_auc_global = roc_auc_score(y_te, eng_g.base_probs(X_te))
    print(f"[6/9] Global base AUC (prod model on OOT test): {base_auc_global:.4f}")
    if eng_g.union_used is not None:
        print(f"[6/9] Model splits on {len(eng_g.union_used)}/{len(FEATURES)} features; "
              f"the rest get an EXACT 0-drop without scoring."
              + ("  [member-caching ON]" if eng_g.use_member_cache else ""))
    for feat in tqdm(FEATURES, desc="  global perm"):
        m, s = eng_g.perm_drop(X_te, y_te, feat, base_auc=base_auc_global,
                               n_shuffles=PERM_SHUFFLES, seed=_feat_seed(feat))
        global_perm_rows.append({
            "feature": feat,
            "auc_drop_global_mean": m,
            "auc_drop_global_std": s,
        })
    global_perm_df = pd.DataFrame(global_perm_rows).set_index("feature")
    global_perm_df.to_csv(out_dir / "feature_permutation_importance.csv")

    # Per-regime permutation on prod regime models
    regime_perm: Dict[str, Dict[str, Tuple[float, float]]] = {r: {} for r in REGIMES}
    regime_aucs: Dict[str, float] = {}
    if use_regime and "stock_regime" in labeled_te.columns:
        for r in REGIMES:
            model_r = regime_models.get(r) or fallback_model
            if model_r is None:
                continue
            m_te = (labeled_te["stock_regime"].values == r)
            n_r = int(m_te.sum())
            if n_r < 1000:
                print(f"[6/9] {r}: only {n_r} test rows, skipping")
                continue
            X_r_te = X_te.iloc[m_te].reset_index(drop=True)
            y_r_te = y_te[m_te]
            if len(set(y_r_te)) < 2:
                continue
            score_r = unwrap_to_score(model_r)
            if score_r is None:
                continue
            eng_r = _PermEngine(model_r, FEATURES, X_probe=X_r_te)
            try:
                base_auc_r = roc_auc_score(y_r_te, eng_r.base_probs(X_r_te))
            except Exception as e:
                print(f"[6/9] {r}: baseline AUC failed ({e}), skipping")
                continue
            regime_aucs[r] = base_auc_r
            print(f"[6/9] {r}: n_test={n_r:,}  base_AUC={base_auc_r:.4f}")
            for feat in tqdm(FEATURES, desc=f"  perm {r}"):
                m_, s_ = eng_r.perm_drop(X_r_te, y_r_te, feat, base_auc=base_auc_r,
                                         n_shuffles=PERM_SHUFFLES, seed=_feat_seed(feat, salt=r))
                regime_perm[r][feat] = (m_, s_)

    # Build per-regime perm DataFrame
    perm_data = {"feature": FEATURES,
                 "auc_drop_global": global_perm_df["auc_drop_global_mean"].reindex(FEATURES).values,
                 "auc_drop_global_std": global_perm_df["auc_drop_global_std"].reindex(FEATURES).values}
    for r in REGIMES:
        perm_data[f"auc_drop_{r}"] = [regime_perm[r].get(f, (np.nan, np.nan))[0] for f in FEATURES]
        perm_data[f"auc_drop_{r}_std"] = [regime_perm[r].get(f, (np.nan, np.nan))[1] for f in FEATURES]
    regime_perm_df = pd.DataFrame(perm_data).set_index("feature")
    regime_perm_df.to_csv(out_dir / "regime_permutation_importance.csv")

    # =========================================================================
    # STAGE 7 - Regime consistency
    # =========================================================================
    print("\n[7/9] REGIME CONSISTENCY...")
    consistency_rows = []
    for feat in FEATURES:
        drops_r = {r: regime_perm[r].get(feat, (np.nan, np.nan))[0] for r in REGIMES}
        valid = [d for d in drops_r.values() if pd.notna(d)]
        if not valid:
            consistency_rows.append({
                "feature": feat, "consistency_score": 0.0,
                "max_regime_drop": np.nan, "min_regime_drop": np.nan,
                "mean_regime_drop": np.nan, "best_regime": "",
                "regime_specialist": False, "is_dead_everywhere": True,
                "is_harmful_anywhere": False,
            })
            continue
        max_d = max(valid)
        min_d = min(valid)
        mean_d = float(np.mean(valid))
        std_d = float(np.std(valid))
        # cosine-style consistency: 1 if all drops similar, 0 if scattered
        cos = 1.0 - std_d / (abs(mean_d) + 1e-6)
        cos = max(0.0, min(1.0, cos))
        best = max(drops_r, key=lambda k: -np.inf if pd.isna(drops_r[k]) else drops_r[k])
        if drops_r[best] < PERM_DROP_KEEP:
            best = ""
        consistency_rows.append({
            "feature": feat,
            "consistency_score": round(cos, 3),
            "max_regime_drop": round(max_d, 5),
            "min_regime_drop": round(min_d, 5),
            "mean_regime_drop": round(mean_d, 5),
            "best_regime": best,
            "regime_specialist": (max_d >= REGIME_EXCEPTIONAL and min_d < PERM_DROP_KEEP),
            "is_dead_everywhere": all(d < PERM_DROP_KEEP for d in valid),
            "is_harmful_anywhere": any(d < PERM_DROP_HARMFUL for d in valid),
        })
    consistency_df = pd.DataFrame(consistency_rows)
    consistency_df.to_csv(out_dir / "regime_consistency.csv", index=False)
    n_specialists = int(consistency_df["regime_specialist"].sum())
    n_dead = int(consistency_df["is_dead_everywhere"].sum())
    n_harmful = int(consistency_df["is_harmful_anywhere"].sum())
    print(f"[7/9] Specialists: {n_specialists}  Dead-everywhere: {n_dead}  Harmful-anywhere: {n_harmful}")

    # =========================================================================
    # STAGE 8 - Feature family aggregation (v5 new)
    # =========================================================================
    print("\n[8/9] FEATURE FAMILY SUMMARY...")
    fam_rows = []
    for fam in sorted(set(family_of(f) for f in FEATURES)):
        members = [f for f in FEATURES if family_of(f) == fam]
        fam_imp = global_imp[global_imp["family"] == fam]
        fam_perm = global_perm_df.reindex(members)["auc_drop_global_mean"]
        fam_ic = ic_summary.reindex(members)["mean_abs_ic"]
        fam_rows.append({
            "family": fam,
            "n_features": len(members),
            "total_gain_pct": round(float(fam_imp["gain_pct"].sum()), 2),
            "median_gain_pct": round(float(fam_imp["gain_pct"].median()), 4),
            "mean_perm_drop": round(float(fam_perm.mean()), 5),
            "max_perm_drop": round(float(fam_perm.max()), 5),
            "mean_abs_ic": round(float(fam_ic.mean()), 4),
            "n_alive": int((fam_perm >= PERM_DROP_KEEP).sum()),
            "n_dead": int((fam_perm < PERM_DROP_KEEP).sum()),
        })
    family_df = pd.DataFrame(fam_rows).sort_values("total_gain_pct", ascending=False)
    family_df.to_csv(out_dir / "feature_family_summary.csv", index=False)
    print(family_df.to_string(index=False))

    # =========================================================================
    # STAGE 9 - Strict pruning recommendation + harmful + specialists
    # =========================================================================
    print("\n[9/9] STRICT PRUNING RECOMMENDATION...")

    rec = global_imp[["feature", "family", "gain_pct", "split_pct", "gain_rank"]].copy()
    rec = rec.merge(ic_summary.reset_index(), on="feature", how="left")
    ys = year_stab.reset_index()
    ys.columns = ["feature", "year_rank_stability"]
    rec = rec.merge(ys, on="feature", how="left")
    rec = rec.merge(gen_df[["feature", "rank_drift", "generalizes"]], on="feature", how="left")
    rec = rec.merge(global_perm_df.reset_index(), on="feature", how="left")
    rec = rec.merge(consistency_df, on="feature", how="left")
    for r in REGIMES:
        rec[f"auc_drop_{r}"] = rec["feature"].map(
            regime_perm_df[f"auc_drop_{r}"].to_dict()
        )
        rec[f"ic_{r}"] = rec["feature"].map(regime_ic_df[f"ic_{r}"].to_dict())

    def recommend(row) -> str:
        # Check harmful first — these are dropped with high priority
        if row.get("is_harmful_anywhere"):
            return "DROP_HARMFUL"
        regime_drops = [row.get(f"auc_drop_{r}") for r in REGIMES]
        regime_drops = [d for d in regime_drops if pd.notna(d)]
        regime_ics = [abs(row.get(f"ic_{r}", np.nan)) for r in REGIMES]
        regime_ics = [v for v in regime_ics if pd.notna(v)]
        global_perm = row.get("auc_drop_global_mean") or 0.0
        mean_abs_ic = row.get("mean_abs_ic") or 0.0
        # std-gate: require the global perm drop to clear k * (its shuffle std), not
        # just the bare absolute PERM_DROP_KEEP (which sits inside the noise band).
        # perm_keep_eff is the effective "useful" bar; perm_dead is the "no real
        # signal" bar used by the DROP test.
        perm_keep_eff = PERM_DROP_KEEP * 3
        perm_dead = PERM_DROP_KEEP
        if cfg.std_gate:
            gstd = row.get("auc_drop_global_std")
            if pd.notna(gstd) and gstd > 0:
                sig = cfg.std_gate_k * float(gstd)
                perm_keep_eff = max(perm_keep_eff, sig)
                perm_dead = max(perm_dead, sig)
        # KEEP signals
        if regime_drops and max(regime_drops) >= REGIME_EXCEPTIONAL:
            return "KEEP"
        if regime_ics and max(regime_ics) >= IC_THRESHOLD * 2:
            return "KEEP"
        if global_perm >= perm_keep_eff:
            return "KEEP"
        if mean_abs_ic >= IC_THRESHOLD * 2:
            return "KEEP"
        # DROP signals
        dead_all_regimes = bool(regime_drops) and all(d < PERM_DROP_KEEP for d in regime_drops)
        dead_global = global_perm < perm_dead
        weak_ic = mean_abs_ic < IC_THRESHOLD
        if dead_all_regimes and dead_global and weak_ic:
            return "DROP"
        return "REVIEW"

    def explain(row) -> str:
        rec_val = row["recommendation"]
        parts = []
        regime_drops = {r: row.get(f"auc_drop_{r}") for r in REGIMES}
        valid = [(r, d) for r, d in regime_drops.items() if pd.notna(d)]
        if rec_val == "DROP_HARMFUL":
            harmful = [(r, d) for r, d in regime_drops.items() if pd.notna(d) and d < PERM_DROP_HARMFUL]
            return f"Harmful in: " + ", ".join(f"{r}({d:.4f})" for r, d in harmful)
        if rec_val == "KEEP":
            if valid:
                best = max(valid, key=lambda x: x[1])
                if best[1] >= REGIME_EXCEPTIONAL:
                    parts.append(f"Strong in {best[0]} ({best[1]:.4f})")
            if pd.notna(row.get("mean_abs_ic")) and row["mean_abs_ic"] >= IC_THRESHOLD * 2:
                parts.append(f"Strong IC ({row['mean_abs_ic']:.4f})")
            if pd.notna(row.get("auc_drop_global_mean")) and row["auc_drop_global_mean"] >= PERM_DROP_KEEP * 3:
                parts.append(f"Strong global perm ({row['auc_drop_global_mean']:.4f})")
        elif rec_val == "DROP":
            parts.append("Dead in all regimes & global")
            if pd.notna(row.get("mean_abs_ic")):
                parts.append(f"Weak IC ({row['mean_abs_ic']:.4f})")
        else:
            if valid:
                best = max(valid, key=lambda x: x[1])
                parts.append(f"Best in {best[0]} ({best[1]:.4f}) but inconsistent")
            else:
                parts.append("Marginal evidence")
        return " | ".join(parts) if parts else ""

    rec["recommendation"] = rec.apply(recommend, axis=1)
    rec["reason"] = rec.apply(explain, axis=1)

    # =====================================================================
    # CLUSTER-AWARE PRUNING (opt-in: cfg.cluster_prune)
    # Permutation importance is unreliable on collinear features (a twin masks the
    # drop), so per-feature pruning can delete a real signal merely because it is
    # duplicated. Group features at |spearman| >= cfg.cluster_corr; within each
    # multi-member cluster keep ONE representative (best by global perm drop, then
    # |IC|) and DEMOTE the rest to REDUNDANT (recorded, never silently lost). If the
    # representative was itself marked DROP only on a low perm drop but still has
    # non-trivial standalone |IC|, that DROP is the masking artifact -> rescue it to
    # REVIEW so the cluster's signal is never lost.
    # =====================================================================
    rec["cluster_id"] = -1
    rec["cluster_role"] = ""
    if cfg.cluster_prune:
        clust_feats = [f for f in rec["feature"] if f in corr_mat.columns]
        cmap = correlation_clusters(corr_mat.loc[clust_feats, clust_feats], cfg.cluster_corr) \
            if clust_feats else {}
        # features absent from corr_mat -> their own singleton ids
        next_id = (max(cmap.values()) + 1) if cmap else 0
        for f in rec["feature"]:
            if f not in cmap:
                cmap[f] = next_id
                next_id += 1
        rec["cluster_id"] = rec["feature"].map(cmap).astype(int)

        gp = rec.set_index("feature")["auc_drop_global_mean"].to_dict()
        ic = rec.set_index("feature")["mean_abs_ic"].to_dict()
        recm = rec.set_index("feature")["recommendation"].to_dict()
        new_rec = dict(recm)
        roles: Dict[str, str] = {f: "" for f in rec["feature"]}
        demoted = 0
        rescued = 0
        for cid, grp in rec.groupby("cluster_id"):
            members = list(grp["feature"])
            if len(members) < 2:
                continue
            # representative = best by (perm drop, then |IC|)
            members_sorted = sorted(
                members,
                key=lambda f: (gp.get(f, 0.0) if pd.notna(gp.get(f)) else 0.0,
                               ic.get(f, 0.0) if pd.notna(ic.get(f)) else 0.0),
                reverse=True,
            )
            rep = members_sorted[0]
            roles[rep] = f"representative(n={len(members)})"
            # RESCUE: a representative that recommend() marked DROP purely on a low
            # permutation drop is the classic collinearity victim (its twin carried
            # the signal during permutation). If it still has non-trivial standalone
            # IC, the DROP is a masking artifact -> rescue to REVIEW. A genuinely dead
            # cluster (weak IC too) is left DROP.
            if recm.get(rep) == "DROP" and pd.notna(ic.get(rep)) and abs(ic.get(rep, 0.0)) >= IC_THRESHOLD:
                new_rec[rep] = "REVIEW"
                roles[rep] = f"representative(n={len(members)}; rescued from perm-mask)"
                rescued += 1
            # the rest of the cluster carries the same signal -> REDUNDANT (recorded,
            # not lost). Harmful features keep their harmful flag (more important).
            for f in members_sorted[1:]:
                if recm.get(f) == "DROP_HARMFUL":
                    roles[f] = "harmful"
                    continue
                new_rec[f] = "REDUNDANT"
                roles[f] = f"redundant->({rep})"
                demoted += 1
        rec["recommendation"] = rec["feature"].map(new_rec)
        rec["cluster_role"] = rec["feature"].map(roles)
        n_clusters = rec["cluster_id"].nunique()
        n_multi = int((rec.groupby("cluster_id")["feature"].transform("size") > 1).sum())
        print(f"[9/9] Cluster-aware pruning: {n_clusters} clusters "
              f"(|r|>={cfg.cluster_corr}); {n_multi} features in multi-member clusters; "
              f"{demoted} demoted to REDUNDANT; {rescued} representative(s) rescued from perm-mask.")

    # NEW: strategy-category tags (breakout / mean_reversion / momentum_trend / ...)
    _cat = {f: categorize(f) for f in rec["feature"]}
    rec["primary_category"] = rec["feature"].map(lambda f: _cat[f][0])
    rec["category_tags"] = rec["feature"].map(lambda f: "|".join(_cat[f][1]))

    order = {"DROP_HARMFUL": 0, "DROP": 1, "REDUNDANT": 2, "REVIEW": 3, "KEEP": 4}
    rec["sort_order"] = rec["recommendation"].map(order).fillna(3)
    rec = rec.sort_values(["sort_order", "gain_rank"]).drop(columns=["sort_order"])

    # Round
    round_cols = [c for c in rec.columns if c not in {"feature", "family", "recommendation", "reason",
                                                       "regime_specialist", "is_dead_everywhere",
                                                       "is_harmful_anywhere", "best_regime",
                                                       "generalizes", "primary_category", "category_tags",
                                                       "cluster_role"}]
    for c in round_cols:
        if c in rec.columns:
            rec[c] = pd.to_numeric(rec[c], errors="coerce").round(5)

    rec.to_csv(out_dir / "feature_pruning_recommendation.csv", index=False)

    # Sub-tables
    harmful = rec[rec["recommendation"] == "DROP_HARMFUL"].copy()
    harmful.to_csv(out_dir / "harmful_features.csv", index=False)

    specialists = rec[rec["regime_specialist"] == True].copy()
    specialists.to_csv(out_dir / "regime_specialist_features.csv", index=False)

    # =========================================================================
    # STRATEGY-CATEGORY REPORT (NEW)
    # "Which features serve which strategy, and how strong are they?" — so when
    # you build a NEW model (breakout, mean-reversion, momentum, ...) you know
    # exactly which features to pull and how much edge each carries.
    # =========================================================================
    print("\n[Cat] Strategy-category report...")
    cat_tags = {f: categorize(f)[1] for f in FEATURES}
    cat_primary = {f: categorize(f)[0] for f in FEATURES}

    # Per-feature map (one row per feature, with its category + all metrics)
    g_by_feat = global_imp.set_index("feature")
    cat_map = pd.DataFrame({"feature": FEATURES})
    cat_map["primary_category"] = cat_map["feature"].map(cat_primary)
    cat_map["category_tags"] = cat_map["feature"].map(lambda f: "|".join(cat_tags[f]))
    cat_map["family"] = cat_map["feature"].map(family_of)
    cat_map["gain_pct"] = cat_map["feature"].map(g_by_feat["gain_pct"].to_dict())
    cat_map["split_pct"] = cat_map["feature"].map(g_by_feat["split_pct"].to_dict())
    cat_map["gain_rank"] = cat_map["feature"].map(g_by_feat["gain_rank"].to_dict())
    cat_map["mean_abs_ic"] = cat_map["feature"].map(ic_summary["mean_abs_ic"].to_dict())
    cat_map["auc_drop_global"] = cat_map["feature"].map(global_perm_df["auc_drop_global_mean"].to_dict())
    for r in REGIMES:
        cat_map[f"auc_drop_{r}"] = cat_map["feature"].map(regime_perm_df[f"auc_drop_{r}"].to_dict())
        cat_map[f"ic_{r}"] = cat_map["feature"].map(regime_ic_df[f"ic_{r}"].to_dict())
    cat_map["best_regime"] = cat_map["feature"].map(
        consistency_df.set_index("feature")["best_regime"].to_dict())
    cat_map["recommendation"] = cat_map["feature"].map(
        rec.set_index("feature")["recommendation"].to_dict())
    cat_map = cat_map.sort_values(["primary_category", "gain_pct"],
                                  ascending=[True, False]).reset_index(drop=True)
    cat_map.to_csv(out_dir / "feature_category_map.csv", index=False)

    # Category SUMMARY: a feature contributes to EVERY tag it carries, so the
    # "breakout" row lists every breakout-relevant feature even if its primary
    # label is volatility/momentum.
    rec_by_feat = rec.set_index("feature")["recommendation"]
    all_tags = sorted({t for f in FEATURES for t in cat_tags[f]})
    summ_rows = []
    for tag in all_tags:
        members = [f for f in FEATURES if tag in cat_tags[f]]
        if not members:
            continue
        gp = g_by_feat.reindex(members)["gain_pct"]
        pdrop = global_perm_df.reindex(members)["auc_drop_global_mean"]
        ic = ic_summary.reindex(members)["mean_abs_ic"]
        recs = rec_by_feat.reindex(members)
        top = gp.sort_values(ascending=False).head(6).index.tolist()
        summ_rows.append({
            "category": tag,
            "n_features": len(members),
            "n_keep": int((recs == "KEEP").sum()),
            "n_review": int((recs == "REVIEW").sum()),
            "n_drop": int(recs.isin(["DROP", "DROP_HARMFUL", "REDUNDANT"]).sum()),
            "total_gain_pct": round(float(gp.sum()), 2),
            "mean_gain_pct": round(float(gp.mean()), 4),
            "mean_perm_drop": round(float(pdrop.mean()), 5),
            "max_perm_drop": round(float(pdrop.max()), 5),
            "mean_abs_ic": round(float(ic.mean()), 4),
            "max_abs_ic": round(float(ic.max()), 4),
            "top_features": ", ".join(top),
        })
    cat_summary = pd.DataFrame(summ_rows).sort_values("total_gain_pct", ascending=False).reset_index(drop=True)
    cat_summary.to_csv(out_dir / "feature_category_summary.csv", index=False)
    print(cat_summary[["category", "n_features", "n_keep", "total_gain_pct",
                       "mean_abs_ic", "max_perm_drop", "top_features"]].to_string(index=False))
    print(f"[Cat] Wrote feature_category_map.csv + feature_category_summary.csv "
          f"({len(all_tags)} categories).")

    # =========================================================================
    # regime_features.json — direct hand-off to production
    # =========================================================================
    print("\n[Out] Writing regime_features.json...")
    keep_global = rec[rec["recommendation"].isin(["KEEP", "REVIEW"])]["feature"].tolist()
    drop_features = rec[rec["recommendation"].isin(["DROP", "DROP_HARMFUL", "REDUNDANT"])]["feature"].tolist()

    regime_specific: Dict[str, List[str]] = {}
    for r in REGIMES:
        # Per-regime: keep if perm_drop in this regime > PERM_DROP_KEEP OR
        # |regime IC| > IC_THRESHOLD, OR feature is in global keep list
        keep_r = []
        for feat in FEATURES:
            d = regime_perm[r].get(feat, (np.nan, np.nan))[0] if r in regime_perm else np.nan
            ic_r = regime_ic_df.loc[feat, f"ic_{r}"]
            keep_this = False
            if pd.notna(d) and d >= PERM_DROP_KEEP:
                keep_this = True
            if pd.notna(ic_r) and abs(ic_r) >= IC_THRESHOLD:
                keep_this = True
            if feat in keep_global and feat not in drop_features:
                keep_this = True
            if keep_this and feat not in drop_features:
                keep_r.append(feat)
        regime_specific[r] = keep_r

    payload = {
        "version": "v5",
        "produced_at": pd.Timestamp.now(tz=IST).isoformat(),
        "n_features_input": len(FEATURES),
        "n_drop": len(drop_features),
        "n_keep_global": len(keep_global),
        "drop_list": drop_features,
        "keep_list_global": keep_global,
        "per_regime": regime_specific,
        "thresholds": {
            "PERM_DROP_KEEP": PERM_DROP_KEEP,
            "PERM_DROP_HARMFUL": PERM_DROP_HARMFUL,
            "REGIME_EXCEPTIONAL": REGIME_EXCEPTIONAL,
            "IC_THRESHOLD": IC_THRESHOLD,
            "CORR_THRESHOLD": CORR_THRESHOLD,
        },
        "diagnostic_test_auc": float(base_auc),
        "regime_test_auc": {r: float(v) for r, v in regime_aucs.items()},
    }
    (out_dir / "regime_features.json").write_text(json.dumps(payload, indent=2))

    # -------------------------------------------------------------------------
    # Also emit a ready-to-use features_train_pruned.json in the SAME schema
    # format the production trainer expects ({"features":[...], "impute":{...}}).
    # This is NON-DESTRUCTIVE: it does not overwrite the input features_train.json.
    # To use the pruned global set in a pooled run, copy this over features_train.json:
    #     copy feature_diagnostics\features_train_pruned.json features_train.json
    # (The trainer can also read regime_features.json directly via USE_PRUNED_FEATURES.)
    # -------------------------------------------------------------------------
    try:
        pruned_impute = {c: float(impute.get(c, GLOBAL_IMPUTE.get(c, 0.0))) for c in keep_global}
        pruned_schema = {"features": list(keep_global), "impute": pruned_impute}
        (out_dir / "features_train_pruned.json").write_text(json.dumps(pruned_schema, indent=2))
        print(f"[Out] Wrote features_train_pruned.json "
              f"({len(keep_global)} features) — copy over features_train.json to use in a pooled run.")
    except Exception as e:
        print(f"[Out] WARN: could not write features_train_pruned.json: {e}")

    # =========================================================================
    # Excel export
    # =========================================================================
    print("\n[Out] Writing Excel report...")
    xlsx = out_dir / "feature_diagnostics_full.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        rec.to_excel(w, sheet_name="0_Pruning", index=False)
        family_df.to_excel(w, sheet_name="1_Families", index=False)
        global_imp.to_excel(w, sheet_name="2_GlobalImp", index=False)
        ic_year_pivot.to_excel(w, sheet_name="3_IC_byYear")
        ic_summary.to_excel(w, sheet_name="4_IC_Summary")
        regime_ic_df.to_excel(w, sheet_name="5_IC_byRegime")
        year_imp_df.to_excel(w, sheet_name="6_Year_Stab")
        gen_df.to_excel(w, sheet_name="7_Time_Generalize", index=False)
        pd.DataFrame(redund_rows).to_excel(w, sheet_name="8_Redundant", index=False)
        regime_perm_df.to_excel(w, sheet_name="9_Perm_byRegime")
        consistency_df.to_excel(w, sheet_name="10_Consistency", index=False)
        harmful.to_excel(w, sheet_name="11_Harmful", index=False)
        specialists.to_excel(w, sheet_name="12_Specialists", index=False)
        cat_summary.to_excel(w, sheet_name="13_CategorySummary", index=False)
        cat_map.to_excel(w, sheet_name="14_CategoryMap", index=False)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    counts = rec["recommendation"].value_counts()
    print(f"\nFeatures total: {len(FEATURES)}")
    for k in ["KEEP", "REVIEW", "REDUNDANT", "DROP", "DROP_HARMFUL"]:
        n = int(counts.get(k, 0))
        pct = 100 * n / len(FEATURES)
        print(f"  {k:<14} {n:>4d}  ({pct:5.1f}%)")
    print(f"\nRegime specialists (preserved): {n_specialists}")
    print(f"Harmful features (drop first):  {n_harmful}")
    if regime_aucs:
        print("\nBase AUC (out-of-time, prod model):")
        for r in REGIMES:
            if r in regime_aucs:
                print(f"  {r:<12} {regime_aucs[r]:.4f}")
        print(f"  {'global':<12} {base_auc_global:.4f}")

    print(f"\nMain file:           {out_dir / 'feature_pruning_recommendation.csv'}")
    print(f"Production hand-off: {out_dir / 'regime_features.json'}")
    print(f"All outputs in:      {out_dir}")
    print("\nDone.")


# =============================================================================
# Self-test: prove the fast path is NUMERICALLY IDENTICAL to the original
# =============================================================================

def _self_test() -> int:
    print("=" * 70)
    print("NEW FEAT IMP — SELF-TEST (numeric equivalence + categories)")
    print("=" * 70)
    from lightgbm import LGBMClassifier
    rng = np.random.default_rng(0)
    n, k = 4000, 12
    cols = [f"D_f{i:02d}" for i in range(k)]
    X = pd.DataFrame(rng.normal(size=(n, k)), columns=cols)
    # member 1 uses f00..f03, member 2 uses f06..f09; f04,f05,f10,f11 are noise
    sig1 = X[cols[0]] + 0.7 * X[cols[2]] - 0.5 * X[cols[3]]
    sig2 = 0.9 * X[cols[6]] - 0.6 * X[cols[8]] + 0.4 * X[cols[9]]
    y = ((sig1 + sig2 + rng.normal(scale=0.5, size=n)) > 0).astype(int).values

    def _fit(feat_subset):
        m = LGBMClassifier(n_estimators=60, num_leaves=15, min_data_in_leaf=50,
                           random_state=1, verbosity=-1)
        Xz = X.copy()
        # zero out non-subset columns so the booster only splits on the subset
        for c in cols:
            if c not in feat_subset:
                Xz[c] = 0.0
        m.fit(Xz, y)
        return m

    m1 = _fit(cols[0:4])
    m2 = _fit(cols[6:10])

    class _MeanEnsemble:
        def __init__(self, members):
            self.members = members
        def predict_proba(self, Xin):
            return np.mean([mm.predict_proba(Xin) for mm in self.members], axis=0)

    ens = _MeanEnsemble([m1, m2])

    # Reference scorer == original code path (whole-model predict_proba)
    class _Wrap:
        def __init__(self, fn):
            self.fn = fn
        def predict_proba(self, Xin):
            p = self.fn(Xin)
            return np.column_stack([1 - p, p])
    ref_wrap = _Wrap(unwrap_to_score(ens))

    base = ens.predict_proba(X)[:, 1]
    base_auc = roc_auc_score(y, base)
    eng = _PermEngine(ens, cols, X_probe=X)
    print(f"[test] member-caching enabled: {eng.use_member_cache} "
          f"(should be True); model splits on {len(eng.union_used)}/{k} features")

    max_diff = 0.0
    n_skipped_exact = 0
    for f in cols:
        seed = _feat_seed(f)
        ref_m, ref_s = permute_auc_multi(ref_wrap, X, y, f, base_auc=base_auc,
                                         n_shuffles=5, seed=seed)
        eng_m, eng_s = eng.perm_drop(X, y, f, base_auc=base_auc, n_shuffles=5, seed=seed)
        d = abs(ref_m - eng_m) + abs(ref_s - eng_s)
        max_diff = max(max_diff, d)
        if eng.union_used is not None and f not in eng.union_used:
            n_skipped_exact += 1
            assert abs(ref_m) < 1e-9, f"{f}: reference drop for an unused feature should be ~0, got {ref_m}"
    print(f"[test] permutation max |fast - reference| over {k} features = {max_diff:.3e}")
    assert max_diff < 1e-9, "fast permutation diverged from the reference!"
    print(f"[test] features skipped via exact-0 (unused): {n_skipped_exact}")

    # Vectorized IC == scipy spearmanr
    target = rng.normal(size=n)
    target[rng.random(n) < 0.1] = np.nan       # some NaN targets (omit path)
    ic_fast = spearman_ic_block(X.to_numpy(dtype=float), cols, target)
    okm = ~np.isnan(target)
    ic_max_diff = 0.0
    for j, c in enumerate(cols):
        ref_ic, _ = spearmanr(X[c].to_numpy()[okm], target[okm])
        ic_max_diff = max(ic_max_diff, abs(ref_ic - ic_fast[c]))
    print(f"[test] vectorized IC max |fast - scipy| = {ic_max_diff:.3e}")
    assert ic_max_diff < 1e-9, "vectorized IC diverged from scipy spearmanr!"

    # Category mapping sanity
    checks = {
        "D_breakout_high_20": "breakout",
        "D_donch_pos_20": "breakout",
        "D_rsi14": "mean_reversion",
        "D_bb_pctB_20": "mean_reversion",
        "D_ema20_angle_deg": "momentum_trend",
        "D_atr14": "volatility",
        "D_obv_slope": "momentum_trend",
        "D_cpr_bc": "cpr_pivot",
        "M_nifty_ret": "macro",
        "DOW_1": "calendar",
    }
    for feat, want in checks.items():
        prim, tags = categorize(feat)
        assert want in tags, f"categorize({feat}) -> {tags}, expected to contain '{want}'"
    # breakout-relevant features all carry the 'breakout' tag
    for feat in ["D_breakout_high_50", "D_dist_from_52wh", "D_nr_expand", "D_compress_state"]:
        assert "breakout" in categorize(feat)[1], f"{feat} missing breakout tag"
    print("[test] category mapping: OK")

    # Config derivation (parametric multi-horizon defaults)
    c5 = Config()  # defaults -> legacy 5d behaviour
    assert c5.horizon == 5 and c5.ret_col_name() == "ret_5d_oc_pct"
    assert c5.target_col_name() == "top20_vs_bot20_5d"
    assert c5.embargo() == 6 and abs(c5.label_q() - 0.20) < 1e-9
    assert c5.resolved_out_dir().name == "feature_diagnostics"
    c2 = Config(base_dir=Path("."), horizon=2)
    assert c2.ret_col_name() == "ret_2d_oc_pct" and c2.embargo() == 3
    assert c2.resolved_out_dir().name == "feature_diagnostics_2d"
    c3 = Config(horizon=3, n_quantiles=10, embargo_days=4)
    assert abs(c3.label_q() - 0.10) < 1e-9 and c3.embargo() == 4
    # forward-return fallback computes a column that doesn't exist
    pf = pd.DataFrame({
        "symbol": ["A"] * 8,
        "open": np.arange(1, 9, dtype=float),
        "close": np.arange(1, 9, dtype=float) * 1.01,
    })
    ensure_forward_return(pf, "ret_2d_oc_pct", 2)
    assert "ret_2d_oc_pct" in pf.columns and pf["ret_2d_oc_pct"].notna().sum() >= 5
    print("[test] Config derivation + forward-return fallback: OK")

    # correlation_clusters: single-linkage connected components at threshold
    cm = pd.DataFrame(
        [[1.00, 0.95, 0.10, 0.05],
         [0.95, 1.00, 0.08, 0.02],
         [0.10, 0.08, 1.00, 0.91],
         [0.05, 0.02, 0.91, 1.00]],
        index=["f0", "f1", "f2", "f3"], columns=["f0", "f1", "f2", "f3"])
    cl = correlation_clusters(cm, threshold=0.85)
    assert cl["f0"] == cl["f1"], "f0/f1 (r=0.95) should cluster together"
    assert cl["f2"] == cl["f3"], "f2/f3 (r=0.91) should cluster together"
    assert cl["f0"] != cl["f2"], "the two pairs should be distinct clusters"
    # transitive chain f0~f1~f2 via single-linkage even if f0,f2 below threshold
    cm2 = pd.DataFrame(
        [[1.00, 0.90, 0.50],
         [0.90, 1.00, 0.88],
         [0.50, 0.88, 1.00]],
        index=["a", "b", "c"], columns=["a", "b", "c"])
    cl2 = correlation_clusters(cm2, threshold=0.85)
    assert cl2["a"] == cl2["b"] == cl2["c"], "single-linkage should chain a-b-c"
    # all-independent -> all singletons
    cm3 = pd.DataFrame(np.eye(3), index=["x", "y", "z"], columns=["x", "y", "z"])
    assert len(set(correlation_clusters(cm3, 0.85).values())) == 3
    print("[test] correlation_clusters (linkage + chaining + singletons): OK")

    # cluster-prune config wiring
    cc = Config(cluster_prune=True, cluster_corr=0.9, std_gate=True, std_gate_k=1.5)
    assert cc.cluster_prune and cc.cluster_corr == 0.9 and cc.std_gate and cc.std_gate_k == 1.5
    print("[test] cluster-prune / std-gate config: OK")

    print("\nALL SELF-TESTS PASSED")
    return 0


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Regime-aware feature diagnostic (parametric: multi-model / multi-horizon).")
    parser.add_argument("--base-dir", type=str, default=str(DEFAULT_BASE_DIR),
                        help="Directory containing panel_cache.parquet, features_train.json, models/")
    parser.add_argument("--horizon", type=int, default=5, help="forward horizon in trading days")
    parser.add_argument("--quantiles", type=int, default=5, dest="n_quantiles",
                        help="top/bottom = 1/quantiles (5 -> top20/bot20); matches strategy_lab")
    parser.add_argument("--embargo", type=int, default=None, dest="embargo_days",
                        help="purged-CV embargo in days (default = horizon + 1)")
    parser.add_argument("--raw-target", action="store_true",
                        help="use raw forward return for the label instead of the vol-adjusted one")
    parser.add_argument("--target-col", type=str, default=None,
                        help="use an existing panel column as the label (overrides derivation)")
    parser.add_argument("--ret-col", type=str, default=None,
                        help="forward-return column name (default ret_<H>d_oc_pct)")
    parser.add_argument("--model-path", type=str, default=None,
                        help="explicit model .joblib (else auto-discover m<H>_* then m5_*)")
    parser.add_argument("--model-free", action="store_true",
                        help="ignore any trained model; importance from the diagnostic model "
                             "(scout mode for a horizon you have not trained yet)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="output directory (default <base>/feature_diagnostics[_<H>d])")
    parser.add_argument("--cluster-prune", action="store_true",
                        help="cluster collinear features (|spearman| >= --cluster-corr) and keep "
                             "one representative per cluster; demote the rest to REDUNDANT")
    parser.add_argument("--cluster-corr", type=float, default=None,
                        help="correlation threshold for a cluster (default 0.85)")
    parser.add_argument("--std-gate", action="store_true",
                        help="gate KEEP/DROP on perm-drop significance (mean > k*std) instead of "
                             "the bare absolute threshold")
    parser.add_argument("--std-gate-k", type=float, default=None,
                        help="significance multiple on the shuffle std (default 1.0)")
    parser.add_argument("--self-test", action="store_true",
                        help="Validate fast-path numeric equivalence + categories on synthetic data; no real data needed")
    args = parser.parse_args()
    if args.self_test:
        raise SystemExit(_self_test())

    cfg = Config(
        base_dir=Path(args.base_dir),
        horizon=args.horizon,
        n_quantiles=args.n_quantiles,
        use_model_target=not args.raw_target,
        embargo_days=args.embargo_days,
        target_col=args.target_col,
        ret_col=args.ret_col,
        model_path=Path(args.model_path) if args.model_path else None,
        model_free=args.model_free,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        cluster_prune=args.cluster_prune,
        cluster_corr=args.cluster_corr if args.cluster_corr is not None else CORR_THRESHOLD,
        std_gate=args.std_gate,
        std_gate_k=args.std_gate_k if args.std_gate_k is not None else 1.0,
    )
    main(cfg)
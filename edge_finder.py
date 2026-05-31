#!/usr/bin/env python3
"""
edge_finder.py — Strategy screener and edge logger for v20 cache panels.

Reads the leak-free per-symbol parquets produced by `Daily cache.py` (v20+),
assembles a (timestamp, symbol) panel, generates hundreds-to-thousands of
candidate strategies (single-feature, pair interactions, regime-gated,
composites, cross-sectional WQ alphas), scores each strategy on uniform
out-of-sample edge metrics, and writes a sortable catalog so you can decide
which strategies are worth pursuing.

WHAT EACH STRATEGY IS SCORED ON
-------------------------------
  * Daily cross-sectional Spearman IC -> mean / std / IR (annualized) / hit rate
  * Long-short quintile portfolio: daily PnL, ann. Sharpe (overlapping &
    non-overlapping), bootstrap 95% CI, max drawdown, turnover
  * Deflated Sharpe (Bailey & López de Prado), accounting for the total
    number of strategies tested in this run
  * Regime breakdown: IR within trend up/down, vol low/high, liquidity
    tertile, DOW, year — so you can see WHEN an edge is alive
  * IC decay: same strategy at horizons 1d / 5d / 10d
  * Stability: fraction of calendar years with positive IR
  * Decision flag: STRONG / PROMISING / MARGINAL / NO_EDGE / AVOID

LEAK SAFETY
-----------
  * Inputs come from v20 leak-free per-symbol parquets.
  * Panel cs_rank_* columns are computed by groupby('timestamp') (causal).
  * Forward-return columns ret_h{N} are computed but ONLY used on the RHS
    of IC/scoring; they are never inputs to a signal.
  * Cross-sectional WQ alphas (`WQ_*_cs`) are built from the cs_rank_*
    columns at panel level, not from the per-symbol tampered v19 series.

USAGE
-----
    # Real cache panel, all generator families, top-30 features for pairs:
    python edge_finder.py --cache-dir /path/to/cache_daily_new --out edge_out \\
                          --start 2018-01-01 --end 2026-05-30 \\
                          --horizons 1,5,10 --pairs-topk 30 \\
                          --max-strategies 3000 --workers 4

    # Self-test on synthetic data (no real parquets needed):
    python edge_finder.py --self-test --out edge_out_synth
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import math
import os
import sqlite3
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats as _scipy_stats
except Exception:
    _scipy_stats = None

try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Forward-return horizons we score by default (in trading days)
DEFAULT_HORIZONS: Tuple[int, ...] = (1, 5, 10)

# Cross-sectional rank candidates: raw OHLCV + the most informative D_* columns
# These get cs_rank_* siblings at panel time.
CS_RANK_INPUT_COLS: Tuple[str, ...] = (
    "open", "high", "low", "close", "volume",
    "D_atr_pct", "D_dollar_vol", "D_obv_slope",
    "D_macd_hist", "D_rsi14", "D_ema20_angle_deg",
    "D_donch_pos_20", "D_donch_pos_50",
    "D_bb_pctB_20", "D_bb_bw_20",
    "D_cmf20", "D_atr_ratio_14_30",
    "D_range_pct", "D_gap_pct",
    "D_close_roll_slope_20", "D_close_roll_slope_50",
    "D_slope_stability", "D_body_ratio", "D_wick_skew",
)

# Columns that are forbidden as features (labels / time / ids / forward-looking)
FORBIDDEN_FEATURE_PREFIXES: Tuple[str, ...] = ("ret_h", "fwd_", "future_")
FORBIDDEN_FEATURE_NAMES: Tuple[str, ...] = (
    "timestamp", "symbol", "date", "ret_5d_close_pct",
)

# Quintile portfolio bins
N_QUINTILES = 5

# Trading days per year (for annualization)
TDPY = 252.0


# ──────────────────────────────────────────────────────────────────────────────
# Section 1: Panel assembly
# ──────────────────────────────────────────────────────────────────────────────

def load_panel(
    cache_dir: Path | str,
    start: Optional[dt.date] = None,
    end: Optional[dt.date] = None,
    symbols: Optional[Sequence[str]] = None,
    max_symbols: Optional[int] = None,
    min_history_rows: int = 250,
    min_nonnull_close: float = 0.95,
) -> pd.DataFrame:
    """
    Load all `<SYM>_daily.parquet` files in `cache_dir` into a long panel.
    Filter symbols with too little history. Return a frame sorted by
    (timestamp, symbol) with columns: timestamp, symbol, open, high, low,
    close, volume, and every D_/W_/Comb_ feature column the parquets contain.
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        raise FileNotFoundError(f"cache dir not found: {cache_dir}")
    files = sorted(cache_dir.glob("*_daily.parquet"))
    if symbols:
        wanted = {s.upper() for s in symbols}
        files = [f for f in files if f.stem.replace("_daily", "").upper() in wanted]
    if max_symbols:
        files = files[: int(max_symbols)]
    if not files:
        raise RuntimeError(f"no *_daily.parquet found under {cache_dir}")

    frames: List[pd.DataFrame] = []
    skipped: List[str] = []
    for fp in files:
        sym = fp.stem.replace("_daily", "")
        try:
            df = pd.read_parquet(fp)
        except Exception as e:
            skipped.append(f"{sym}:read_error:{e}")
            continue
        if df is None or df.empty or "timestamp" not in df.columns or "close" not in df.columns:
            skipped.append(f"{sym}:bad_schema")
            continue
        if len(df) < min_history_rows:
            skipped.append(f"{sym}:short_history:{len(df)}")
            continue
        n_close_ok = float(pd.to_numeric(df["close"], errors="coerce").notna().mean())
        if n_close_ok < min_nonnull_close:
            skipped.append(f"{sym}:sparse_close:{n_close_ok:.2f}")
            continue
        df["symbol"] = sym
        frames.append(df)
    if not frames:
        raise RuntimeError("after filtering, no symbols survived")

    panel = pd.concat(frames, ignore_index=True, copy=False)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], errors="coerce", utc=False)
    if panel["timestamp"].dt.tz is not None:
        panel["timestamp"] = panel["timestamp"].dt.tz_convert(None)
    panel = panel.dropna(subset=["timestamp", "close"])

    if start is not None:
        panel = panel[panel["timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        panel = panel[panel["timestamp"] <= pd.Timestamp(end)]

    panel = panel.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    # Memory: downcast obvious float64 columns to float32
    float64_cols = [c for c in panel.columns if str(panel[c].dtype) == "float64" and c != "volume"]
    for c in float64_cols:
        panel[c] = panel[c].astype("float32")

    if skipped:
        print(f"[panel] loaded {len(frames)} symbols; skipped {len(skipped)}; first skips: {skipped[:5]}")
    print(f"[panel] rows={len(panel):,}  symbols={panel['symbol'].nunique()}  "
          f"dates={panel['timestamp'].nunique()}  cols={len(panel.columns)}")
    return panel


def add_forward_returns(panel: pd.DataFrame, horizons: Sequence[int] = DEFAULT_HORIZONS) -> pd.DataFrame:
    """
    Compute forward log-returns and arithmetic returns at each horizon, per
    symbol. ret_h{H} = close.shift(-H) / close - 1, computed within each
    symbol's time series. These are scoring labels; never use as inputs.
    """
    out = panel.sort_values(["symbol", "timestamp"]).copy()
    grp = out.groupby("symbol", sort=False)["close"]
    for h in horizons:
        out[f"ret_h{int(h)}"] = grp.transform(lambda s, h=int(h): s.shift(-h) / s - 1.0).astype("float32")
    return out.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def add_cross_sectional_ranks(panel: pd.DataFrame, cols: Sequence[str] = CS_RANK_INPUT_COLS) -> pd.DataFrame:
    """
    Add cs_rank_<col> = panel.groupby('timestamp')[col].rank(pct=True) for
    each requested col. This is the missing piece for genuine WQ-style
    cross-sectional alphas — done at panel time, strictly causal.
    """
    available = [c for c in cols if c in panel.columns]
    if not available:
        return panel
    print(f"[panel] adding {len(available)} cross-sectional ranks")
    g = panel.groupby("timestamp")
    for c in available:
        col = pd.to_numeric(panel[c], errors="coerce")
        panel[f"cs_rank_{c}"] = g[c].transform(lambda s: pd.to_numeric(s, errors="coerce").rank(pct=True)).astype("float32")
    return panel


def _ts_corr_per_symbol(a: pd.Series, b: pd.Series, sym: pd.Series, w: int) -> pd.Series:
    """Per-symbol rolling correlation, panel-flattened back."""
    df = pd.DataFrame({"a": a, "b": b, "sym": sym})
    return df.groupby("sym", sort=False).apply(
        lambda g: g["a"].rolling(w, min_periods=max(5, w // 2)).corr(g["b"])
    ).reset_index(level=0, drop=True).reindex(a.index)


def _ts_apply_per_symbol(s: pd.Series, sym: pd.Series, fn: Callable[[pd.Series], pd.Series]) -> pd.Series:
    df = pd.DataFrame({"s": s, "sym": sym})
    return df.groupby("sym", sort=False)["s"].transform(fn)


def add_panel_wq_alphas(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Build the *real* cross-sectional WQ alphas from the cs_rank_* columns
    that already exist on the panel. The names end in `_cs` to distinguish
    them from the v20 per-symbol expanding-rank versions in the parquet
    cache (D_WQ_*).

    Only a subset is built (the ones with cs_rank-friendly inputs already
    materialized). Add more by computing more cs_rank_* columns first.
    """
    needed = {"cs_rank_open", "cs_rank_volume", "cs_rank_close", "cs_rank_high"}
    if not needed.issubset(panel.columns):
        print(f"[panel] skipping cs WQ alphas — need {needed - set(panel.columns)}")
        return panel

    sym = panel["symbol"]
    ro = panel["cs_rank_open"]
    rv = panel["cs_rank_volume"]
    rc = panel["cs_rank_close"]
    rh = panel["cs_rank_high"]

    # WQ_3_cs : -ts_corr(cs_rank(open), cs_rank(volume), 10)
    panel["WQ_3_cs"] = (-_ts_corr_per_symbol(ro, rv, sym, 10)).astype("float32")
    # WQ_44_cs: -ts_corr(high, cs_rank(volume), 5)
    high = pd.to_numeric(panel["high"], errors="coerce")
    panel["WQ_44_cs"] = (-_ts_corr_per_symbol(high, rv, sym, 5)).astype("float32")
    # WQ_40_cs: -cs_rank(ts_std(high,10)) * ts_corr(high, volume, 10)
    vol = pd.to_numeric(panel["volume"], errors="coerce")
    ts_std_high_10 = _ts_apply_per_symbol(high, sym, lambda s: s.rolling(10, min_periods=5).std())
    cs_rank_ts_std_h10 = ts_std_high_10.groupby(panel["timestamp"]).rank(pct=True)
    ts_corr_hv = _ts_corr_per_symbol(high, vol, sym, 10)
    panel["WQ_40_cs"] = (-cs_rank_ts_std_h10 * ts_corr_hv).astype("float32")

    # WQ_38_cs: -cs_rank(ts_rank(close,10)) * cs_rank(close/open)
    close = pd.to_numeric(panel["close"], errors="coerce")
    open_ = pd.to_numeric(panel["open"], errors="coerce")
    ts_rank_c10 = _ts_apply_per_symbol(close, sym, lambda s: s.rolling(10, min_periods=5).rank(pct=True))
    cs_rank_tsrc10 = ts_rank_c10.groupby(panel["timestamp"]).rank(pct=True)
    co = (close / open_.replace(0, np.nan))
    cs_rank_co = co.groupby(panel["timestamp"]).rank(pct=True)
    panel["WQ_38_cs"] = (-cs_rank_tsrc10 * cs_rank_co).astype("float32")

    # WQ_33_cs: cs_rank(-1 + open/close)
    panel["WQ_33_cs"] = (
        (-1 + open_ / close.replace(0, np.nan)).groupby(panel["timestamp"]).rank(pct=True)
    ).astype("float32")

    return panel


def add_regime_columns(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add categorical regime columns (per row) used both as conditioning gates
    and as breakdown dimensions for scoring.

      reg_trend     in {-1: down, 0: range, +1: up}     (close vs sma200; sma50 vs sma200)
      reg_vol       in {0: low, 1: mid, 2: high}        (cs tertile of D_atr_pct that day)
      reg_liq       in {0: low, 1: mid, 2: high}        (cs tertile of D_dollar_vol that day)
      reg_dow       0..4
      reg_year      calendar year (int)
    """
    out = panel
    has_sma50 = "D_sma50" in out.columns
    has_sma200 = "D_sma200" in out.columns
    if has_sma50 and has_sma200:
        c = pd.to_numeric(out["close"], errors="coerce")
        s50 = pd.to_numeric(out["D_sma50"], errors="coerce")
        s200 = pd.to_numeric(out["D_sma200"], errors="coerce")
        up = (c > s200) & (s50 > s200)
        dn = (c < s200) & (s50 < s200)
        reg = np.where(up, 1, np.where(dn, -1, 0)).astype("int8")
        out["reg_trend"] = reg
    else:
        out["reg_trend"] = 0

    if "D_atr_pct" in out.columns:
        out["reg_vol"] = (
            out.groupby("timestamp")["D_atr_pct"]
            .transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=False, duplicates="drop"))
            .astype("Int8")
            .fillna(1)
            .astype("int8")
        )
    else:
        out["reg_vol"] = 1

    if "D_dollar_vol" in out.columns:
        out["reg_liq"] = (
            out.groupby("timestamp")["D_dollar_vol"]
            .transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=False, duplicates="drop"))
            .astype("Int8")
            .fillna(1)
            .astype("int8")
        )
    else:
        out["reg_liq"] = 1

    out["reg_dow"] = out["timestamp"].dt.dayofweek.astype("int8")
    out["reg_year"] = out["timestamp"].dt.year.astype("int16")
    return out


def pick_feature_columns(
    panel: pd.DataFrame,
    *,
    max_features: Optional[int] = None,
    max_nan_rate: float = 0.30,
    min_unique: int = 8,
    drop_correlated_above: float = 0.97,
) -> List[str]:
    """
    Pick numeric feature columns from the panel:
      - prefix D_, W_, Comb_, cs_rank_, WQ_, plus core structural/Wq cols
      - exclude forbidden labels / time / id / forward-looking
      - NaN rate <= max_nan_rate
      - >= min_unique distinct values
      - de-dupe by absolute Spearman correlation > drop_correlated_above
        (sample 50k rows for speed)
    """
    candidates: List[str] = []
    for c in panel.columns:
        if c in FORBIDDEN_FEATURE_NAMES:
            continue
        if any(c.startswith(p) for p in FORBIDDEN_FEATURE_PREFIXES):
            continue
        if str(panel[c].dtype) == "object":
            continue
        if not (
            c.startswith(("D_", "W_", "Comb_", "cs_rank_", "WQ_"))
            or c in ("open", "high", "low", "close", "volume")
        ):
            continue
        s = pd.to_numeric(panel[c], errors="coerce")
        if s.isna().mean() > max_nan_rate:
            continue
        if s.dropna().nunique() < min_unique:
            continue
        candidates.append(c)

    if not candidates:
        return []

    # De-dupe by correlation on a random sample
    sample = panel[candidates].sample(min(50_000, len(panel)), random_state=0)
    corr = sample.astype("float32").corr(method="pearson").abs()
    keep: List[str] = []
    dropped: set = set()
    for c in candidates:
        if c in dropped:
            continue
        keep.append(c)
        # drop later columns with corr > threshold to c
        same = corr.loc[c, candidates]
        for c2 in candidates:
            if c2 == c or c2 in dropped or c2 in keep:
                continue
            v = same.get(c2, 0.0)
            if pd.notna(v) and abs(v) > drop_correlated_above:
                dropped.add(c2)

    if max_features and len(keep) > max_features:
        keep = keep[:max_features]
    print(f"[features] candidates={len(candidates)}  kept_after_dedupe={len(keep)}")
    return keep


# ──────────────────────────────────────────────────────────────────────────────
# Section 2: Strategy primitives
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StrategySpec:
    """
    A strategy is a recipe to materialize a per-row signal series from the
    panel. `direction=+1` means larger signal -> long; `-1` flips it.
    """
    name: str
    family: str          # 'single' | 'pair' | 'regime' | 'composite' | 'cs_alpha'
    horizon: int
    direction: int
    feature_cols: Tuple[str, ...]
    transform: str       # 'identity' | 'product' | 'sign_cond' | 'gate' | 'zsum' | 'within_regime'
    transform_args: Tuple[Tuple[str, Any], ...] = ()  # immutable kv pairs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "family": self.family, "horizon": self.horizon,
            "direction": self.direction, "feature_cols": list(self.feature_cols),
            "transform": self.transform,
            "transform_args": dict(self.transform_args),
        }


def materialize_signal(spec: StrategySpec, panel: pd.DataFrame) -> pd.Series:
    """
    Compute the signal column for `spec` on `panel`. Returns a float32
    Series aligned to panel.index. NaN where the signal is undefined.
    """
    cols = spec.feature_cols
    args = dict(spec.transform_args)

    if spec.transform == "identity":
        s = pd.to_numeric(panel[cols[0]], errors="coerce")

    elif spec.transform == "product":
        a = pd.to_numeric(panel[cols[0]], errors="coerce")
        b = pd.to_numeric(panel[cols[1]], errors="coerce")
        s = a * b

    elif spec.transform == "sign_cond":
        # sign(A) * B
        a = pd.to_numeric(panel[cols[0]], errors="coerce")
        b = pd.to_numeric(panel[cols[1]], errors="coerce")
        s = np.sign(a) * b

    elif spec.transform == "gate":
        # `cols[0]` is the signal; `cols[1]` is the gate; only fire when gate
        # is in the top quintile of its same-day cs distribution
        a = pd.to_numeric(panel[cols[0]], errors="coerce")
        b = pd.to_numeric(panel[cols[1]], errors="coerce")
        b_rank = b.groupby(panel["timestamp"]).rank(pct=True)
        s = a.where(b_rank >= 0.8)

    elif spec.transform == "within_regime":
        regime_col = args.get("regime_col", "reg_trend")
        regime_val = args.get("regime_val", 1)
        a = pd.to_numeric(panel[cols[0]], errors="coerce")
        s = a.where(panel[regime_col] == regime_val)

    elif spec.transform == "zsum":
        # equal-weight z-score sum (per-day cross-sectional z) of cols
        parts = []
        for c in cols:
            x = pd.to_numeric(panel[c], errors="coerce")
            mean = x.groupby(panel["timestamp"]).transform("mean")
            std = x.groupby(panel["timestamp"]).transform("std").replace(0, np.nan)
            parts.append((x - mean) / std)
        s = sum(parts) / len(parts)

    else:
        raise ValueError(f"unknown transform: {spec.transform}")

    s = pd.to_numeric(s, errors="coerce").astype("float32")
    if spec.direction == -1:
        s = -s
    return s


# ──────────────────────────────────────────────────────────────────────────────
# Section 3: Strategy generators
# ──────────────────────────────────────────────────────────────────────────────

def gen_single_feature_strategies(
    feature_cols: Sequence[str],
    horizons: Sequence[int],
    *,
    both_directions: bool = False,
) -> Iterator[StrategySpec]:
    for c in feature_cols:
        for h in horizons:
            yield StrategySpec(
                name=f"single::{c}::h{h}", family="single", horizon=int(h),
                direction=+1, feature_cols=(c,), transform="identity",
            )
            if both_directions:
                yield StrategySpec(
                    name=f"single::-{c}::h{h}", family="single", horizon=int(h),
                    direction=-1, feature_cols=(c,), transform="identity",
                )


def gen_pair_interaction_strategies(
    top_features: Sequence[str],
    horizons: Sequence[int],
    *,
    modes: Sequence[str] = ("product", "sign_cond", "gate"),
    max_pairs: Optional[int] = None,
) -> Iterator[StrategySpec]:
    pairs = list(itertools.combinations(top_features, 2))
    if max_pairs:
        pairs = pairs[: int(max_pairs)]
    for a, b in pairs:
        for mode in modes:
            for h in horizons:
                if mode == "product":
                    yield StrategySpec(
                        name=f"pair::{a}*{b}::h{h}", family="pair", horizon=int(h),
                        direction=+1, feature_cols=(a, b), transform="product",
                    )
                elif mode == "sign_cond":
                    yield StrategySpec(
                        name=f"pair::sign({a})*{b}::h{h}", family="pair", horizon=int(h),
                        direction=+1, feature_cols=(a, b), transform="sign_cond",
                    )
                elif mode == "gate":
                    yield StrategySpec(
                        name=f"pair::{a}_gated_by_top20({b})::h{h}", family="pair", horizon=int(h),
                        direction=+1, feature_cols=(a, b), transform="gate",
                    )


def gen_regime_conditioned_strategies(
    top_features: Sequence[str],
    regimes: Sequence[Tuple[str, Any, str]],
    horizons: Sequence[int],
) -> Iterator[StrategySpec]:
    """
    `regimes`: list of (regime_col, regime_val, label) e.g.
       ("reg_trend", 1, "trend_up"), ("reg_vol", 0, "vol_low"), ...
    """
    for c in top_features:
        for (rcol, rval, rlabel) in regimes:
            for h in horizons:
                yield StrategySpec(
                    name=f"regime::{c}|{rlabel}::h{h}", family="regime", horizon=int(h),
                    direction=+1, feature_cols=(c,), transform="within_regime",
                    transform_args=(("regime_col", rcol), ("regime_val", rval), ("regime_label", rlabel)),
                )


def gen_composite_strategies(
    top_features: Sequence[str],
    horizons: Sequence[int],
    *,
    sizes: Sequence[int] = (3, 5, 8),
) -> Iterator[StrategySpec]:
    """Equal-weight z-score sums of the top-K features, for K in `sizes`."""
    for k in sizes:
        if k > len(top_features):
            continue
        members = tuple(top_features[:k])
        for h in horizons:
            yield StrategySpec(
                name=f"composite::zsum_top{k}::h{h}", family="composite", horizon=int(h),
                direction=+1, feature_cols=members, transform="zsum",
            )


def gen_cs_wq_alpha_strategies(
    panel: pd.DataFrame,
    horizons: Sequence[int],
) -> Iterator[StrategySpec]:
    """Wraps any WQ_*_cs columns that exist on the panel as single-feature strategies."""
    for c in [col for col in panel.columns if col.startswith("WQ_") and col.endswith("_cs")]:
        for h in horizons:
            yield StrategySpec(
                name=f"cs_alpha::{c}::h{h}", family="cs_alpha", horizon=int(h),
                direction=+1, feature_cols=(c,), transform="identity",
            )


# ──────────────────────────────────────────────────────────────────────────────
# Section 4: Scoring engine
# ──────────────────────────────────────────────────────────────────────────────

def daily_ic(
    panel_subset: pd.DataFrame,
    signal_col: str,
    fwd_ret_col: str,
    method: str = "spearman",
) -> pd.Series:
    """
    Per-day cross-sectional rank IC, fully vectorized.

    Drops days with <3 finite pairs. Uses per-day rank-transform + Pearson
    on the ranks (mathematically identical to Spearman, ~30x faster than
    groupby.apply because we avoid the Python per-group call).
    """
    sub = panel_subset[["timestamp", signal_col, fwd_ret_col]].dropna()
    if sub.empty:
        return pd.Series([], dtype="float64")

    if method == "spearman":
        # Per-day rank transform of both columns -> Pearson IC
        ts = sub["timestamp"].to_numpy()
        sub = sub.assign(
            _s=sub.groupby("timestamp")[signal_col].rank(method="average", pct=True),
            _r=sub.groupby("timestamp")[fwd_ret_col].rank(method="average", pct=True),
        )
    else:
        sub = sub.assign(_s=sub[signal_col].astype("float64"), _r=sub[fwd_ret_col].astype("float64"))

    g = sub.groupby("timestamp", sort=True)
    n = g["_s"].transform("count")
    sm = g["_s"].transform("mean")
    rm = g["_r"].transform("mean")
    sd = sub["_s"] - sm
    rd = sub["_r"] - rm
    num = (sd * rd).groupby(sub["timestamp"], sort=True).sum()
    sss = (sd * sd).groupby(sub["timestamp"], sort=True).sum()
    rss = (rd * rd).groupby(sub["timestamp"], sort=True).sum()
    counts = n.groupby(sub["timestamp"], sort=True).first()
    ic = num / np.sqrt(sss * rss)
    ic = ic.where(counts >= 3)
    return ic.dropna()


def quintile_long_short_pnl(
    panel_subset: pd.DataFrame,
    signal_col: str,
    fwd_ret_col: str,
    n_quintiles: int = N_QUINTILES,
) -> pd.Series:
    """
    Daily long-top-quintile, short-bottom-quintile return series.
    Vectorized — uses per-day rank then bin = floor(rank * Q), no qcut.
    """
    sub = panel_subset[["timestamp", "symbol", signal_col, fwd_ret_col]].dropna()
    if sub.empty:
        return pd.Series([], dtype="float64")

    # Rank within each day (pct -> [0,1)); bin = floor(pct * n_quintiles)
    sub = sub.assign(
        _r=sub.groupby("timestamp")[signal_col].rank(method="first", pct=True)
    )
    bins = np.minimum((sub["_r"].to_numpy() * n_quintiles).astype("int64"), n_quintiles - 1)
    sub = sub.assign(_q=bins)

    long_ret = sub.loc[sub["_q"] == n_quintiles - 1].groupby("timestamp")[fwd_ret_col].mean()
    short_ret = sub.loc[sub["_q"] == 0].groupby("timestamp")[fwd_ret_col].mean()
    return long_ret.sub(short_ret, fill_value=0.0).dropna()


def turnover_per_horizon(panel_subset: pd.DataFrame, signal_col: str, n_quintiles: int = N_QUINTILES) -> float:
    """Average fraction of names whose quintile bin changes day-over-day."""
    sub = panel_subset[["timestamp", "symbol", signal_col]].dropna().copy()
    if sub.empty:
        return float("nan")
    sub["_r"] = sub.groupby("timestamp")[signal_col].rank(method="first", pct=True)
    sub["q"] = np.minimum((sub["_r"].to_numpy() * n_quintiles).astype("int64"), n_quintiles - 1)
    sub = sub.sort_values(["symbol", "timestamp"])
    sub["q_prev"] = sub.groupby("symbol")["q"].shift(1)
    sub = sub.dropna(subset=["q_prev"])
    return float((sub["q"] != sub["q_prev"]).mean())


def annualized_sharpe(daily_ret: pd.Series, periods_per_year: float = TDPY) -> float:
    if daily_ret.empty:
        return float("nan")
    mu = float(daily_ret.mean())
    sd = float(daily_ret.std(ddof=1))
    if sd == 0 or not np.isfinite(sd):
        return float("nan")
    return mu / sd * math.sqrt(periods_per_year)


def max_drawdown(daily_ret: pd.Series) -> float:
    if daily_ret.empty:
        return float("nan")
    cum = (1.0 + daily_ret.fillna(0.0)).cumprod()
    peak = cum.cummax()
    dd = (cum / peak - 1.0).min()
    return float(dd) if np.isfinite(dd) else float("nan")


def bootstrap_sharpe_ci(
    daily_ret: pd.Series, n_boot: int = 500, alpha: float = 0.05, seed: int = 0,
) -> Tuple[float, float]:
    if daily_ret.empty or len(daily_ret) < 30:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = daily_ret.to_numpy(dtype="float64")
    n = arr.size
    sharpes = np.empty(n_boot, dtype="float64")
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        sample = arr[idx]
        sd = sample.std(ddof=1)
        sharpes[i] = (sample.mean() / sd * math.sqrt(TDPY)) if sd > 0 else np.nan
    sharpes = sharpes[np.isfinite(sharpes)]
    if sharpes.size < 10:
        return (float("nan"), float("nan"))
    lo = float(np.quantile(sharpes, alpha / 2))
    hi = float(np.quantile(sharpes, 1 - alpha / 2))
    return (lo, hi)


def _inv_norm_cdf(p: float) -> float:
    """Inverse standard-normal CDF (uses scipy if present, else Acklam approx)."""
    if _scipy_stats is not None:
        return float(_scipy_stats.norm.ppf(p))
    # Acklam's algorithm
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _norm_cdf(z: float) -> float:
    if _scipy_stats is not None:
        return float(_scipy_stats.norm.cdf(z))
    return float(0.5 * (1 + math.erf(z / math.sqrt(2))))


def expected_max_sharpe(n_trials: int, var_sr_trials: float) -> float:
    """
    Expected maximum of `n_trials` Sharpe estimates under H0 (true SR == 0),
    per Bailey & López de Prado (2014), eq. for E[max].

        E[max SR] ≈ sqrt(Var_trials) * [ (1-γ)·Z^{-1}(1 - 1/N) + γ·Z^{-1}(1 - 1/(N·e)) ]

    `var_sr_trials` is the variance of the (per-period) Sharpe estimates ACROSS
    the strategies tested. All quantities are in per-period Sharpe units.
    """
    if n_trials is None or n_trials < 2 or not np.isfinite(var_sr_trials) or var_sr_trials <= 0:
        return 0.0
    gamma = 0.5772156649015329  # Euler-Mascheroni
    z1 = _inv_norm_cdf(1.0 - 1.0 / n_trials)
    z2 = _inv_norm_cdf(1.0 - 1.0 / (n_trials * math.e))
    factor = (1.0 - gamma) * z1 + gamma * z2
    return math.sqrt(var_sr_trials) * factor


def deflated_sharpe_ratio(
    sr_periodic: float,
    n_obs: int,
    sr_star: float,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """
    Deflated Sharpe Ratio (Bailey & López de Prado 2014): the probability that
    the TRUE per-period Sharpe exceeds the multiple-testing benchmark `sr_star`,
    given a non-normal return distribution.

        DSR = Φ[ (SR - SR*) · sqrt(n_obs - 1) / sqrt(1 - γ3·SR + ((γ4-1)/4)·SR²) ]

    ALL Sharpe inputs are PER-PERIOD (e.g. per non-overlapping holding period),
    NOT annualized. `sr_star` comes from expected_max_sharpe() using the
    empirical cross-trial variance. Range [0, 1]; > 0.95 ≈ "survives the
    multiple-testing correction".
    """
    if not np.isfinite(sr_periodic) or n_obs is None or n_obs < 30:
        return float("nan")
    sr2 = sr_periodic * sr_periodic
    denom_var = 1.0 - skew * sr_periodic + ((kurt - 1.0) / 4.0) * sr2
    if not np.isfinite(denom_var) or denom_var <= 0:
        denom_var = 1.0
    z = (sr_periodic - sr_star) * math.sqrt(n_obs - 1) / math.sqrt(denom_var)
    return _norm_cdf(z)


def regime_breakdown(
    panel_subset: pd.DataFrame, signal_col: str, fwd_ret_col: str,
    regime_col: str,
) -> Dict[Any, Dict[str, float]]:
    """Per-regime IC mean / IR / hit-rate / Sharpe of L/S."""
    out: Dict[Any, Dict[str, float]] = {}
    for val, sub in panel_subset.groupby(regime_col):
        ic = daily_ic(sub, signal_col, fwd_ret_col)
        ls = quintile_long_short_pnl(sub, signal_col, fwd_ret_col)
        if ic.empty:
            continue
        ic_mean = float(ic.mean())
        ic_std = float(ic.std(ddof=1)) if len(ic) > 1 else float("nan")
        ic_ir = (ic_mean / ic_std * math.sqrt(TDPY)) if ic_std and np.isfinite(ic_std) and ic_std > 0 else float("nan")
        out[str(val)] = {
            "n_days": int(len(ic)),
            "ic_mean": ic_mean,
            "ic_ir": ic_ir,
            "hit_rate": float((ic > 0).mean()),
            "ls_sharpe": annualized_sharpe(ls),
        }
    return out


def yearly_ir(ic_series: pd.Series) -> Dict[int, float]:
    """IR within each calendar year. Used for stability score."""
    if ic_series.empty:
        return {}
    df = ic_series.to_frame("ic")
    df["yr"] = df.index.year
    out: Dict[int, float] = {}
    for yr, g in df.groupby("yr"):
        s = g["ic"]
        if len(s) < 20:
            continue
        sd = s.std(ddof=1)
        if not np.isfinite(sd) or sd == 0:
            continue
        out[int(yr)] = float(s.mean() / sd * math.sqrt(TDPY))
    return out


def score_strategy(
    panel: pd.DataFrame,
    spec: StrategySpec,
    signal: pd.Series,
    *,
    compute_regimes: bool = True,
    compute_bootstrap: bool = True,
) -> Dict[str, Any]:
    """
    Single-strategy scoring. Builds the (timestamp, symbol, signal, ret) view
    and computes IC + L/S Sharpe + bootstrap CI + regime breakdown + decay.

    `compute_regimes` / `compute_bootstrap` can be disabled for a fast first
    pass; run_edge_finder re-scores the top strategies with them enabled.
    """
    fwd_ret_col = f"ret_h{int(spec.horizon)}"
    if fwd_ret_col not in panel.columns:
        return {"name": spec.name, "error": f"missing {fwd_ret_col}"}

    view = pd.DataFrame({
        "timestamp": panel["timestamp"].values,
        "symbol": panel["symbol"].values,
        "signal": signal.values,
        fwd_ret_col: panel[fwd_ret_col].values,
        "reg_trend": panel["reg_trend"].values,
        "reg_vol": panel["reg_vol"].values,
        "reg_liq": panel["reg_liq"].values,
        "reg_dow": panel["reg_dow"].values,
        "reg_year": panel["reg_year"].values,
    })

    n_finite = int(view[["signal", fwd_ret_col]].dropna().shape[0])
    if n_finite < 500:
        return {
            "name": spec.name, "family": spec.family, "horizon": spec.horizon,
            "direction": spec.direction, "n_obs": n_finite,
            "decision": "INSUFFICIENT_DATA",
            "spec": spec.to_dict(),
        }

    ic = daily_ic(view, "signal", fwd_ret_col)
    ls = quintile_long_short_pnl(view, "signal", fwd_ret_col)

    if ic.empty or ls.empty:
        return {
            "name": spec.name, "family": spec.family, "horizon": spec.horizon,
            "direction": spec.direction, "n_obs": n_finite,
            "decision": "INSUFFICIENT_DATA",
            "spec": spec.to_dict(),
        }

    ic_mean = float(ic.mean())
    ic_std = float(ic.std(ddof=1)) if len(ic) > 1 else float("nan")
    ic_ir = (ic_mean / ic_std * math.sqrt(TDPY)) if ic_std and np.isfinite(ic_std) and ic_std > 0 else float("nan")
    ic_t = (ic_mean / (ic_std / math.sqrt(len(ic)))) if ic_std and np.isfinite(ic_std) and ic_std > 0 else float("nan")
    hit_rate = float((ic > 0).mean())

    # Auto-flip direction if observed IR is meaningfully negative — the signal
    # carries information, just inverted. We log original direction in `spec`.
    # When we flip, we also negate the signal column in `view` so that every
    # downstream metric (regime breakdowns, turnover) is consistent with the
    # reported direction.
    autoflip = False
    if np.isfinite(ic_ir) and ic_ir < -0.1 and spec.direction == +1:
        ic = -ic
        ls = -ls
        ic_mean = -ic_mean
        ic_ir = -ic_ir
        hit_rate = float((ic > 0).mean())
        view["signal"] = -view["signal"]
        autoflip = True

    # Long-short metrics. For overlapping returns at horizon h>1, also compute
    # non-overlapping Sharpe by sub-sampling every h-th day.
    sharpe_overlap = annualized_sharpe(ls)
    if spec.horizon > 1:
        ls_nonoverlap = ls.iloc[:: int(spec.horizon)].dropna()
    else:
        ls_nonoverlap = ls.dropna()
    sd_no = float(ls_nonoverlap.std(ddof=1)) if len(ls_nonoverlap) > 1 else float("nan")
    if sd_no and np.isfinite(sd_no) and sd_no > 0:
        # per-period (non-overlapping) Sharpe — the units the DSR needs
        sr_periodic = float(ls_nonoverlap.mean() / sd_no)
        ppy = TDPY / max(int(spec.horizon), 1)
        sharpe_nonoverlap = sr_periodic * math.sqrt(ppy)
    else:
        sr_periodic = float("nan")
        sharpe_nonoverlap = sharpe_overlap
    n_periodic = int(len(ls_nonoverlap))

    boot_lo, boot_hi = bootstrap_sharpe_ci(ls) if compute_bootstrap else (float("nan"), float("nan"))
    mdd = max_drawdown(ls)

    # Skew/kurt of the non-overlapping L/S series for the DSR correction
    series_for_moments = ls_nonoverlap if len(ls_nonoverlap) >= 30 else ls.dropna()
    if len(series_for_moments) >= 30 and _scipy_stats is not None:
        sk = float(_scipy_stats.skew(series_for_moments))
        kt = float(_scipy_stats.kurtosis(series_for_moments, fisher=False))
    else:
        sk = 0.0
        kt = 3.0

    # Regime breakdowns (skippable for the cheap first pass)
    if compute_regimes:
        rb_trend = regime_breakdown(view, "signal", fwd_ret_col, "reg_trend")
        rb_vol = regime_breakdown(view, "signal", fwd_ret_col, "reg_vol")
        rb_liq = regime_breakdown(view, "signal", fwd_ret_col, "reg_liq")
        rb_dow = regime_breakdown(view, "signal", fwd_ret_col, "reg_dow")
    else:
        rb_trend = rb_vol = rb_liq = rb_dow = {}

    # Year-by-year IR & stability
    yir = yearly_ir(ic)
    if yir:
        stab = float(np.mean([1 if v > 0 else 0 for v in yir.values()]))
    else:
        stab = float("nan")

    turnover = turnover_per_horizon(view, "signal")

    return {
        "name": spec.name,
        "family": spec.family,
        "horizon": int(spec.horizon),
        "direction": spec.direction * (-1 if autoflip else 1),
        "autoflipped": bool(autoflip),
        "feature_cols": list(spec.feature_cols),
        "transform": spec.transform,
        "transform_args": dict(spec.transform_args),
        "n_obs": n_finite,
        "n_days": int(len(ic)),
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_ir": ic_ir,
        "ic_t": ic_t,
        "hit_rate": hit_rate,
        "ls_sharpe": sharpe_overlap,
        "ls_sharpe_nonoverlap": sharpe_nonoverlap,
        "ls_sharpe_boot_lo": boot_lo,
        "ls_sharpe_boot_hi": boot_hi,
        "max_drawdown": mdd,
        "turnover": turnover,
        "sk": sk,
        "kt": kt,
        "sr_periodic": sr_periodic,
        "n_periodic": n_periodic,
        "yearly_ir": {str(k): v for k, v in yir.items()},
        "regime_stability": stab,
        "rb_trend": rb_trend,
        "rb_vol": rb_vol,
        "rb_liq": rb_liq,
        "rb_dow": rb_dow,
        "spec": spec.to_dict(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Section 5: Run + catalog
# ──────────────────────────────────────────────────────────────────────────────

def assign_decision_flag(row: Dict[str, Any]) -> str:
    """
    Combine IC information ratio, the deflated Sharpe (multiple-testing
    survival), and cross-year stability into a single verdict.

    DSR is the gatekeeper: a strategy that does not beat the best-of-N noise
    benchmark (low DSR) cannot be STRONG/PROMISING no matter how good its raw
    IR looks, because raw IR is exactly what gets inflated by screening
    thousands of hypotheses.
    """
    if row.get("decision") == "ERROR":
        return "ERROR"
    if row.get("decision") == "INSUFFICIENT_DATA":
        return "INSUFFICIENT_DATA"
    ir = row.get("ic_ir", float("nan"))
    dsr = row.get("dsr", float("nan"))
    stab = row.get("regime_stability", 0.0) or 0.0
    if not np.isfinite(ir):
        return "INSUFFICIENT_DATA"

    has_dsr = np.isfinite(dsr)
    if ir < -0.1:
        return "AVOID"
    if abs(ir) <= 0.1:
        return "NO_EDGE"
    # DSR gate: survives the best-of-N deflation with high confidence
    if ir > 0.5 and stab >= 0.6 and (dsr >= 0.95 if has_dsr else False):
        return "STRONG"
    if ir > 0.3 and stab >= 0.5 and (dsr >= 0.5 if has_dsr else False):
        return "PROMISING"
    # Has univariate signal but does not survive multiple-testing deflation
    return "MARGINAL"


def run_edge_finder(
    panel: pd.DataFrame,
    strategies: Sequence[StrategySpec],
    *,
    n_jobs: int = 1,
    verbose: bool = True,
    regime_detail_topk: int = 100,
) -> pd.DataFrame:
    """
    Score every strategy in two passes:
      1) cheap pass — IC + L/S Sharpe + DSR inputs, no regime/bootstrap detail
      2) detail pass — recompute the top `regime_detail_topk` strategies (by
         |ic_ir|) WITH regime breakdowns and bootstrap CIs.
    Single-process unless joblib is installed and n_jobs>1.
    """
    t0 = time.perf_counter()

    def _one(spec: StrategySpec, regimes: bool, boot: bool) -> Dict[str, Any]:
        try:
            sig = materialize_signal(spec, panel)
            return score_strategy(panel, spec, sig,
                                  compute_regimes=regimes, compute_bootstrap=boot)
        except Exception as e:
            return {"name": spec.name, "error": f"{type(e).__name__}: {e}",
                    "spec": spec.to_dict(), "decision": "ERROR"}

    # ---- Pass 1: cheap ----
    if n_jobs and n_jobs > 1 and _HAS_JOBLIB:
        if verbose:
            print(f"[run] pass1 (cheap): scoring {len(strategies)} strategies, joblib n_jobs={n_jobs}")
        rows = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_one)(s, False, False) for s in strategies
        )
    else:
        if verbose:
            print(f"[run] pass1 (cheap): scoring {len(strategies)} strategies (serial)")
        rows = []
        for i, s in enumerate(strategies):
            rows.append(_one(s, False, False))
            if verbose and (i + 1) % 200 == 0:
                print(f"  ..{i+1}/{len(strategies)}  ({time.perf_counter()-t0:.1f}s)")

    df = pd.DataFrame(rows)
    by_name = {s.name: s for s in strategies}

    # ---- Pass 2: regime + bootstrap detail on the top-K by |ic_ir| ----
    if regime_detail_topk and "ic_ir" in df.columns:
        ranked = df.dropna(subset=["ic_ir"]).copy()
        ranked["_absir"] = ranked["ic_ir"].abs()
        top_names = ranked.sort_values("_absir", ascending=False).head(int(regime_detail_topk))["name"].tolist()
        if verbose:
            print(f"[run] pass2 (detail): regime + bootstrap on top {len(top_names)} strategies")
        detail_specs = [by_name[n] for n in top_names if n in by_name]
        if n_jobs and n_jobs > 1 and _HAS_JOBLIB:
            detail_rows = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_one)(s, True, True) for s in detail_specs
            )
        else:
            detail_rows = [_one(s, True, True) for s in detail_specs]
        detail_by_name = {r.get("name"): r for r in detail_rows}
        # Merge detail rows back over the cheap rows
        merged = []
        for r in df.to_dict(orient="records"):
            nm = r.get("name")
            merged.append(detail_by_name.get(nm, r))
        df = pd.DataFrame(merged)

    # ---- Deflated Sharpe — uses the empirical cross-trial Sharpe variance ----
    # All in per-period (non-overlapping) Sharpe units.
    n_trials = max(int(len(df)), 1)
    if "sr_periodic" in df.columns and "n_periodic" in df.columns:
        sr_vals = pd.to_numeric(df["sr_periodic"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(sr_vals) >= 3:
            var_sr_trials = float(sr_vals.var(ddof=1))
        else:
            var_sr_trials = float("nan")
        sr_star = expected_max_sharpe(n_trials, var_sr_trials) if np.isfinite(var_sr_trials) else float("nan")
        dsrs = []
        for _, r in df.iterrows():
            sr = r.get("sr_periodic")
            n = r.get("n_periodic") or 0
            sk = r.get("sk", 0.0) or 0.0
            kt = r.get("kt", 3.0) or 3.0
            try:
                if np.isfinite(sr_star):
                    dsrs.append(deflated_sharpe_ratio(float(sr), int(n), float(sr_star), float(sk), float(kt)))
                else:
                    dsrs.append(float("nan"))
            except Exception:
                dsrs.append(float("nan"))
        df["dsr"] = dsrs
        df["sr_star"] = sr_star
        if verbose and np.isfinite(sr_star):
            print(f"[run] DSR benchmark sr_star (per-period) = {sr_star:.4f} "
                  f"from var_sr_trials={var_sr_trials:.4f}, n_trials={n_trials}")

    # Decision flags
    df["decision"] = [assign_decision_flag(r) for r in df.to_dict(orient="records")]
    if verbose:
        print(f"[run] done in {time.perf_counter()-t0:.1f}s; decisions: "
              f"{df['decision'].value_counts().to_dict()}")
    return df


def write_results(results: pd.DataFrame, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parquet — flatten complex columns to JSON strings for portability
    flat = results.copy()
    for c in ("yearly_ir", "rb_trend", "rb_vol", "rb_liq", "rb_dow",
              "feature_cols", "transform_args", "spec"):
        if c in flat.columns:
            flat[c] = flat[c].apply(lambda v: json.dumps(v, default=str) if isinstance(v, (dict, list)) else v)
    flat.to_parquet(out_dir / "results.parquet", index=False)

    # SQLite (queryable)
    conn = sqlite3.connect(out_dir / "results.sqlite")
    flat.to_sql("results", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()

    # Per-strategy detail JSONs
    pdir = out_dir / "per_strategy"
    pdir.mkdir(exist_ok=True)
    for r in results.to_dict(orient="records"):
        sid = (r.get("name") or "unknown").replace("/", "_").replace(":", "_")[:200]
        with open(pdir / f"{sid}.json", "w", encoding="utf-8") as f:
            json.dump(r, f, default=str, indent=2)


def write_leaderboard(results: pd.DataFrame, out_path: Path, top_n: int = 50) -> None:
    """Markdown leaderboard sorted by ic_ir desc, with regime stripes."""
    if results.empty:
        out_path.write_text("# Edge finder leaderboard\n\n_(no results)_\n", encoding="utf-8")
        return
    df = results.copy()
    if "ic_ir" not in df.columns:
        out_path.write_text("# Edge finder leaderboard\n\n_(no scoreable strategies)_\n", encoding="utf-8")
        return

    df = df[df["decision"] != "ERROR"].sort_values("ic_ir", ascending=False, na_position="last")
    head = df.head(top_n)

    lines: List[str] = []
    lines.append("# Edge finder leaderboard")
    lines.append("")
    lines.append(f"_Run timestamp: {dt.datetime.now().isoformat(timespec='seconds')}_")
    lines.append("")
    lines.append(f"Total strategies scored: **{len(df)}**")
    lines.append("")
    lines.append("Decision distribution:")
    for k, v in df["decision"].value_counts().items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"## Top {min(top_n, len(head))} by IC IR")
    lines.append("")
    lines.append(
        "| # | strategy | h | dir | n_days | ic_ir | ic_mean | hit | LS Sharpe | DSR | stab | decision |"
    )
    lines.append(
        "|---|----------|---|-----|--------|------:|--------:|----:|----------:|----:|-----:|---------:|"
    )
    for i, r in enumerate(head.to_dict(orient="records"), start=1):
        lines.append(
            f"| {i} | `{r.get('name','')}` | {r.get('horizon','')} | "
            f"{r.get('direction','')} | {r.get('n_days','')} | "
            f"{_fmt(r.get('ic_ir'))} | {_fmt(r.get('ic_mean'),4)} | "
            f"{_fmt(r.get('hit_rate'),3)} | {_fmt(r.get('ls_sharpe'))} | "
            f"{_fmt(r.get('dsr'),3)} | {_fmt(r.get('regime_stability'),2)} | "
            f"**{r.get('decision','')}** |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Regime sensitivity (top 10 by IR)")
    lines.append("")
    for r in head.head(10).to_dict(orient="records"):
        lines.append(f"### `{r.get('name','')}`")
        lines.append("")
        for label, key in (("Trend", "rb_trend"), ("Vol", "rb_vol"), ("Liquidity", "rb_liq"), ("DOW", "rb_dow")):
            rb = r.get(key) or {}
            if not isinstance(rb, dict) or not rb:
                continue
            lines.append(f"_{label}_:")
            for k, v in rb.items():
                if not isinstance(v, dict):
                    continue
                lines.append(
                    f"- `{k}`  n_days={v.get('n_days','?')}  "
                    f"ic_ir={_fmt(v.get('ic_ir'))}  hit={_fmt(v.get('hit_rate'),3)}  "
                    f"ls_sharpe={_fmt(v.get('ls_sharpe'))}"
                )
            lines.append("")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def _fmt(v: Any, nd: int = 2) -> str:
    if v is None:
        return ""
    try:
        f = float(v)
    except Exception:
        return str(v)
    if not np.isfinite(f):
        return ""
    return f"{f:+.{nd}f}"


# ──────────────────────────────────────────────────────────────────────────────
# Section 6: Helpers used by main / CLI
# ──────────────────────────────────────────────────────────────────────────────

def pick_top_features_by_univariate_ic(
    panel: pd.DataFrame,
    feature_cols: Sequence[str],
    horizon: int,
    k: int = 30,
) -> List[str]:
    """Quick & cheap: rank features by |IC| at one horizon, take top k."""
    fwd = f"ret_h{int(horizon)}"
    if fwd not in panel.columns:
        return list(feature_cols[:k])
    rows: List[Tuple[str, float]] = []
    for c in feature_cols:
        sub = panel[["timestamp", c, fwd]].dropna()
        if len(sub) < 500:
            rows.append((c, 0.0))
            continue
        try:
            ic = daily_ic(sub.rename(columns={c: "_sig"}), "_sig", fwd, method="spearman")
            if ic.empty:
                rows.append((c, 0.0))
                continue
            sd = float(ic.std(ddof=1))
            ir = float(ic.mean() / sd * math.sqrt(TDPY)) if sd and np.isfinite(sd) and sd > 0 else 0.0
        except Exception:
            ir = 0.0
        rows.append((c, ir))
    rows.sort(key=lambda r: abs(r[1]), reverse=True)
    return [r[0] for r in rows[: int(k)]]


# ──────────────────────────────────────────────────────────────────────────────
# Section 7: CLI + self-test
# ──────────────────────────────────────────────────────────────────────────────

def _build_synthetic_panel(
    n_symbols: int = 25,
    n_days: int = 600,
    n_features: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a synthetic panel for the self-test. Half the features are pure
    noise, half have a small (signed) predictive relationship with forward
    returns. So we can verify the edge finder discriminates.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    rows: List[pd.DataFrame] = []
    for s in range(n_symbols):
        sym = f"SYM{s:03d}"
        log_ret = rng.normal(0, 0.012, n_days)
        close = 100.0 * np.exp(np.cumsum(log_ret))
        op = close * (1 + rng.normal(0, 0.003, n_days))
        hi = np.maximum(close, op) * (1 + np.abs(rng.normal(0, 0.005, n_days)))
        lo = np.minimum(close, op) * (1 - np.abs(rng.normal(0, 0.005, n_days)))
        vol = rng.integers(50_000, 1_500_000, n_days).astype("float64")
        df = pd.DataFrame({
            "timestamp": dates, "symbol": sym, "open": op, "high": hi,
            "low": lo, "close": close, "volume": vol,
        })
        # Cheap proxies for the cache feature names
        df["D_atr_pct"] = pd.Series((hi - lo) / close * 100).rolling(14, min_periods=3).mean().values
        df["D_dollar_vol"] = (close * vol)
        df["D_obv_slope"] = pd.Series(np.sign(np.diff(close, prepend=close[0])) * vol).cumsum().diff().values
        df["D_rsi14"] = 50 + 10 * rng.normal(0, 1, n_days)
        df["D_macd_hist"] = rng.normal(0, 1, n_days)
        df["D_bb_pctB_20"] = rng.uniform(-1, 2, n_days)
        df["D_bb_bw_20"] = rng.uniform(0.05, 0.2, n_days)
        df["D_donch_pos_20"] = rng.uniform(0, 1, n_days)
        df["D_donch_pos_50"] = rng.uniform(0, 1, n_days)
        df["D_cmf20"] = rng.normal(0, 0.2, n_days)
        df["D_atr_ratio_14_30"] = rng.uniform(0.5, 1.5, n_days)
        df["D_ema20_angle_deg"] = rng.normal(0, 2, n_days)
        df["D_range_pct"] = (hi - lo) / close * 100
        df["D_gap_pct"] = rng.normal(0, 0.5, n_days)
        df["D_close_roll_slope_20"] = rng.normal(0, 0.1, n_days)
        df["D_close_roll_slope_50"] = rng.normal(0, 0.1, n_days)
        df["D_slope_stability"] = rng.uniform(0, 1, n_days)
        df["D_body_ratio"] = rng.uniform(-1, 1, n_days)
        df["D_wick_skew"] = rng.uniform(-1, 1, n_days)
        df["D_sma50"] = pd.Series(close).rolling(50, min_periods=1).mean().values
        df["D_sma200"] = pd.Series(close).rolling(200, min_periods=1).mean().values
        # Synthetic noise features
        for k in range(n_features):
            df[f"D_noise_{k:02d}"] = rng.normal(0, 1, n_days)
        rows.append(df)
    panel = pd.concat(rows, ignore_index=True)
    panel = panel.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    # Inject a known, REALISTIC signal: D_signal_alpha carries a modest slice
    # of forward-5d-return information (per-day cross-sectional IC ≈ 0.12),
    # D_signal_beta is its negative twin. This mirrors the magnitude of a
    # strong-but-plausible real alpha (not a near-perfect oracle), so the
    # deflation benchmark stays realistic.
    panel = add_forward_returns(panel, horizons=(5,))
    fwd5 = pd.to_numeric(panel["ret_h5"], errors="coerce")
    # Cross-sectional z-score of the forward return (per day)
    fwd5_z = (
        (fwd5 - fwd5.groupby(panel["timestamp"]).transform("mean"))
        / fwd5.groupby(panel["timestamp"]).transform("std").replace(0, np.nan)
    ).fillna(0.0).to_numpy()
    nrows = len(panel)
    coef = 0.12
    panel["D_signal_alpha"] = (coef * fwd5_z + rng.normal(0, 1, nrows)).astype("float32")
    panel["D_signal_beta"] = (-coef * fwd5_z + rng.normal(0, 1, nrows)).astype("float32")
    panel = panel.drop(columns=["ret_h5"])
    return panel


def _run_self_test(out_dir: Path) -> None:
    print("=" * 70)
    print("EDGE FINDER SELF-TEST")
    print("=" * 70)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[selftest] building synthetic panel ...")
    panel = _build_synthetic_panel(n_symbols=25, n_days=600, n_features=30)
    panel = add_forward_returns(panel, horizons=DEFAULT_HORIZONS)
    panel = add_cross_sectional_ranks(panel, CS_RANK_INPUT_COLS)
    panel = add_panel_wq_alphas(panel)
    panel = add_regime_columns(panel)

    feats = pick_feature_columns(panel, max_features=60)
    print(f"[selftest] panel rows={len(panel):,}  features={len(feats)}")

    print("[selftest] picking top features by univariate IR ...")
    top_feats = pick_top_features_by_univariate_ic(panel, feats, horizon=5, k=10)
    print(f"[selftest] top-10 by univariate IR: {top_feats}")

    strategies: List[StrategySpec] = []
    strategies += list(gen_single_feature_strategies(feats, DEFAULT_HORIZONS, both_directions=False))
    strategies += list(gen_pair_interaction_strategies(top_feats, DEFAULT_HORIZONS,
                                                       modes=("product", "sign_cond"), max_pairs=20))
    strategies += list(gen_regime_conditioned_strategies(
        top_features=top_feats[:5], horizons=DEFAULT_HORIZONS,
        regimes=[("reg_trend", 1, "trend_up"), ("reg_trend", -1, "trend_down"),
                 ("reg_vol", 0, "vol_low"), ("reg_vol", 2, "vol_high")],
    ))
    strategies += list(gen_composite_strategies(top_feats, DEFAULT_HORIZONS, sizes=(3, 5)))
    strategies += list(gen_cs_wq_alpha_strategies(panel, DEFAULT_HORIZONS))
    print(f"[selftest] strategies generated: {len(strategies)}")

    results = run_edge_finder(panel, strategies, n_jobs=1, verbose=True, regime_detail_topk=40)
    print(f"[selftest] results shape: {results.shape}")
    write_results(results, out_dir)
    write_leaderboard(results, out_dir / "top_strategies.md", top_n=20)

    # Sanity assertion: D_signal_alpha (positive predictor) should rank near
    # the top, D_signal_beta (negative predictor) should also rank near the
    # top after autoflip (its IR magnitude is what we sort on if needed).
    head = results.dropna(subset=["ic_ir"]).sort_values("ic_ir", ascending=False).head(20)
    top_names = head["name"].tolist()
    assert any("D_signal_alpha" in n for n in top_names), \
        f"D_signal_alpha did not surface in top 20 by IR; got: {top_names[:10]}"
    print(f"[selftest] OK — D_signal_alpha surfaced near the top.")
    print(f"[selftest] outputs in: {out_dir.resolve()}")
    print(f"  - results.parquet")
    print(f"  - results.sqlite")
    print(f"  - top_strategies.md")
    print(f"  - per_strategy/*.json")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Strategy screener / edge logger over a v20 cache panel."
    )
    ap.add_argument("--cache-dir", type=str, default=None,
                    help="Path to the cache_daily_new folder containing *_daily.parquet")
    ap.add_argument("--start", type=str, default=None, help="ISO date (inclusive)")
    ap.add_argument("--end", type=str, default=None, help="ISO date (inclusive)")
    ap.add_argument("--horizons", type=str, default="1,5,10",
                    help="Comma-separated forward-return horizons in days")
    ap.add_argument("--max-symbols", type=int, default=None)
    ap.add_argument("--max-features", type=int, default=120)
    ap.add_argument("--max-strategies", type=int, default=3000)
    ap.add_argument("--pairs-topk", type=int, default=30,
                    help="Generate pair interactions over the top-K univariate features")
    ap.add_argument("--include-singles", action="store_true", default=True)
    ap.add_argument("--include-pairs", action="store_true", default=True)
    ap.add_argument("--include-regimes", action="store_true", default=True)
    ap.add_argument("--include-composites", action="store_true", default=True)
    ap.add_argument("--include-cs-wq", action="store_true", default=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--regime-detail-topk", type=int, default=100,
                    help="Compute regime breakdowns + bootstrap CIs only for the top-K by |IC IR|")
    ap.add_argument("--out", type=str, default="edge_finder_out")
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--self-test", action="store_true",
                    help="Run on a synthetic panel; no real cache needed")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.self_test:
        _run_self_test(out_dir)
        return 0

    if not args.cache_dir:
        print("ERROR: provide --cache-dir or --self-test", file=sys.stderr)
        return 2

    horizons = tuple(int(x.strip()) for x in args.horizons.split(",") if x.strip())

    # 1) Panel
    panel = load_panel(
        args.cache_dir,
        start=pd.Timestamp(args.start).date() if args.start else None,
        end=pd.Timestamp(args.end).date() if args.end else None,
        max_symbols=args.max_symbols,
    )
    panel = add_forward_returns(panel, horizons=horizons)
    panel = add_cross_sectional_ranks(panel, CS_RANK_INPUT_COLS)
    panel = add_panel_wq_alphas(panel)
    panel = add_regime_columns(panel)

    # 2) Features
    feats = pick_feature_columns(panel, max_features=args.max_features)
    if not feats:
        print("ERROR: no usable feature columns after filtering.", file=sys.stderr)
        return 3

    # 3) Strategies
    print("[strats] picking top features by univariate IR for pair/regime/composite generators ...")
    top_feats = pick_top_features_by_univariate_ic(panel, feats, horizon=horizons[0], k=args.pairs_topk)

    strategies: List[StrategySpec] = []
    if args.include_singles:
        strategies += list(gen_single_feature_strategies(feats, horizons, both_directions=False))
    if args.include_pairs:
        strategies += list(gen_pair_interaction_strategies(top_feats, horizons))
    if args.include_regimes:
        strategies += list(gen_regime_conditioned_strategies(
            top_features=top_feats[:10], horizons=horizons,
            regimes=[("reg_trend", 1, "trend_up"), ("reg_trend", -1, "trend_down"),
                     ("reg_vol", 0, "vol_low"), ("reg_vol", 2, "vol_high"),
                     ("reg_liq", 0, "liq_low"), ("reg_liq", 2, "liq_high")],
        ))
    if args.include_composites:
        strategies += list(gen_composite_strategies(top_feats, horizons, sizes=(3, 5, 8)))
    if args.include_cs_wq:
        strategies += list(gen_cs_wq_alpha_strategies(panel, horizons))

    if args.max_strategies and len(strategies) > args.max_strategies:
        # De-dupe by name first, then truncate
        seen = set()
        deduped: List[StrategySpec] = []
        for s in strategies:
            if s.name in seen:
                continue
            seen.add(s.name)
            deduped.append(s)
            if len(deduped) >= args.max_strategies:
                break
        strategies = deduped

    print(f"[strats] total strategies: {len(strategies)}")

    # 4) Score + persist
    results = run_edge_finder(panel, strategies, n_jobs=args.workers,
                              regime_detail_topk=args.regime_detail_topk)
    write_results(results, out_dir)
    write_leaderboard(results, out_dir / "top_strategies.md", top_n=args.top_n)

    # Panel summary
    summary = {
        "rows": int(len(panel)),
        "symbols": int(panel["symbol"].nunique()),
        "dates": int(panel["timestamp"].nunique()),
        "first_date": str(panel["timestamp"].min().date()),
        "last_date": str(panel["timestamp"].max().date()),
        "horizons": list(horizons),
        "n_features": len(feats),
        "n_strategies": len(strategies),
    }
    (out_dir / "panel_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    print(f"\nDone. See {out_dir.resolve()}/top_strategies.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

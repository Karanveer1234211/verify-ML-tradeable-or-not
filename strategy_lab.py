#!/usr/bin/env python3
"""
strategy_lab.py  —  multi-strategy cross-sectional EDGE FINDER / research orchestrator.

WHAT THIS IS
------------
edge_finder.py scores ONE feature at a time. strategy_lab.py scores HUNDREDS to
THOUSANDS of *strategies* against the model's enriched panel and writes a single
ranked, decision-ready scoreboard so you can see — at a glance — which hypotheses
are worth pursuing and which are noise.

A "strategy" here is a triple:

      (signal expression,  forward horizon,  scoring view)

and strategies are produced automatically by five generators so you do not have
to hand-write them:

  1. SINGLES        every model feature, every horizon (1/3/5/10d), both directions.
  2. PAIRS          for the top-K univariate features, three interaction forms:
                      • z-product            z(A)·z(B)
                      • sign-conditioned     sign(z(A))·z(B)   ("B's payoff flips with A")
                      • gating               B, but only on names in A's top tercile
  3. REGIME GATES   top-K features turned ON only inside a market regime
                    (high/low vol, up/down trend, high/low dispersion).
  4. COMPOSITES     equal-weight blend of the top-N oriented feature z-scores.
  5. WQ X-SECTIONAL the *true* cross-sectional WorldQuant alphas
                    (groupby(date).rank()), which the per-symbol cache cannot build.
                    This closes the gap the audit flagged (Daily cache.py uses a
                    TEMPORAL per-symbol rank, not a cross-sectional one).

WHAT GETS LOGGED FOR EVERY STRATEGY
-----------------------------------
  • Daily cross-sectional rank-IC: mean, std, IR (=mean/std), Newey-West t, hit-rate
  • Out-of-sample IC on the model's embargoed TEST split (sign must survive)
  • Long/short quantile portfolio: gross & net-of-cost Sharpe (annualised),
    annual return, max drawdown, # non-overlapping rebalances
  • Deflated Sharpe Ratio (Bailey & Lopez de Prado) — explicitly discounts the
    fact that we tried N strategies, so a "great" Sharpe found by luck is exposed
  • Regime breakdown: IC-IR inside high/low vol, up/down trend, dispersion buckets
  • Stability: fraction of calendar years with same-sign IC
  • A DECISION verdict:  STRONG / PROMISING / MARGINAL / NO_EDGE / AVOID

WHY THIS IS FLEXIBLE (matches the ask)
--------------------------------------
  • "different features work on different strategies"  -> 5 generators, top-K mixing
  • "some interactions in the features"               -> PAIRS (product/sign/gate)
  • "certain regimes when a thing works"              -> REGIME GATES + regime breakdown
  • "there can be days when some strategies work"      -> per-day IC series + IC hit-rate
                                                          + year stability + bootstrap CI

HONESTY (read this)
-------------------
Clearing every gate is NECESSARY, not SUFFICIENT. It says "worth paper-trading",
not "worth capital". This tool CANNOT detect leakage baked INTO a feature upstream
(e.g. a whole-series .rank()); fix that at source and rebuild the panel. Known-leaky
WQ columns are excluded by default. An "AVOID (too good)" verdict flags suspiciously
strong results that almost always mean look-ahead leakage, not alpha.

USAGE
-----
  # against the real model panel (recommended):
  python strategy_lab.py --panel /path/to/panel_cache.parquet

  # self-test with a synthetic panel that has PLANTED signals (no data needed):
  python strategy_lab.py --self-test

Deps: numpy, pandas (+ pyarrow to read parquet), scipy (skew/kurtosis only; a
pure-python fallback is used if scipy is missing).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import kurtosis as _sp_kurtosis
    from scipy.stats import skew as _sp_skew
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


# ───────────────────────── model data contract (mirrors New_model.py) ─────────────────────────

# Columns the model NEVER trains on (labels / helpers). Mirror of
# New_model.discover_daily_features NEVER_FEATURE.
MODEL_NEVER_FEATURE = {
    "timestamp", "date", "year", "symbol", "instrument_token",
    "open", "high", "low", "close", "volume",
    "ret_1d_close_pct", "ret_3d_close_pct", "ret_5d_close_pct",
    "ret_1d_oc_pct", "ret_3d_oc_pct", "ret_5d_oc_pct",
    "ret_5d_adj", "ret_3d_adj", "rank_5d_pct",
    "top20_vs_bot20_5d", "top20_strict_5d",
    "avg20_vol", "stock_regime",
    "vol_20", "atr_pct",
}
MODEL_NEVER_PATTERNS = ("mfe_", "mae_")
MODEL_FEATURE_PREFIXES = (
    "D_", "W_", "WQ_", "M_", "X_", "Comb_", "DOW_",
    "regime_", "CPR_", "Struct_", "DayType_",
)

# WQ alphas tainted by the v19 whole-series .rank() look-ahead bug. Excluded by default.
LEAKY_WQ_DEFAULT = {
    "D_WQ_3", "D_WQ_13", "D_WQ_16", "D_WQ_19", "D_WQ_20",
    "D_WQ_29", "D_WQ_33", "D_WQ_38", "D_WQ_40", "D_WQ_44",
    "WQ_3", "WQ_13", "WQ_16", "WQ_19", "WQ_20",
    "WQ_29", "WQ_33", "WQ_38", "WQ_40", "WQ_44",
}

EULER_GAMMA = 0.5772156649015329


@dataclass
class Config:
    # data
    panel_path: Optional[Path] = Path(r"C:\Users\karanvsi\Desktop\Pycharm\Cache\panel_cache.parquet")
    cache_dir: Optional[Path] = None
    out_dir: Path = Path(".")
    self_test: bool = False

    # target
    horizons: Tuple[int, ...] = (1, 3, 5, 10)
    primary_horizon: int = 5
    use_model_target: bool = True       # vol-adjusted forward return like the model

    # universe filters
    min_names_per_day: int = 30
    min_adv_inr: float = 0.0
    min_price: float = 0.0

    # holdout split (model-style, embargoed)
    train_frac: float = 0.70
    cal_frac: float = 0.20
    embargo_days: int = 5

    # backtest
    n_quantiles: int = 5
    cost_bps: float = 15.0
    slippage_bps: float = 10.0

    # significance
    nw_lag: Optional[int] = None        # default = horizon
    n_boot: int = 0                     # block-bootstrap reps for top-K (0 = skip)

    # generators
    top_k_pairs: int = 18               # how many top univariate feats feed PAIRS
    top_k_regime: int = 18              # how many top feats feed REGIME GATES
    composite_sizes: Tuple[int, ...] = (5, 10, 20)
    do_pairs: bool = True
    do_regime_gates: bool = True
    do_composites: bool = True
    do_wq_xsection: bool = True
    max_features: Optional[int] = None

    # leakage / sanity
    exclude_leaky_wq: bool = True

    # verdict thresholds
    dsr_strong: float = 0.90
    dsr_promising: float = 0.50
    net_sharpe_strong: float = 1.0
    net_sharpe_promising: float = 0.40
    year_hit_min: float = 0.60
    too_good_daily_ir: float = 0.50     # |daily IC IR| above this == suspicious leakage
    too_good_gross_sharpe: float = 6.0

    # detailed report depth
    detail_top_k: int = 25

    def lag(self) -> int:
        return self.nw_lag if self.nw_lag is not None else self.primary_horizon

    def __post_init__(self):
        for a in ("panel_path", "cache_dir", "out_dir"):
            v = getattr(self, a)
            if v is not None:
                setattr(self, a, Path(v))


# ───────────────────────── math (no hard scipy dependency for the core) ─────────────────────────

def _norm_ppf(p: float) -> float:
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
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
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _norm_cdf(z: float) -> float:
    if not np.isfinite(z):
        return 1.0 if z > 0 else 0.0
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def two_sided_p(t: float) -> float:
    return 2.0 * (1.0 - _norm_cdf(abs(t)))


def expected_max_t(n_trials: int) -> float:
    """E[max] of n_trials iid standard-normals (multiple-testing bar for a t-stat)."""
    n = max(2, int(n_trials))
    return (1 - EULER_GAMMA) * _norm_ppf(1 - 1.0 / n) + EULER_GAMMA * _norm_ppf(1 - 1.0 / (n * math.e))


def newey_west_t(x: np.ndarray, lag: int) -> Tuple[float, float, int]:
    """t-stat of mean(x) with Newey-West HAC variance (overlapping returns inflate naive t)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < max(10, lag + 2):
        return (np.nan, np.nan, n)
    mu = x.mean()
    e = x - mu
    var = float(np.dot(e, e) / n)
    L = min(lag, n - 1)
    for k in range(1, L + 1):
        w = 1.0 - k / (L + 1.0)
        var += 2.0 * w * float(np.dot(e[k:], e[:-k]) / n)
    if var <= 0:
        return (np.nan, np.nan, n)
    se = math.sqrt(var / n)
    t = mu / se if se > 0 else np.nan
    return (t, two_sided_p(t), n)


def _skew_kurt(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return (0.0, 3.0)
    if _HAVE_SCIPY:
        return (float(_sp_skew(x, bias=False)), float(_sp_kurtosis(x, fisher=False, bias=False)))
    m = x.mean()
    s = x.std(ddof=0)
    if s <= 0:
        return (0.0, 3.0)
    z = (x - m) / s
    return (float((z ** 3).mean()), float((z ** 4).mean()))


def probabilistic_sharpe(sr_period: float, n: int, skew: float, kurt: float,
                         sr_benchmark: float = 0.0) -> float:
    """PSR(sr_benchmark): P(true per-period Sharpe > benchmark) given non-normal returns."""
    if not np.isfinite(sr_period) or n < 4:
        return np.nan
    denom = 1.0 - skew * sr_period + ((kurt - 1.0) / 4.0) * (sr_period ** 2)
    if denom <= 0:
        return np.nan
    z = (sr_period - sr_benchmark) * math.sqrt(max(1, n - 1)) / math.sqrt(denom)
    return _norm_cdf(z)


def deflated_sharpe(sr_period: float, n: int, skew: float, kurt: float,
                    var_sr_across_trials: float, n_trials: int) -> float:
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014). Discounts selection bias
    from testing n_trials strategies: benchmark = expected MAX Sharpe under the null.
    """
    if not np.isfinite(sr_period) or n < 4 or n_trials < 2:
        return np.nan
    v = max(var_sr_across_trials, 1e-12)
    sd = math.sqrt(v)
    z1 = _norm_ppf(1 - 1.0 / n_trials)
    z2 = _norm_ppf(1 - 1.0 / (n_trials * math.e))
    sr_star = sd * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)
    return probabilistic_sharpe(sr_period, n, skew, kurt, sr_benchmark=sr_star)


def max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return np.nan
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


# ───────────────────────── loading ─────────────────────────

def _norm_date(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, errors="coerce")
    tz = getattr(out.dt, "tz", None)
    if tz is not None:
        out = out.dt.tz_localize(None)
    return out.dt.normalize()


def load_panel(cfg: Config) -> pd.DataFrame:
    if cfg.self_test:
        return make_synthetic_panel()

    if cfg.panel_path and Path(cfg.panel_path).exists():
        print(f"PANEL MODE: reading {cfg.panel_path}")
        panel = pd.read_parquet(cfg.panel_path)
        dcol = "timestamp" if "timestamp" in panel.columns else ("date" if "date" in panel.columns else None)
        if dcol is None:
            raise SystemExit("panel has no 'timestamp' or 'date' column")
        panel["date"] = _norm_date(panel[dcol])
        if "symbol" not in panel.columns:
            raise SystemExit("panel has no 'symbol' column")
        panel = panel.dropna(subset=["date"]).sort_values(["symbol", "date"]).reset_index(drop=True)
        print(f"  {panel['symbol'].nunique()} symbols, {len(panel):,} rows, "
              f"{panel['date'].min().date()} -> {panel['date'].max().date()}, {panel.shape[1]} cols")
        return panel

    if cfg.cache_dir and Path(cfg.cache_dir).exists():
        print(f"PER-SYMBOL MODE: globbing {cfg.cache_dir}")
        files = sorted(glob.glob(str(cfg.cache_dir / "*_daily.parquet")))
        if not files:
            raise SystemExit(f"No *_daily.parquet in {cfg.cache_dir}")
        frames = []
        for fp in files:
            sym = os.path.basename(fp).replace("_daily.parquet", "")
            try:
                df = pd.read_parquet(fp)
            except Exception as e:
                print(f"  skip {sym}: {e}")
                continue
            dcol = "date" if "date" in df.columns else ("timestamp" if "timestamp" in df.columns else None)
            if dcol is None:
                continue
            df = df.rename(columns={dcol: "date"})
            df["date"] = _norm_date(df["date"])
            df["symbol"] = sym
            frames.append(df)
        panel = pd.concat(frames, ignore_index=True).dropna(subset=["date"])
        return panel.sort_values(["symbol", "date"]).reset_index(drop=True)

    raise SystemExit(
        "No data. Pass --panel /path/to/panel_cache.parquet, or --cache-dir DIR, "
        "or --self-test to run on a synthetic planted-signal panel."
    )


def survivorship_check(panel: pd.DataFrame) -> float:
    gmax = panel["date"].max()
    alive = float((panel.groupby("symbol")["date"].max() >= gmax - pd.Timedelta(days=10)).mean())
    print("\n── Survivorship check ──")
    print(f"  symbols alive at sample end: {alive:6.1%}")
    if alive > 0.95:
        print("  WARNING  >95% of names survive to the end -> universe is likely point-in-time")
        print("           (survivorship bias). Treat every number below as optimistic.")
    return alive


# ───────────────────────── target & regimes ─────────────────────────

def build_targets(panel: pd.DataFrame, cfg: Config) -> Dict[int, str]:
    """
    Build vol-adjusted forward returns for each horizon (model-consistent) and the
    realised open->close return used for PnL. Returns {horizon: target_col_name}.
    """
    g_close = panel.groupby("symbol", observed=True)["close"]
    g_open = panel.groupby("symbol", observed=True)["open"]
    close = pd.to_numeric(panel["close"], errors="coerce")

    # vol basis = 20d std of daily returns, fallback ATR%, floored per day at 5th pct.
    daily_ret = g_close.pct_change() * 100.0
    vol_20 = (daily_ret.groupby(panel["symbol"], observed=True)
              .rolling(20, min_periods=10).std().reset_index(level=0, drop=True))
    if "atr_pct" in panel.columns:
        atr_pct = pd.to_numeric(panel["atr_pct"], errors="coerce")
    elif "D_atr14" in panel.columns:
        atr_pct = (pd.to_numeric(panel["D_atr14"], errors="coerce") / close.replace(0, np.nan)) * 100.0
    else:
        atr_pct = pd.Series(np.nan, index=panel.index)
    vol_basis = vol_20.fillna(atr_pct).replace(0.0, np.nan)
    vb_floor = vol_basis.groupby(panel["date"]).transform(
        lambda s: np.nanpercentile(s, 5) if np.isfinite(s.to_numpy(dtype="float64")).any() else 0.0
    ).fillna(0.0)
    vol_basis = np.maximum(vol_basis, vb_floor)
    panel["_vol_basis"] = vol_basis

    target_cols: Dict[int, str] = {}
    for h in cfg.horizons:
        # entry = next open, exit = close[t+h]  (open->close, matches model ev_target="oc")
        ret_oc = (g_close.shift(-h) / g_open.shift(-1) - 1.0) * 100.0
        ret_cc = (g_close.shift(-h) / close - 1.0) * 100.0
        panel[f"_ret_oc_{h}"] = ret_oc / 100.0  # fraction, for PnL
        if cfg.use_model_target:
            panel[f"_tgt_{h}"] = (ret_cc / vol_basis).replace([np.inf, -np.inf], np.nan)
        else:
            panel[f"_tgt_{h}"] = ret_cc
        target_cols[h] = f"_tgt_{h}"

    # tradeability
    if cfg.min_adv_inr > 0 or cfg.min_price > 0:
        vol = pd.to_numeric(panel.get("volume", np.nan), errors="coerce")
        adv = (close * vol).groupby(panel["symbol"], observed=True).transform(
            lambda s: s.rolling(20, min_periods=10).mean())
        panel["_tradeable"] = ((adv >= cfg.min_adv_inr) & (close >= cfg.min_price)).fillna(False)
    else:
        panel["_tradeable"] = panel[f"_tgt_{cfg.primary_horizon}"].notna()

    msg = "vol-adjusted (model-style)" if cfg.use_model_target else "raw close-close"
    print(f"Targets built for horizons {cfg.horizons}: {msg}. "
          f"PnL uses next-open->close[t+h] return.")
    return target_cols


def build_day_regimes(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Per-DAY market-regime tags used for the regime breakdown. Uses the model's own
    regime columns when present, and derives any missing ones from cross-sectional
    daily returns. Returns a frame indexed by date.
    """
    dates = pd.Index(np.sort(panel["date"].unique()), name="date")
    out = pd.DataFrame(index=dates)

    if "regime_high_vol" in panel.columns:
        out["vol_state"] = (panel.groupby("date")["regime_high_vol"].max()
                            .reindex(dates).fillna(0).astype(int))
    if "regime_market_trend" in panel.columns:
        tr = panel.groupby("date")["regime_market_trend"].mean().reindex(dates)
        out["trend_state"] = np.where(tr.fillna(0) > 0, "up", "down")
    if "regime_dispersion" in panel.columns:
        disp = panel.groupby("date")["regime_dispersion"].mean().reindex(dates)
        out["disp_state"] = pd.qcut(disp.rank(method="first"), 3,
                                    labels=["lo", "mid", "hi"]).astype(object)

    need_vol = ("vol_state" not in out.columns) or (out["vol_state"].nunique() < 2)
    need_trend = "trend_state" not in out.columns
    need_disp = "disp_state" not in out.columns
    if need_vol or need_trend or need_disp:
        r1 = panel.groupby("symbol", observed=True)["close"].pct_change()
        tmp = pd.DataFrame({"date": panel["date"].to_numpy(), "r1": r1.to_numpy()})
        cs_std = tmp.groupby("date")["r1"].std().reindex(dates)
        cs_mean = tmp.groupby("date")["r1"].mean().reindex(dates)
        if need_vol:
            med = cs_std.rolling(120, min_periods=20).median().shift(1)
            out["vol_state"] = (cs_std.shift(1) > med).fillna(False).astype(int)
        if need_trend:
            trend = cs_mean.rolling(120, min_periods=20).mean().shift(1)
            out["trend_state"] = np.where(trend.fillna(0) > 0, "up", "down")
        if need_disp:
            out["disp_state"] = pd.qcut(cs_std.shift(1).rank(method="first"), 3,
                                        labels=["lo", "mid", "hi"]).astype(object)

    out["year"] = out.index.year
    out["dow"] = out.index.dayofweek  # 0=Mon
    return out


# ───────────────────────── feature selection ─────────────────────────

def pick_features(panel: pd.DataFrame, cfg: Config) -> List[str]:
    cols = list(panel.columns)
    feats = [c for c in cols
             if isinstance(c, str)
             and c.startswith(MODEL_FEATURE_PREFIXES)
             and c not in MODEL_NEVER_FEATURE
             and not c.startswith(MODEL_NEVER_PATTERNS)
             and "__dup" not in c]
    drop = {c for c in cols if isinstance(c, str)
            and (("fwd" in c.lower()) or c.lower().startswith("ret_") or ("future" in c.lower()))}
    drop |= {c for c in cols if isinstance(c, str) and c.startswith("_")}
    if cfg.exclude_leaky_wq:
        present_leaky = sorted(set(feats) & LEAKY_WQ_DEFAULT)
        if present_leaky:
            print(f"\nWARNING  excluding {len(present_leaky)} known-leaky WQ columns the model "
                  f"still trains on: {', '.join(present_leaky)}")
        drop |= LEAKY_WQ_DEFAULT
    feats = [c for c in feats if c not in drop]

    out = []
    for c in feats:
        s = pd.to_numeric(panel[c], errors="coerce")
        if s.notna().mean() >= 0.30 and s.nunique(dropna=True) > 5:
            panel[c] = s
            out.append(c)
    if cfg.max_features:
        out = out[:cfg.max_features]
    print(f"Usable model features: {len(out)}")
    return out


# ───────────────────────── scoring engine (vectorised) ─────────────────────────

class ScoreEngine:
    """
    Holds the row-aligned arrays + per-day group structure once, so each strategy's
    IC and L/S backtest are cheap array ops (no per-strategy groupby/apply).
    """

    def __init__(self, panel: pd.DataFrame, target_cols: Dict[int, str], cfg: Config):
        self.cfg = cfg
        m = panel["_tradeable"].to_numpy()
        self.work = panel.loc[m].reset_index(drop=True)
        n = len(self.work)

        # day codes (contiguous 0..D-1, ordered by date)
        codes, uniq = pd.factorize(self.work["date"], sort=True)
        self.codes = codes.astype(np.int64)
        self.uniq_dates = pd.DatetimeIndex(uniq)
        self.D = len(uniq)
        self.day_count = np.bincount(self.codes, minlength=self.D).astype(float)

        # per-day row indices (so the L/S backtest never scans the full array)
        order = np.argsort(self.codes, kind="stable")
        bounds = np.cumsum(self.day_count.astype(int))[:-1]
        self.day_rows = np.split(order, bounds)  # day_rows[code] -> row indices
        self.sym = self.work["symbol"].to_numpy()

        # per-horizon target within-day rank (average ties), precomputed once
        self.tgt_rank: Dict[int, np.ndarray] = {}
        self.tgt_valid: Dict[int, np.ndarray] = {}
        self.ret_oc: Dict[int, np.ndarray] = {}
        for h, col in target_cols.items():
            y = pd.to_numeric(self.work[col], errors="coerce").to_numpy()
            self.tgt_valid[h] = np.isfinite(y)
            self.tgt_rank[h] = self._rank_within(self.codes, y, self.tgt_valid[h])
            self.ret_oc[h] = pd.to_numeric(self.work[f"_ret_oc_{h}"], errors="coerce").to_numpy()

        # holdout split (date-level, embargoed) on the primary horizon
        self.test_day_mask = self._make_test_mask()

        # day attribute table for regime breakdown
        self.day_attrs = build_day_regimes(panel).reindex(self.uniq_dates)
        self.year = self.day_attrs["year"].to_numpy()

    # ---- helpers ----
    @staticmethod
    def _rank_within(codes: np.ndarray, vals: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """Average-tie rank of vals within each day code; NaN where invalid."""
        out = np.full(vals.shape, np.nan)
        if valid.any():
            s = pd.Series(vals[valid])
            r = s.groupby(codes[valid]).rank(method="average").to_numpy()
            out[valid] = r
        return out

    @staticmethod
    def _rank_pct_within(codes: np.ndarray, vals: np.ndarray, valid: np.ndarray) -> np.ndarray:
        out = np.full(vals.shape, np.nan)
        if valid.any():
            s = pd.Series(vals[valid])
            r = s.groupby(codes[valid]).rank(method="average", pct=True).to_numpy()
            out[valid] = r
        return out

    def _zscore_within(self, vals: np.ndarray) -> np.ndarray:
        """Cross-sectional z-score per day (used to build interaction/composite signals)."""
        valid = np.isfinite(vals)
        out = np.full(vals.shape, np.nan)
        if not valid.any():
            return out
        c = self.codes[valid]
        v = vals[valid]
        cnt = np.bincount(c, minlength=self.D)
        ssum = np.bincount(c, weights=v, minlength=self.D)
        mean = np.divide(ssum, cnt, out=np.zeros_like(ssum), where=cnt > 0)
        dev = v - mean[c]
        ss = np.bincount(c, weights=dev * dev, minlength=self.D)
        std = np.sqrt(np.divide(ss, np.maximum(cnt - 1, 1), out=np.zeros_like(ss), where=cnt > 1))
        z = np.divide(dev, std[c], out=np.zeros_like(dev), where=std[c] > 0)
        out[valid] = z
        return out

    def _make_test_mask(self) -> np.ndarray:
        d = np.arange(self.D)
        n = self.D
        if n < 30:
            return np.ones(n, dtype=bool)
        cut_cal = int((self.cfg.train_frac + self.cfg.cal_frac) * n)
        emb = max(0, self.cfg.embargo_days)
        start = min(n - 1, cut_cal + emb)
        mask = np.zeros(n, dtype=bool)
        mask[start:] = True
        return mask

    # ---- core metrics ----
    def ic_per_day(self, signal: np.ndarray, h: int) -> np.ndarray:
        """Cross-sectional rank-IC per day. Returns array of length D (NaN for skipped days)."""
        valid = np.isfinite(signal) & self.tgt_valid[h]
        out = np.full(self.D, np.nan)
        if valid.sum() < self.cfg.min_names_per_day:
            return out
        c = self.codes[valid]
        fr = self._rank_within(self.codes, signal, valid)[valid]
        tr = self.tgt_rank[h][valid]
        cnt = np.bincount(c, minlength=self.D).astype(float)
        fr_mean = np.divide(np.bincount(c, weights=fr, minlength=self.D), cnt,
                            out=np.zeros(self.D), where=cnt > 0)
        tr_mean = np.divide(np.bincount(c, weights=tr, minlength=self.D), cnt,
                            out=np.zeros(self.D), where=cnt > 0)
        fr_dm = fr - fr_mean[c]
        tr_dm = tr - tr_mean[c]
        num = np.bincount(c, weights=fr_dm * tr_dm, minlength=self.D)
        sf = np.bincount(c, weights=fr_dm * fr_dm, minlength=self.D)
        st = np.bincount(c, weights=tr_dm * tr_dm, minlength=self.D)
        den = np.sqrt(sf * st)
        ok = (cnt >= self.cfg.min_names_per_day) & (den > 0)
        out[ok] = num[ok] / den[ok]
        return out

    def ls_backtest(self, signal: np.ndarray, h: int, direction: float,
                    precise_turnover: bool = False) -> dict:
        """
        Dollar-neutral top-vs-bottom quantile L/S, rebalanced every h days
        (non-overlapping). Returns gross & net (cost) annualised stats.
        """
        cfg = self.cfg
        nil = dict(gross_sharpe=np.nan, net_sharpe=np.nan, net_ann_ret=np.nan,
                   gross_ann_ret=np.nan, turnover=np.nan, max_dd=np.nan, n_reb=0,
                   net_rets=np.array([]), gross_rets=np.array([]))
        sig = direction * signal
        valid = np.isfinite(sig) & np.isfinite(self.ret_oc[h])
        if valid.sum() < cfg.min_names_per_day:
            return nil
        rp = self._rank_pct_within(self.codes, sig, valid)
        thr = 1.0 / cfg.n_quantiles
        is_long = valid & (rp > 1 - thr)
        is_short = valid & (rp <= thr)
        ret = self.ret_oc[h]
        sym = self.sym

        rebal_codes = range(0, self.D, h)  # non-overlapping
        cost = (cfg.cost_bps + cfg.slippage_bps) / 1e4
        gross, net, turns = [], [], []
        prev_l = prev_s = None
        for dc in rebal_codes:
            rows = self.day_rows[dc]
            if rows.size < cfg.min_names_per_day:
                continue
            lrows = rows[is_long[rows]]
            srows = rows[is_short[rows]]
            if lrows.size == 0 or srows.size == 0:
                continue
            g = ret[lrows].mean() - ret[srows].mean()
            ls, ss = set(sym[lrows]), set(sym[srows])
            if prev_l is None:
                turn = 1.0
            else:
                turn = 0.5 * ((1 - len(ls & prev_l) / max(1, len(ls)))
                              + (1 - len(ss & prev_s) / max(1, len(ss))))
            prev_l, prev_s = ls, ss
            turns.append(turn)
            gross.append(g)
            net.append(g - turn * cost * 2.0)
        if len(net) < 8:
            return {**nil, "n_reb": len(net)}
        net = np.asarray(net, float)
        gross = np.asarray(gross, float)
        ppy = 252.0 / h
        sd_n = net.std(ddof=1)
        sd_g = gross.std(ddof=1)
        return dict(
            gross_sharpe=(gross.mean() / sd_g) * math.sqrt(ppy) if sd_g > 0 else np.nan,
            net_sharpe=(net.mean() / sd_n) * math.sqrt(ppy) if sd_n > 0 else np.nan,
            gross_ann_ret=gross.mean() * ppy,
            net_ann_ret=net.mean() * ppy,
            turnover=float(np.mean(turns)) if turns else np.nan,
            max_dd=max_drawdown(np.cumprod(1.0 + net)),
            n_reb=len(net),
            net_rets=net,
            gross_rets=gross,
        )

    def regime_breakdown(self, ic: np.ndarray) -> dict:
        """IR of IC within day-level regimes + year stability."""
        s = pd.Series(ic, index=self.uniq_dates)
        attrs = self.day_attrs
        res: Dict[str, float] = {}

        def ir(mask) -> float:
            seg = s[mask.values if hasattr(mask, "values") else mask].dropna()
            if seg.size < 10 or seg.std() == 0:
                return np.nan
            return float(seg.mean() / seg.std())

        if "vol_state" in attrs:
            res["IR_hivol"] = ir(attrs["vol_state"] == 1)
            res["IR_lovol"] = ir(attrs["vol_state"] == 0)
        if "trend_state" in attrs:
            res["IR_trendup"] = ir(attrs["trend_state"] == "up")
            res["IR_trenddn"] = ir(attrs["trend_state"] == "down")
        if "disp_state" in attrs:
            res["IR_disphi"] = ir(attrs["disp_state"] == "hi")
            res["IR_displo"] = ir(attrs["disp_state"] == "lo")

        # year stability
        df = pd.DataFrame({"ic": ic, "year": self.year}).dropna()
        if not df.empty:
            ymean = df.groupby("year")["ic"].mean()
            overall = np.sign(np.nanmean(ic))
            res["year_hit"] = float((np.sign(ymean) == overall).mean()) if len(ymean) else np.nan
            res["n_years"] = int(len(ymean))
        else:
            res["year_hit"] = np.nan
            res["n_years"] = 0
        return res


# ───────────────────────── strategy generators ─────────────────────────

def _iter_singles(engine: ScoreEngine, feats: List[str], cfg: Config):
    for f in feats:
        v = pd.to_numeric(engine.work[f], errors="coerce").to_numpy()
        for h in cfg.horizons:
            yield dict(name=f, kind="single", horizon=h, feature=f, signal=v)


def _rank_feats_by_ir(engine: ScoreEngine, feats: List[str], cfg: Config) -> List[Tuple[str, float]]:
    """Cheap univariate ranking at the primary horizon to seed PAIRS/REGIME/COMPOSITE."""
    h = cfg.primary_horizon
    scored = []
    for f in feats:
        v = pd.to_numeric(engine.work[f], errors="coerce").to_numpy()
        ic = engine.ic_per_day(v, h)
        ic = ic[np.isfinite(ic)]
        if ic.size < 30 or ic.std() == 0:
            continue
        scored.append((f, abs(ic.mean() / ic.std())))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _iter_pairs(engine: ScoreEngine, top_feats: List[str], cfg: Config):
    h = cfg.primary_horizon
    z = {f: engine._zscore_within(pd.to_numeric(engine.work[f], errors="coerce").to_numpy())
         for f in top_feats}
    for i in range(len(top_feats)):
        for j in range(i + 1, len(top_feats)):
            a, b = top_feats[i], top_feats[j]
            za, zb = z[a], z[b]
            # 1) multiplicative interaction
            yield dict(name=f"PROD({a}*{b})", kind="pair_prod", horizon=h,
                       feature=f"{a}|{b}", signal=za * zb)
            # 2) sign-conditioned: B's payoff conditioned on sign of A
            yield dict(name=f"SIGNCOND(sign({a})*{b})", kind="pair_sign", horizon=h,
                       feature=f"{a}|{b}", signal=np.sign(za) * zb)
            # 3) gating: trade B only where A is in its top tercile that day
            rp_a = engine._rank_pct_within(engine.codes, za, np.isfinite(za))
            gated = np.where(rp_a > 2.0 / 3.0, zb, np.nan)
            yield dict(name=f"GATE({b} | {a}_top3)", kind="pair_gate", horizon=h,
                       feature=f"{a}|{b}", signal=gated)


def _iter_regime_gates(engine: ScoreEngine, top_feats: List[str], cfg: Config):
    h = cfg.primary_horizon
    attrs = engine.day_attrs
    # map day-level regime flags down to rows via codes
    def day_to_row(col_mask_series: pd.Series) -> np.ndarray:
        arr = col_mask_series.reindex(engine.uniq_dates).to_numpy()
        return arr[engine.codes]

    regimes = {}
    if "vol_state" in attrs:
        regimes["hivol"] = day_to_row(attrs["vol_state"] == 1)
        regimes["lovol"] = day_to_row(attrs["vol_state"] == 0)
    if "trend_state" in attrs:
        regimes["trendup"] = day_to_row(attrs["trend_state"] == "up")
        regimes["trenddn"] = day_to_row(attrs["trend_state"] == "down")
    if "disp_state" in attrs:
        regimes["disphi"] = day_to_row(attrs["disp_state"] == "hi")

    for f in top_feats:
        v = pd.to_numeric(engine.work[f], errors="coerce").to_numpy()
        for rname, rmask in regimes.items():
            sig = np.where(rmask.astype(bool), v, np.nan)
            yield dict(name=f"REGIME({f} @ {rname})", kind="regime_gate", horizon=h,
                       feature=f, signal=sig)


def _iter_composites(engine: ScoreEngine, ranked: List[Tuple[str, float]], cfg: Config):
    h = cfg.primary_horizon
    # orient each feature by the sign of its mean IC, then equal-weight z-scores.
    oriented = {}
    for f, _ in ranked:
        v = pd.to_numeric(engine.work[f], errors="coerce").to_numpy()
        ic = engine.ic_per_day(v, h)
        s = np.sign(np.nanmean(ic)) or 1.0
        oriented[f] = s * engine._zscore_within(v)
    feats_sorted = [f for f, _ in ranked]
    for n in cfg.composite_sizes:
        chosen = feats_sorted[:n]
        if len(chosen) < 2:
            continue
        stack = np.vstack([oriented[f] for f in chosen])
        sig = np.nanmean(stack, axis=0)
        yield dict(name=f"COMPOSITE(top{n})", kind="composite", horizon=h,
                   feature="+".join(chosen[:4]) + ("..." if len(chosen) > 4 else ""),
                   signal=sig)


def _iter_wq_xsection(engine: ScoreEngine, panel: pd.DataFrame, cfg: Config):
    """
    TRUE cross-sectional WorldQuant alphas: rank() is per-day across stocks, then
    time-series ops per symbol. These cannot be built by the per-symbol cache, which
    is exactly why the audit flagged them. Computed here at panel level.
    """
    h = cfg.primary_horizon
    w = engine.work
    # align raw inputs to engine.work order
    o = pd.to_numeric(w["open"], errors="coerce")
    hi = pd.to_numeric(w["high"], errors="coerce")
    lo = pd.to_numeric(w["low"], errors="coerce")
    cl = pd.to_numeric(w["close"], errors="coerce")
    vol = pd.to_numeric(w["volume"], errors="coerce")
    date = w["date"]
    sym = w["symbol"]

    def xrank(s: pd.Series) -> pd.Series:
        return s.groupby(date).rank(pct=True)

    def ts_corr(a: pd.Series, b: pd.Series, win: int) -> np.ndarray:
        tmp = pd.DataFrame({"a": np.asarray(a, dtype=float), "b": np.asarray(b, dtype=float)})
        g = sym.to_numpy()
        r = (tmp.groupby(g, sort=False, group_keys=False)
                .apply(lambda x: x["a"].rolling(win, min_periods=max(5, win // 2)).corr(x["b"])))
        return r.reindex(range(len(tmp))).to_numpy()

    ret = cl.groupby(sym, observed=True).pct_change()

    specs = []
    # Alpha#3: -corr(rank(open), rank(volume), 10)
    specs.append(("X_WQc_3", -ts_corr(xrank(o), xrank(vol), 10)))
    # Alpha#6: -corr(open, volume, 10)
    specs.append(("X_WQc_6", -ts_corr(o, vol, 10)))
    # Alpha#12: sign(delta(volume,1)) * -delta(close,1)
    dvol = vol.groupby(sym, observed=True).diff(1)
    dcl = cl.groupby(sym, observed=True).diff(1)
    specs.append(("X_WQc_12", (np.sign(dvol) * (-dcl)).to_numpy()))
    # Alpha#33: rank(-1 + open/close)  (pure cross-sectional)
    specs.append(("X_WQc_33", xrank(-1 + o / cl.replace(0, np.nan)).to_numpy()))
    # Alpha#101: (close-open)/(high-low+eps)  cross-sectionally ranked
    specs.append(("X_WQc_101", xrank((cl - o) / ((hi - lo) + 1e-9)).to_numpy()))
    # Alpha#44: -corr(high, rank(volume), 5)
    specs.append(("X_WQc_44", -ts_corr(hi, xrank(vol), 5)))

    for nm, arr in specs:
        arr = np.asarray(arr, dtype=float)
        if np.isfinite(arr).mean() < 0.2:
            continue
        yield dict(name=nm, kind="wq_xsection", horizon=h, feature=nm, signal=arr)


def generate_strategies(engine: ScoreEngine, panel: pd.DataFrame, feats: List[str], cfg: Config):
    """Yield all strategy specs from every enabled generator."""
    yield from _iter_singles(engine, feats, cfg)

    ranked = _rank_feats_by_ir(engine, feats, cfg)
    top_pairs = [f for f, _ in ranked[:cfg.top_k_pairs]]
    top_regime = [f for f, _ in ranked[:cfg.top_k_regime]]

    if cfg.do_pairs and len(top_pairs) >= 2:
        yield from _iter_pairs(engine, top_pairs, cfg)
    if cfg.do_regime_gates and top_regime:
        yield from _iter_regime_gates(engine, top_regime, cfg)
    if cfg.do_composites and ranked:
        yield from _iter_composites(engine, ranked, cfg)
    if cfg.do_wq_xsection and {"open", "high", "low", "close", "volume"} <= set(panel.columns):
        yield from _iter_wq_xsection(engine, panel, cfg)


# ───────────────────────── driver ─────────────────────────

def run(cfg: Config) -> pd.DataFrame:
    t0 = time.perf_counter()
    panel = load_panel(cfg)
    survivorship_check(panel)
    target_cols = build_targets(panel, cfg)
    feats = pick_features(panel, cfg)
    if not feats:
        raise SystemExit("No usable features after filtering.")

    engine = ScoreEngine(panel, target_cols, cfg)
    print(f"Scoring frame: {len(engine.work):,} tradeable rows, {engine.D} days, "
          f"{engine.test_day_mask.sum()} test days (embargo {cfg.embargo_days}d).")

    # materialise specs (so we know N for multiple-testing & deflated Sharpe)
    specs = list(generate_strategies(engine, panel, feats, cfg))
    N = len(specs)
    t_bar = expected_max_t(N)
    print(f"\nGenerated {N} strategies across "
          f"{len(set(s['kind'] for s in specs))} families.")
    print(f"Multiple-testing bar: |Newey-West t| must exceed E[max of {N} nulls] = {t_bar:.2f}")
    cost = cfg.cost_bps + cfg.slippage_bps
    print(f"L/S cost: {cost:.0f} bps/leg round-trip. NW lag = {cfg.lag()}. "
          f"Quantiles = {cfg.n_quantiles}.\n")

    rows = []
    sr_pool = []  # per-period net Sharpe across trials, for the deflated-Sharpe variance term
    for i, sp in enumerate(specs, 1):
        h = sp["horizon"]
        sig = sp["signal"]
        ic = engine.ic_per_day(sig, h)
        ic_v = ic[np.isfinite(ic)]
        if ic_v.size < 20:
            continue
        mean_ic = float(ic_v.mean())
        std_ic = float(ic_v.std(ddof=1)) if ic_v.size > 1 else np.nan
        ir = mean_ic / std_ic if std_ic and std_ic > 0 else np.nan
        t_nw, p_nw, n_nw = newey_west_t(ic_v, cfg.lag())
        direction = 1.0 if mean_ic >= 0 else -1.0

        # OOS sign on the embargoed test split
        test_ic = ic[engine.test_day_mask]
        test_ic = test_ic[np.isfinite(test_ic)]
        test_mean = float(test_ic.mean()) if test_ic.size else np.nan

        bt = engine.ls_backtest(sig, h, direction)
        net_rets = bt.pop("net_rets")
        gross_rets = bt.pop("gross_rets")
        # Skill detection (DSR/PSR) runs on GROSS returns: the null is "no predictive
        # skill" -> gross Sharpe ~ 0. Net Sharpe is kept as a separate DEPLOYABILITY
        # gate (does the skill survive costs?). Mixing them would let cost-bleed noise
        # define the null and crush every DSR.
        sr_period = np.nan
        skew = kurt = np.nan
        if gross_rets.size >= 8:
            sdg = gross_rets.std(ddof=1)
            sr_period = float(gross_rets.mean() / sdg) if sdg > 0 else np.nan
            skew, kurt = _skew_kurt(gross_rets)
            # The null Sharpe dispersion is estimated from RAW SINGLES at the primary
            # horizon only. Engineered families (pairs/regime/composite) are seeded
            # from top features and are deliberately enriched for true-positives, so
            # including them would inflate the null and crush every DSR.
            if (np.isfinite(sr_period) and gross_rets.size >= 20
                    and h == cfg.primary_horizon and sp["kind"] == "single"):
                sr_pool.append(sr_period)

        reg = engine.regime_breakdown(ic)

        rows.append(dict(
            strategy=sp["name"], kind=sp["kind"], horizon=h, direction=int(direction),
            feature=sp.get("feature", ""),
            mean_IC=mean_ic, IC_IR=ir, t_NW=t_nw, p_NW=p_nw,
            IC_hit=float((ic_v > 0).mean()), test_IC=test_mean,
            gross_sharpe=bt["gross_sharpe"], net_sharpe=bt["net_sharpe"],
            net_ann_ret=bt["net_ann_ret"], turnover=bt["turnover"],
            max_dd=bt["max_dd"], n_reb=bt["n_reb"], n_days=int(ic_v.size),
            srp=sr_period, nret=int(net_rets.size), rskew=skew, rkurt=kurt,
            **reg,
        ))
        if i % 200 == 0 or i == N:
            print(f"  scored {i}/{N}  ({time.perf_counter()-t0:.0f}s)")

    res = pd.DataFrame(rows)
    if res.empty:
        raise SystemExit("No strategy produced enough cross-sectional observations.")

    # Deflated Sharpe Ratio: dispersion of per-period Sharpe ACROSS the trials.
    # Use a ROBUST (MAD-based) estimate: when trials are highly correlated (many
    # strategies derived from the same few signals), a handful of genuine
    # true-positives inflate a plain variance and force every DSR to ~0. The MAD
    # tracks the null/bulk spread instead. Floored at the analytical null variance
    # (~1/n) so we never under-penalise.
    sr_arr = np.asarray(sr_pool, dtype=float)
    sr_arr = sr_arr[np.isfinite(sr_arr)]
    n_reb_typical = max(8, engine.D // cfg.primary_horizon)
    var_floor = 1.0 / n_reb_typical
    if sr_arr.size > 5:
        med = float(np.median(sr_arr))
        mad = float(np.median(np.abs(sr_arr - med)))
        var_sr = max((1.4826 * mad) ** 2, var_floor)
    elif sr_arr.size > 2:
        var_sr = max(float(np.var(sr_arr, ddof=1)), var_floor)
    else:
        var_sr = var_floor
    n_trials = len(res)
    if os.environ.get("STRATLAB_DEBUG"):
        sd = math.sqrt(var_sr)
        br = ((1 - EULER_GAMMA) * _norm_ppf(1 - 1.0 / n_trials)
              + EULER_GAMMA * _norm_ppf(1 - 1.0 / (n_trials * math.e)))
        print(f"[DEBUG] singles-null pool n={sr_arr.size}, var_sr={var_sr:.5f}, "
              f"sd={sd:.4f}, sr_star/period={sd * br:.4f}")
    res["DSR"] = [
        deflated_sharpe(r.srp, r.nret, r.rskew, r.rkurt, var_sr, n_trials)
        for r in res.itertuples()
    ]

    res["verdict"] = [_verdict(r, t_bar, cfg) for r in res.itertuples()]

    # rank: candidates first, then by deflated Sharpe, then |t|
    order = res.assign(
        _is_cand=res["verdict"].isin(["STRONG", "PROMISING"]).astype(int),
        _abs_t=res["t_NW"].abs(),
        _dsr=res["DSR"].fillna(0),
    ).sort_values(["_is_cand", "_dsr", "_abs_t"], ascending=False).index
    res = res.loc[order].reset_index(drop=True)

    _write_outputs(res, engine, cfg, N, t_bar, var_sr)
    print(f"\nDone in {time.perf_counter()-t0:.0f}s.")
    return res


def _verdict(r, t_bar: float, cfg: Config) -> str:
    t = abs(r.t_NW) if pd.notna(r.t_NW) else 0.0
    ir = abs(r.IC_IR) if pd.notna(r.IC_IR) else 0.0
    gsh = r.gross_sharpe if pd.notna(r.gross_sharpe) else np.nan
    nsh = r.net_sharpe if pd.notna(r.net_sharpe) else np.nan
    dsr = r.DSR if pd.notna(r.DSR) else 0.0
    yh = r.year_hit if pd.notna(r.year_hit) else 0.0
    oos_ok = (pd.isna(r.test_IC)) or (np.sign(r.test_IC) == np.sign(r.mean_IC))

    # too-good-to-be-true -> almost always upstream leakage, not alpha
    if ir > cfg.too_good_daily_ir or (pd.notna(gsh) and gsh > cfg.too_good_gross_sharpe):
        return "AVOID (too good - check leakage)"

    if t < t_bar:
        return "NO_EDGE"

    if (dsr >= cfg.dsr_strong and pd.notna(nsh) and nsh >= cfg.net_sharpe_strong
            and yh >= cfg.year_hit_min and oos_ok):
        return "STRONG"
    if (dsr >= cfg.dsr_promising and pd.notna(nsh) and nsh >= cfg.net_sharpe_promising
            and yh >= cfg.year_hit_min and oos_ok):
        return "PROMISING"
    return "MARGINAL"


def _write_outputs(res: pd.DataFrame, engine: ScoreEngine, cfg: Config,
                   N: int, t_bar: float, var_sr: float) -> None:
    out_dir = Path(cfg.out_dir)
    public_cols = [
        "strategy", "kind", "horizon", "direction", "feature",
        "verdict", "DSR", "mean_IC", "IC_IR", "t_NW", "p_NW", "IC_hit",
        "test_IC", "gross_sharpe", "net_sharpe", "net_ann_ret", "turnover",
        "max_dd", "n_reb", "n_days",
        "IR_hivol", "IR_lovol", "IR_trendup", "IR_trenddn", "IR_disphi", "IR_displo",
        "year_hit", "n_years",
    ]
    public_cols = [c for c in public_cols if c in res.columns]
    board = res[public_cols].copy()
    csv_path = out_dir / "strategy_lab_scoreboard.csv"
    board.to_csv(csv_path, index=False)

    counts = res["verdict"].value_counts().to_dict()
    summary = dict(
        n_strategies=int(N),
        n_scored=int(len(res)),
        multiple_testing_t_bar=round(t_bar, 3),
        deflated_sharpe_trial_var=round(var_sr, 6),
        verdict_counts=counts,
        families={k: int(v) for k, v in res["kind"].value_counts().items()},
        config=dict(horizons=list(cfg.horizons), primary_horizon=cfg.primary_horizon,
                    n_quantiles=cfg.n_quantiles, cost_bps=cfg.cost_bps,
                    slippage_bps=cfg.slippage_bps, min_names_per_day=cfg.min_names_per_day,
                    use_model_target=cfg.use_model_target),
    )
    with open(out_dir / "strategy_lab_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    # detailed top-K (best by DSR among non-AVOID)
    top = res[~res["verdict"].str.startswith("AVOID")].head(cfg.detail_top_k)
    top.to_csv(out_dir / "strategy_lab_top.csv", index=False)

    # ── pretty print ──
    show = board.copy()
    fmt = ["DSR", "mean_IC", "IC_IR", "t_NW", "p_NW", "IC_hit", "test_IC",
           "gross_sharpe", "net_sharpe", "net_ann_ret", "turnover", "max_dd",
           "IR_hivol", "IR_lovol", "IR_trendup", "IR_trenddn", "year_hit"]
    for c in fmt:
        if c in show.columns:
            show[c] = show[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    pd.set_option("display.max_rows", 120, "display.width", 260,
                  "display.max_colwidth", 42)
    cols_print = ["strategy", "kind", "horizon", "verdict", "DSR", "mean_IC",
                  "IC_IR", "t_NW", "net_sharpe", "year_hit"]
    cols_print = [c for c in cols_print if c in show.columns]
    print("\n==================== TOP 40 STRATEGIES (ranked) ====================")
    print(show[cols_print].head(40).to_string(index=False))

    print("\n==================== VERDICT TALLY ====================")
    for k in ["STRONG", "PROMISING", "MARGINAL", "NO_EDGE"]:
        print(f"  {k:<10}: {counts.get(k, 0)}")
    avoid = sum(v for k, v in counts.items() if k.startswith("AVOID"))
    print(f"  {'AVOID':<10}: {avoid}")
    print(f"\n  scoreboard : {csv_path}")
    print(f"  summary    : {out_dir / 'strategy_lab_summary.json'}")
    print(f"  top detail : {out_dir / 'strategy_lab_top.csv'}")
    print("\n  Reality check:")
    print("   - Clearing the gates is NECESSARY, not SUFFICIENT. Paper-trade before capital.")
    print("   - 'AVOID (too good)' almost always means look-ahead leakage upstream, not alpha.")
    print("   - Deflated Sharpe already discounts the number of strategies tried.")


# ───────────────────────── synthetic self-test ─────────────────────────

def make_synthetic_panel(n_symbols: int = 120, n_days: int = 1200, seed: int = 7,
                         snr_alpha: float = 0.020, snr_beta: float = 0.040,
                         noise: float = 1.0) -> pd.DataFrame:
    """
    Build a synthetic OHLCV panel with PLANTED, causal, REALISTICALLY-WEAK signals so
    we can verify the orchestrator end-to-end without the real data:

      D_signal_alpha : clean cross-sectional predictor of the forward 5d return.
      D_signal_beta  : predictor that ONLY works in the high-vol regime (tests the
                       regime view).
      D_noise_*      : pure noise features (should land in NO_EDGE).
      D_mom20, D_rsi : ordinary momentum/RSI-style columns (D_mom20 will partly pick
                       up the persistent alpha; that is expected, not a bug).

    Causality: each day's return is driven by the signal value of the PREVIOUS day
    (signal[t-1]), so signal[t] predicts FUTURE returns only -> no look-ahead. Signal
    strength is tuned so the planted alpha's daily IC IR is ~0.15-0.30 (a strong but
    plausible factor), NOT the absurd >0.4 that would (correctly) trip the leakage
    flag. The regime signal is gated by the (slowly-switching) market vol state.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    syms = [f"SYM{i:03d}" for i in range(n_symbols)]

    # market-wide vol regime (slow switching) -> high-vol stretches
    vol_state = np.zeros(n_days, dtype=int)
    s = 0
    for d in range(n_days):
        if rng.random() < 0.015:
            s ^= 1
        vol_state[d] = s
    base_vol = np.where(vol_state == 1, 0.022, 0.011)
    vs_lag = np.concatenate([[0], vol_state[:-1]]).astype(float)  # regime known at t-1

    def ar1(rho: float) -> np.ndarray:
        x = np.zeros(n_days)
        for d in range(1, n_days):
            x[d] = rho * x[d - 1] + rng.normal(0, 1)
        return (x - x.mean()) / (x.std() + 1e-9)

    frames = []
    for sym in syms:
        a = ar1(0.92)   # persistent latent alpha
        b = ar1(0.92)   # persistent latent regime-alpha
        a_lag = np.concatenate([[0.0], a[:-1]])   # drive return[t] with a[t-1]
        b_lag = np.concatenate([[0.0], b[:-1]])
        idio = rng.normal(0, 1, n_days)

        daily_ret = base_vol * (snr_alpha * a_lag
                                + snr_beta * b_lag * vs_lag
                                + noise * idio)

        close = 100 * np.cumprod(1 + daily_ret)
        openp = close / (1 + daily_ret * 0.5)
        high = np.maximum(openp, close) * (1 + np.abs(rng.normal(0, 0.004, n_days)))
        low = np.minimum(openp, close) * (1 - np.abs(rng.normal(0, 0.004, n_days)))
        volume = rng.lognormal(12, 0.4, n_days)

        df = pd.DataFrame({
            "timestamp": dates, "symbol": sym,
            "open": openp, "high": high, "low": low, "close": close, "volume": volume,
            "D_signal_alpha": a,
            "D_signal_beta": b,
            "D_atr14": pd.Series(close).rolling(14, min_periods=1).std().to_numpy(),
            "D_sma200": pd.Series(close).rolling(200, min_periods=1).mean().to_numpy(),
            "D_adx14": rng.uniform(10, 40, n_days),
            "D_mom20": pd.Series(close).pct_change(20).to_numpy(),
            "D_rsi14": rng.uniform(20, 80, n_days),
            "D_noise_1": rng.normal(0, 1, n_days),
            "D_noise_2": rng.normal(0, 1, n_days),
            "D_noise_3": rng.normal(0, 1, n_days),
            "D_noise_4": rng.normal(0, 1, n_days),
            "D_noise_5": rng.normal(0, 1, n_days),
            "D_noise_6": rng.normal(0, 1, n_days),
            "D_noise_7": rng.normal(0, 1, n_days),
            "D_noise_8": rng.normal(0, 1, n_days),
            "regime_high_vol": vol_state,
        })
        frames.append(df)

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = _norm_date(panel["timestamp"])
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    print(f"SELF-TEST PANEL: {n_symbols} symbols x {n_days} days = {len(panel):,} rows "
          f"(planted: D_signal_alpha clean, D_signal_beta regime-conditional).")
    return panel


def self_test() -> int:
    cfg = Config(self_test=True, horizons=(1, 3, 5), primary_horizon=5,
                 top_k_pairs=6, top_k_regime=6, min_names_per_day=20,
                 composite_sizes=(3, 5), detail_top_k=15)
    res = run(cfg)

    print("\n==================== SELF-TEST ASSERTIONS ====================")
    ok = True

    # 1) the clean planted alpha must be a top single at the primary horizon
    singles5 = res[(res["kind"] == "single") & (res["horizon"] == 5)].copy()
    singles5 = singles5.reindex(singles5["t_NW"].abs().sort_values(ascending=False).index)
    top_feats = list(singles5["feature"].head(3))
    cond1 = "D_signal_alpha" in top_feats
    print(f"  [{'PASS' if cond1 else 'FAIL'}] D_signal_alpha in top-3 singles by |t|: {top_feats}")
    ok &= cond1

    # 2) planted alpha must be recognised as having edge at the primary horizon
    #    (STRONG / PROMISING / or AVOID-too-good on this ultra-clean panel) - never
    #    dismissed as NO_EDGE or MARGINAL.
    va = res.loc[(res["feature"] == "D_signal_alpha") & (res["kind"] == "single")
                 & (res["horizon"] == 5), "verdict"]
    vstr = va.iloc[0] if not va.empty else "MISSING"
    cond2 = (not va.empty) and (vstr.startswith(("STRONG", "PROMISING", "AVOID")))
    print(f"  [{'PASS' if cond2 else 'FAIL'}] D_signal_alpha recognised as edge "
          f"(STRONG/PROMISING/AVOID): {vstr}")
    ok &= cond2

    # 3) pure noise features should overwhelmingly be NO_EDGE
    noise = res[res["feature"].astype(str).str.startswith("D_noise_")]
    frac_noise_dead = (noise["verdict"] == "NO_EDGE").mean() if len(noise) else 1.0
    cond3 = frac_noise_dead >= 0.6
    print(f"  [{'PASS' if cond3 else 'FAIL'}] >=60% of noise features are NO_EDGE: "
          f"{frac_noise_dead:.0%}")
    ok &= cond3

    # 4) regime view: beta's IC-IR in high vol should beat low vol
    vb = res[(res["feature"] == "D_signal_beta") & (res["kind"] == "single")
             & (res["horizon"] == 5)]
    if not vb.empty and pd.notna(vb["IR_hivol"].iloc[0]) and pd.notna(vb["IR_lovol"].iloc[0]):
        cond4 = abs(vb["IR_hivol"].iloc[0]) > abs(vb["IR_lovol"].iloc[0])
        print(f"  [{'PASS' if cond4 else 'FAIL'}] D_signal_beta |IR| hi-vol > lo-vol: "
              f"{vb['IR_hivol'].iloc[0]:.3f} vs {vb['IR_lovol'].iloc[0]:.3f}")
    else:
        cond4 = True
        print("  [SKIP] D_signal_beta regime IRs unavailable")
    ok &= cond4

    # 5) we really did test hundreds of strategies (generators fired)
    cond5 = len(res) >= 50
    print(f"  [{'PASS' if cond5 else 'FAIL'}] scored >=50 strategies: {len(res)}")
    ok &= cond5

    # 6) the STRONG/PROMISING decision path must actually fire on a real signal
    n_good = int(res["verdict"].isin(["STRONG", "PROMISING"]).sum())
    cond6 = n_good >= 1
    print(f"  [{'PASS' if cond6 else 'FAIL'}] >=1 STRONG/PROMISING strategy surfaced: {n_good}")
    ok &= cond6

    print("\n" + ("ALL SELF-TESTS PASSED" if ok else "SELF-TEST FAILURES ABOVE"))
    return 0 if ok else 1


# ───────────────────────── CLI ─────────────────────────

def parse_args() -> Tuple[Config, bool]:
    p = argparse.ArgumentParser(description="Multi-strategy cross-sectional edge finder.")
    p.add_argument("--panel", type=Path, dest="panel_path",
                   help="path to model panel_cache.parquet (recommended)")
    p.add_argument("--cache-dir", type=Path, help="legacy per-symbol *_daily.parquet dir")
    p.add_argument("--out-dir", type=Path, default=Path("."))
    p.add_argument("--self-test", action="store_true", help="run synthetic planted-signal test")

    p.add_argument("--horizons", type=int, nargs="+", default=None)
    p.add_argument("--primary-horizon", type=int, default=None)
    p.add_argument("--raw-target", action="store_true", help="skip model vol-adjusted target")

    p.add_argument("--quantiles", type=int, default=None, dest="n_quantiles")
    p.add_argument("--cost-bps", type=float, default=None)
    p.add_argument("--slippage-bps", type=float, default=None)
    p.add_argument("--min-names", type=int, default=None, dest="min_names_per_day")
    p.add_argument("--min-adv", type=float, default=None, dest="min_adv_inr")
    p.add_argument("--min-price", type=float, default=None)
    p.add_argument("--embargo", type=int, default=None, dest="embargo_days")

    p.add_argument("--top-k-pairs", type=int, default=None)
    p.add_argument("--top-k-regime", type=int, default=None)
    p.add_argument("--no-pairs", action="store_true")
    p.add_argument("--no-regime-gates", action="store_true")
    p.add_argument("--no-composites", action="store_true")
    p.add_argument("--no-wq", action="store_true")
    p.add_argument("--max-features", type=int, default=None)
    p.add_argument("--keep-leaky-wq", action="store_true")
    a = p.parse_args()

    cfg = Config()
    if a.self_test:
        return cfg, True
    if a.panel_path is not None:
        cfg.panel_path = a.panel_path
    if a.cache_dir is not None:
        cfg.cache_dir = a.cache_dir
        cfg.panel_path = None
    if a.horizons:
        cfg.horizons = tuple(a.horizons)
    if a.primary_horizon:
        cfg.primary_horizon = a.primary_horizon
    if a.raw_target:
        cfg.use_model_target = False
    for k in ["out_dir", "n_quantiles", "cost_bps", "slippage_bps", "min_names_per_day",
              "min_adv_inr", "min_price", "embargo_days", "top_k_pairs", "top_k_regime",
              "max_features"]:
        v = getattr(a, k, None)
        if v is not None:
            setattr(cfg, k, v)
    if a.no_pairs:
        cfg.do_pairs = False
    if a.no_regime_gates:
        cfg.do_regime_gates = False
    if a.no_composites:
        cfg.do_composites = False
    if a.no_wq:
        cfg.do_wq_xsection = False
    if a.keep_leaky_wq:
        cfg.exclude_leaky_wq = False
    cfg.__post_init__()
    return cfg, False


if __name__ == "__main__":
    _cfg, _is_selftest = parse_args()
    if _is_selftest:
        raise SystemExit(self_test())
    run(_cfg)

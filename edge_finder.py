#!/usr/bin/env python3
"""
edge_finder.py  (v2 — panel-native)

An HONEST cross-sectional edge scanner. v2 reads the MODEL's enriched
panel_cache.parquet directly (recommended), so it scores features against the
exact target, feature set, and holdout the model itself uses. It still supports
the old per-symbol *_daily.parquet mode.

WHY PANEL MODE IS MORE ACCURATE (vs v1 per-symbol mode):
  • Same target  : the model optimizes the cross-sectional rank of the
                   VOL-ADJUSTED 5d return  ret_5d_adj = ret_5d_oc_pct / atr_pct
                   (entry = next open, exit = close[t+5]). v2 scores IC against
                   ret_5d_adj, not raw close-to-close return.
  • Same features: scans the model's whole feature set (D_/W_/WQ_/M_/X_/X_regime_
                   /regime_/Comb_/CPR_/Struct_/DayType_/DOW_), minus the model's
                   NEVER_FEATURE label/helper columns.
  • Same holdout : reports IC on the model's day-aligned, embargoed TEST split
                   and requires the sign to survive there.

HONESTY GATES (a feature is "CANDIDATE" only if it clears ALL):
  1. Newey-West t-stat  (overlapping 5d returns inflate naive t by ~sqrt(H))
  2. Multiple-testing   (best |t| must beat E[max t] of N null trials)
  3. Net-of-cost L/S    (dollar-neutral quantile spread, India costs+slippage,
                         realized open->close return, non-overlapping rebalance)
  4. Purged walk-forward stability (same IC sign in >=60% of folds)
  5. Model holdout      (test-split IC same sign as full-sample IC)

Clearing every gate is NECESSARY, NOT SUFFICIENT. It means "worth paper-trading",
not "worth capital". It CANNOT detect leakage baked into a feature upstream
(e.g. the whole-series _rank_cs WQ bug) — fix that at source and rebuild.

Deps: numpy, pandas (+ pyarrow to read parquet). No scipy/sklearn/statsmodels.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# WQ alphas that leak via whole-series .rank() in Daily_cache.py. Excluded by default.
LEAKY_WQ_DEFAULT = [
    "D_WQ_3", "D_WQ_13", "D_WQ_16", "D_WQ_19", "D_WQ_20",
    "D_WQ_29", "D_WQ_33", "D_WQ_38", "D_WQ_40", "D_WQ_44",
    "WQ_3", "WQ_13", "WQ_16", "WQ_19", "WQ_20",
    "WQ_29", "WQ_33", "WQ_38", "WQ_40", "WQ_44",
]

# Mirror of New_model__3_.discover_daily_features NEVER_FEATURE (labels/helpers).
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


@dataclass
class Config:
    panel_path: Optional[Path] = Path(r"C:\Users\karanvsi\Desktop\Pycharm\Cache\panel_cache.parquet")
    cache_dir: Optional[Path] = None
    macro_path: Optional[Path] = None
    out_dir: Path = Path(".")

    horizon: int = 5
    use_model_target: bool = True

    train_frac: float = 0.70
    cal_frac: float = 0.20
    embargo_days: int = 5

    n_quantiles: int = 5
    n_folds: int = 5

    min_adv_inr: float = 0.0
    min_price: float = 0.0
    min_names_per_day: int = 30

    cost_bps: float = 15.0
    slippage_bps: float = 10.0
    nw_lag: Optional[int] = None
    exclude_leaky_wq: bool = True
    max_features: Optional[int] = None

    def lag(self) -> int:
        return self.nw_lag if self.nw_lag is not None else self.horizon

    def __post_init__(self):
        for a in ("panel_path", "cache_dir", "macro_path", "out_dir"):
            v = getattr(self, a)
            if v is not None:
                setattr(self, a, Path(v))


# ───────────────────────── Math (no scipy) ─────────────────────────

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
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def two_sided_p(t: float) -> float:
    return 2.0 * (1.0 - _norm_cdf(abs(t)))


def expected_max_t(n_trials: int) -> float:
    n = max(2, int(n_trials))
    g = 0.5772156649015329
    return (1 - g) * _norm_ppf(1 - 1.0 / n) + g * _norm_ppf(1 - 1.0 / (n * math.e))


def newey_west_t(x: pd.Series, lag: int) -> Tuple[float, float, int]:
    x = pd.Series(x).dropna().to_numpy(dtype=float)
    n = x.size
    if n < max(10, lag + 2):
        return (np.nan, np.nan, n)
    mu = x.mean()
    e = x - mu
    var = np.dot(e, e) / n
    L = min(lag, n - 1)
    for k in range(1, L + 1):
        w = 1.0 - k / (L + 1.0)
        var += 2.0 * w * (np.dot(e[k:], e[:-k]) / n)
    if var <= 0:
        return (np.nan, np.nan, n)
    se = math.sqrt(var / n)
    return (mu / se if se > 0 else np.nan, two_sided_p(mu / se) if se > 0 else np.nan, n)


def max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return np.nan
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


# ───────────────────────── Loading ─────────────────────────

def _norm_date(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, errors="coerce")
    tz = getattr(out.dt, "tz", None)
    if tz is not None:
        out = out.dt.tz_localize(None)
    return out.dt.normalize()


def load_panel(cfg: Config) -> pd.DataFrame:
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

    if not cfg.cache_dir:
        raise SystemExit("No panel_path and no cache_dir. Provide --panel or --cache-dir.")
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
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    if cfg.macro_path and Path(cfg.macro_path).exists():
        macro = pd.read_parquet(cfg.macro_path)
        macro["date"] = _norm_date(macro["date"])
        panel = panel.merge(macro, on="date", how="left")
    print(f"  {len(files)} symbols, {len(panel):,} rows")
    return panel


def survivorship_check(panel: pd.DataFrame) -> None:
    gmax = panel["date"].max()
    alive = (panel.groupby("symbol")["date"].max() >= gmax - pd.Timedelta(days=10)).mean()
    print("\n── Survivorship check ──")
    print(f"  symbols alive at sample end: {alive:6.1%}")
    if alive > 0.95:
        print("  ⚠  >95% survive to the end -> universe is almost certainly point-in-time")
        print("     = survivorship bias. Every result below is optimistic until you")
        print("     rebuild on a delisting-aware universe.")
    else:
        print("  ok-ish: a meaningful fraction of names die mid-sample.")


# ───────────────────────── Target & features ─────────────────────────

def build_target(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    close = pd.to_numeric(panel.get("close"), errors="coerce")

    if cfg.use_model_target and "ret_5d_oc_pct" in panel.columns:
        r5 = pd.to_numeric(panel["ret_5d_oc_pct"], errors="coerce")
        if "atr_pct" in panel.columns:
            atr_pct = pd.to_numeric(panel["atr_pct"], errors="coerce")
        elif "D_atr14" in panel.columns and close is not None:
            atr_pct = (pd.to_numeric(panel["D_atr14"], errors="coerce") /
                       close.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan) * 100.0
        else:
            atr_pct = None
        if atr_pct is not None:
            panel["_fwd"] = (r5 / atr_pct.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
            print("Target: model's vol-adjusted 5d return  (ret_5d_oc_pct / atr_pct).")
        else:
            panel["_fwd"] = r5
            print("Target: raw ret_5d_oc_pct (no ATR column found for vol-adjust).")
        panel["_pnl"] = r5 / 100.0
    else:
        g = panel.groupby("symbol", sort=False)["close"]
        fwd = g.shift(-cfg.horizon) / close - 1.0
        panel["_fwd"] = fwd
        panel["_pnl"] = fwd
        print("Target: close-to-close forward return (legacy mode).")

    if cfg.min_adv_inr > 0 or cfg.min_price > 0:
        vol = pd.to_numeric(panel.get("volume", np.nan), errors="coerce")
        adv = (close * vol).groupby(panel["symbol"]).transform(
            lambda s: s.rolling(20, min_periods=10).mean())
        panel["_tradeable"] = ((adv >= cfg.min_adv_inr) & (close >= cfg.min_price)).fillna(False)
    else:
        panel["_tradeable"] = panel["_fwd"].notna()
    return panel


def pick_features(panel: pd.DataFrame, cfg: Config, panel_mode: bool) -> List[str]:
    cols = list(panel.columns)
    if panel_mode:
        drop = set(MODEL_NEVER_FEATURE)
        feats = [c for c in cols
                 if c.startswith(MODEL_FEATURE_PREFIXES)
                 and c not in drop
                 and not c.startswith(MODEL_NEVER_PATTERNS)]
    else:
        feats = [c for c in cols if c.startswith(("D_", "W_", "M_"))]

    forward_like = [c for c in cols if ("fwd" in c.lower()) or c.lower().startswith("ret_")
                    or ("future" in c.lower()) or c in ("_fwd", "_pnl", "_tradeable")]
    drop_all = set(forward_like)
    if cfg.exclude_leaky_wq:
        drop_all |= set(LEAKY_WQ_DEFAULT)
    feats = [c for c in feats if c not in drop_all]

    leaky_present = [c for c in cols if c in set(LEAKY_WQ_DEFAULT)]
    if leaky_present:
        print(f"\n⚠  Panel contains {len(leaky_present)} known-leaky WQ columns the MODEL "
              f"trains on: {', '.join(leaky_present)}")
        if cfg.exclude_leaky_wq:
            print("   (edge_finder excludes them. Consider excluding them in the model too.)")

    out = []
    for c in feats:
        s = pd.to_numeric(panel[c], errors="coerce")
        if s.notna().mean() >= 0.30 and s.nunique(dropna=True) > 5:
            panel[c] = s
            out.append(c)
    if cfg.max_features:
        out = out[:cfg.max_features]
    print(f"Testing {len(out)} features.")
    return out


def date_splits(panel: pd.DataFrame, cfg: Config) -> Tuple[set, set]:
    d = np.sort(panel["date"].unique())
    n = len(d)
    if n < 30:
        return set(d), set(d)
    cut_tr = int(cfg.train_frac * n)
    cut_cal = int((cfg.train_frac + cfg.cal_frac) * n)
    emb = max(0, cfg.embargo_days)
    train = set(d[:max(0, cut_tr - emb)])
    test = set(d[min(n - 1, cut_cal + emb):])
    return train, test


# ───────────────────────── Core stats ─────────────────────────

def ic_series(panel: pd.DataFrame, feat: str, cfg: Config) -> pd.Series:
    d = panel.loc[panel["_tradeable"], ["date", feat, "_fwd"]].dropna()
    if d.empty:
        return pd.Series(dtype=float)
    cnt = d.groupby("date")[feat].transform("size")
    d = d[cnt >= cfg.min_names_per_day]
    if d.empty:
        return pd.Series(dtype=float)
    fr = d.groupby("date")[feat].rank()
    tr = d.groupby("date")["_fwd"].rank()
    by = d["date"]
    fr_dm = fr - fr.groupby(by).transform("mean")
    tr_dm = tr - tr.groupby(by).transform("mean")
    num = (fr_dm * tr_dm).groupby(by).sum()
    den = np.sqrt((fr_dm ** 2).groupby(by).sum() * (tr_dm ** 2).groupby(by).sum())
    return (num / den).replace([np.inf, -np.inf], np.nan).dropna()


def fold_stability(ic: pd.Series, cfg: Config) -> float:
    if ic.size < cfg.n_folds * 5:
        return np.nan
    ic = ic.sort_index()
    overall = np.sign(ic.mean())
    same = used = 0
    for k, block in enumerate(np.array_split(np.arange(ic.size), cfg.n_folds)):
        if block.size == 0:
            continue
        lo, hi = block[0], block[-1]
        if k > 0:
            lo = min(lo + cfg.horizon, hi)
        seg = ic.iloc[lo:hi + 1]
        if seg.size == 0:
            continue
        used += 1
        same += int(np.sign(seg.mean()) == overall)
    return same / used if used else np.nan


def quantile_ls(panel: pd.DataFrame, feat: str, cfg: Config) -> dict:
    d = panel.loc[panel["_tradeable"], ["date", "symbol", feat, "_pnl"]].dropna()
    nil = {"net_sharpe": np.nan, "net_ann_ret": np.nan, "turnover": np.nan, "max_dd": np.nan, "n_reb": 0}
    if d.empty:
        return nil
    dates = np.sort(d["date"].unique())[::cfg.horizon]
    cost = (cfg.cost_bps + cfg.slippage_bps) / 1e4
    prev_l, prev_s = set(), set()
    net, turns = [], []
    for dte in dates:
        day = d[d["date"] == dte]
        if len(day) < cfg.min_names_per_day:
            continue
        q = pd.qcut(day[feat].rank(method="first"), cfg.n_quantiles, labels=False, duplicates="drop")
        if q is None or q.nunique() < cfg.n_quantiles:
            continue
        longs, shorts = day.loc[q == cfg.n_quantiles - 1], day.loc[q == 0]
        if longs.empty or shorts.empty:
            continue
        g = longs["_pnl"].mean() - shorts["_pnl"].mean()
        ls, ss = set(longs["symbol"]), set(shorts["symbol"])
        turn = 1.0 if not (prev_l or prev_s) else 0.5 * (
            (1 - len(ls & prev_l) / max(1, len(ls))) + (1 - len(ss & prev_s) / max(1, len(ss))))
        prev_l, prev_s = ls, ss
        turns.append(turn)
        net.append(g - turn * cost * 2.0)
    if len(net) < 8:
        return {**nil, "n_reb": len(net)}
    net = np.asarray(net, float)
    ppy = 252.0 / cfg.horizon
    sd = net.std(ddof=1)
    return {
        "net_sharpe": (net.mean() / sd) * math.sqrt(ppy) if sd > 0 else np.nan,
        "net_ann_ret": net.mean() * ppy,
        "turnover": float(np.mean(turns)) if turns else np.nan,
        "max_dd": max_drawdown(np.cumprod(1.0 + net)),
        "n_reb": len(net),
    }


# ───────────────────────── Driver ─────────────────────────

def run(cfg: Config) -> pd.DataFrame:
    panel = load_panel(cfg)
    panel_mode = bool(cfg.panel_path and Path(cfg.panel_path).exists())
    survivorship_check(panel)
    panel = build_target(panel, cfg)
    feats = pick_features(panel, cfg, panel_mode)
    if not feats:
        raise SystemExit("No usable features after filtering.")

    train_dates, test_dates = date_splits(panel, cfg)
    print(f"Model-style split: {len(train_dates)} train days, {len(test_dates)} test days "
          f"(embargo {cfg.embargo_days}d). IC sign must survive the test split.")

    N = len(feats)
    t_bar = expected_max_t(N)
    print(f"\nMultiple-testing bar: best |t| must exceed E[max of {N} nulls] = {t_bar:.2f}")
    print(f"Cost: {cfg.cost_bps:.0f}bps + {cfg.slippage_bps:.0f}bps slippage per leg. "
          f"Newey-West lag = {cfg.lag()}.\n")

    rows = []
    for i, f in enumerate(feats, 1):
        ic = ic_series(panel, f, cfg)
        if ic.size < 30:
            continue
        t, p, n = newey_west_t(ic, cfg.lag())
        test_ic = ic[ic.index.isin(test_dates)]
        bt = quantile_ls(panel, f, cfg)
        rows.append({
            "feature": f,
            "mean_IC": ic.mean(),
            "IC_IR": ic.mean() / ic.std() if ic.std() > 0 else np.nan,
            "t_NW": t, "p_NW": p,
            "hit_rate": (ic > 0).mean(),
            "test_IC": test_ic.mean() if test_ic.size else np.nan,
            "fold_stab": fold_stability(ic, cfg),
            "net_LS_sharpe": bt["net_sharpe"],
            "net_ann_ret": bt["net_ann_ret"],
            "turnover": bt["turnover"],
            "max_dd": bt["max_dd"],
            "n_days": n, "n_reb": bt["n_reb"],
        })
        if i % 25 == 0 or i == len(feats):
            print(f"  scanned {i}/{len(feats)}")

    res = pd.DataFrame(rows)
    if res.empty:
        raise SystemExit("No feature produced enough cross-sectional observations.")

    def verdict(r) -> str:
        if pd.isna(r["t_NW"]):
            return "insufficient_data"
        if abs(r["t_NW"]) < t_bar:
            return "noise (fails multiple-testing)"
        if pd.notna(r["net_LS_sharpe"]) and r["net_LS_sharpe"] <= 0:
            return "fails net-of-cost"
        if pd.notna(r["fold_stab"]) and r["fold_stab"] < 0.6:
            return "unstable across folds"
        if pd.notna(r["test_IC"]) and np.sign(r["test_IC"]) != np.sign(r["mean_IC"]):
            return "decays out-of-sample"
        return "CANDIDATE"

    res["verdict"] = res.apply(verdict, axis=1)
    res = res.reindex(res["t_NW"].abs().sort_values(ascending=False).index).reset_index(drop=True)

    out_path = Path(cfg.out_dir) / "edge_finder_results.csv"
    res.to_csv(out_path, index=False)

    show = res.copy()
    for c in ["mean_IC", "IC_IR", "t_NW", "p_NW", "hit_rate", "test_IC", "fold_stab",
              "net_LS_sharpe", "net_ann_ret", "turnover", "max_dd"]:
        show[c] = show[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    pd.set_option("display.max_rows", 80, "display.width", 220)
    print("\n================= RESULTS (sorted by |Newey-West t|) =================")
    print(show.head(40).to_string(index=False))

    n_cand = (res["verdict"] == "CANDIDATE").sum()
    print("\n================= HONEST SUMMARY =================")
    print(f"  features tested : {N}")
    print(f"  candidates      : {n_cand}")
    print(f"  results written : {out_path}")
    print("\n  Reality check:")
    print("   • Clearing every gate is NECESSARY, not SUFFICIENT — not proof of edge.")
    print("   • Cannot detect leakage baked INTO a feature upstream (the _rank_cs WQ")
    print("     bug). Fix it in Daily_cache.py and rebuild the panel.")
    print("   • If the survivorship warning fired, every number above is optimistic.")
    print("   • Next step is PAPER trading, not capital. Expect live IC well below this.")
    return res


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Honest panel-native edge finder.")
    p.add_argument("--panel", type=Path, dest="panel_path",
                   help="path to model panel_cache.parquet (recommended)")
    p.add_argument("--cache-dir", type=Path, help="legacy per-symbol *_daily.parquet dir")
    p.add_argument("--macro", type=Path, dest="macro_path")
    p.add_argument("--out-dir", type=Path, default=Path("."))
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--raw-target", action="store_true", help="skip model vol-adjusted target")
    p.add_argument("--quantiles", type=int, default=5, dest="n_quantiles")
    p.add_argument("--folds", type=int, default=5, dest="n_folds")
    p.add_argument("--embargo", type=int, default=5, dest="embargo_days")
    p.add_argument("--cost-bps", type=float, default=15.0)
    p.add_argument("--slippage-bps", type=float, default=10.0)
    p.add_argument("--min-adv", type=float, default=0.0, dest="min_adv_inr")
    p.add_argument("--min-price", type=float, default=0.0)
    p.add_argument("--min-names", type=int, default=30, dest="min_names_per_day")
    p.add_argument("--max-features", type=int, default=None)
    p.add_argument("--keep-leaky-wq", action="store_true")
    a = p.parse_args()

    cfg = Config()
    if a.panel_path is not None:
        cfg.panel_path = a.panel_path
    if a.cache_dir is not None:
        cfg.cache_dir = a.cache_dir
        cfg.panel_path = None
    for k in ["macro_path", "out_dir", "horizon", "n_quantiles", "n_folds",
              "embargo_days", "cost_bps", "slippage_bps", "min_adv_inr",
              "min_price", "min_names_per_day", "max_features"]:
        v = getattr(a, k, None)
        if v is not None:
            setattr(cfg, k, v)
    if a.raw_target:
        cfg.use_model_target = False
    if a.keep_leaky_wq:
        cfg.exclude_leaky_wq = False
    cfg.__post_init__()
    return cfg


if __name__ == "__main__":
    run(parse_args())

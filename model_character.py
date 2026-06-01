#!/usr/bin/env python3
"""
model_character.py — is the model momentum or mean-reversion?

Quantifies, at the PREDICTION level (not just features), how contrarian the
model is. It correlates the model's probability with each name's RECENT
trailing strength, cross-sectionally per day, and splits the result by regime.

    corr(prob, recent_return) << 0   ->  CONTRARIAN / short-term mean-reversion
                                         (high prob = recently beaten down)
    corr(prob, recent_return) >> 0   ->  MOMENTUM
                                         (high prob = recent winner)

WHY THIS DESIGN
  The trained model is scored in-process and is not reliably dumped to disk,
  so instead of re-loading it this reads the model's OWN output (the watchlist
  CSV, which already carries prob_5d_mean + stock_regime + every panel feature)
  or any scored file you pass. If you also pass the enriched panel parquet it
  computes TRUE trailing returns (close/close.shift(N)-1) per symbol and merges
  them onto the scored rows for the cleanest read.

INPUTS (all optional; sensible defaults)
  --scored   path to a scored file with a probability column + symbol + date.
             Default: watchlist_5d_signal.csv in --out-dir or CWD.
             A FULL-panel scored file (many dates) gives the most robust read;
             the default watchlist is a single latest cross-section, which is
             exactly the snapshot you were eyeballing.
  --panel    path to panel_cache.parquet (the enriched panel). Used to compute
             true trailing returns ret_{5,20,60}d_past. If omitted, the script
             falls back to the extension features already present in --scored.
  --prob-col / --regime-col / --date-col / --symbol-col : auto-detected if unset.

USAGE
  python model_character.py --scored out\\watchlist_5d_signal.csv \\
                            --panel  C:\\...\\Cache\\panel_cache.parquet
  python model_character.py --self-test      # synthetic proof it works

OUTPUT
  - console table: corr(prob, signal) overall + per regime, for every recent-
    strength signal found, plus a probability-decile profile.
  - model_character_report.csv (the same numbers, machine-readable).

Deps: numpy, pandas (+ pyarrow for parquet). No scipy/sklearn.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── column auto-detection ───────────────────────────────────────────────────
PROB_CANDIDATES = ["prob_5d_mean", "prob_top20_5d", "prob_5d", "prob", "score", "y_prob"]
REGIME_CANDIDATES = ["stock_regime", "regime_used", "regime"]
DATE_CANDIDATES = ["timestamp", "date"]
SYMBOL_CANDIDATES = ["symbol", "ticker", "tradingsymbol"]

# Trailing-strength features already in the panel. Higher value = "more recently
# strong / extended". sign = +1 means a positive raw value means recent strength
# (so corr<0 with prob => contrarian). All of these are oriented that way.
EXTENSION_FEATURES = [
    "W_ret_4w", "W_ret_13w",        # trailing multi-week return
    "D_rsi7", "D_rsi14",            # overbought oscillators
    "D_bb_pctB_20",                 # position in Bollinger band
    "D_donch_pos_20", "D_donch_pos_50",   # position in Donchian channel
    "D_dist_from_20h", "D_dist_from_52wh",  # proximity to recent/annual high
    "D_ema20_angle_deg",            # slope of EMA20
    "D_close_roll_slope_20",        # rolling OLS slope of price
    "D_macd_hist",                  # MACD histogram
]

# Forward-looking / label columns that must NEVER be used as a "recent" signal.
FORBIDDEN = {
    "ret_1d_close_pct", "ret_3d_close_pct", "ret_5d_close_pct",
    "ret_1d_oc_pct", "ret_3d_oc_pct", "ret_5d_oc_pct",
    "ret_5d_adj", "ret_3d_adj", "rank_5d_pct",
    "top20_vs_bot20_5d", "top20_strict_5d",
    "expected_ret_5d", "expected_ret_3d", "expected_ret_5d_adj", "expected_sharpe_5d",
}


def _pick(cols: List[str], candidates: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    return None


def _norm_date(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, errors="coerce")
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_localize(None)
    return out.dt.normalize()


def _load_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


# ── core stat: per-day cross-sectional Spearman corr(prob, signal) ───────────
def xs_spearman(df: pd.DataFrame, date_col: str, a: str, b: str,
                min_names: int = 10) -> Tuple[float, int, int]:
    """Mean over days of the within-day rank correlation between a and b.

    Returns (mean_corr, n_days_used, n_obs). Spearman == Pearson of per-day ranks.
    """
    d = df[[date_col, a, b]].replace([np.inf, -np.inf], np.nan).dropna()
    if d.empty:
        return (np.nan, 0, 0)
    cnt = d.groupby(date_col)[a].transform("size")
    d = d[cnt >= min_names]
    if d.empty:
        return (np.nan, 0, 0)
    by = d[date_col]
    ra = d.groupby(by)[a].rank()
    rb = d.groupby(by)[b].rank()
    am = ra - ra.groupby(by).transform("mean")
    bm = rb - rb.groupby(by).transform("mean")
    num = (am * bm).groupby(by).sum()
    den = np.sqrt((am ** 2).groupby(by).sum() * (bm ** 2).groupby(by).sum())
    ic = (num / den).replace([np.inf, -np.inf], np.nan).dropna()
    return (float(ic.mean()) if ic.size else np.nan, int(ic.size), int(len(d)))


def add_trailing_returns(panel: pd.DataFrame, symbol_col: str, date_col: str,
                         horizons=(5, 20, 60)) -> pd.DataFrame:
    """Causal trailing returns ret_{H}d_past = close/close.shift(H) - 1, per symbol."""
    if "close" not in panel.columns:
        return panel
    p = panel.sort_values([symbol_col, date_col]).copy()
    g = p.groupby(symbol_col, observed=True)["close"]
    for h in horizons:
        p[f"ret_{h}d_past"] = g.transform(lambda s, h=h: s / s.shift(h) - 1.0)
    return p


def analyse(df: pd.DataFrame, prob_col: str, date_col: str, regime_col: Optional[str],
            signals: List[str], min_names: int) -> pd.DataFrame:
    rows = []
    groups: List[Tuple[str, pd.DataFrame]] = [("ALL", df)]
    if regime_col and regime_col in df.columns:
        for rv, sub in df.groupby(regime_col):
            groups.append((str(rv), sub))
    for sig in signals:
        if sig not in df.columns:
            continue
        for gname, gdf in groups:
            corr, ndays, nobs = xs_spearman(gdf, date_col, prob_col, sig, min_names)
            rows.append({"signal": sig, "group": gname,
                         "corr_prob_vs_signal": corr, "n_days": ndays, "n_obs": nobs})
    return pd.DataFrame(rows)


def prob_decile_profile(df: pd.DataFrame, prob_col: str, date_col: str,
                        signals: List[str], n_bins: int = 10) -> pd.DataFrame:
    """Mean of each recent-strength signal by within-day probability decile."""
    present = [s for s in signals if s in df.columns]
    if not present:
        return pd.DataFrame()
    d = df[[date_col, prob_col] + present].copy()
    def _bin(s: pd.Series) -> pd.Series:
        try:
            return pd.qcut(s.rank(method="first"), n_bins, labels=False, duplicates="drop")
        except ValueError:
            return pd.Series(np.nan, index=s.index)
    d["pdec"] = d.groupby(date_col)[prob_col].transform(_bin)
    prof = d.dropna(subset=["pdec"]).groupby("pdec")[present].mean()
    prof.index = [f"D{int(i)+1}" for i in prof.index]
    return prof


# ── self-test ─────────────────────────────────────────────────────────────--
def _build_synthetic(seed=0, contrarian=True, n_days=40, n_names=120) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    parts = []
    for d in dates:
        ret_20 = rng.normal(0, 0.08, n_names)          # recent trailing return
        rsi = 50 + 200 * ret_20 + rng.normal(0, 5, n_names)
        # prob is (anti-)correlated with recent return + noise
        sign = -1.0 if contrarian else 1.0
        latent = sign * ret_20 + rng.normal(0, 0.03, n_names)
        prob = 1.0 / (1.0 + np.exp(-8 * latent))       # squash to (0,1)
        regime = rng.choice(["bull_trend", "bull_range", "bear_trend", "bear_range"], n_names)
        parts.append(pd.DataFrame({
            "timestamp": d, "symbol": [f"S{i:03d}" for i in range(n_names)],
            "prob_5d_mean": prob, "stock_regime": regime,
            "ret_20d_past": ret_20, "D_rsi14": rsi,
        }))
    return pd.concat(parts, ignore_index=True)


def _run_self_test() -> int:
    print("=" * 64)
    print("MODEL_CHARACTER SELF-TEST")
    print("=" * 64)
    df = _build_synthetic(contrarian=True)
    sigs = ["ret_20d_past", "D_rsi14"]
    res = analyse(df, "prob_5d_mean", "timestamp", "stock_regime", sigs, min_names=10)
    allc = res[res["group"] == "ALL"].set_index("signal")["corr_prob_vs_signal"]
    print(res.to_string(index=False))
    assert allc["ret_20d_past"] < -0.3, allc["ret_20d_past"]
    assert allc["D_rsi14"] < -0.3, allc["D_rsi14"]
    verdict = classify(allc.mean())
    print(f"\naggregate corr={allc.mean():.3f} -> {verdict}")
    assert "MEAN-REVERSION" in verdict or "CONTRARIAN" in verdict
    # momentum control
    dfm = _build_synthetic(contrarian=False, seed=1)
    resm = analyse(dfm, "prob_5d_mean", "timestamp", None, sigs, min_names=10)
    allm = resm[resm["group"] == "ALL"].set_index("signal")["corr_prob_vs_signal"]
    assert allm["ret_20d_past"] > 0.3, allm["ret_20d_past"]
    print(f"momentum control corr={allm.mean():.3f} -> {classify(allm.mean())}")
    print("\nALL SELF-TESTS PASSED")
    return 0


def classify(mean_corr: float) -> str:
    if not np.isfinite(mean_corr):
        return "UNDETERMINED (no data)"
    if mean_corr <= -0.30:
        return "STRONGLY CONTRARIAN  (short-term MEAN-REVERSION)"
    if mean_corr <= -0.10:
        return "MILDLY CONTRARIAN  (lean MEAN-REVERSION)"
    if mean_corr < 0.10:
        return "NEUTRAL  (neither clearly momentum nor reversion)"
    if mean_corr < 0.30:
        return "MILDLY MOMENTUM"
    return "STRONGLY MOMENTUM"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Is the model momentum or mean-reversion?")
    ap.add_argument("--scored", type=Path, default=None,
                    help="scored file (watchlist or full-panel) with a prob column")
    ap.add_argument("--panel", type=Path, default=None,
                    help="panel_cache.parquet to compute true trailing returns")
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    ap.add_argument("--prob-col", default=None)
    ap.add_argument("--regime-col", default=None)
    ap.add_argument("--date-col", default=None)
    ap.add_argument("--symbol-col", default=None)
    ap.add_argument("--min-names", type=int, default=10,
                    help="min names per day to compute a cross-sectional corr")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)

    if a.self_test:
        return _run_self_test()

    scored_path = a.scored
    if scored_path is None:
        for cand in (a.out_dir / "watchlist_5d_signal.csv", Path("watchlist_5d_signal.csv")):
            if cand.exists():
                scored_path = cand
                break
    if scored_path is None or not Path(scored_path).exists():
        print("ERROR: no scored file. Pass --scored <watchlist_5d_signal.csv or a "
              "full-panel scored file>.", file=sys.stderr)
        return 2

    df = _load_any(Path(scored_path))
    prob_col = a.prob_col or _pick(list(df.columns), PROB_CANDIDATES)
    date_col = a.date_col or _pick(list(df.columns), DATE_CANDIDATES)
    sym_col = a.symbol_col or _pick(list(df.columns), SYMBOL_CANDIDATES)
    regime_col = a.regime_col or _pick(list(df.columns), REGIME_CANDIDATES)
    if not prob_col or not date_col or not sym_col:
        print(f"ERROR: could not locate prob/date/symbol columns. "
              f"Found prob={prob_col} date={date_col} symbol={sym_col}. "
              f"Pass them explicitly.", file=sys.stderr)
        return 3
    df[date_col] = _norm_date(df[date_col])
    df[prob_col] = pd.to_numeric(df[prob_col], errors="coerce")

    n_dates = df[date_col].nunique()
    print(f"Scored file: {scored_path}")
    print(f"  prob='{prob_col}'  regime='{regime_col}'  date='{date_col}'  symbol='{sym_col}'")
    print(f"  rows={len(df):,}  symbols={df[sym_col].nunique()}  dates={n_dates}"
          + ("  (single cross-section — a snapshot)" if n_dates == 1 else ""))

    # Trailing returns from the panel (cleanest signal), merged onto scored rows.
    trailing_cols: List[str] = []
    if a.panel and Path(a.panel).exists():
        panel = _load_any(Path(a.panel))
        pdate = _pick(list(panel.columns), DATE_CANDIDATES)
        psym = _pick(list(panel.columns), SYMBOL_CANDIDATES)
        if pdate and psym and "close" in panel.columns:
            panel[pdate] = _norm_date(panel[pdate])
            panel = add_trailing_returns(panel, psym, pdate)
            trailing_cols = [c for c in panel.columns if c.endswith("d_past")]
            merge = panel[[psym, pdate] + trailing_cols].rename(
                columns={psym: sym_col, pdate: date_col})
            df = df.merge(merge, on=[sym_col, date_col], how="left")
            print(f"  merged trailing returns from panel: {trailing_cols}")
        else:
            print("  WARN: panel missing symbol/date/close; skipping trailing returns")
    else:
        print("  (no --panel given; using extension features already in the scored file)")

    # Build the recent-strength signal list (trailing returns first, then
    # whatever extension features are present), excluding forbidden/forward cols.
    signals = [c for c in (trailing_cols + EXTENSION_FEATURES)
               if c in df.columns and c not in FORBIDDEN]
    signals = list(dict.fromkeys(signals))  # de-dupe, keep order
    if not signals:
        print("ERROR: no recent-strength signals available (need --panel or "
              "extension features like D_rsi14/W_ret_4w in the scored file).",
              file=sys.stderr)
        return 4

    res = analyse(df, prob_col, date_col, regime_col, signals, a.min_names)

    # Pretty print: pivot signal x group
    piv = res.pivot(index="signal", columns="group", values="corr_prob_vs_signal")
    cols = ["ALL"] + [c for c in piv.columns if c != "ALL"]
    piv = piv.reindex(columns=[c for c in cols if c in piv.columns])
    print("\n================ corr(prob, recent-strength)  [<0 = contrarian] ================")
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", lambda x: f"{x:+.3f}"):
        print(piv.to_string())

    overall = piv["ALL"].dropna()
    agg = float(overall.mean()) if len(overall) else float("nan")
    print("\n================ PROBABILITY-DECILE PROFILE (within-day) ================")
    prof = prob_decile_profile(df, prob_col, date_col, signals)
    if not prof.empty:
        with pd.option_context("display.width", 200, "display.max_columns", 30,
                               "display.float_format", lambda x: f"{x:+.4f}"):
            print(prof.to_string())
        print("  (D1 = lowest prob, D10 = highest prob. If recent-strength signals "
              "DECREASE D1->D10, the model favours recently-weak names = contrarian.)")

    print("\n================ VERDICT ================")
    print(f"  mean corr(prob, recent-strength) over {len(overall)} signals = {agg:+.3f}")
    print(f"  -> {classify(agg)}")
    if regime_col and regime_col in df.columns:
        per_reg = piv.drop(columns=["ALL"], errors="ignore").mean(axis=0)
        if len(per_reg):
            strongest = per_reg.idxmin()
            print(f"  most contrarian regime: {strongest} (mean corr {per_reg.min():+.3f}); "
                  f"least: {per_reg.idxmax()} ({per_reg.max():+.3f})")
    if n_dates == 1:
        print("  NOTE: single cross-section (the watchlist snapshot). For a robust, "
              "multi-day read, score the full panel and pass it via --scored.")

    out_csv = Path(a.out_dir) / "model_character_report.csv"
    res.to_csv(out_csv, index=False)
    print(f"\n  written: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

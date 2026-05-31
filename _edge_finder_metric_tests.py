"""Focused correctness tests for edge_finder's metric math.

Run: python _edge_finder_metric_tests.py
These validate the numbers the screener reports, on controlled inputs where
the right answer is known analytically.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd

import edge_finder as ef


def _panel(n_days=200, n_sym=20, seed=0, signal_strength=1.0):
    """Panel where `sig` predicts `ret_h1` with a controllable strength."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=n_days, freq="B")
    rows = []
    for s in range(n_sym):
        sig = rng.normal(0, 1, n_days)
        noise = rng.normal(0, 1, n_days)
        ret = signal_strength * sig + noise  # ret correlated with sig
        rows.append(pd.DataFrame({
            "timestamp": dates, "symbol": f"S{s:02d}",
            "sig": sig, "ret_h1": ret,
        }))
    return pd.concat(rows, ignore_index=True).sort_values(["timestamp", "symbol"])


def test_daily_ic_perfect_positive():
    # ret == sig exactly -> per-day rank IC == 1.0
    df = _panel(signal_strength=1.0, seed=1)
    df = df.copy()
    df["ret_h1"] = df["sig"]
    ic = ef.daily_ic(df, "sig", "ret_h1", method="spearman")
    assert abs(ic.mean() - 1.0) < 1e-6, f"expected IC≈1, got {ic.mean()}"
    print(f"  daily_ic perfect positive: IC mean = {ic.mean():.6f}  OK")


def test_daily_ic_perfect_negative():
    df = _panel(seed=2).copy()
    df["ret_h1"] = -df["sig"]
    ic = ef.daily_ic(df, "sig", "ret_h1", method="spearman")
    assert abs(ic.mean() + 1.0) < 1e-6, f"expected IC≈-1, got {ic.mean()}"
    print(f"  daily_ic perfect negative: IC mean = {ic.mean():.6f}  OK")


def test_daily_ic_matches_scipy_spearman():
    df = _panel(signal_strength=0.5, seed=3)
    ic_fast = ef.daily_ic(df, "sig", "ret_h1", method="spearman")
    # Brute force with scipy per day
    from scipy.stats import spearmanr
    by_day = []
    for ts, g in df.groupby("timestamp"):
        if len(g) >= 3:
            r, _ = spearmanr(g["sig"], g["ret_h1"])
            by_day.append((ts, r))
    ic_ref = pd.Series({ts: r for ts, r in by_day}).dropna()
    aligned = ic_fast.reindex(ic_ref.index)
    max_diff = float((aligned - ic_ref).abs().max())
    assert max_diff < 1e-6, f"fast IC differs from scipy by {max_diff}"
    print(f"  daily_ic vs scipy spearman: max abs diff = {max_diff:.2e}  OK")


def test_quintile_ls_monotonic_positive():
    # Strong positive signal -> long-short PnL should be clearly positive
    df = _panel(signal_strength=2.0, seed=4)
    ls = ef.quintile_long_short_pnl(df, "sig", "ret_h1")
    assert ls.mean() > 0, f"expected positive LS PnL, got {ls.mean()}"
    sharpe = ef.annualized_sharpe(ls)
    assert sharpe > 0
    print(f"  quintile LS positive signal: mean={ls.mean():.4f} sharpe={sharpe:.2f}  OK")


def test_sharpe_zero_for_noise():
    df = _panel(signal_strength=0.0, seed=5)
    ls = ef.quintile_long_short_pnl(df, "sig", "ret_h1")
    sharpe = ef.annualized_sharpe(ls)
    assert abs(sharpe) < 3.0, f"noise should have ~0 Sharpe, got {sharpe}"
    print(f"  noise LS Sharpe near zero: {sharpe:.2f}  OK")


def test_max_drawdown_sign():
    ret = pd.Series([0.1, -0.5, 0.1, 0.1])
    mdd = ef.max_drawdown(ret)
    assert mdd < 0, f"max_drawdown should be negative, got {mdd}"
    print(f"  max_drawdown sign: {mdd:.4f}  OK")


def test_expected_max_sharpe_grows_with_trials():
    # With fixed cross-trial variance, the expected max Sharpe under H0 must
    # increase as we test more strategies.
    e10 = ef.expected_max_sharpe(10, var_sr_trials=0.04)
    e1000 = ef.expected_max_sharpe(1000, var_sr_trials=0.04)
    assert e1000 > e10 > 0, f"E[max] should grow with trials: {e10} vs {e1000}"
    print(f"  E[max SR] grows with trials: N=10 -> {e10:.4f}  N=1000 -> {e1000:.4f}  OK")


def test_deflated_sharpe_monotonic_in_benchmark():
    # Higher multiple-testing benchmark sr_star -> lower DSR for the same SR.
    dsr_low_bar = ef.deflated_sharpe_ratio(0.20, n_obs=500, sr_star=0.02)
    dsr_high_bar = ef.deflated_sharpe_ratio(0.20, n_obs=500, sr_star=0.15)
    assert dsr_low_bar > dsr_high_bar, f"DSR should drop as sr_star rises: {dsr_low_bar} vs {dsr_high_bar}"
    assert 0.0 <= dsr_high_bar <= 1.0 and 0.0 <= dsr_low_bar <= 1.0
    print(f"  DSR monotonic in benchmark: sr*=0.02 -> {dsr_low_bar:.4f}   sr*=0.15 -> {dsr_high_bar:.4f}  OK")


def test_deflated_sharpe_higher_sr_higher_conf():
    # For a fixed benchmark, a higher observed per-period SR -> higher DSR.
    dsr_lo = ef.deflated_sharpe_ratio(0.05, n_obs=500, sr_star=0.08)
    dsr_hi = ef.deflated_sharpe_ratio(0.25, n_obs=500, sr_star=0.08)
    assert dsr_hi > dsr_lo
    print(f"  DSR rises with SR: SR=0.05 -> {dsr_lo:.4f}   SR=0.25 -> {dsr_hi:.4f}  OK")


def test_bootstrap_ci_brackets_point_estimate():
    df = _panel(signal_strength=1.5, seed=6)
    ls = ef.quintile_long_short_pnl(df, "sig", "ret_h1")
    point = ef.annualized_sharpe(ls)
    lo, hi = ef.bootstrap_sharpe_ci(ls, n_boot=400, seed=0)
    assert lo <= point <= hi, f"point {point} not in CI [{lo}, {hi}]"
    print(f"  bootstrap CI brackets point: [{lo:.2f}, {hi:.2f}] contains {point:.2f}  OK")


def test_no_lookahead_in_cs_rank():
    """cs_rank computed per-timestamp must not change a past row when a future
    row is mutated (cross-sectional rank only mixes same-day names)."""
    rng = np.random.default_rng(7)
    dates = pd.date_range("2021-01-01", periods=50, freq="B")
    rows = []
    for s in range(10):
        rows.append(pd.DataFrame({
            "timestamp": dates, "symbol": f"S{s}",
            "open": rng.normal(100, 5, 50), "high": rng.normal(101, 5, 50),
            "low": rng.normal(99, 5, 50), "close": rng.normal(100, 5, 50),
            "volume": rng.integers(1e5, 1e6, 50).astype(float),
        }))
    panel = pd.concat(rows, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    a = ef.add_cross_sectional_ranks(panel.copy(), ["close"])
    tamp = panel.copy()
    tamp.loc[tamp["timestamp"] >= dates[40], "close"] *= 3.0  # mutate future
    b = ef.add_cross_sectional_ranks(tamp, ["close"])
    past = panel["timestamp"] < dates[40]
    diff = float((a.loc[past, "cs_rank_close"].to_numpy()
                  - b.loc[past, "cs_rank_close"].to_numpy()).__abs__().max())
    assert diff < 1e-9, f"cs_rank leaked the future: max past diff = {diff}"
    print(f"  cs_rank no-lookahead: max past diff = {diff:.2e}  OK")


def _synth_ohlcv_panel(n_sym=12, n_days=120, seed=11):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=n_days, freq="B")
    rows = []
    for s in range(n_sym):
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n_days)))
        op = close * (1 + rng.normal(0, 0.003, n_days))
        hi = np.maximum(close, op) * (1 + np.abs(rng.normal(0, 0.005, n_days)))
        lo = np.minimum(close, op) * (1 - np.abs(rng.normal(0, 0.005, n_days)))
        vol = rng.integers(1e5, 1e6, n_days).astype(float)
        rows.append(pd.DataFrame({"timestamp": dates, "symbol": f"S{s:02d}",
                                  "open": op, "high": hi, "low": lo,
                                  "close": close, "volume": vol}))
    return pd.concat(rows, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def test_panel_wq_alphas_count_and_finite():
    # 300+ days so the 250-window alpha (WQ_19) clears its min_periods warmup.
    panel = _synth_ohlcv_panel(n_sym=12, n_days=320)
    panel = ef.add_panel_wq_alphas(panel)
    wq = [c for c in panel.columns if c.startswith("WQ_") and c.endswith("_cs")]
    assert len(wq) == 16, f"expected 16 WQ alphas, got {len(wq)}: {sorted(wq)}"
    # each alpha should have a meaningful number of finite values (post warmup)
    for c in wq:
        finite = int(np.isfinite(pd.to_numeric(panel[c], errors="coerce")).sum())
        assert finite > 100, f"{c} has too few finite values ({finite})"
    print(f"  panel WQ alphas: built {len(wq)}, all populated  OK")


def test_wq_cs_no_lookahead():
    """A WQ_*_cs value at (symbol, day t) must not change when only FUTURE
    rows (day > t) are mutated. Validates both the per-symbol ts ops and the
    per-day cross-sectional ranks are causal."""
    panel = _synth_ohlcv_panel(n_sym=12, n_days=420, seed=3)
    dates = np.sort(panel["timestamp"].unique())
    cut = dates[360]
    a = ef.add_panel_wq_alphas(panel.copy())
    tamp = panel.copy()
    fut = tamp["timestamp"] >= cut
    rng = np.random.default_rng(0)
    for col in ["open", "high", "low", "close", "volume"]:
        tamp.loc[fut, col] = tamp.loc[fut, col].to_numpy() * rng.uniform(0.3, 3.0, int(fut.sum()))
    b = ef.add_panel_wq_alphas(tamp)
    past = panel["timestamp"] < cut
    wq = [c for c in a.columns if c.startswith("WQ_") and c.endswith("_cs")]
    worst = 0.0
    worst_col = None
    for c in wq:
        av = pd.to_numeric(a.loc[past, c], errors="coerce").to_numpy()
        bv = pd.to_numeric(b.loc[past, c], errors="coerce").to_numpy()
        an, bn = np.isnan(av), np.isnan(bv)
        assert np.array_equal(an, bn), f"{c}: NaN pattern changed in the past"
        d = float(np.nanmax(np.where(an, 0.0, np.abs(av - bv))) if (~an).any() else 0.0)
        if d > worst:
            worst, worst_col = d, c
    assert worst < 1e-6, f"WQ cs alpha leaked future: {worst_col} max past diff = {worst}"
    print(f"  WQ_*_cs no-lookahead: worst past diff = {worst:.2e} ({worst_col})  OK")


def test_panel_cache_roundtrip(tmp_dir="_pc_test"):
    """get_panel must MISS then HIT, and return identical data from cache."""
    import os, shutil
    base = tmp_dir
    cache_dir = os.path.join(base, "cache_daily")
    os.makedirs(cache_dir, exist_ok=True)
    try:
        # Write per-symbol parquets (need >=250 rows to survive load filter)
        p = _synth_ohlcv_panel(n_sym=8, n_days=300, seed=5)
        for sym, g in p.groupby("symbol"):
            g.drop(columns=["symbol"]).to_parquet(os.path.join(cache_dir, f"{sym}_daily.parquet"), index=False)
        pc = os.path.join(base, "panel.parquet")
        # First call: MISS -> builds + saves
        panel1 = ef.get_panel(cache_dir, horizons=(1, 5), panel_cache=pc)
        assert os.path.exists(pc) and os.path.exists(pc + ".meta.json")
        # Second call: HIT -> loads cached parquet
        panel2 = ef.get_panel(cache_dir, horizons=(1, 5), panel_cache=pc)
        assert panel1.shape == panel2.shape, f"shape mismatch {panel1.shape} vs {panel2.shape}"
        # Values identical for a WQ alpha column
        wqcol = [c for c in panel1.columns if c.startswith("WQ_") and c.endswith("_cs")][0]
        a = pd.to_numeric(panel1[wqcol], errors="coerce").to_numpy()
        b = pd.to_numeric(panel2[wqcol], errors="coerce").to_numpy()
        an, bn = np.isnan(a), np.isnan(b)
        assert np.array_equal(an, bn)
        assert float(np.nanmax(np.where(an, 0.0, np.abs(a - b)))) < 1e-6
        # Stale detection: different horizons must rebuild (params changed)
        panel3 = ef.get_panel(cache_dir, horizons=(1, 5, 10), panel_cache=pc)
        assert "ret_h10" in panel3.columns
        print(f"  panel cache roundtrip: MISS->HIT identical, stale-detect OK  ({panel1.shape[0]} rows)")
    finally:
        shutil.rmtree(base, ignore_errors=True)


def main():
    print("EDGE FINDER METRIC TESTS")
    print("-" * 50)
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print("-" * 50)
    print(f"ALL {len(tests)} METRIC TESTS PASSED")


if __name__ == "__main__":
    main()

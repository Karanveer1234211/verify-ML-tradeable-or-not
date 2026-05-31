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

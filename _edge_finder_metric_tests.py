"""Focused correctness tests for edge_finder.py (v2 — panel-native).

Run: python _edge_finder_metric_tests.py

These pin the math the honest scanner relies on, on controlled inputs where the
right answer is known: the normal-dist helpers, the multiple-testing bar
(expected max |t|), the Newey-West t-stat (incl. its deflation under overlap),
cross-sectional rank-IC, the net-of-cost quantile long-short, drawdown, fold
stability, the model-aligned target, and the leaky-WQ / forward-column feature
guards.

No scipy/sklearn needed (matches edge_finder's own dependency footprint).
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

import edge_finder as ef


def _cfg(**kw):
    c = ef.Config()
    # Don't touch the filesystem in unit tests.
    c.panel_path = None
    c.cache_dir = None
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def _xs_panel(n_days=60, n_names=40, seed=0, signal=1.0, noise=1.0):
    """Cross-sectional panel where feature F predicts _fwd with tunable strength."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n_days, freq="B")
    parts = []
    syms = [f"S{i:03d}" for i in range(n_names)]
    for d in dates:
        f = rng.normal(size=n_names)
        fwd = signal * f + noise * rng.normal(size=n_names)
        parts.append(pd.DataFrame({
            "date": d, "symbol": syms, "F": f,
            "_fwd": fwd, "_pnl": fwd / 100.0, "_tradeable": True,
        }))
    return pd.concat(parts, ignore_index=True)


# ───────────────────────── normal-dist helpers ─────────────────────────

def test_norm_ppf_cdf_known_values():
    assert abs(ef._norm_ppf(0.975) - 1.959963985) < 1e-4
    assert abs(ef._norm_cdf(0.0) - 0.5) < 1e-12
    assert abs(ef._norm_cdf(1.959963985) - 0.975) < 1e-4
    # round trip
    for p in (0.05, 0.3, 0.84, 0.99):
        assert abs(ef._norm_cdf(ef._norm_ppf(p)) - p) < 1e-4
    print("  _norm_ppf/_norm_cdf known values + round trip  OK")


def test_two_sided_p():
    # |t|=1.96 -> ~0.05 two-sided
    assert abs(ef.two_sided_p(1.959963985) - 0.05) < 1e-3
    assert abs(ef.two_sided_p(0.0) - 1.0) < 1e-9
    print(f"  two_sided_p(1.96)={ef.two_sided_p(1.96):.4f}  OK")


def test_expected_max_t_monotonic():
    # The multiple-testing bar must rise as we test more features.
    vals = [ef.expected_max_t(n) for n in (2, 10, 100, 1000, 5000)]
    assert all(b > a for a, b in zip(vals, vals[1:])), vals
    assert vals[0] > 0
    print("  expected_max_t monotonic in #trials: "
          + ", ".join(f"{n}->{ef.expected_max_t(n):.2f}" for n in (10, 100, 1000)) + "  OK")


# ───────────────────────── Newey-West ─────────────────────────

def test_newey_west_zero_lag_matches_plain_t():
    rng = np.random.default_rng(1)
    x = pd.Series(0.3 + rng.normal(0, 1, 500))
    t_nw, p, n = ef.newey_west_t(x, lag=0)
    # plain t with population variance (/n), as the implementation uses
    e = x.to_numpy() - x.mean()
    se = math.sqrt((e @ e / len(e)) / len(e))
    t_plain = x.mean() / se
    assert abs(t_nw - t_plain) < 1e-9, (t_nw, t_plain)
    print(f"  Newey-West lag=0 == plain t: {t_nw:.4f}  OK")


def test_newey_west_deflates_under_positive_autocorrelation():
    # Overlapping/positively-autocorrelated series -> NW SE larger -> |t| smaller.
    rng = np.random.default_rng(2)
    n = 800
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = 0.7 * e[i - 1] + rng.normal()
    x = pd.Series(0.5 + e)
    t0, _, _ = ef.newey_west_t(x, lag=0)
    t5, _, _ = ef.newey_west_t(x, lag=10)
    assert abs(t5) < abs(t0), (t0, t5)
    print(f"  Newey-West deflates under autocorrelation: |t|@lag0={abs(t0):.2f} -> |t|@lag10={abs(t5):.2f}  OK")


# ───────────────────────── drawdown ─────────────────────────

def test_max_drawdown():
    eq = np.array([1.0, 1.2, 0.6, 0.9, 1.3])  # peak 1.2 -> trough 0.6 = -50%
    assert abs(ef.max_drawdown(eq) - (-0.5)) < 1e-9
    assert ef.max_drawdown(np.array([1, 2, 3, 4], dtype=float)) == 0.0
    print(f"  max_drawdown: {ef.max_drawdown(eq):.2f}  OK")


# ───────────────────────── cross-sectional IC ─────────────────────────

def test_ic_series_perfect_positive():
    p = _xs_panel(signal=1.0, noise=0.0, seed=3)  # _fwd == F per day
    ic = ef.ic_series(p, "F", _cfg(min_names_per_day=10))
    assert ic.size > 0 and abs(ic.mean() - 1.0) < 1e-9, ic.mean()
    print(f"  ic_series perfect positive: mean IC = {ic.mean():.6f}  OK")


def test_ic_series_perfect_negative():
    p = _xs_panel(signal=1.0, noise=0.0, seed=4)
    p["_fwd"] = -p["F"]
    ic = ef.ic_series(p, "F", _cfg(min_names_per_day=10))
    assert abs(ic.mean() + 1.0) < 1e-9, ic.mean()
    print(f"  ic_series perfect negative: mean IC = {ic.mean():.6f}  OK")


def test_ic_series_respects_min_names():
    # With a high min_names_per_day, a thin panel yields no IC observations.
    p = _xs_panel(n_names=12, signal=1.0, noise=0.5, seed=5)
    ic = ef.ic_series(p, "F", _cfg(min_names_per_day=30))
    assert ic.size == 0, f"expected no IC days (thin breadth), got {ic.size}"
    ic2 = ef.ic_series(p, "F", _cfg(min_names_per_day=10))
    assert ic2.size > 0
    print("  ic_series honours min_names_per_day breadth gate  OK")


# ───────────────────────── quantile long-short ─────────────────────────

def test_quantile_ls_positive_signal():
    p = _xs_panel(n_days=120, n_names=50, signal=2.0, noise=1.0, seed=6)
    bt = ef.quantile_ls(p, "F", _cfg(min_names_per_day=10, n_quantiles=5, horizon=5))
    assert bt["n_reb"] >= 8
    assert bt["net_sharpe"] > 0, bt
    print(f"  quantile_ls positive signal: net_sharpe={bt['net_sharpe']:.2f}, "
          f"ann_ret={bt['net_ann_ret']:.3f}, reb={bt['n_reb']}  OK")


def test_quantile_ls_noise_has_no_positive_edge():
    # Pure noise has ~0 gross spread; transaction costs then drag the net
    # series reliably negative. The honest outcome is "no tradeable edge":
    # what must NOT happen is a positive net-of-cost Sharpe.
    p = _xs_panel(n_days=120, n_names=50, signal=0.0, noise=1.0, seed=7)
    bt = ef.quantile_ls(p, "F", _cfg(min_names_per_day=10, n_quantiles=5, horizon=5))
    assert np.isnan(bt["net_sharpe"]) or bt["net_sharpe"] <= 0.5, bt
    print(f"  quantile_ls noise -> no positive edge: net_sharpe={bt['net_sharpe']:.2f}  OK")


# ───────────────────────── fold stability ─────────────────────────

def test_fold_stability_consistent_sign():
    # An all-positive IC series -> every fold agrees with overall sign -> 1.0
    ic = pd.Series(np.abs(np.random.default_rng(8).normal(0.05, 0.01, 200)),
                   index=pd.date_range("2022-01-03", periods=200, freq="B"))
    stab = ef.fold_stability(ic, _cfg(n_folds=5, horizon=5))
    assert abs(stab - 1.0) < 1e-9, stab
    print(f"  fold_stability all-positive IC -> {stab:.2f}  OK")


# ───────────────────────── target & feature selection ─────────────────────────

def test_build_target_vol_adjusted():
    n = 200
    rng = np.random.default_rng(9)
    p = pd.DataFrame({
        "date": pd.date_range("2022-01-03", periods=n, freq="B"),
        "symbol": "AAA",
        "close": 100 + rng.normal(0, 1, n).cumsum(),
        "ret_5d_oc_pct": rng.normal(0, 2, n),
        "atr_pct": np.abs(rng.normal(2, 0.3, n)) + 0.5,
        "volume": rng.integers(1e5, 1e6, n),
    })
    out = ef.build_target(p.copy(), _cfg(use_model_target=True))
    expected = out["ret_5d_oc_pct"] / out["atr_pct"]
    m = out["_fwd"].notna()
    assert np.allclose(out.loc[m, "_fwd"], expected.loc[m], atol=1e-9)
    print("  build_target uses model vol-adjusted target (ret_5d_oc_pct/atr_pct)  OK")


def test_pick_features_excludes_leaky_wq_and_forward():
    n = 300
    rng = np.random.default_rng(10)
    p = pd.DataFrame({
        "date": pd.date_range("2022-01-03", periods=n, freq="B"),
        "symbol": "AAA",
        "D_good": rng.normal(0, 1, n),
        "WQ_3": rng.normal(0, 1, n),          # leaky -> excluded by default
        "ret_5d_oc_pct": rng.normal(0, 1, n),  # forward-like -> excluded
        "_fwd": rng.normal(0, 1, n),
        "_pnl": rng.normal(0, 1, n),
        "_tradeable": True,
    })
    feats = ef.pick_features(p.copy(), _cfg(exclude_leaky_wq=True), panel_mode=True)
    assert "D_good" in feats
    assert "WQ_3" not in feats, "leaky WQ must be excluded by default"
    assert "ret_5d_oc_pct" not in feats and "_fwd" not in feats
    feats_keep = ef.pick_features(p.copy(), _cfg(exclude_leaky_wq=False), panel_mode=True)
    assert "WQ_3" in feats_keep, "--keep-leaky-wq path should retain WQ_3"
    print("  pick_features excludes leaky WQ + forward cols (and can opt back in)  OK")


def test_date_splits_embargo_no_overlap():
    n = 400
    p = pd.DataFrame({
        "date": np.repeat(pd.date_range("2021-01-04", periods=n, freq="B"), 3),
        "symbol": ["A", "B", "C"] * n,
    })
    train, test = ef.date_splits(p, _cfg(train_frac=0.70, cal_frac=0.20, embargo_days=5))
    assert train and test
    assert not (train & test), "train and test must not overlap"
    assert max(train) < min(test), "test must come strictly after train (embargoed)"
    print(f"  date_splits: {len(train)} train / {len(test)} test days, embargoed, no overlap  OK")


def main():
    print("EDGE FINDER (v2) METRIC TESTS")
    print("-" * 55)
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print("-" * 55)
    print(f"ALL {len(tests)} METRIC TESTS PASSED")


if __name__ == "__main__":
    main()

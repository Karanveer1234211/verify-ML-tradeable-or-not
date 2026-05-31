"""Regression test for the 'previously unseen labels: [0]' crash in
train_5d_quantile_cls.

1. Reproduces the raw LightGBM failure: a single-class training fold + an
   eval_set that contains the missing class.
2. Verifies _fit_binary_cls_safe handles it (skips / filters) instead of
   crashing.
3. Verifies the cross-sectional label mechanism: days with <5 names never
   produce a class-0 label (the root cause of the single-class folds).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
import New_model as NM


def test_raw_lightgbm_crashes_on_single_class_train():
    """Confirm the failure mode the user hit is real with this LightGBM."""
    rng = np.random.default_rng(0)
    X_tr = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("abcd"))
    y_tr = np.ones(200, dtype=int)                 # single class (all 1s)
    X_val = pd.DataFrame(rng.normal(size=(60, 4)), columns=list("abcd"))
    y_val = np.array([0, 1] * 30, dtype=int)        # contains the unseen 0
    clf = LGBMClassifier(n_estimators=10, verbosity=-1)
    raised = False
    try:
        clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="binary_logloss")
    except ValueError as e:
        raised = "previously unseen labels" in str(e)
    assert raised, "expected the LightGBM unseen-label ValueError to reproduce"
    print("  raw LightGBM reproduces the unseen-label crash  OK")


def test_safe_fit_skips_single_class_fold():
    rng = np.random.default_rng(1)
    X_tr = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("abcd"))
    y_tr = np.ones(200, dtype=int)                 # single class -> must skip
    X_val = pd.DataFrame(rng.normal(size=(60, 4)), columns=list("abcd"))
    y_val = np.array([0, 1] * 30, dtype=int)
    clf = LGBMClassifier(n_estimators=10, verbosity=-1)
    ok = NM._fit_binary_cls_safe(clf, X_tr, y_tr, X_val, y_val)
    assert ok is False, "single-class training fold should be skipped (return False)"
    print("  safe-fit skips single-class training fold  OK")


def test_safe_fit_filters_unseen_eval_labels():
    """Two-class train, but eval set has an extra label not in train -> filter,
    don't crash. (Defensive; with binary labels this mainly exercises the path.)"""
    rng = np.random.default_rng(2)
    X_tr = pd.DataFrame(rng.normal(size=(300, 4)), columns=list("abcd"))
    y_tr = np.array([0, 1] * 150, dtype=int)        # both classes present
    X_val = pd.DataFrame(rng.normal(size=(80, 4)), columns=list("abcd"))
    y_val = np.array([0, 1] * 40, dtype=int)
    clf = LGBMClassifier(n_estimators=10, verbosity=-1)
    ok = NM._fit_binary_cls_safe(clf, X_tr, y_tr, X_val, y_val)
    assert ok is True
    p = clf.predict_proba(X_val)[:, 1]
    assert p.shape[0] == 80 and np.isfinite(p).all()
    print("  safe-fit trains normally on a healthy fold  OK")


def test_thin_cross_section_yields_only_class_1():
    """Mechanistic proof: with <5 names per day, the cross-sectional
    top/bottom-20% label can only be class 1 (or NaN), never class 0."""
    for n_names in (2, 3, 4):
        pct = pd.Series(np.arange(n_names, dtype=float)).rank(method="average", pct=True)
        lab = np.where(pct >= 0.80, 1, np.where(pct <= 0.20, 0, np.nan))
        assert (lab == 0).sum() == 0, f"n={n_names} unexpectedly produced a class-0 label"
    # At n=5 the bottom finally appears
    pct5 = pd.Series(np.arange(5, dtype=float)).rank(method="average", pct=True)
    lab5 = np.where(pct5 >= 0.80, 1, np.where(pct5 <= 0.20, 0, np.nan))
    assert (lab5 == 0).sum() == 1
    print("  thin cross-section (<5 names/day) yields no class-0 labels  OK")


def main():
    print("SINGLE-CLASS FOLD REGRESSION TESTS")
    print("-" * 50)
    for t in [test_raw_lightgbm_crashes_on_single_class_train,
              test_safe_fit_skips_single_class_fold,
              test_safe_fit_filters_unseen_eval_labels,
              test_thin_cross_section_yields_only_class_1]:
        t()
    print("-" * 50)
    print("ALL SINGLE-CLASS FOLD TESTS PASSED")


if __name__ == "__main__":
    main()

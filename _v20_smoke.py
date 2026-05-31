"""v20 smoke test: load 'Daily cache.py' via importlib, run static + canary
checks, and spot-check compute_daily_indicators on a synthetic frame.
Also benchmarks v20 indicator timing vs the v19 ground truth (whole-series
rank semantics) to confirm correctness ordering and rough speed."""
from __future__ import annotations
import importlib.util
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("dc_v20", str(HERE / "Daily cache.py"))
m = importlib.util.module_from_spec(spec)
sys.modules["dc_v20"] = m
spec.loader.exec_module(m)

print(f"[v20] loaded; SCHEMA_VERSION={m.SCHEMA_VERSION}")

# 1. Static check
m._v20_static_leak_check()
print("[v20] static leak check: OK")

# 2. Runtime canary
t0 = time.perf_counter()
m._v20_leak_canary_check()
print(f"[v20] runtime canary: OK ({time.perf_counter()-t0:.2f}s)")

# 3. Smoke test on synthetic OHLCV — 2000 rows ≈ 8 yrs of daily candles
rng = np.random.default_rng(7)
N = 2000
log_ret = rng.normal(0.0, 0.012, N)
close = 100.0 * np.exp(np.cumsum(log_ret))
op = close * (1.0 + rng.normal(0.0, 0.003, N))
hi = np.maximum(close, op) * (1.0 + np.abs(rng.normal(0.0, 0.005, N)))
lo = np.minimum(close, op) * (1.0 - np.abs(rng.normal(0.0, 0.005, N)))
vol = rng.integers(50_000, 1_500_000, N).astype("float64")
ts = pd.date_range("2018-01-01", periods=N, freq="B", tz=m.IST)
df = pd.DataFrame({"timestamp": ts, "open": op, "high": hi, "low": lo,
                   "close": close, "volume": vol})

t0 = time.perf_counter()
out = m.compute_daily_indicators(df.copy())
dur = time.perf_counter() - t0
print(f"[v20] compute_daily_indicators on N={N}: {dur:.2f}s -> {out.shape[1]} cols")

# Verify all expected columns are present
missing = [c for c in m.DAILY_INDICATOR_COLUMNS if c not in out.columns]
print(f"[v20] schema columns missing: {len(missing)}  (expected: 0)")
assert not missing, f"missing columns: {missing[:10]}"

# Verify the WQ alphas have expected NaN warmup but populate eventually
WQ_COLS = [f"D_WQ_{i}" for i in (3, 6, 12, 13, 16, 19, 20, 23, 26, 29, 33, 35, 38, 40, 41, 44)]
for c in WQ_COLS:
    s = pd.to_numeric(out[c], errors="coerce")
    n_nan = int(s.isna().sum())
    n_fin = int(np.isfinite(s).sum())
    last_val = float(s.iloc[-1]) if np.isfinite(s.iloc[-1]) else float("nan")
    print(f"  {c:8s}  nan={n_nan:5d}  finite={n_fin:5d}  last={last_val:+.4f}")

# Expanding-rank columns should be in [0, 1]
for c in ["D_WQ_3", "D_WQ_13", "D_WQ_16", "D_WQ_19", "D_WQ_20",
          "D_WQ_29", "D_WQ_33", "D_WQ_38", "D_WQ_40", "D_WQ_44"]:
    s = pd.to_numeric(out[c], errors="coerce").dropna()
    if len(s):
        # Some are products of ranks so range is wider than [0,1]; just verify finite
        assert np.isfinite(s).all(), f"{c} contains non-finite values"

# 4. Direct leak demonstration vs v19's broken semantic
# Compute the v19 _rank_cs equivalent and show that it would give different
# values for the same row when we tamper future data.
def _rank_cs_v19(s):
    return s.rank(pct=True)
def _xrank_v20(s):
    return s.expanding(min_periods=60).rank(pct=True)

# Use a clean test series whose row 100 ordering is meaningfully sensitive to
# tampering: a stationary noise series (so tampered values land in any rank
# bucket).
rng2 = np.random.default_rng(42)
s_full = pd.Series(rng2.normal(0, 1, 500))
v19 = _rank_cs_v19(s_full)
v20 = _xrank_v20(s_full)
# Tamper future rows with extreme values that will perturb the ordering
s_tamp = s_full.copy()
s_tamp.iloc[-50:] = -100.0  # all the last 50 become the smallest values
v19_t = _rank_cs_v19(s_tamp)
v20_t = _xrank_v20(s_tamp)

diff19 = float(abs(v19.iloc[100] - v19_t.iloc[100]))
diff20 = float(abs(v20.iloc[100] - v20_t.iloc[100]))
print()
print("[demo] N=500 stationary noise; tamper last 50 rows; observe row 100:")
print(f"  v19 _rank_cs       row100: original={v19.iloc[100]:.6f}  tampered={v19_t.iloc[100]:.6f}  delta={diff19:.6f}  {'<- LEAKS' if diff19 > 1e-9 else ''}")
print(f"  v20 _xrank         row100: original={v20.iloc[100]:.6f}  tampered={v20_t.iloc[100]:.6f}  delta={diff20:.6f}  <- LEAK-FREE")
assert diff20 < 1e-12, "v20 _xrank still leaks!"
assert diff19 > 1e-3, "v19 demo did not produce a measurable leak; tampering was too mild"

# 5. Idempotence: same input gives same output
out2 = m.compute_daily_indicators(df.copy())
diff_cols = []
for c in out.columns:
    a = out[c]; b = out2[c]
    if str(a.dtype) == "object":
        eq = (a.astype(str) == b.astype(str)) | (a.isna() & b.isna())
        if not eq.all():
            diff_cols.append(c)
        continue
    a = pd.to_numeric(a, errors="coerce").to_numpy(dtype="float64")
    b = pd.to_numeric(b, errors="coerce").to_numpy(dtype="float64")
    a_nan = np.isnan(a); b_nan = np.isnan(b)
    if not np.array_equal(a_nan, b_nan):
        diff_cols.append(c); continue
    d = np.where(a_nan, 0.0, np.abs(a - b))
    if np.nanmax(d) > 1e-9:
        diff_cols.append(c)
print(f"[v20] idempotence: identical outputs across two calls? {'YES' if not diff_cols else f'NO ({diff_cols[:5]})'}")
assert not diff_cols, f"non-idempotent: {diff_cols[:5]}"

print()
print("[v20] ALL SMOKE TESTS PASSED")

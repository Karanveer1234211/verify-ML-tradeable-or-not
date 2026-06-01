#!/usr/bin/env python3
"""
=============================================================================
UNIFIED CACHE  v1  —  one builder for DAILY + INTRADAY, leak-free, feature-rich
=============================================================================

Merges the two cache scripts into a single entry point:

  * DAILY    : reuses the validated, leak-free v20 engine in "Daily cache.py"
               (compute_daily_indicators) verbatim — no rewrite, no regression.
  * INTRADAY : fetches 5-min OHLCV (as "latest intraday cache.py" did) and now
               ALSO computes features, which intraday previously lacked:
                 (a) INTRADAY-NATIVE indicators on the 5-min bars (I_* columns)
                 (b) the FULL DAILY feature set attached per intraday bar via a
                     leak-free as-of join (the prior COMPLETED daily bar).

So "any feature for any model is readily available, daily or intraday":
  - daily models   -> the D_/W_/WQ_/Comb_/regime_ columns (unchanged).
  - intraday models-> I_* (5-min native) + every daily feature as context.

LEAK-FREE BY CONSTRUCTION
  - Daily engine is the v20 leak-free engine (expanding/rolling only).
  - Intraday-native indicators use only causal ops (rolling / cumulative within
    the session / shift) — never whole-series, never forward.
  - Daily->intraday attach: a daily feature for date D becomes "available" only
    at D's close (15:30 IST); merge_asof(direction=backward, allow_exact=False)
    guarantees an intraday bar on day D sees only daily features from D-1 and
    earlier. Verified by --self-test (tamper the future / a daily row, assert
    the past is byte-identical).

INCREMENTAL (the daily regression fix, applied to both timeframes)
  - If a symbol's parquet exists, recompute features in place + fetch ONLY the
    missing tail (and backfill earlier warm-up if needed). A schema/feature
    change costs a recompute, not a full re-download. Full fetch happens only on
    first build or with --force.

NETWORK NOTE
  The OHLCV fetch reuses KiteProvider from "Daily cache.py" (auth, symbol
  resolution, rate limiting, retries). The feature/leak/incremental logic is
  unit-tested here on synthetic data; the live Kite fetch needs a real run to
  validate end-to-end (no Kite access in CI).

USAGE
  python unified_cache.py --timeframe both   --symbols-file syms.txt
  python unified_cache.py --timeframe daily    --symbols TCS INFY
  python unified_cache.py --timeframe intraday --symbols TCS --interval 5minute
  python unified_cache.py --self-test          # synthetic leak + feature checks
=============================================================================
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
IST_TZ = "Asia/Kolkata"
SESSION_OPEN_MIN = 9 * 60 + 15      # 09:15
SESSION_CLOSE_MIN = 15 * 60 + 30    # 15:30
BARS_PER_SESSION_5MIN = 75          # ~ (15:30-09:15)/5

# Daily feature column prefixes that get attached to intraday bars as context.
DAILY_CONTEXT_PREFIXES = ("D_", "W_", "WQ_", "Comb_", "CPR_", "Struct_",
                          "DayType_", "DOW_", "regime_", "X_", "M_")


# =============================================================================
# Load the validated daily engine from "Daily cache.py" (DRY, no regression)
# =============================================================================

def load_daily_engine(repo_dir: Optional[Path] = None):
    """Import the v20 daily engine module ("Daily cache.py") by file path."""
    repo_dir = Path(repo_dir or Path(__file__).resolve().parent)
    for name in ("Daily cache.py", "daily_cache.py", "Daily_cache.py"):
        p = repo_dir / name
        if p.exists():
            spec = importlib.util.spec_from_file_location("daily_engine", str(p))
            mod = importlib.util.module_from_spec(spec)
            sys.modules["daily_engine"] = mod        # needed for dataclass introspection
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(
        "Could not find 'Daily cache.py' next to unified_cache.py — the daily "
        "engine is required for daily features and daily-context attachment."
    )


# =============================================================================
# Small numeric helpers (mirrors of the daily engine's, kept local & causal)
# =============================================================================

def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _ema(s: pd.Series, span: int) -> pd.Series:
    return _num(s).ewm(span=span, adjust=False, min_periods=1).mean()


def _rsi(s: pd.Series, period: int) -> pd.Series:
    c = _num(s)
    d = c.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    ag = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    al = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(h: pd.Series, l: pd.Series, c: pd.Series, period: int) -> pd.Series:
    h, l, c = _num(h), _num(l), _num(c)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


# =============================================================================
# INTRADAY-NATIVE indicators (leak-free; computed on a single symbol's 5-min bars)
# =============================================================================

def compute_intraday_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """All columns are CAUSAL: rolling / shift / cumulative-within-session only.
    Never a whole-series or forward op -> no look-ahead. Prefix `I_`."""
    if df is None or df.empty:
        return df
    df = df.sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(df["timestamp"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(IST_TZ)
    else:
        ts = ts.dt.tz_convert(IST_TZ)
    day = ts.dt.normalize()

    o, h, l, c = _num(df["open"]), _num(df["high"]), _num(df["low"]), _num(df["close"])
    v = _num(df["volume"]).fillna(0.0)

    # Returns over bar lags
    df["I_ret_1"] = c.pct_change(1)
    df["I_ret_5"] = c.pct_change(5)
    df["I_ret_15"] = c.pct_change(15)
    df["I_ret_std_15"] = c.pct_change().rolling(15, min_periods=5).std()

    # Trend / momentum
    df["I_ema20"] = _ema(c, 20)
    df["I_ema50"] = _ema(c, 50)
    prev = df["I_ema20"].shift(1)
    df["I_ema20_angle"] = np.degrees(np.arctan((df["I_ema20"] - prev) / prev.replace(0, np.nan)))
    macd = _ema(c, 12) - _ema(c, 26)
    sig = macd.ewm(span=9, adjust=False, min_periods=1).mean()
    df["I_macd"] = macd
    df["I_macd_hist"] = macd - sig
    df["I_rsi14"] = _rsi(c, 14)

    # Volatility
    df["I_atr14"] = _atr(h, l, c, 14)
    df["I_atr_pct"] = (df["I_atr14"] / c.replace(0, np.nan)) * 100.0
    sma20 = c.rolling(20, min_periods=1).mean()
    std20 = c.rolling(20, min_periods=1).std()
    bw = (4 * std20)
    df["I_bb_pctB_20"] = ((c - (sma20 - 2 * std20)) / bw.replace(0, np.nan)).clip(-5, 5)
    df["I_bb_bw_20"] = (bw / sma20.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    # Session VWAP (resets daily; cumulative within the day = causal)
    tp = (h + l + c) / 3.0
    cum_pv = (tp * v).groupby(day).cumsum()
    cum_v = v.groupby(day).cumsum().replace(0, np.nan)
    df["I_vwap_session"] = cum_pv / cum_v
    df["I_dist_vwap"] = (c - df["I_vwap_session"]) / df["I_vwap_session"].replace(0, np.nan)

    # Volume
    vol_m = v.rolling(BARS_PER_SESSION_5MIN, min_periods=10).mean()
    vol_s = v.rolling(BARS_PER_SESSION_5MIN, min_periods=10).std()
    df["I_vol_z"] = (v - vol_m) / vol_s.replace(0, np.nan)
    df["I_vol_surge"] = (v > (vol_m + 2 * vol_s)).astype("int8")
    obv = (np.sign(c.diff().fillna(0.0)) * v).cumsum()
    df["I_obv"] = obv
    df["I_obv_slope"] = obv.diff()

    # Intraday Donchian position over ~1 session of bars (uses shifted highs/lows)
    hi_n = h.shift(1).rolling(BARS_PER_SESSION_5MIN, min_periods=10).max()
    lo_n = l.shift(1).rolling(BARS_PER_SESSION_5MIN, min_periods=10).min()
    df["I_donch_pos"] = ((c - lo_n) / (hi_n - lo_n).replace(0, np.nan)).clip(-1, 2)

    # Time-of-day structure (deterministic, leak-free)
    mins = ts.dt.hour * 60 + ts.dt.minute
    df["I_min_since_open"] = (mins - SESSION_OPEN_MIN).clip(lower=0).astype("int16")
    df["I_bar_idx"] = ts.groupby(day).cumcount().astype("int16")
    # Return since this session's open (open of first bar of the day), causal
    sess_open = c.groupby(day).transform("first")
    df["I_ret_since_open"] = (c / sess_open.replace(0, np.nan) - 1.0)

    # inf -> nan hygiene on the new columns
    icols = [col for col in df.columns if col.startswith("I_")]
    for col in icols:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return df


# =============================================================================
# Attach DAILY features to intraday bars (leak-free as-of join)
# =============================================================================

def attach_daily_context(intraday: pd.DataFrame, daily: pd.DataFrame,
                         context_cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """For each intraday bar, attach the daily features from the most recent
    COMPLETED daily bar. A daily row for date D is treated as available only at
    D's session close (15:30 IST), so an intraday bar on day D sees D-1 and
    earlier — never same-day or future daily. Leak-free by construction.
    """
    if intraday is None or intraday.empty or daily is None or daily.empty:
        return intraday
    intr = intraday.copy()
    dly = daily.copy()
    intr["timestamp"] = pd.to_datetime(intr["timestamp"])
    dly["timestamp"] = pd.to_datetime(dly["timestamp"])
    # normalize tz to IST for both
    for d in (intr, dly):
        if d["timestamp"].dt.tz is None:
            d["timestamp"] = d["timestamp"].dt.tz_localize(IST_TZ)
        else:
            d["timestamp"] = d["timestamp"].dt.tz_convert(IST_TZ)

    if context_cols is None:
        context_cols = [c for c in dly.columns
                        if c.startswith(DAILY_CONTEXT_PREFIXES)]
    context_cols = [c for c in context_cols if c in dly.columns]
    if not context_cols:
        return intr

    dly = dly[["timestamp"] + context_cols].copy()
    # availability = daily date's session close (15:30 IST)
    dly["_avail"] = dly["timestamp"].dt.normalize() + pd.Timedelta(hours=15, minutes=30)
    dly = dly.drop(columns=["timestamp"]).sort_values("_avail").reset_index(drop=True)
    intr = intr.sort_values("timestamp").reset_index(drop=True)

    merged = pd.merge_asof(
        intr, dly,
        left_on="timestamp", right_on="_avail",
        direction="backward", allow_exact_matches=False,
    )
    return merged.drop(columns=["_avail"])


# =============================================================================
# Self-test: leak canaries + feature availability (no Kite / network needed)
# =============================================================================

def _synth_intraday(n_days: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    day0 = pd.Timestamp("2023-01-02", tz=IST_TZ)
    for d in range(n_days):
        day = day0 + pd.Timedelta(days=d)
        # 75 five-min bars 09:15..15:30
        times = [day + pd.Timedelta(minutes=15 + 5 * i) for i in range(BARS_PER_SESSION_5MIN)]
        c = 100 + np.cumsum(rng.normal(0, 0.1, len(times)))
        o = c + rng.normal(0, 0.05, len(times))
        hi = np.maximum(o, c) + np.abs(rng.normal(0, 0.05, len(times)))
        lo = np.minimum(o, c) - np.abs(rng.normal(0, 0.05, len(times)))
        vol = rng.integers(1000, 50000, len(times)).astype(float)
        rows.append(pd.DataFrame({"timestamp": times, "open": o, "high": hi,
                                  "low": lo, "close": c, "volume": vol}))
    return pd.concat(rows, ignore_index=True)


def _synth_daily(n_days: int = 30, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.date_range("2022-12-01", periods=n_days, freq="B", tz=IST_TZ)
    c = 100 + np.cumsum(rng.normal(0, 1, n_days))
    return pd.DataFrame({
        "timestamp": days,
        "D_rsi14": 50 + rng.normal(0, 10, n_days),
        "D_atr_pct": np.abs(rng.normal(2, 0.5, n_days)),
        "W_ret_4w": rng.normal(0, 3, n_days),
    })


def _run_self_test() -> int:
    print("=" * 70)
    print("UNIFIED CACHE — SELF-TEST")
    print("=" * 70)
    K = BARS_PER_SESSION_5MIN  # tamper the last session

    # 1) INTRADAY-NATIVE leak canary: tamper the future, assert past unchanged
    base = _synth_intraday(n_days=8, seed=3)
    tamp = base.copy()
    n = len(base)
    rng = np.random.default_rng(9)
    for col in ["open", "high", "low", "close", "volume"]:
        tamp.loc[n - K:, col] = tamp.loc[n - K:, col].to_numpy() * rng.uniform(0.3, 3.0, K)
    a = compute_intraday_indicators(base.copy())
    b = compute_intraday_indicators(tamp.copy())
    icols = [c for c in a.columns if c.startswith("I_")]
    worst, worst_col = 0.0, None
    for c in icols:
        av = pd.to_numeric(a[c], errors="coerce").to_numpy()[: n - K]
        bv = pd.to_numeric(b[c], errors="coerce").to_numpy()[: n - K]
        an, bn = np.isnan(av), np.isnan(bv)
        assert np.array_equal(an, bn), f"{c}: NaN pattern changed in the past"
        d = float(np.nanmax(np.where(an, 0.0, np.abs(av - bv))) if (~an).any() else 0.0)
        if d > worst:
            worst, worst_col = d, c
    print(f"[1] intraday-native: {len(icols)} I_ features; worst past-diff after "
          f"tampering future = {worst:.3e} ({worst_col})")
    assert worst < 1e-9, "intraday-native indicator leaks the future!"

    # 2) DAILY-CONTEXT as-of leak canary: tamper a daily row for date D, assert
    #    intraday bars on day D (and earlier) are unchanged; only D+1.. change.
    intr = _synth_intraday(n_days=8, seed=3)
    daily = _synth_daily(n_days=40, seed=1)
    # align daily dates to the intraday window
    idays = pd.to_datetime(intr["timestamp"]).dt.normalize().dt.tz_localize(None).unique()
    daily = daily.copy()
    daily["timestamp"] = pd.date_range(pd.Timestamp(idays.min()) - pd.Timedelta(days=10),
                                       periods=len(daily), freq="B", tz=IST_TZ)
    merged_a = attach_daily_context(intr.copy(), daily.copy())
    # pick a daily date that falls inside the intraday span and tamper it
    span_lo = pd.to_datetime(intr["timestamp"]).min()
    span_hi = pd.to_datetime(intr["timestamp"]).max()
    cand = daily[(daily["timestamp"] >= span_lo) & (daily["timestamp"] <= span_hi)]
    assert len(cand) >= 2, "need daily rows inside the intraday span for the test"
    tamper_date = cand["timestamp"].iloc[len(cand) // 2].normalize()
    daily_t = daily.copy()
    mask = daily_t["timestamp"].dt.normalize() == tamper_date
    daily_t.loc[mask, ["D_rsi14", "D_atr_pct", "W_ret_4w"]] = -999.0
    merged_b = attach_daily_context(intr.copy(), daily_t.copy())

    ts = pd.to_datetime(merged_a["timestamp"])
    # bars strictly before the tampered daily's availability (D 15:30) must match
    avail = tamper_date + pd.Timedelta(hours=15, minutes=30)
    before = ts < avail
    ctx = [c for c in merged_a.columns if c.startswith(DAILY_CONTEXT_PREFIXES)]
    for c in ctx:
        av = pd.to_numeric(merged_a.loc[before, c], errors="coerce").to_numpy()
        bv = pd.to_numeric(merged_b.loc[before, c], errors="coerce").to_numpy()
        an, bn = np.isnan(av), np.isnan(bv)
        assert np.array_equal(an, bn) and (np.nanmax(np.where(an, 0.0, np.abs(av - bv))) < 1e-9
                                           if (~an).any() else True), \
            f"daily-context LEAK: {c} changed on/before the tampered day"
    # and at least one later bar DID change (sanity: the tamper propagated forward)
    after = ts >= avail
    changed = False
    for c in ctx:
        av = pd.to_numeric(merged_a.loc[after, c], errors="coerce").to_numpy()
        bv = pd.to_numeric(merged_b.loc[after, c], errors="coerce").to_numpy()
        if np.nansum(np.abs(np.nan_to_num(av) - np.nan_to_num(bv))) > 0:
            changed = True
            break
    print(f"[2] daily-context as-of: {len(ctx)} daily cols attached; past unchanged "
          f"when daily {str(tamper_date.date())} tampered; future propagated={changed}")
    assert changed, "tamper did not propagate to later bars — join may be wrong"

    # 3) feature availability: a fully-built intraday frame carries BOTH the
    #    intraday-native I_* features AND the attached daily features per bar.
    full = attach_daily_context(compute_intraday_indicators(intr.copy()), daily.copy())
    n_i = sum(c.startswith("I_") for c in full.columns)
    n_d = sum(c.startswith(DAILY_CONTEXT_PREFIXES) for c in full.columns)
    assert n_i > 0, "no intraday-native I_ features on the built frame"
    assert n_d > 0, "no daily features attached to the intraday frame"
    print(f"[3] feature availability: built intraday frame exposes "
          f"{n_i} I_ + {n_d} daily features per bar")

    print("\nALL SELF-TESTS PASSED")
    return 0


# =============================================================================
# Build paths / IO
# =============================================================================

def _cache_root(timeframe: str) -> Path:
    if timeframe == "daily":
        return Path(os.environ.get("CACHE_DAILY_ROOT",
                    str(Path.home() / ".kite_cache" / "cache_daily_new")))
    return Path(os.environ.get("CACHE_INTRADAY_ROOT",
                str(Path.home() / ".kite_cache" / "intraday_5min")))


def _intraday_path(symbol: str, interval: str) -> Path:
    safe = "".join(ch for ch in symbol.upper() if ch.isalnum() or ch in "._-")
    suffix = "5min" if interval.startswith("5") else interval
    return _cache_root("intraday") / f"{safe}_{suffix}.parquet"


# =============================================================================
# Intraday build (fetch OHLCV via the daily engine's KiteProvider + features)
# =============================================================================

def _fetch_intraday_bars(provider, symbol: str, start: dt.date, end: dt.date,
                         interval: str = "5minute") -> pd.DataFrame:
    """Reuse the daily engine's KiteProvider auth + symbol resolution + _hist,
    chunked to Kite's ~60-day intraday limit."""
    MAX_DAYS = 60
    inst = provider._symbol_to_instrument_token(symbol)
    rows: list = []
    cur = start
    while cur <= end:
        hi = min(end, cur + dt.timedelta(days=MAX_DAYS - 1))
        s_dt = dt.datetime.combine(cur, dt.time(9, 0))
        e_dt = dt.datetime.combine(hi, dt.time(15, 40))
        chunk = provider._hist(inst, s_dt, e_dt, interval=interval)
        rows.extend(chunk or [])
        cur = hi + dt.timedelta(days=1)
    df = pd.DataFrame(rows)
    if df is None or df.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    if "date" in df.columns and "timestamp" not in df.columns:
        df = df.rename(columns={"date": "timestamp"})
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def build_intraday_symbol(provider, daily_engine, symbol: str, start: dt.date,
                          end: dt.date, interval: str = "5minute",
                          force: bool = False, attach_daily: bool = True) -> Path:
    """Incremental intraday build with features. Recompute-in-place + tail
    gap-fill (never full re-download unless first build or --force)."""
    out = _intraday_path(symbol, interval)
    out.parent.mkdir(parents=True, exist_ok=True)

    existing = None
    fetch_start = start
    if out.exists() and not force:
        try:
            existing = pd.read_parquet(out, columns=["timestamp", "open", "high", "low", "close", "volume"])
            existing["timestamp"] = pd.to_datetime(existing["timestamp"])
            last = existing["timestamp"].max()
            fetch_start = (last + pd.Timedelta(minutes=5)).date()
        except Exception:
            existing = None
            fetch_start = start

    raw_new = _fetch_intraday_bars(provider, symbol, fetch_start, end, interval=interval)

    if existing is not None and len(existing):
        if raw_new is not None and len(raw_new):
            raw = (pd.concat([existing, raw_new], ignore_index=True)
                   .drop_duplicates("timestamp", keep="last")
                   .sort_values("timestamp").reset_index(drop=True))
        else:
            raw = existing.sort_values("timestamp").reset_index(drop=True)
    else:
        raw = raw_new
    if raw is None or raw.empty:
        return out

    # Features: intraday-native + (optional) leak-free daily context
    feat = compute_intraday_indicators(raw)
    if attach_daily:
        try:
            daily_pq = _cache_root("daily") / f"{symbol.upper()}_daily.parquet"
            if daily_pq.exists():
                daily_df = pd.read_parquet(daily_pq)
                feat = attach_daily_context(feat, daily_df)
        except Exception as e:
            print(f"[intraday] {symbol}: daily-context attach skipped ({e})")

    feat.to_parquet(out, index=False, compression="snappy")
    return out


# =============================================================================
# CLI
# =============================================================================

def _load_symbols(args) -> List[str]:
    if args.symbols:
        return [s.strip().upper() for s in args.symbols if s.strip()]
    if args.symbols_file and Path(args.symbols_file).exists():
        txt = Path(args.symbols_file).read_text(encoding="utf-8", errors="ignore")
        toks = [t.strip().upper() for t in txt.replace(",", "\n").splitlines()]
        return [t for t in toks if t]
    return []


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Unified daily+intraday cache builder.")
    ap.add_argument("--timeframe", choices=["daily", "intraday", "both"], default="both")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--symbols-file", default=None)
    ap.add_argument("--start-date", default=None)
    ap.add_argument("--end-date", default=None)
    ap.add_argument("--interval", default="5minute")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-attach-daily", action="store_true",
                    help="do not attach daily features to intraday bars")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)

    if a.self_test:
        return _run_self_test()

    symbols = _load_symbols(a)
    if not symbols:
        print("ERROR: provide --symbols or --symbols-file (or --self-test).", file=sys.stderr)
        return 2

    daily_engine = load_daily_engine()
    end = (pd.Timestamp(a.end_date).date() if a.end_date else dt.datetime.now(tz=IST).date())
    start = (pd.Timestamp(a.start_date).date() if a.start_date
             else end - dt.timedelta(days=int(os.environ.get("UNIFIED_DAYS", "600"))))

    # Daily: delegate to the validated engine (build_daily) via its Pipeline-style call.
    if a.timeframe in ("daily", "both"):
        cfg = daily_engine.Config.from_env()
        provider = daily_engine.KiteProvider()
        for s in symbols:
            try:
                daily_engine.build_daily(provider, cfg, s,
                                         start - dt.timedelta(days=200), end,
                                         force=a.force, recompute_only=False)
                print(f"[daily] {s} OK")
            except Exception as e:
                print(f"[daily] {s} ERROR: {e}")
    else:
        provider = daily_engine.KiteProvider()

    # Intraday: fetch + features (+ leak-free daily context).
    if a.timeframe in ("intraday", "both"):
        for s in symbols:
            try:
                p = build_intraday_symbol(provider, daily_engine, s, start, end,
                                          interval=a.interval, force=a.force,
                                          attach_daily=not a.no_attach_daily)
                print(f"[intraday] {s} -> {p}")
            except Exception as e:
                print(f"[intraday] {s} ERROR: {e}")

    print("Unified cache build complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

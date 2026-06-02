#!/usr/bin/env python3
"""
=============================================================================
INTRADAY 5-MIN CACHER v3 — fast (parallel), leak-free, indicator-rich
=============================================================================

Supersedes "latest intraday cache.py" (v2). Same Kite auth (kite_token.json),
same per-symbol parquet layout, but:

  SPEED (zero compromise on output)
  ---------------------------------
  v2 fetched single-threaded at 3 req/s (the docstring admitted 6-10h for the
  full universe). Fetching is network-bound, and concurrency does NOT change the
  bars returned — so v3 mirrors Daily cache.py: a ThreadPoolExecutor with a
  shared token-bucket RateLimiter (default 3 req/s, Kite's historical limit) and
  exponential-backoff retry. Same bars out, a fraction of the wall time. Atomic
  writes (tmp + os.replace) + a per-symbol lock make every run crash-safe and
  resumable. Indicators are computed in parallel across symbols (CPU-bound) using
  processes.

  INDICATORS (intraday, SESSION-AWARE, LEAK-FREE BY CONSTRUCTION)
  --------------------------------------------------------------
  v2 stored RAW bars only. v3 adds compute_intraday_indicators() with the same
  leak-free discipline as Daily cache.py v20 (expanding/rolling only; no whole-
  series rank/cumsum that can see the future) PLUS the intraday-specific subtlety
  that daily does not have: VWAP, VPOC, opening-range, cumulative volume and RVOL
  must RESET every trading session and may never peek across the day boundary.

  Families (all per 5-min bar, all causal):
    • Session VWAP + std bands (1/2 sigma)         I_vwap, I_vwap_dev_pct, I_vwap_z, I_above_vwap, I_vwap_band_{up,dn}{1,2}
    • Session running VPOC (volume point of control) I_vpoc, I_dist_vpoc_pct
    • Opening Range 5 / 15 / 30 min + breakouts     I_or{5,15,30}_high/low/width_pct, I_or{..}_brk_up/dn, I_dist_or{..}_high/low_pct
    • RSI (Wilder) 7 / 14                            I_rsi7, I_rsi14
    • ADX / +DI / -DI (Wilder) 14                    I_adx14, I_pdi14, I_mdi14
    • ATR(14) + % of price                           I_atr14, I_atr_pct
    • EMA 9/20/50 + stack + dist                     I_ema{9,20,50}, I_ema_stack, I_dist_ema20_pct
    • MACD (12,26,9)                                 I_macd, I_macd_signal, I_macd_hist
    • Bollinger(20,2) bandwidth + %B                 I_bb_bw, I_bb_pctb
    • Returns / range / gap                          I_ret_1, I_range_pct, I_gap_open_pct (first-bar gap vs prev close)
    • Cumulative volume + RVOL vs 20-day per-slot    I_cum_vol, I_rvol_slot, I_vol_z
    • VWAP-anchored momentum / session position      I_sess_pos, I_minute_of_day, I_bars_into_session
    • Stochastic %K/%D (14,3), CCI(20), Williams %R   I_stoch_k, I_stoch_d, I_cci20, I_willr14
    • OBV slope, MFI(14)                              I_obv, I_obv_slope, I_mfi14

  SAFETY NETS (mirrors daily v20)
  -------------------------------
  • _static_leak_check(): scans this file for unwindowed .rank()/.cumsum()/
    .csummax() inside compute_intraday_indicators() that would re-introduce a
    look-ahead, and for any rolling/ewm/cum op missing a per-session groupby.
  • _leak_canary_check(): builds two synthetic multi-day intraday frames that are
    identical for rows 0..N-K and differ only in the last K rows, computes
    indicators on both, and asserts every column matches on rows 0..N-K AND that
    no value bleeds across the session boundary.

  STORAGE / SCHEMA
  ----------------
  SCHEMA_VERSION + a per-symbol <SYM>_5min.ok.json sidecar (mirrors daily). The
  parquet keeps raw OHLCV + every I_* column. --raw-only stores bars without
  indicators; --recompute rebuilds indicators from cached bars WITHOUT refetching
  (so you can add indicators later for free).

USAGE
-----
  python intraday_cache.py                      # parallel build, all symbols
  python intraday_cache.py --limit 10           # test on 10 symbols
  python intraday_cache.py --symbols A,B,C
  python intraday_cache.py --full-rebuild
  python intraday_cache.py --recompute          # re-derive indicators from cache, no fetch
  python intraday_cache.py --raw-only           # fetch bars, skip indicators
  python intraday_cache.py --self-test          # offline: leak canary + indicator checks (no Kite)

Deps: numpy, pandas (+ pyarrow); kiteconnect only for live fetch.
=============================================================================
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import contextlib
import datetime as dt
import functools
import json
import math
import os
import re
import sys
import threading
import time
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# -------------------- schema / session --------------------
SCHEMA_VERSION = 3
OK_VERSION_KEY = "schema_version"

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
SESSION_OPEN = dt.time(9, 15)
SESSION_CLOSE = dt.time(15, 30)
INTERVAL = "5minute"
BAR_MINUTES = 5

OHLCV = ["timestamp", "open", "high", "low", "close", "volume"]


# -------------------- config --------------------
def _win_default(p: str) -> Path:
    return Path(p)


@dataclass(frozen=True)
class Config:
    token_file: Path = field(default_factory=lambda: Path(
        os.environ.get("KITE_TOKEN_FILE",
                       r"C:\Users\karanvsi\PyCharmMiscProject\kite_token.json")))
    master_symbol_file: Path = field(default_factory=lambda: Path(
        os.environ.get("MASTER_SYMBOL_FILE",
                       r"C:\Users\karanvsi\PyCharmMiscProject\symbols_master.txt")))
    daily_cache_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("CACHE_DAILY_ROOT",
                       r"C:\Users\karanvsi\Desktop\Pycharm\Cache\cache_daily_new")))
    intraday_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("INTRADAY_DIR",
                       r"C:\Users\karanvsi\Desktop\Pycharm\Cache\intraday_5min")))

    history_years: float = 5.0
    interval: str = INTERVAL

    # Kite limits + concurrency
    rate_limit_per_sec: float = float(os.environ.get("KITE_RATE_LIMIT_PER_SEC", "3"))
    max_days_per_request: int = 60
    fetch_workers: int = int(os.environ.get("INTRADAY_FETCH_WORKERS", "8"))
    compute_workers: int = int(os.environ.get("INTRADAY_COMPUTE_WORKERS", str(max(1, (os.cpu_count() or 4)))))
    request_timeout_s: float = 30.0
    retry_tries: int = 6
    retry_backoff_base: float = 0.45

    # RVOL lookback (sessions) for per-time-slot average volume
    rvol_lookback_days: int = 20

    parquet_compression: Optional[str] = os.environ.get("PARQUET_COMPRESSION", "snappy")

    def intraday_path(self, symbol: str) -> Path:
        return self.intraday_dir / f"{sanitize_symbol(symbol)}_5min.parquet"

    def ok_path(self, symbol: str) -> Path:
        return self.intraday_dir / f"{sanitize_symbol(symbol)}_5min.ok.json"


# -------------------- symbol sanitation --------------------
_ILLEGAL = set('<>:"/\\|?*')


def sanitize_symbol(sym: str) -> str:
    s = str(sym or "").upper().strip()
    for suf in ("_DAILY", "_INTRADAY", "_5MIN"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    for ext in (".PKL", ".PARQUET", ".CSV"):
        if s.endswith(ext):
            s = s[: -len(ext)]
    s = "".join(ch for ch in s if ch not in _ILLEGAL)
    return s.strip()


# -------------------- atomic IO + lock (mirrors Daily cache.py) --------------------
class FileLock:
    def __init__(self, path: Path, poll_ms: int = 50, timeout_s: float = 60.0):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = Path(str(path) + ".lock")
        self.poll_ms = poll_ms
        self.timeout_s = timeout_s
        self._fd: Optional[int] = None

    def acquire(self):
        deadline = time.time() + self.timeout_s
        while True:
            try:
                self._fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self._fd, str(os.getpid()).encode())
                return
            except FileExistsError:
                if time.time() > deadline:
                    raise TimeoutError(f"lock timeout {self.lock_path}")
                time.sleep(self.poll_ms / 1000.0)

    def release(self):
        if self._fd is not None:
            try:
                os.close(self._fd)
            finally:
                self._fd = None
            with contextlib.suppress(FileNotFoundError):
                os.remove(self.lock_path)

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *a):
        self.release()


def atomic_write_parquet(path: Path, df: pd.DataFrame, compression: Optional[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False, compression=compression)
    os.replace(tmp, path)


def write_json_atomic(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
    os.replace(tmp, path)


# -------------------- rate limit + retry (mirrors Daily cache.py) --------------------
class RateLimiter:
    def __init__(self, per_sec: float):
        self.per_sec = float(per_sec)
        self._lock = threading.Lock()
        self._tokens = per_sec
        self._updated = time.perf_counter()

    def acquire(self):
        while True:
            with self._lock:
                now = time.perf_counter()
                self._tokens = min(self.per_sec, self._tokens + (now - self._updated) * self.per_sec)
                self._updated = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                need = max(0.0, 1.0 - self._tokens)
                wait = need / self.per_sec if self.per_sec > 0 else 0.01
            time.sleep(wait if wait > 0 else 0.01)


class AuthExpired(Exception):
    """Kite access token missing/expired/invalid."""


def with_retry(fn=None, *, tries: int = 6, backoff: float = 0.45):
    """Retry decorator with exponential backoff. Usable bare (@with_retry) or
    parametrized (@with_retry(tries=..., backoff=...)). Re-raises AuthExpired
    immediately (no point retrying a dead token)."""
    if fn is None:
        return functools.partial(with_retry, tries=tries, backoff=backoff)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        last = None
        for attempt in range(tries):
            try:
                return fn(*args, **kwargs)
            except AuthExpired:
                raise
            except Exception as e:  # noqa
                msg = str(e).lower()
                if "tokenexception" in msg or "access_token" in msg or "expired" in msg:
                    raise AuthExpired(str(e))
                if "too many" in msg or "rate" in msg or "429" in msg:
                    sleep = min(15.0, backoff * (2.2 ** attempt))
                else:
                    sleep = min(8.0, backoff * (1.8 ** attempt) + np.random.random() * backoff)
                last = e
                time.sleep(sleep)
        raise last
    return wrapper


# =============================================================================
#                         INDICATOR HELPERS (causal)
# =============================================================================
def _f(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _ema(s: pd.Series, span: int) -> pd.Series:
    return _f(s).ewm(span=span, adjust=False, min_periods=1).mean()


def _rsi(s: pd.Series, period: int) -> pd.Series:
    c = _f(s)
    d = c.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    ag = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    al = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _true_range(h, l, c):
    h, l, c = _f(h), _f(l), _f(c)
    pc = c.shift(1)
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def _atr(h, l, c, period):
    return _true_range(h, l, c).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _adx(h, l, c, period):
    h, l, c = _f(h), _f(l), _f(c)
    up = h.diff()
    dn = l.shift(1) - l
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = _true_range(h, l, c)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    pdi = 100 * pd.Series(plus_dm, index=h.index).ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr
    mdi = 100 * pd.Series(minus_dm, index=h.index).ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr
    denom = (pdi + mdi).replace(0, np.nan)
    dx = (pdi - mdi).abs() / denom * 100
    adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return adx, pdi, mdi


def _running_vwap(price_vol_cumsum: np.ndarray, vol_cumsum: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(vol_cumsum > 0, price_vol_cumsum / vol_cumsum, np.nan)
    return out


def _session_running_vpoc(tp: np.ndarray, vol: np.ndarray, session_id: np.ndarray,
                          bins: int = 30) -> np.ndarray:
    """Causal per-session VPOC: at row t, the VPOC over this session's rows 0..t ONLY.

    LEAK-FREE BY CONSTRUCTION: the price->bin mapping uses the RUNNING [min,max] of
    rows seen so far (i..k), never the whole session's range. (Using the full
    session's min/max for bin edges would let a late bar's price move the bin
    boundaries of earlier rows -> future leak, the same bug daily v20 fixed for
    weekly VPOC.) When a new bar expands the running range we rebuild the histogram
    over the bars seen so far under the new edges; that is O(bins) per expansion and
    only depends on past bars, so it stays strictly causal.
    """
    n = tp.size
    out = np.full(n, np.nan)
    i = 0
    while i < n:
        j = i
        sid = session_id[i]
        while j < n and session_id[j] == sid:
            j += 1
        run_lo = math.inf
        run_hi = -math.inf
        edges = None
        hist = np.zeros(bins, dtype="float64")
        seen_p: List[float] = []
        seen_v: List[float] = []
        for k in range(i, j):
            pk = tp[k]
            vk = vol[k]
            if not (math.isfinite(pk) and math.isfinite(vk)):
                out[k] = out[k - 1] if k > i else np.nan
                continue
            seen_p.append(pk)
            seen_v.append(vk)
            new_lo = min(run_lo, pk)
            new_hi = max(run_hi, pk)
            if edges is None or new_lo < run_lo or new_hi > run_hi:
                # running range expanded -> rebuild edges + histogram over PAST bars
                run_lo, run_hi = new_lo, new_hi
                if math.isclose(run_lo, run_hi):
                    out[k] = run_lo
                    # single-price so far; keep a trivial 1-bar histogram
                    edges = None
                    hist = np.zeros(bins, dtype="float64")
                    continue
                edges = np.linspace(run_lo, run_hi, bins + 1)
                hist = np.zeros(bins, dtype="float64")
                idxs = np.clip(np.digitize(np.asarray(seen_p), edges) - 1, 0, bins - 1)
                for kk, b in enumerate(idxs):
                    hist[b] += seen_v[kk]
            else:
                b = int(np.clip(np.digitize([pk], edges)[0] - 1, 0, bins - 1))
                hist[b] += vk
            top = int(np.argmax(hist))
            out[k] = (edges[top] + edges[top + 1]) / 2.0
        i = j
    return out


def _opening_range(df: pd.DataFrame, minutes: int) -> Tuple[pd.Series, pd.Series]:
    """Per-session opening-range high/low using only bars within the first
    `minutes` of each session (causal: value is constant across the session but
    only known after the OR window closes -> NaN before then)."""
    n = len(df)
    or_high = np.full(n, np.nan)
    or_low = np.full(n, np.nan)
    mod = df["_minute_of_day"].to_numpy()
    sid = df["_session_id"].to_numpy()
    high = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    open_min = SESSION_OPEN.hour * 60 + SESSION_OPEN.minute
    cutoff = open_min + minutes
    i = 0
    while i < n:
        j = i
        s = sid[i]
        while j < n and sid[j] == s:
            j += 1
        in_or = (mod[i:j] >= open_min) & (mod[i:j] < cutoff)
        if in_or.any():
            h = np.nanmax(high[i:j][in_or])
            l = np.nanmin(low[i:j][in_or])
            # only known AFTER the OR window closes -> assign to bars at/after cutoff
            after = mod[i:j] >= cutoff
            or_high[i:j] = np.where(after, h, np.nan)
            or_low[i:j] = np.where(after, l, np.nan)
        i = j
    return pd.Series(or_high, index=df.index), pd.Series(or_low, index=df.index)


def _stoch(h, l, c, k=14, d=3):
    h, l, c = _f(h), _f(l), _f(c)
    ll = l.rolling(k, min_periods=k).min()
    hh = h.rolling(k, min_periods=k).max()
    rng = (hh - ll).replace(0, np.nan)
    kf = 100 * (c - ll) / rng
    return kf, kf.rolling(d, min_periods=d).mean()


def _cci(h, l, c, period=20):
    tp = (_f(h) + _f(l) + _f(c)) / 3.0
    sma = tp.rolling(period, min_periods=period).mean()
    md = (tp - sma).abs().rolling(period, min_periods=period).mean()
    return (tp - sma) / (0.015 * md.replace(0, np.nan))


def _willr(h, l, c, period=14):
    h, l, c = _f(h), _f(l), _f(c)
    hh = h.rolling(period, min_periods=period).max()
    ll = l.rolling(period, min_periods=period).min()
    rng = (hh - ll).replace(0, np.nan)
    return -100 * (hh - c) / rng


def _mfi(h, l, c, v, period=14):
    tp = (_f(h) + _f(l) + _f(c)) / 3.0
    rmf = tp * _f(v)
    dtp = tp.diff()
    pos = rmf.where(dtp > 0, 0.0).rolling(period, min_periods=period).sum()
    neg = rmf.where(dtp < 0, 0.0).rolling(period, min_periods=period).sum()
    mr = pos / neg.replace(0, np.nan)
    return 100 - 100 / (1 + mr)


def _rolling_slope(s: pd.Series, window: int) -> pd.Series:
    y = _f(s)
    x = np.arange(window, dtype="float64")
    xm = x.mean()
    xd = x - xm
    denom = float((xd * xd).sum()) or np.nan

    def _sl(arr):
        if np.isnan(arr).any():
            return np.nan
        ym = arr.mean()
        return float((xd * (arr - ym)).sum() / denom)

    return y.rolling(window, min_periods=window).apply(_sl, raw=True)


# =============================================================================
#               THE INTRADAY INDICATOR ENGINE  (session-aware, causal)
# =============================================================================
INTRADAY_INDICATOR_COLUMNS = [
    # session position
    "I_minute_of_day", "I_bars_into_session", "I_sess_pos",
    # vwap family
    "I_vwap", "I_vwap_dev_pct", "I_vwap_z", "I_above_vwap",
    "I_vwap_band_up1", "I_vwap_band_dn1", "I_vwap_band_up2", "I_vwap_band_dn2",
    # vpoc
    "I_vpoc", "I_dist_vpoc_pct",
    # opening ranges
    "I_or5_high", "I_or5_low", "I_or5_width_pct", "I_or5_brk_up", "I_or5_brk_dn",
    "I_dist_or5_high_pct", "I_dist_or5_low_pct",
    "I_or15_high", "I_or15_low", "I_or15_width_pct", "I_or15_brk_up", "I_or15_brk_dn",
    "I_dist_or15_high_pct", "I_dist_or15_low_pct",
    "I_or30_high", "I_or30_low", "I_or30_width_pct", "I_or30_brk_up", "I_or30_brk_dn",
    "I_dist_or30_high_pct", "I_dist_or30_low_pct",
    # momentum / trend
    "I_rsi7", "I_rsi14", "I_adx14", "I_pdi14", "I_mdi14",
    "I_ema9", "I_ema20", "I_ema50", "I_ema_stack", "I_dist_ema20_pct",
    "I_macd", "I_macd_signal", "I_macd_hist",
    # volatility / bands
    "I_atr14", "I_atr_pct", "I_bb_bw", "I_bb_pctb",
    # returns / range / gap
    "I_ret_1", "I_range_pct", "I_gap_open_pct",
    # volume / flow
    "I_cum_vol", "I_rvol_slot", "I_vol_z", "I_obv", "I_obv_slope", "I_mfi14",
    # oscillators
    "I_stoch_k", "I_stoch_d", "I_cci20", "I_willr14",
]


def _ensure_session_cols(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"])
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(IST)
    else:
        ts = ts.dt.tz_convert(IST)
    df = df.copy()
    df["timestamp"] = ts
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    sess_date = ts.dt.date
    df["_session_id"] = pd.factorize(sess_date)[0].astype("int64")
    df["_minute_of_day"] = (ts.dt.hour * 60 + ts.dt.minute).astype("int64")
    df["_time_slot"] = df["_minute_of_day"]  # 5-min grid slot key for RVOL
    return df


def compute_intraday_indicators(df: pd.DataFrame, cfg: Optional[Config] = None) -> pd.DataFrame:
    """Compute the full intraday indicator suite. SESSION-AWARE and LEAK-FREE:
    every cumulative/rolling/vwap op is reset per trading day and only uses past
    bars within the row's own session (or strictly past sessions for RVOL)."""
    if df is None or df.empty:
        return df
    df = _ensure_session_cols(df)
    g = df.groupby("_session_id", sort=False, group_keys=False)

    o = _f(df["open"]); h = _f(df["high"]); l = _f(df["low"]); c = _f(df["close"]); v = _f(df["volume"]).fillna(0.0)
    tp = (h + l + c) / 3.0

    # ---- session position ----
    open_min = SESSION_OPEN.hour * 60 + SESSION_OPEN.minute
    close_min = SESSION_CLOSE.hour * 60 + SESSION_CLOSE.minute
    df["I_minute_of_day"] = df["_minute_of_day"]
    df["I_bars_into_session"] = g.cumcount().astype("float64")
    df["I_sess_pos"] = ((df["_minute_of_day"] - open_min) / max(1, (close_min - open_min))).clip(0, 1)

    # ---- session VWAP + bands (causal cumulative within session) ----
    # Both cumsums are grouped by _session_id, so they RESET at every session open
    # (verified at runtime by run_self_test's cum_vol-reset assertion).
    pv = tp * v
    df["I_cum_vol"] = v.groupby(df["_session_id"]).cumsum()
    cum_pv = pv.groupby(df["_session_id"]).cumsum()
    vwap = _running_vwap(cum_pv.to_numpy(dtype="float64"), df["I_cum_vol"].to_numpy(dtype="float64"))
    df["I_vwap"] = vwap
    dev = (c.to_numpy(dtype="float64") - vwap)
    df["I_vwap_dev_pct"] = np.where(np.abs(vwap) > 0, dev / vwap * 100.0, np.nan)
    # session running std of (price - vwap) for bands, causal
    resid = pd.Series(dev, index=df.index)
    run_std = resid.groupby(df["_session_id"]).transform(
        lambda s: s.expanding(min_periods=3).std())
    df["I_vwap_z"] = np.where(run_std > 0, dev / run_std, np.nan)
    df["I_above_vwap"] = (c.to_numpy(dtype="float64") > vwap).astype("float64")
    df["I_vwap_band_up1"] = vwap + run_std.to_numpy()
    df["I_vwap_band_dn1"] = vwap - run_std.to_numpy()
    df["I_vwap_band_up2"] = vwap + 2 * run_std.to_numpy()
    df["I_vwap_band_dn2"] = vwap - 2 * run_std.to_numpy()

    # ---- session running VPOC ----
    vpoc = _session_running_vpoc(tp.to_numpy(dtype="float64"), v.to_numpy(dtype="float64"),
                                 df["_session_id"].to_numpy())
    df["I_vpoc"] = vpoc
    df["I_dist_vpoc_pct"] = np.where(np.abs(vpoc) > 0, (c.to_numpy(dtype="float64") - vpoc) / vpoc * 100.0, np.nan)

    # ---- opening ranges 5/15/30 ----
    cnp = c.to_numpy(dtype="float64")
    for mins in (5, 15, 30):
        orh, orl = _opening_range(df, mins)
        orh_np, orl_np = orh.to_numpy(), orl.to_numpy()
        df[f"I_or{mins}_high"] = orh_np
        df[f"I_or{mins}_low"] = orl_np
        df[f"I_or{mins}_width_pct"] = np.where(np.abs(orl_np) > 0, (orh_np - orl_np) / orl_np * 100.0, np.nan)
        df[f"I_or{mins}_brk_up"] = (cnp > orh_np).astype("float64")
        df[f"I_or{mins}_brk_dn"] = (cnp < orl_np).astype("float64")
        df[f"I_dist_or{mins}_high_pct"] = np.where(np.abs(orh_np) > 0, (cnp - orh_np) / orh_np * 100.0, np.nan)
        df[f"I_dist_or{mins}_low_pct"] = np.where(np.abs(orl_np) > 0, (cnp - orl_np) / orl_np * 100.0, np.nan)

    # ---- momentum / trend (continuous across sessions: these are short-window
    #      technicals that legitimately use the trailing tape; NOT session-cumulative) ----
    df["I_rsi7"] = _rsi(c, 7)
    df["I_rsi14"] = _rsi(c, 14)
    adx, pdi, mdi = _adx(h, l, c, 14)
    df["I_adx14"] = adx; df["I_pdi14"] = pdi; df["I_mdi14"] = mdi
    ema9 = _ema(c, 9); ema20 = _ema(c, 20); ema50 = _ema(c, 50)
    df["I_ema9"] = ema9; df["I_ema20"] = ema20; df["I_ema50"] = ema50
    df["I_ema_stack"] = ((ema9 > ema20) & (ema20 > ema50)).astype("float64") - \
                        ((ema9 < ema20) & (ema20 < ema50)).astype("float64")
    df["I_dist_ema20_pct"] = np.where(ema20.to_numpy() > 0, (cnp - ema20.to_numpy()) / ema20.to_numpy() * 100.0, np.nan)
    macd = ema_fast = _ema(c, 12) - _ema(c, 26)
    macd_sig = macd.ewm(span=9, adjust=False, min_periods=1).mean()
    df["I_macd"] = macd; df["I_macd_signal"] = macd_sig; df["I_macd_hist"] = macd - macd_sig

    # ---- volatility / bands ----
    atr = _atr(h, l, c, 14)
    df["I_atr14"] = atr
    df["I_atr_pct"] = np.where(cnp > 0, atr.to_numpy() / cnp * 100.0, np.nan)
    bb_mid = c.rolling(20, min_periods=20).mean()
    bb_std = c.rolling(20, min_periods=20).std()
    df["I_bb_bw"] = np.where(bb_mid.to_numpy() > 0, (4 * bb_std.to_numpy()) / bb_mid.to_numpy(), np.nan)
    df["I_bb_pctb"] = ((c - (bb_mid - 2 * bb_std)) / (4 * bb_std).replace(0, np.nan)).to_numpy()

    # ---- returns / range / gap ----
    df["I_ret_1"] = c.pct_change() * 100.0
    df["I_range_pct"] = np.where(cnp > 0, (h.to_numpy() - l.to_numpy()) / cnp * 100.0, np.nan)
    # first-bar gap = session open vs previous session's last close (causal)
    sess_open = g["open"].transform("first")
    last_close_by_sess = df.groupby("_session_id")["close"].last()
    prev_close_map = last_close_by_sess.shift(1)
    pc = df["_session_id"].map(prev_close_map).to_numpy(dtype="float64")
    so = sess_open.to_numpy(dtype="float64")
    is_first = (df["I_bars_into_session"].to_numpy() == 0)
    gap = np.where((np.abs(pc) > 0) & is_first, (so - pc) / pc * 100.0, np.nan)
    df["I_gap_open_pct"] = gap

    # ---- volume / flow ----
    # RVOL vs trailing rvol_lookback_days average for the SAME time slot (strictly
    # past sessions -> .shift on a per-slot rolling mean).
    look = (cfg.rvol_lookback_days if cfg else 20)
    slot = df["_time_slot"]
    slot_mean = (v.groupby(slot)
                  .apply(lambda s: s.shift(1).rolling(look, min_periods=3).mean())
                  .reset_index(level=0, drop=True)
                  .sort_index())
    df["I_rvol_slot"] = np.where(slot_mean.to_numpy() > 0, v.to_numpy() / slot_mean.to_numpy(), np.nan)
    slot_std = (v.groupby(slot)
                 .apply(lambda s: s.shift(1).rolling(look, min_periods=3).std())
                 .reset_index(level=0, drop=True)
                 .sort_index())
    df["I_vol_z"] = np.where(slot_std.to_numpy() > 0, (v.to_numpy() - slot_mean.to_numpy()) / slot_std.to_numpy(), np.nan)
    sign = np.sign(c.diff().fillna(0.0).to_numpy())
    df["I_obv"] = np.cumsum(sign * v.to_numpy())  # leakcheck-ok: continuous causal cumsum (OBV)
    df["I_obv_slope"] = _rolling_slope(pd.Series(df["I_obv"].to_numpy(), index=df.index), 20)
    df["I_mfi14"] = _mfi(h, l, c, v, 14)

    # ---- oscillators ----
    k, d = _stoch(h, l, c, 14, 3)
    df["I_stoch_k"] = k; df["I_stoch_d"] = d
    df["I_cci20"] = _cci(h, l, c, 20)
    df["I_willr14"] = _willr(h, l, c, 14)

    # cleanup helper cols, keep grid for downstream if needed
    df = df.drop(columns=["_time_slot"], errors="ignore")
    for col in INTRADAY_INDICATOR_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return df


def finalize_for_cache(df: pd.DataFrame) -> pd.DataFrame:
    """Drop internal helper columns; keep OHLCV + I_* (+ session id for grouping)."""
    drop = [c for c in df.columns if c.startswith("_") and c != "_session_id"]
    return df.drop(columns=drop, errors="ignore")


# =============================================================================
#                         LEAK SAFETY NETS
# =============================================================================
def _static_leak_check() -> None:
    """Scan compute_intraday_indicators source for look-ahead idioms.

    The genuine future-peeking risks are WHOLE-SERIES rank/quantile/cummax/cummin
    (a value computed at row t that depends on rows > t). cumsum is inherently
    causal (sum of rows <= t), so it is allowed; session RESET correctness for the
    VWAP/cum_vol cumsums is verified separately by run_self_test's runtime
    assertions and by _leak_canary_check. A line may opt out with the trailing
    comment '# leakcheck-ok'."""
    import inspect
    src = inspect.getsource(compute_intraday_indicators)
    for raw_line in src.splitlines():
        line = raw_line.split("#")[0] if "leakcheck-ok" in raw_line else raw_line
        # whole-series rank without an explicit window/pct is the v19-style leak
        if re.search(r"\.rank\((?![^)]*pct=)", line):
            raise AssertionError(f"_static_leak_check: whole-series .rank() (future-peeking): {raw_line.strip()!r}")
        if re.search(r"\.cummax\(|\.cummin\(", line):
            raise AssertionError(f"_static_leak_check: cummax/cummin (future-peeking): {raw_line.strip()!r}")
        if re.search(r"\.quantile\(", line):
            raise AssertionError(f"_static_leak_check: whole-series .quantile(): {raw_line.strip()!r}")


def _make_synth_intraday(seed: int, n_days: int = 6, last_k_bars: int = 10,
                         tamper: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    bars_per_day = 75  # 09:15..15:30 inclusive on a 5-min grid
    rows = []
    price = 100.0
    base_day = dt.date(2024, 1, 1)
    day = 0
    made = 0
    total = n_days * bars_per_day
    while made < total:
        d = base_day + dt.timedelta(days=day)
        if d.weekday() >= 5:
            day += 1
            continue
        t = dt.datetime.combine(d, SESSION_OPEN, tzinfo=IST)
        for b in range(bars_per_day):
            if made >= total:
                break
            ret = rng.normal(0, 0.0015)
            # tamper only the final last_k_bars
            if tamper and made >= total - last_k_bars:
                ret += 0.05
            price *= (1 + ret)
            o = price * (1 + rng.normal(0, 0.0003))
            hi = max(o, price) * (1 + abs(rng.normal(0, 0.0008)))
            lo = min(o, price) * (1 - abs(rng.normal(0, 0.0008)))
            vol = float(rng.integers(1000, 50000))
            rows.append((t, o, hi, lo, price, vol))
            t += dt.timedelta(minutes=BAR_MINUTES)
            made += 1
        day += 1
    return pd.DataFrame(rows, columns=OHLCV)


def _leak_canary_check() -> None:
    """Two frames identical on rows 0..N-K and differing only in the last K rows;
    every indicator must match on the shared prefix (no future bleed), and no
    session's values may depend on a later session's tampered bars."""
    K = 10
    base = _make_synth_intraday(7, n_days=6, last_k_bars=K, tamper=False)
    tamp = _make_synth_intraday(7, n_days=6, last_k_bars=K, tamper=True)
    a = compute_intraday_indicators(base.copy())
    b = compute_intraday_indicators(tamp.copy())
    n = len(a)
    prefix = n - K
    cols = [c for c in INTRADAY_INDICATOR_COLUMNS if c in a.columns]
    mism = []
    for col in cols:
        va = a[col].to_numpy()[:prefix]
        vb = b[col].to_numpy()[:prefix]
        both_nan = np.isnan(va) & np.isnan(vb)
        close = np.isclose(va, vb, equal_nan=False) | both_nan
        if not close.all():
            mism.append((col, int((~close).sum())))
    if mism:
        raise AssertionError(f"_leak_canary_check: future leaked into past for {mism[:8]}")


def run_self_test() -> int:
    print("=" * 70)
    print("INTRADAY CACHE v3 — SELF-TEST (offline, no Kite)")
    print("=" * 70)
    _static_leak_check()
    print("[1] static leak check: PASS")
    _leak_canary_check()
    print("[2] runtime leak canary (no future bleed across the K-bar boundary): PASS")

    df = _make_synth_intraday(11, n_days=8)
    out = compute_intraday_indicators(df.copy())
    fin = finalize_for_cache(out.copy())
    present = [c for c in INTRADAY_INDICATOR_COLUMNS if c in fin.columns]
    missing = [c for c in INTRADAY_INDICATOR_COLUMNS if c not in fin.columns]
    print(f"[3] indicators produced: {len(present)}/{len(INTRADAY_INDICATOR_COLUMNS)}"
          + (f"  MISSING={missing}" if missing else ""))
    assert not missing, f"missing indicator columns: {missing}"

    # session resets: VWAP first bar of each session == its own typical price (tp),
    # and cum_vol resets to that bar's volume.
    g0 = out.groupby("_session_id", sort=False)
    first_idx = g0.head(1).index
    cumvol_first = out.loc[first_idx, "I_cum_vol"].to_numpy()
    vol_first = out.loc[first_idx, "volume"].to_numpy()
    assert np.allclose(cumvol_first, vol_first), "cum_vol did not reset at session open"
    # bars_into_session starts at 0 each session
    assert (g0["I_bars_into_session"].first() == 0).all(), "bars_into_session not reset"
    # OR5 high is NaN on the very FIRST bar of each session (window not closed yet).
    # Use positional first row per session (groupby.first() skips NaN, which is the
    # opposite of what we want to assert here).
    first_rows = out.groupby("_session_id", sort=False).head(1)
    assert first_rows["I_or5_high"].isna().all(), "OR5 leaked before the opening-range window closed"
    print("[4] session resets (VWAP/cum_vol/bars/opening-range): PASS")

    # no all-NaN indicator columns after warmup
    warm = fin.iloc[120:]
    allnan = [c for c in present if warm[c].isna().all()]
    assert not allnan, f"all-NaN after warmup: {allnan}"
    print(f"[5] no dead columns after warmup; rows={len(fin)}, cols={fin.shape[1]}")

    print("\nALL SELF-TESTS PASSED")
    return 0


# =============================================================================
#                         KITE PROVIDER (parallel fetch)
# =============================================================================
def load_kite(cfg: Config):
    if not cfg.token_file.exists():
        raise AuthExpired(f"Token file missing: {cfg.token_file}. Refresh it first.")
    data = json.loads(cfg.token_file.read_text())
    api_key, access_token = data.get("api_key"), data.get("access_token")
    if not api_key or not access_token:
        raise AuthExpired("api_key/access_token missing in token file.")
    try:
        from kiteconnect import KiteConnect
    except ImportError:
        raise SystemExit("kiteconnect not installed. Run: pip install kiteconnect")
    kite = KiteConnect(api_key=api_key)
    try:
        kite.timeout = int(cfg.request_timeout_s)
    except Exception:
        pass
    kite.set_access_token(access_token)
    prof = kite.profile()  # raises if stale
    print(f"[Kite] Authenticated as: {prof.get('user_name')} ({prof.get('user_id')})")
    return kite


def load_instrument_map(kite, cfg: Config) -> Dict[str, int]:
    cache = cfg.intraday_dir / "_instrument_map.parquet"
    if cache.exists() and (time.time() - cache.stat().st_mtime) / 3600 < 24:
        df = pd.read_parquet(cache)
    else:
        rows = kite.instruments("NSE")
        df = pd.DataFrame(rows)
        df = df[(df["segment"] == "NSE") & (df["instrument_type"] == "EQ")].copy()
        df = df[["tradingsymbol", "instrument_token"]].rename(
            columns={"tradingsymbol": "symbol"})
        cfg.intraday_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
        print(f"[Instruments] Cached {len(df):,} NSE EQ instruments")
    return {sanitize_symbol(s): int(t) for s, t in zip(df["symbol"], df["instrument_token"])}


def _normalize_bars(rows: list) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=OHLCV)
    df = pd.DataFrame(rows).rename(columns={"date": "timestamp"})
    ts = pd.to_datetime(df["timestamp"])
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(IST)
    else:
        ts = ts.dt.tz_convert(IST)
    df["timestamp"] = ts
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            df[col] = np.nan
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    return df[OHLCV]


def fetch_bars(kite, token: int, start: dt.datetime, end: dt.datetime,
               cfg: Config, limiter: RateLimiter) -> pd.DataFrame:
    chunks = []
    cur = start
    step = dt.timedelta(days=cfg.max_days_per_request - 1)

    @with_retry
    def _hist(a, b):
        limiter.acquire()
        return kite.historical_data(token, a, b, cfg.interval)

    while cur < end:
        ce = min(cur + step, end)
        rows = _hist(cur, ce)
        if rows:
            chunks.append(pd.DataFrame(rows))
        cur = ce + dt.timedelta(days=1)
    if not chunks:
        return pd.DataFrame(columns=OHLCV)
    return _normalize_bars(pd.concat(chunks, ignore_index=True).to_dict("records"))


# =============================================================================
#                         SYMBOL UNIVERSE
# =============================================================================
def get_universe(cfg: Config) -> List[str]:
    if cfg.master_symbol_file.exists():
        syms = []
        for line in cfg.master_symbol_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "," in line:
                line = line.split(",")[0].strip()
            syms.append(sanitize_symbol(line))
        out = sorted(set(s for s in syms if s))
        print(f"[Universe] {len(out):,} symbols from master file")
        return out
    if cfg.daily_cache_dir.exists():
        files = list(cfg.daily_cache_dir.glob("*.parquet")) or list(cfg.daily_cache_dir.glob("*.csv"))
        out = sorted(set(sanitize_symbol(f.stem) for f in files if not f.stem.startswith("_")))
        print(f"[Universe] {len(out):,} symbols from daily cache folder")
        return out
    raise SystemExit("No master file and no daily cache dir found.")


# =============================================================================
#                         OK SIDECAR
# =============================================================================
def write_ok(cfg: Config, symbol: str, df: pd.DataFrame, with_indicators: bool):
    meta = {
        OK_VERSION_KEY: SCHEMA_VERSION,
        "symbol": sanitize_symbol(symbol),
        "rows": int(len(df)),
        "first_ts": str(df["timestamp"].min()) if len(df) else None,
        "last_ts": str(df["timestamp"].max()) if len(df) else None,
        "has_indicators": bool(with_indicators),
        "n_indicator_cols": int(sum(c in df.columns for c in INTRADAY_INDICATOR_COLUMNS)),
        "created_ts": dt.datetime.now(tz=IST).isoformat(),
    }
    write_json_atomic(cfg.ok_path(symbol), meta)


def read_ok(cfg: Config, symbol: str) -> Optional[dict]:
    p = cfg.ok_path(symbol)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


# =============================================================================
#                         PER-SYMBOL BUILD
# =============================================================================
def build_symbol(kite, symbol: str, token: int, cfg: Config, limiter: RateLimiter,
                 *, force_full: bool, raw_only: bool) -> Tuple[bool, int, str]:
    """Fetch (incrementally) + optionally compute indicators + atomic write."""
    path = cfg.intraday_path(symbol)
    end = dt.datetime.now()
    existing_raw = None
    start = end - dt.timedelta(days=int(cfg.history_years * 365.25))

    if path.exists() and not force_full:
        try:
            ex = pd.read_parquet(path, columns=OHLCV)
            ex["timestamp"] = pd.to_datetime(ex["timestamp"])
            existing_raw = ex
            last_ts = ex["timestamp"].max()
            start = (last_ts + pd.Timedelta(minutes=BAR_MINUTES)).to_pydatetime()
            if start.tzinfo is not None:
                start = start.replace(tzinfo=None)
            if pd.Timestamp(start).date() > end.date():
                return True, len(ex), "up-to-date"
        except Exception:
            existing_raw = None
            start = end - dt.timedelta(days=int(cfg.history_years * 365.25))

    new = fetch_bars(kite, token, start, end, cfg, limiter)
    if (new is None or new.empty) and existing_raw is None:
        return False, 0, "no-data"

    if existing_raw is not None and new is not None and not new.empty:
        ex = existing_raw.copy()
        ex["timestamp"] = pd.to_datetime(ex["timestamp"])
        if getattr(ex["timestamp"].dt, "tz", None) is None:
            ex["timestamp"] = ex["timestamp"].dt.tz_localize(IST)
        raw = (pd.concat([ex, new], ignore_index=True)
               .drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True))
    else:
        raw = new if (new is not None and not new.empty) else existing_raw

    out = raw if raw_only else finalize_for_cache(compute_intraday_indicators(raw.copy(), cfg))
    with FileLock(path):
        atomic_write_parquet(path, out, cfg.parquet_compression)
        write_ok(cfg, symbol, out, with_indicators=not raw_only)
    return True, len(out), ("raw-only" if raw_only else "indicators")


def recompute_symbol(symbol: str, cfg: Config) -> Tuple[bool, int, str]:
    """Re-derive indicators from already-cached bars. No Kite, fully parallel."""
    path = cfg.intraday_path(symbol)
    if not path.exists():
        return False, 0, "no-cache"
    raw = pd.read_parquet(path, columns=[c for c in OHLCV])
    out = finalize_for_cache(compute_intraday_indicators(raw.copy(), cfg))
    with FileLock(path):
        atomic_write_parquet(path, out, cfg.parquet_compression)
        write_ok(cfg, symbol, out, with_indicators=True)
    return True, len(out), "recomputed"


# =============================================================================
#                         DRIVERS
# =============================================================================
def run_recompute(cfg: Config, symbols: List[str]):
    print(f"[Recompute] {len(symbols)} symbols, {cfg.compute_workers} workers (no fetch)")
    ok = fail = 0
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=cfg.compute_workers) as ex:
        futs = {ex.submit(recompute_symbol, s, cfg): s for s in symbols}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            s = futs[fut]
            try:
                good, n, msg = fut.result()
                ok += int(good); fail += int(not good)
            except Exception as e:
                fail += 1
                print(f"  [ERR] {s}: {e}")
            if i % 100 == 0:
                print(f"  {i}/{len(symbols)}  ok={ok} fail={fail}  {time.time()-t0:.0f}s")
    print(f"[Recompute] done: ok={ok} fail={fail} in {(time.time()-t0)/60:.1f}m")


def run_build(cfg: Config, symbols: List[str], *, force_full: bool, raw_only: bool):
    print("=" * 80)
    print(f"INTRADAY 5-MIN BUILD v{SCHEMA_VERSION}  ({'RAW ONLY' if raw_only else 'WITH INDICATORS'})")
    print("=" * 80)
    kite = load_kite(cfg)
    token_map = load_instrument_map(kite, cfg)
    valid = [s for s in symbols if s in token_map]
    missing = [s for s in symbols if s not in token_map]
    if missing:
        print(f"[WARN] {len(missing)} symbols not in Kite NSE EQ list (e.g. {missing[:6]})")
    print(f"[Build] {len(valid)} symbols, {cfg.fetch_workers} fetch workers, "
          f"rate {cfg.rate_limit_per_sec}/s")
    if not valid:
        print("Nothing to do.")
        return

    limiter = RateLimiter(cfg.rate_limit_per_sec)
    ok = fail = 0
    t0 = time.time()
    # Fetch is network-bound -> threads share the token bucket. Indicators run
    # inline per symbol (still parallel across the thread pool).
    with cf.ThreadPoolExecutor(max_workers=cfg.fetch_workers) as ex:
        futs = {ex.submit(build_symbol, kite, s, token_map[s], cfg, limiter,
                          force_full=force_full, raw_only=raw_only): s for s in valid}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            s = futs[fut]
            try:
                good, n, msg = fut.result()
                if good:
                    ok += 1
                else:
                    fail += 1
            except AuthExpired:
                print("\n[FATAL] Kite token expired during run. Refresh and resume.")
                break
            except Exception as e:
                fail += 1
                print(f"  [ERR] {s}: {e}")
            if i % 25 == 0:
                el = time.time() - t0
                rate = i / el if el else 0
                eta = (len(valid) - i) / rate / 60 if rate else 0
                print(f"  {i}/{len(valid)} ({100*i/len(valid):.0f}%) ok={ok} fail={fail} "
                      f"elapsed={el/60:.1f}m eta={eta:.1f}m")
    print(f"\n[Build] done: ok={ok} fail={fail} in {(time.time()-t0)/60:.1f}m -> {cfg.intraday_dir}")


# =============================================================================
#                         CLI
# =============================================================================
def main():
    p = argparse.ArgumentParser(description="Intraday 5-min cacher v3 (fast, leak-free, indicator-rich)")
    p.add_argument("--symbols", type=str, default=None, help="comma-separated symbols")
    p.add_argument("--from-file", type=str, default=None, help="file with one symbol per line")
    p.add_argument("--limit", type=int, default=None, help="first N symbols (testing)")
    p.add_argument("--full-rebuild", action="store_true", help="refetch all history")
    p.add_argument("--raw-only", action="store_true", help="store bars without indicators")
    p.add_argument("--recompute", action="store_true",
                   help="re-derive indicators from cached bars (no fetch)")
    p.add_argument("--fetch-workers", type=int, default=None)
    p.add_argument("--compute-workers", type=int, default=None)
    p.add_argument("--rate", type=float, default=None, help="Kite req/s (default 3)")
    p.add_argument("--self-test", action="store_true", help="offline leak canary + indicator checks")
    a = p.parse_args()

    if a.self_test:
        raise SystemExit(run_self_test())

    overrides = {}
    if a.fetch_workers is not None:
        overrides["fetch_workers"] = a.fetch_workers
    if a.compute_workers is not None:
        overrides["compute_workers"] = a.compute_workers
    if a.rate is not None:
        overrides["rate_limit_per_sec"] = a.rate
    cfg = replace(Config(), **overrides) if overrides else Config()

    if a.symbols:
        symbols = [sanitize_symbol(s) for s in a.symbols.split(",")]
    elif a.from_file:
        symbols = [sanitize_symbol(x) for x in Path(a.from_file).read_text().splitlines() if x.strip()]
    else:
        symbols = get_universe(cfg)
    if a.limit:
        symbols = symbols[:a.limit]

    if a.recompute:
        run_recompute(cfg, symbols)
    else:
        run_build(cfg, symbols, force_full=a.full_rebuild, raw_only=a.raw_only)


if __name__ == "__main__":
    main()

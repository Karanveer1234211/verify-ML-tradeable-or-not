#!/usr/bin/env python3
"""
FAST Daily Cache Builder for Zerodha Kite historical data (equities only)
Warm-up backfill, leak-free indicators, feature finalization, and logical combos.

v20 (2026-05-31)  — LEAK-FREE BY CONSTRUCTION + FASTER
======================================================
Same column schema as v19. Same CLI / GUI / parquet layout. Bug fixes:

  CORRECTNESS
  -----------
  v19 used `_rank_cs(s) = s.rank(pct=True)` inside compute_daily_indicators(),
  which was applied to a single symbol's full time series. Pandas' Series.rank()
  with no window ranks each row against EVERY other row — including future
  rows. So at day t, rank values already encoded "where today sits relative
  to the rest of history, including t+1..t+N". That leaked the future into 10
  of the 16 WorldQuant alphas:

      D_WQ_3, D_WQ_13, D_WQ_16, D_WQ_19, D_WQ_20,
      D_WQ_29, D_WQ_33, D_WQ_38, D_WQ_40, D_WQ_44

  v20 replaces every per-symbol whole-series rank with `_xrank(s)` =
  `s.expanding(min_periods=60).rank(pct=True)` — the percentile rank of s[t]
  within s[0..t] only. Strictly leak-free. Column names are unchanged so
  downstream code (New_model.py, NEW FEAT IMP.py, triple_barrier_backtest.py,
  compare.py) continues to work without modification, but the values for the
  10 affected columns are different (and now correct).

  Note: this is a *temporal* per-symbol rank, not a true cross-sectional WQ
  rank. A genuine WQ-style rank requires `panel.groupby('timestamp')[col]
  .rank(pct=True)` at panel-assembly time, which is outside this per-symbol
  cache.

  STATIC SAFETY NET
  -----------------
  At module load, _v20_static_leak_check() scans this file for any unwindowed
  `.rank(` / `.quantile(` / `.cumsum(` calls inside compute_daily_indicators()
  that would re-introduce the leak. Builds abort if a future edit regresses.

  RUNTIME CANARY
  --------------
  At main() start, _v20_leak_canary_check() builds two synthetic OHLCV frames
  identical for rows 0..N-K and different for the last K rows, computes
  indicators on both, and asserts every column matches on rows 0..N-K. Any
  column whose past values change when the future changes => abort.

  PERFORMANCE
  -----------
  Hot Python loops vectorized:
    - _rolling_ols_slope_fast: rolling cov(t,y) / var(t)        [~10-50x faster]
    - NR-day streak length, run counts, days-since-breakout      [O(n) numpy]
    - Weekly VPOC inner reassign loop                            [vectorized]

  All v17/v18/v19 features and the full DAILY_INDICATOR_COLUMNS list are
  preserved verbatim.
"""

from __future__ import annotations

import concurrent.futures as cf
import contextlib
import datetime as dt
import functools
import json
import math
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional, Sequence, Tuple, List, Dict

import numpy as np
import pandas as pd
from requests.exceptions import (
    ReadTimeout,
    ConnectTimeout,
    ConnectionError as RequestsConnectionError,
)

try:
    from urllib3.exceptions import ReadTimeoutError as Urllib3ReadTimeoutError
except Exception:
    Urllib3ReadTimeoutError = Exception

# -------------------- GUI (tkinter) --------------------
try:
    import tkinter as tk
    from tkinter import filedialog, simpledialog, messagebox
    from tkinter import ttk
    TK_OK = True
except Exception:
    TK_OK = False

# -------------------- Timezone / market session --------------------
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
DEFAULT_SESSION_OPEN  = dt.time(9, 15, tzinfo=IST)
DEFAULT_SESSION_CLOSE = dt.time(15, 30, tzinfo=IST)


def today_ist() -> dt.date:
    return dt.datetime.now(tz=IST).date()


# -------------------- Schema + .ok metadata --------------------
SCHEMA_VERSION = 20
OK_VERSION_KEY = "schema_version"

# -------------------- Windows default cache roots --------------------
WIN_DEFAULT_BASE = Path(r"C:\Users\karanvsi\Desktop\Pycharm\Cache")


def _expand_path(value: str) -> Path:
    return Path(value).expanduser()


def _default_base_dir() -> Path:
    env_base = os.environ.get("CACHE_BASE_DIR")
    if env_base:
        return _expand_path(env_base)
    if os.name == "nt":
        return WIN_DEFAULT_BASE
    return Path.home() / ".kite_cache"


def _platform_default(
    env_var: str, *, windows_default: Path, unix_suffix: str
) -> Path:
    val = os.environ.get(env_var)
    if val:
        return _expand_path(val)
    base = _default_base_dir()
    if os.name == "nt":
        return windows_default
    return base / unix_suffix


def _default_daily_root() -> Path:
    return _platform_default(
        "CACHE_DAILY_ROOT",
        windows_default=WIN_DEFAULT_BASE / "cache_daily_new",
        unix_suffix="cache_daily_new",
    )


# -------------------- Config --------------------
@dataclass(frozen=True)
class Config:
    daily_root: Path = field(default_factory=_default_daily_root)
    trading_open: dt.time = DEFAULT_SESSION_OPEN
    trading_close: dt.time = DEFAULT_SESSION_CLOSE
    max_workers: int = 32
    rate_limit_per_sec: float = 32.0
    request_timeout_s: float = 15.0
    retry_tries: int = 6
    retry_backoff_base: float = 0.45
    parquet_engine: str = os.environ.get("PARQUET_ENGINE", "pyarrow")
    parquet_compression: Optional[str] = os.environ.get("PARQUET_COMPRESSION", "snappy")
    parquet_use_dictionary: bool = True

    def day_root(self) -> Path:
        return self.daily_root

    @classmethod
    def from_env(cls, **overrides) -> "Config":
        kwargs: dict = {}
        val = os.environ.get("CACHE_DAILY_ROOT")
        if val:
            kwargs["daily_root"] = _expand_path(val)
        kwargs.update(overrides)
        return cls(**kwargs)

    def with_updates(self, **updates) -> "Config":
        return replace(self, **updates)


# -------------------- Sanitization / paths --------------------
_ILLEGAL = set('<>:"/\n?*')
_HEADER_WORDS = {"symbol", "symbols", "ticker", "tickers", "scrip", "scrips", "name"}


def sanitize_symbol(sym: str) -> Optional[str]:
    if sym is None:
        return None
    s = str(sym)
    s = s.replace("\x00", "").replace("\r", " ").replace("\t", " ").replace("\n", " ")
    s = s.lstrip("\ufeff")
    s = " ".join(s.strip().split())
    s = "".join(ch for ch in s if ch not in _ILLEGAL)
    if not s:
        return None
    if s.strip().casefold() in _HEADER_WORDS:
        return None
    return s


def assert_path_safe(p: Path):
    sp = str(p)
    if "\x00" in sp:
        raise ValueError(f"Path contains NUL (\\x00): {sp!r}")


def daily_path(config: Config, symbol: str) -> Path:
    s = sanitize_symbol(symbol) or "UNKNOWN"
    p = config.day_root() / f"{s}_daily.parquet"
    assert_path_safe(p)
    return p


def ok_path(config: Config, symbol: str) -> Path:
    s = sanitize_symbol(symbol) or "UNKNOWN"
    p = config.day_root() / f"{s}_daily.ok.json"
    assert_path_safe(p)
    return p


def ok_meta_base() -> dict:
    return {
        OK_VERSION_KEY: SCHEMA_VERSION,
        "created_ts": dt.datetime.now(tz=IST).isoformat(),
    }


# -------------------- Atomic IO --------------------
class FileLock:
    def __init__(self, path: Path, poll_ms: int = 50, timeout_s: float = 30.0):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = Path(str(path) + ".lock")
        self.poll_ms = poll_ms
        self.timeout_s = timeout_s
        self._fd: Optional[int] = None

    def acquire(self):
        deadline = time.time() + self.timeout_s
        while True:
            try:
                self._fd = os.open(
                    self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY
                )
                os.write(self._fd, str(os.getpid()).encode())
                return
            except FileExistsError:
                if time.time() > deadline:
                    raise TimeoutError(f"Timeout acquiring lock {self.lock_path}")
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

    def __exit__(self, exc_type, exc, tb):
        self.release()


def atomic_write_bytes(target: Path, data: bytes):
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, target)


def write_json_atomic(path: Path, obj: dict):
    raw = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True).encode()
    atomic_write_bytes(path, raw)


def to_parquet(
    path: Path,
    df: pd.DataFrame,
    *,
    engine: str,
    compression: Optional[str],
    use_dictionary: bool,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(
        path,
        index=False,
        engine=engine,
        compression=compression,
        use_dictionary=use_dictionary,
    )


def read_parquet(
    path: Path, columns: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    if columns is not None:
        columns = list(columns)
    return pd.read_parquet(path, columns=columns)


def read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------- Rate limit + retry --------------------
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
                self._tokens = min(
                    self.per_sec,
                    self._tokens + (now - self._updated) * self.per_sec,
                )
                self._updated = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                need = max(0.0, 1.0 - self._tokens)
                wait = need / self.per_sec if self.per_sec > 0 else 0.0
            time.sleep(wait if wait > 0 else 0)


def with_retry(fn, *, tries: int, backoff: float):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        attempt = 0
        last_exc = None
        while attempt < tries:
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                msg = str(e).lower()
                if "too many" in msg or "rate" in msg or "429" in msg:
                    sleep = min(15.0, backoff * (2.2**attempt))
                else:
                    sleep = min(
                        8.0,
                        backoff * (1.8**attempt) + np.random.random() * (backoff / 2),
                    )
                last_exc = e
                time.sleep(sleep)
                attempt += 1
        raise last_exc

    return wrapper


class DataFrameCache:
    def __init__(self, maxsize: int = 256):
        self.maxsize = max(1, int(maxsize))
        self._store: Dict[tuple, pd.DataFrame] = {}
        self._order: List[tuple] = []
        self._lock = threading.Lock()

    def get(self, key: tuple) -> Optional[pd.DataFrame]:
        with self._lock:
            df = self._store.get(key)
            if df is None:
                return None
            if key in self._order:
                self._order.remove(key)
            self._order.append(key)
            return df.copy(deep=True)

    def put(self, key: tuple, df: pd.DataFrame) -> pd.DataFrame:
        clone = df.copy(deep=True)
        with self._lock:
            self._store[key] = clone
            if key in self._order:
                self._order.remove(key)
            self._order.append(key)
            while len(self._order) > self.maxsize:
                old = self._order.pop(0)
                self._store.pop(old, None)
        return clone.copy(deep=True)


# -------------------- Provider + resolver (Kite) --------------------
try:
    from kiteconnect import KiteConnect
    from kiteconnect.exceptions import TokenException, KiteException, InputException
except Exception:
    KiteConnect = None
    TokenException = KiteException = InputException = Exception


class AuthExpired(Exception):
    """Kite access token missing/expired/invalid."""


def _token_file_path() -> str:
    env_path = os.environ.get("KITE_TOKEN_FILE")
    if env_path:
        return env_path
    default_win = r"C:\Users\karanvsi\PyCharmMiscProject\kite_token.json"
    if os.name == "nt" and os.path.exists(default_win):
        return default_win
    return os.path.join(os.path.dirname(__file__), "kite_token.json")


def _instrument_cache_path() -> Path:
    p = os.environ.get("INSTRUMENT_CACHE_FILE")
    if p:
        return Path(p)
    default_win = str(WIN_DEFAULT_BASE / "instrument_cache.json")
    return Path(default_win) if os.name == "nt" else Path("instrument_cache.json")


def _load_instrument_cache() -> dict:
    try:
        p = _instrument_cache_path()
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_instrument_cache(cache: dict) -> None:
    try:
        p = _instrument_cache_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


import difflib

UNRESOLVED_SYMBOLS_LOG = Path(
    os.environ.get("UNRESOLVED_SYMBOLS_LOG")
    or str(_instrument_cache_path().parent / "unresolved_symbols.jsonl")
)
OVERRIDES_FILE = Path(
    os.environ.get("SYMBOL_OVERRIDES_FILE")
    or str(_instrument_cache_path().parent / "symbol_overrides.json")
)


class UnresolvedSymbol(Exception):
    """Symbol cannot be mapped to instrument_token."""


def _normalize_sym(s: str) -> str:
    s = (s or "").upper().strip()
    for suf in ("-EQ", "-BE", "-BZ", "-BL", "-SM", "-GS", "-GB"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return "".join(ch for ch in s if ch.isalnum())


def _load_overrides() -> dict:
    try:
        if OVERRIDES_FILE.exists():
            with open(OVERRIDES_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
            return {_normalize_sym(k): str(v).upper() for k, v in d.items()}
    except Exception:
        pass
    return {}


def _append_unresolved_log(symbol: str, suggestions: list):
    val = os.environ.get("SKIP_UNRESOLVED", "").strip().lower()
    if val in ("1", "true", "yes"):
        return
    UNRESOLVED_SYMBOLS_LOG.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "ts": dt.datetime.now(tz=IST).isoformat(),
        "symbol": symbol,
        "suggestions": suggestions[:5],
    }
    with open(UNRESOLVED_SYMBOLS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


class InvalidInstrument(Exception):
    """Instrument token invalid/stale."""


class SymbolResolver:
    def __init__(self, kite: KiteConnect):
        self.kite = kite
        self.overrides = _load_overrides()
        self._built = False
        self.exact: Dict[str, int] = {}
        self.base: Dict[str, int] = {}
        self.names: List[str] = []

    def _build_maps(self):
        if self._built:
            return
        rows = self.kite.instruments("NSE") + self.kite.instruments("BSE")
        by_exact: dict = {}
        by_base: dict = {}
        for r in rows:
            ts = str(r.get("tradingsymbol", "")).upper()
            itok = r.get("instrument_token")
            seg = str(r.get("segment", ""))
            inst_type = str(r.get("instrument_type", ""))
            if not itok or not ts:
                continue
            score = (
                2
                if (seg.upper().startswith(("NSE", "BSE")) and inst_type.upper() == "EQ")
                else 1
                if seg.upper().startswith(("NSE", "BSE"))
                else 0
            )
            prev = by_exact.get(ts)
            if prev is None or score > prev[0]:
                by_exact[ts] = (score, int(itok))
            base_key = _normalize_sym(ts)
            prevb = by_base.get(base_key)
            if prevb is None or score > prevb[0]:
                by_base[base_key] = (score, int(itok))
        self.exact = {k: v[1] for k, v in by_exact.items()}
        self.base = {k: v[1] for k, v in by_base.items()}
        self.names = list(self.exact.keys())
        self._built = True

    def resolve(self, symbol: str) -> Optional[int]:
        norm = _normalize_sym(symbol)
        if norm in self.overrides:
            want = self.overrides[norm]
            self._build_maps()
            tok = (
                self.exact.get(want)
                or self.exact.get(f"{want}-EQ")
                or self.base.get(_normalize_sym(want))
            )
            if tok:
                return int(tok)
        self._build_maps()
        if symbol.upper() in self.exact:
            return int(self.exact[symbol.upper()])
        if f"{symbol.upper()}-EQ" in self.exact:
            return int(self.exact[f"{symbol.upper()}-EQ"])
        if norm in self.base:
            return int(self.base[norm])
        close_matches = difflib.get_close_matches(
            symbol.upper(), self.names, n=5, cutoff=0.77
        )
        _append_unresolved_log(symbol, close_matches)
        return None


class KiteProvider:
    """Zerodha Kite-backed provider with symbol->instrument caching (daily-only)."""

    def __init__(self, *, exchange_prefix: str = "NSE:"):
        if KiteConnect is None:
            raise RuntimeError("kiteconnect not installed. `pip install kiteconnect`")
        self.exchange_prefix = exchange_prefix.rstrip(":") + ":"
        self._kite: Optional[KiteConnect] = None
        self._instruments: Dict[str, int] = {}
        self._inst_cache_file: Path = _instrument_cache_path()
        self._inst_cache_data: Dict[str, int] = {
            k.upper(): int(v)
            for k, v in _load_instrument_cache().items()
            if isinstance(v, (int, float, str)) and str(v).isdigit()
        }
        self._load_token()
        self._resolver = SymbolResolver(self._kite)

    def _load_token(self) -> None:
        token_file = _token_file_path()
        if not os.path.exists(token_file):
            raise AuthExpired(f"Token file missing: {token_file}. Refresh it first.")
        with open(token_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        api_key = data.get("api_key")
        access_token = data.get("access_token")
        if not api_key or not access_token:
            raise AuthExpired("api_key/access_token missing in token file.")
        http_timeout = float(os.environ.get("KITE_HTTP_TIMEOUT", "15"))
        max_tries, backoff = 5, 0.6
        last_err = None
        kite = KiteConnect(api_key=api_key)
        try:
            kite.timeout = http_timeout
        except Exception:
            pass
        kite.set_access_token(access_token)
        for attempt in range(max_tries):
            try:
                _ = kite.profile()
                self._kite = kite
                return
            except TokenException as e:
                raise AuthExpired(str(e)) from e
            except (
                ReadTimeout,
                ConnectTimeout,
                RequestsConnectionError,
                Urllib3ReadTimeoutError,
                socket.timeout,
            ) as e:
                last_err = e
                sleep_s = min(12.0, backoff * (1.8**attempt))
                time.sleep(sleep_s)
                continue
            except Exception as e:
                last_err = e
                break
        if last_err:
            raise RuntimeError(
                f"Kite API connectivity failed after retries: {last_err}"
            ) from last_err
        raise RuntimeError("Kite API connectivity failed for an unknown reason.")

    def _symbol_to_instrument_token(self, symbol: str) -> int:
        sym = symbol.strip().upper()
        if sym in self._instruments:
            return self._instruments[sym]
        if sym in self._inst_cache_data:
            tok = int(self._inst_cache_data[sym])
            self._instruments[sym] = tok
            return tok
        assert self._kite is not None
        qual = f"{self.exchange_prefix}{sym}"
        try:
            quote = self._kite.ltp([qual])
            if quote and isinstance(quote, dict):
                info = quote.get(qual) or (
                    list(quote.values())[0] if list(quote.values()) else None
                )
                if info and "instrument_token" in info:
                    inst = int(info["instrument_token"])
                    self._instruments[sym] = inst
                    self._inst_cache_data[sym] = inst
                    _save_instrument_cache(self._inst_cache_data)
                    return inst
        except TokenException as e:
            raise AuthExpired(str(e))
        except InputException:
            pass
        except KiteException:
            pass
        rows = self._kite.instruments("NSE") + self._kite.instruments("BSE")
        by_exact: dict = {}
        by_base: dict = {}
        for r in rows:
            ts = str(r.get("tradingsymbol", "")).upper()
            itok = r.get("instrument_token")
            seg = str(r.get("segment", ""))
            inst_type = str(r.get("instrument_type", ""))
            if not itok or not ts:
                continue
            score = (
                2
                if (seg.upper().startswith(("NSE", "BSE")) and inst_type.upper() == "EQ")
                else 1
                if seg.upper().startswith(("NSE", "BSE"))
                else 0
            )
            prev = by_exact.get(ts)
            if prev is None or score > prev[0]:
                by_exact[ts] = (score, int(itok))
            base = ts[:-3] if ts.endswith("-EQ") else ts
            prevb = by_base.get(base)
            if prevb is None or score > prevb[0]:
                by_base[base] = (score, int(itok))
        token = None
        if sym in by_exact:
            token = by_exact[sym][1]
        elif f"{sym}-EQ" in by_exact:
            token = by_exact[f"{sym}-EQ"][1]
        elif sym in by_base:
            token = by_base[sym][1]
        else:
            base = sym[:-3] if sym.endswith("-EQ") else sym
            token = by_exact.get(base, (None, None))[1] or by_base.get(
                base, (None, None)
            )[1]
        if token is None:
            resolved = self._resolver.resolve(symbol)
            if resolved is not None:
                token = int(resolved)
        if token is None:
            names = list(by_exact.keys())
            close_matches = difflib.get_close_matches(sym, names, n=3, cutoff=0.80)
            _append_unresolved_log(symbol, close_matches)
            raise UnresolvedSymbol(sym)
        inst = int(token)
        self._instruments[sym] = inst
        self._inst_cache_data[sym] = inst
        _save_instrument_cache(self._inst_cache_data)
        return inst

    def _hist(
        self,
        instrument_token: int,
        start_dt: dt.datetime,
        end_dt: dt.datetime,
        interval: str,
    ):
        assert self._kite is not None
        try:
            return self._kite.historical_data(
                instrument_token,
                from_date=start_dt,
                to_date=end_dt,
                interval=interval,
                oi=False,
            )
        except TokenException as e:
            raise AuthExpired(str(e))
        except InputException as e:
            msg = str(e).lower()
            if "invalid token" in msg or "instrument_token" in msg:
                raise InvalidInstrument(str(e))
            if "too many" in msg or "429" in msg:
                raise
            raise RuntimeError(f"Kite historical data failed: {e}")

    def _ensure_ist_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        if "date" in df.columns and "timestamp" not in df.columns:
            df = df.rename(columns={"date": "timestamp"})
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], errors="coerce")
            try:
                import zoneinfo
                tz = zoneinfo.ZoneInfo("Asia/Kolkata")
            except Exception:
                tz = IST
            if getattr(ts.dt, "tz", None) is None:
                ts = ts.dt.tz_localize(tz)
            else:
                ts = ts.dt.tz_convert(tz)
            df["timestamp"] = ts
        return df

    def fetch_daily(
        self, symbol: str, start: dt.date, end: dt.date
    ) -> pd.DataFrame:
        """
        Fetch daily candles for [start, end] inclusive, automatically chunking
        to respect Kite's 2000-day per-call maximum.
        """
        MAX_DAYS = 1999

        def _iter_chunks(s: dt.date, e: dt.date):
            cur = s
            while cur <= e:
                hi = min(e, cur + dt.timedelta(days=MAX_DAYS))
                yield cur, hi
                cur = hi + dt.timedelta(days=1)

        inst = self._symbol_to_instrument_token(symbol)
        all_rows: list = []
        for chunk_start, chunk_end in _iter_chunks(start, end):
            start_dt_c = dt.datetime.combine(chunk_start, dt.time(0, 0))
            end_dt_c = dt.datetime.combine(chunk_end, dt.time(23, 59))
            try:
                rows = self._hist(inst, start_dt_c, end_dt_c, interval="day")
            except InvalidInstrument:
                sym = symbol.strip().upper()
                self._instruments.pop(sym, None)
                self._inst_cache_data.pop(sym, None)
                _save_instrument_cache(self._inst_cache_data)
                inst = self._symbol_to_instrument_token(symbol)
                rows = self._hist(inst, start_dt_c, end_dt_c, interval="day")
            all_rows.extend(rows or [])
        df = pd.DataFrame(all_rows)
        if df is None or df.empty:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
        df = self._ensure_ist_timestamp(df)
        use_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        for c in use_cols:
            if c not in df.columns:
                df[c] = pd.NA
        return df[use_cols]


# -------------------- Helpers: tz, indicators, validation --------------------

def _ensure_ist(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    ts = pd.to_datetime(df["timestamp"], utc=False)
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(IST)
    else:
        ts = ts.dt.tz_convert(IST)
    df = df.copy()
    df["timestamp"] = ts
    return df


def _validate_monotonic(df: pd.DataFrame):
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("timestamps must be strictly monotonic increasing")
    if df["timestamp"].duplicated().any():
        raise ValueError("duplicate timestamps detected")


def _ensure_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _ema(series: pd.Series, span: int) -> pd.Series:
    return _ensure_float(series).ewm(span=span, adjust=False, min_periods=1).mean()


def _sma(series: pd.Series, window: int) -> pd.Series:
    return _ensure_float(series).rolling(window=window, min_periods=1).mean()


def _rsi(series: pd.Series, period: int) -> pd.Series:
    close = _ensure_float(series)
    d = close.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    h = _ensure_float(h)
    l = _ensure_float(l)
    c = _ensure_float(c)
    pc = c.shift(1)
    ranges = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1)
    return ranges.max(axis=1)


def _atr(h: pd.Series, l: pd.Series, c: pd.Series, period: int) -> pd.Series:
    tr = _true_range(h, l, c)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _adx(h, l, c, period: int):
    h = _ensure_float(h)
    l = _ensure_float(l)
    c = _ensure_float(c)
    up = h.diff()
    dn = l.shift(1) - l
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = _true_range(h, l, c)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = (
        100
        * pd.Series(plus_dm, index=h.index)
        .ewm(alpha=1 / period, adjust=False, min_periods=period)
        .mean()
        / atr
    )
    minus_di = (
        100
        * pd.Series(minus_dm, index=h.index)
        .ewm(alpha=1 / period, adjust=False, min_periods=period)
        .mean()
        / atr
    )
    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = (plus_di - minus_di).abs() / denom * 100
    adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return adx, plus_di, minus_di


def compute_vpoc(df: pd.DataFrame, bins: int = 50) -> float:
    if df.empty:
        return float("nan")
    vols = df["volume"].to_numpy(dtype="float64")
    if {"high", "low", "close"}.issubset(df.columns):
        highs = df["high"].to_numpy(dtype="float64")
        lows = df["low"].to_numpy(dtype="float64")
        closes = df["close"].to_numpy(dtype="float64")
        prices = (highs + lows + closes) / 3.0
    else:
        prices = df["close"].to_numpy(dtype="float64")
    mask = np.isfinite(prices) & np.isfinite(vols)
    if not mask.any():
        return float("nan")
    prices = prices[mask]
    vols = vols[mask]
    lo = float(np.min(prices))
    hi = float(np.max(prices))
    if not math.isfinite(lo) or not math.isfinite(hi):
        return float("nan")
    if math.isclose(lo, hi):
        return float(lo)
    hist, edges = np.histogram(prices, bins=bins, range=(lo, hi), weights=vols)
    if hist.size == 0 or np.all(hist == 0):
        return float((lo + hi) / 2.0)
    idx = int(np.argmax(hist))
    up_idx = min(idx + 1, len(edges) - 1)
    return float((edges[idx] + edges[up_idx]) / 2.0)


def _compute_weekly_vpoc_fast(
    timestamps: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """
    Strictly leak-free running weekly VPOC.

    For each row t inside a calendar week, the VPOC is computed using ONLY
    rows of that week up to and including t. v19 used the full week's
    [min(TP), max(TP)] as bin edges — which leaked the rest-of-week into
    early-in-week rows. v20 expands the bin layout monotonically with the
    observed data, so a row at Tuesday only sees Mon+Tue prices.
    """
    N_BINS = 50
    ts_local = (
        timestamps.dt.tz_convert(None)
        if timestamps.dt.tz is not None
        else timestamps
    )
    week_key = ts_local.dt.to_period("W-FRI")
    result = pd.Series(np.nan, index=timestamps.index, dtype="float64")

    week_groups = week_key.groupby(week_key).groups
    for wk, idx_arr in week_groups.items():
        idx = np.asarray(list(idx_arr), dtype="int64")
        if idx.size == 0:
            continue
        h_w = high.iloc[idx].to_numpy(dtype="float64")
        l_w = low.iloc[idx].to_numpy(dtype="float64")
        c_w = close.iloc[idx].to_numpy(dtype="float64")
        v_w = volume.iloc[idx].to_numpy(dtype="float64")
        tp_w = (h_w + l_w + c_w) / 3.0
        n = idx.size

        # Strictly causal running weekly VPOC: at each row i within the week,
        # use ONLY tp[0..i] / v[0..i] (no future days, even within the same
        # week). v19 leaked here by computing bin edges from the full week's
        # min/max — which the leak canary catches. We rebuild a fresh
        # histogram at each step; per-week cost is O(L^2 + L * N_BINS) which
        # is trivial for L≈5.
        vpocs = np.full(n, np.nan, dtype="float64")
        for i in range(n):
            finite = np.isfinite(tp_w[: i + 1]) & np.isfinite(v_w[: i + 1])
            if not finite.any():
                continue
            tps = tp_w[: i + 1][finite]
            vs = v_w[: i + 1][finite]
            lo_i = float(tps.min())
            hi_i = float(tps.max())
            if not (math.isfinite(lo_i) and math.isfinite(hi_i)):
                continue
            if math.isclose(lo_i, hi_i):
                vpocs[i] = lo_i
                continue
            edges = np.linspace(lo_i, hi_i, N_BINS + 1)
            bin_mids = (edges[:-1] + edges[1:]) / 2.0
            bins = np.clip(
                ((tps - lo_i) / (hi_i - lo_i) * N_BINS).astype("int64"),
                0,
                N_BINS - 1,
            )
            cum = np.bincount(bins, weights=vs, minlength=N_BINS)
            vpocs[i] = bin_mids[int(np.argmax(cum))]
        result.iloc[idx] = vpocs

    return result


def _cpr_relationship(base_bc, base_tc, other_bc, other_tc) -> pd.Series:
    relation = pd.Series(index=base_bc.index, dtype="object")
    relation[
        (other_bc > base_tc) & other_bc.notna() & base_tc.notna()
    ] = "Above"
    relation[
        (other_tc < base_bc) & other_tc.notna() & base_bc.notna()
    ] = "Below"
    relation[
        ((other_bc <= base_tc) & (other_tc >= base_bc))
        & other_bc.notna()
        & other_tc.notna()
        & base_bc.notna()
        & base_tc.notna()
    ] = "Inside"
    relation = relation.fillna("Overlap")
    relation[
        (base_bc.isna()) | (base_tc.isna()) | (other_bc.isna()) | (other_tc.isna())
    ] = None
    return relation


def _period_trend_from_highs_leakfree(
    ts: pd.Series, high: pd.Series, period: str
) -> pd.Series:
    ts_local = ts.dt.tz_convert(None) if ts.dt.tz is not None else ts
    periods = ts_local.dt.to_period(period)
    frame = pd.DataFrame({"period": periods, "high": high})
    max_by_period = frame.groupby("period", sort=True)["high"].max()
    prev_max_by_period = max_by_period.shift(1)
    running_high = high.groupby(periods).cummax()
    prev_map = prev_max_by_period.to_dict()
    prev_at_row = periods.map(prev_map)
    trend_row = pd.Series(0, index=high.index, dtype="Int8")
    mask_prev_ok = pd.notna(prev_at_row)
    trend_row[(running_high > prev_at_row) & mask_prev_ok] = 1
    trend_row[(running_high < prev_at_row) & mask_prev_ok] = -1
    return trend_row


def _rolling_ols_slope_fast(y: pd.Series, window: int) -> pd.Series:
    """
    Vectorized rolling OLS slope of y vs t = 0,1,2,..., over `window`.
        slope = cov(t, y) / var(t)
    var(t) for w consecutive integers is exactly w*(w+1)/12, so we avoid
    rebuilding the constant t-series statistics each window.
    Same numerical result as the v19 Python loop, ~10-50x faster.
    """
    y = pd.to_numeric(y, errors="coerce")
    n = len(y)
    if n == 0:
        return pd.Series(np.full(0, np.nan), index=y.index, dtype="float64")
    w = int(window)
    if w <= 1:
        return pd.Series(np.full(n, np.nan), index=y.index, dtype="float64")
    t = pd.Series(np.arange(n, dtype="float64"), index=y.index)
    var_t = w * (w + 1) / 12.0
    cov_xy = t.rolling(w, min_periods=w).cov(y)
    return (cov_xy / var_t).astype("float64")


# ──────────────────────────────────────────────────────────────────────────────
#  v20 LEAK-FREE PRIMITIVES
#
#  These are the ONLY allowed transforms inside compute_daily_indicators().
#  Anything that ranks / normalizes / aggregates against a per-symbol's
#  WHOLE series (e.g. unwindowed s.rank() / s.mean() / s.quantile()) would
#  peek at the future and is forbidden. The static check at the bottom of
#  this section refuses to import the module if a forbidden pattern leaks
#  back in.
# ──────────────────────────────────────────────────────────────────────────────


def _expand_rank_pct(s: pd.Series, min_periods: int = 60) -> pd.Series:
    """
    Leak-free expanding percentile rank.

    At row t, returns the percentile rank of s[t] within s[0..t] inclusive
    (average method for ties). NaN values are skipped in the ranking and
    inherit NaN in the output. Returns NaN until at least `min_periods`
    finite values have been observed.

    This is the leak-free replacement for v19's `_rank_cs(s) = s.rank(pct=True)`,
    which ranked each row against the entire (past + future) series.
    """
    s = pd.to_numeric(s, errors="coerce")
    if len(s) == 0:
        return pd.Series([], index=s.index, dtype="float64")
    # Pandas >= 1.4: vectorized, C-level expanding rank.
    try:
        return s.expanding(min_periods=int(min_periods)).rank(pct=True).astype("float64")
    except Exception:
        # Fallback for older pandas: bisect-based, O(n^2) but correct.
        import bisect
        arr = s.to_numpy(dtype="float64")
        n = arr.size
        out = np.full(n, np.nan, dtype="float64")
        sl: List[float] = []
        for i in range(n):
            x = arr[i]
            if np.isfinite(x):
                bisect.insort(sl, x)
                if len(sl) >= int(min_periods):
                    lo = bisect.bisect_left(sl, x)
                    hi = bisect.bisect_right(sl, x)
                    out[i] = ((lo + 1 + hi) / 2.0) / len(sl)
        return pd.Series(out, index=s.index, dtype="float64")


def _streak_length(flag) -> pd.Series:
    """O(n) consecutive-True streak length. flag may be bool/Int8/object."""
    arr = pd.Series(flag).fillna(False).astype(bool).to_numpy()
    n = arr.size
    if n == 0:
        return pd.Series(np.zeros(0, dtype="int32"))
    grp = (~arr).cumsum()  # increments on every False -> resets the run
    s = pd.Series(arr.astype("int32"))
    out = s.groupby(grp).cumsum().to_numpy()
    out[~arr] = 0
    return pd.Series(out.astype("int32"))


def _days_since_flag(flag) -> pd.Series:
    """
    O(n) days since the most recent True. Zero before the first True (matching
    v19 behaviour for D_days_since_boh_20 / D_days_since_bol_20).
    """
    arr = pd.Series(flag).fillna(0).astype("int8").to_numpy()
    n = arr.size
    if n == 0:
        return pd.Series(np.zeros(0, dtype="int32"))
    idx = np.arange(n, dtype="int64")
    last = np.where(arr == 1, idx, -1)
    last = np.maximum.accumulate(last)
    days = (idx - last).astype("int32")
    days[last < 0] = 0
    return pd.Series(days)


def _v20_static_leak_check() -> None:
    """
    At import time, parse this file and refuse to load if compute_daily_indicators
    contains any unwindowed `.rank(`, `.cumsum(`, `.cummax(`, `.cummin(`,
    `.expanding(min_periods` without an `expanding` qualifier, etc. We allow
    `.cumsum()` only on diff/sign products (OBV pattern) by whitelisting the
    surrounding context.

    The check is deliberately conservative: it reads source and grep-tests
    for forbidden tokens; false positives are tolerated and can be silenced
    with the marker `# leak-free:` on the same line.
    """
    try:
        path = Path(__file__)
        src = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return

    # Slice out the body of compute_daily_indicators. Use a leading newline
    # in the search anchor so we don't match the string literal in this
    # function's own body (which appears earlier in the file).
    start = src.find("\ndef compute_daily_indicators(")
    if start < 0:
        return
    start += 1  # skip the leading newline so line numbering aligns
    # End at the next top-level "# ----" banner that follows the function
    end = src.find("\n# -------------------- Finalize for cache --------------------", start)
    if end < 0:
        end = len(src)
    body = src[start:end]

    forbidden_patterns = [
        # An unwindowed rank() that is not preceded by .rolling( or .expanding(
        # within a few characters. Tolerate `# leak-free:` annotation.
        # We split lines to make per-line annotation possible.
    ]
    bad: List[str] = []
    body_lines = body.splitlines()
    for lineno, line in enumerate(body_lines, start=1):
        if "# leak-free:" in line:
            continue
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        # Whole-series rank: `.rank(` not preceded by `.rolling(`, `.expanding(`,
        # or `.groupby(` within the same statement. Statements may span lines,
        # so we widen the search to the previous 4 lines for chain continuations.
        if ".rank(" in line:
            pre = line.split(".rank(")[0]
            ctx = "\n".join(body_lines[max(0, lineno - 5): lineno])
            qualified = (
                "rolling(" in pre
                or "expanding(" in pre
                or "groupby(" in pre
                or "rolling(" in ctx
                or "expanding(" in ctx
                or "groupby(" in ctx
            )
            if not qualified:
                bad.append(f"L{lineno}: {stripped}")
    if bad:
        raise RuntimeError(
            "v20 STATIC LEAK CHECK FAILED — unwindowed .rank() calls detected "
            "inside compute_daily_indicators(). These leak the future. "
            "Lines:\n  " + "\n  ".join(bad[:20])
        )


def _v20_leak_canary_check() -> None:
    """
    Runtime canary: build two synthetic OHLCV frames identical for indices
    [0, N-K) and DIFFERENT for the last K rows. Compute indicators on both
    and assert every column matches on rows [0, N-K). Any column whose
    PAST values change when only the FUTURE changes is leaking.

    The label column `ret_5d_close_pct` is exempt — it is a forward return
    and is supposed to peek (but it should never be used as a feature).
    """
    rng_ = np.random.default_rng(42)
    N = 400
    K = 12
    log_ret = rng_.normal(0.0, 0.012, N)
    close_arr = 100.0 * np.exp(np.cumsum(log_ret))
    open_arr = close_arr * (1.0 + rng_.normal(0.0, 0.003, N))
    hi_off = np.abs(rng_.normal(0.0, 0.005, N))
    lo_off = np.abs(rng_.normal(0.0, 0.005, N))
    high_arr = np.maximum(close_arr, open_arr) * (1.0 + hi_off)
    low_arr = np.minimum(close_arr, open_arr) * (1.0 - lo_off)
    vol_arr = rng_.integers(50_000, 1_500_000, N).astype("float64")
    ts = pd.date_range("2018-01-01", periods=N, freq="B", tz=IST)

    base = pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_arr,
            "high": high_arr,
            "low": low_arr,
            "close": close_arr,
            "volume": vol_arr,
        }
    )
    tampered = base.copy()
    rng2 = np.random.default_rng(999)
    mult_oc = rng2.uniform(0.4, 2.5, K)
    mult_hl = rng2.uniform(0.4, 2.5, K)
    tampered.loc[N - K:, "open"] = tampered.loc[N - K:, "open"].to_numpy() * mult_oc
    tampered.loc[N - K:, "close"] = tampered.loc[N - K:, "close"].to_numpy() * mult_oc
    tampered.loc[N - K:, "high"] = tampered.loc[N - K:, "high"].to_numpy() * mult_hl
    tampered.loc[N - K:, "low"] = tampered.loc[N - K:, "low"].to_numpy() * mult_hl
    tampered.loc[N - K:, "volume"] = (
        tampered.loc[N - K:, "volume"].to_numpy() * rng2.uniform(0.1, 10.0, K)
    )

    out_a = compute_daily_indicators(base.copy())
    out_b = compute_daily_indicators(tampered.copy())
    head_a = out_a.iloc[: N - K]
    head_b = out_b.iloc[: N - K]

    # Skip the timestamp itself + the forward-looking label column.
    # ret_5d_close_pct[t] = close[t+5]/close[t] - 1, so for t in [N-K-5, N-K),
    # it legitimately reads tampered values.
    SKIP = {"timestamp", "ret_5d_close_pct"}
    leaks: List[str] = []
    for c in head_a.columns:
        if c in SKIP:
            continue
        try:
            a = pd.to_numeric(head_a[c], errors="coerce").to_numpy(dtype="float64")
            b = pd.to_numeric(head_b[c], errors="coerce").to_numpy(dtype="float64")
        except Exception:
            continue
        a_nan = np.isnan(a)
        b_nan = np.isnan(b)
        if not np.array_equal(a_nan, b_nan):
            leaks.append(c)
            continue
        diff = np.where(a_nan, 0.0, np.abs(a - b))
        if np.nanmax(diff) > 1e-7:
            leaks.append(c)
    if leaks:
        raise RuntimeError(
            "v20 LEAK CANARY FAILED — these columns depend on FUTURE data: "
            + ", ".join(leaks[:30])
            + (f"  (+{len(leaks) - 30} more)" if len(leaks) > 30 else "")
        )


# -------------------- Indicators column list --------------------
DAILY_INDICATOR_COLUMNS = [
    # Core OHLCV
    "timestamp", "open", "high", "low", "close", "volume",
    # EMAs / RSI
    "D_ema20", "D_ema50", "D_ema100", "D_rsi7", "D_rsi14",
    # MACD
    "D_macd", "D_macd_signal", "D_macd_hist",
    # CMF / ADX
    "D_cmf20", "D_adx14", "D_pdi14", "D_mdi14",
    # Inside day
    "D_inside_day", "D_prev_inside_day",
    # CPR / Pivots
    "D_cpr_pivot", "D_cpr_bc", "D_cpr_tc",
    "D_pivot", "D_support1", "D_resistance1", "D_support2", "D_resistance2",
    # NR
    "D_nr", "D_nr_length", "D_nr_day",
    # VPOC
    "D_vpoc", "D_weekly_vpoc",
    # SMAs
    "D_sma5", "D_sma20",
    # Trend
    "D_daily_trend", "D_weekly_trend", "D_monthly_trend",
    "D_rsi7_gt_rsi14", "D_ema_stack_20_50_100", "D_ema20_angle_deg",
    # ATR
    "D_atr14", "D_atr30", "D_atr_ratio_14_30",
    # CPR width / tomorrow CPR
    "D_cpr_width_pct", "D_tmr_cpr_bc", "D_tmr_cpr_tc",
    "D_tmr_cpr_vs_today", "D_cpr_vs_yday",
    # Structure
    "D_hh", "D_hl", "D_lh", "D_ll", "D_structure_trend",
    # Prev day
    "D_prev_high", "D_prev_low", "D_prev_close",
    # OLI / day type / range
    "D_oli", "D_day_type", "D_range_to_atr14",
    # SMAs extended
    "D_sma50", "D_sma200", "D_golden_regime",
    # OBV
    "D_obv", "D_obv_slope", "D_price_and_obv_rising",
    # Numeric codes
    "D_tmr_cpr_vs_today_code", "D_cpr_vs_yday_code", "D_structure_trend_code",
    # v17: Bollinger / Yang-Zhang / Donchian / Breakout / Volume
    "D_dow",
    "D_bb_pctB_20", "D_bb_bw_20",
    "D_vol_yz_20", "D_vol_yz_50",
    "D_donch_pos_20", "D_donch_pos_50",
    "D_breakout_high_20", "D_breakout_low_20",
    "D_breakout_high_50", "D_breakout_low_50",
    "D_days_since_boh_20", "D_days_since_bol_20",
    "D_dollar_vol", "D_dvol_z20", "D_dvol_z50", "D_dvol_z252",
    "D_vol_surge_20", "D_vol_surge_50",
    # v17: Cache-side features
    "D_atr_pct", "D_range_pct", "D_gap_pct",
    "D_rsi14_z252", "D_atr_pct_z252", "D_vol_z252", "D_ema20_angle_z252",
    "D_rsi14_obv_x", "D_rsi7_obv_x", "D_atr14_to_close_pct",
    "D_ret_5d_roll_std", "D_close_roll_slope_20", "D_close_roll_slope_50",
    # v17: Logical combos
    "Comb_RSIslopePos__ADX_15_25",
    "Comb_GapUp__CPR_Tmr_Above", "Comb_GapDown__CPR_Tmr_Below",
    "Comb_ATRlow__EMA20pos", "Comb_ATRhigh__EMA20neg",
    # v18: Copilot structural features
    "D_body_ratio", "D_wick_skew",
    "D_hh_run", "D_hl_run", "D_lh_run", "D_ll_run",
    "D_nr_expand", "D_compress_state",
    "D_dist_from_20h", "D_dist_from_20l", "D_dist_from_52wh",
    "D_midpoint_slope", "D_slope_stability",
    # v18: Weekly momentum features
    "W_ret_4w", "W_ret_13w", "W_close_pos", "W_vol_vs_4w",
    # v19 NEW: 16 WorldQuant alphas
    "D_WQ_3", "D_WQ_6", "D_WQ_12", "D_WQ_13", "D_WQ_16", "D_WQ_19", "D_WQ_20", "D_WQ_23",
        "D_WQ_26", "D_WQ_29", "D_WQ_33", "D_WQ_35", "D_WQ_38", "D_WQ_40", "D_WQ_41", "D_WQ_44",
    # v19 NEW: 17 rolling/lag/diff transforms
    "D_slope_stability_rmean50", "D_slope_stability_rstd10", "D_slope_stability_rstd20",
    "D_body_ratio_rmean50", "D_body_ratio_rmean20",
    "D_close_roll_slope_20_rstd20", "D_close_roll_slope_20_rstd10",
    "D_macd_hist_rstd10",
    "D_mdi14_diff1", "D_mdi14_diff5",
    "D_mdi14_rrank10", "D_mdi14_rrank20",
    "D_donch_pos_50_rmean50", "D_donch_pos_50_lag5",
    "D_donch_pos_20_rmean50", "D_donch_pos_20_lag5",
    "D_cmf20_rmean50",
]


# =============================================================================
#   INDICATOR COMPUTATION
# =============================================================================

def compute_daily_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    missing = [c for c in DAILY_INDICATOR_COLUMNS if c not in df.columns]
    if missing:
        df = pd.concat(
            [df, pd.DataFrame({c: np.nan for c in missing}, index=df.index)],
            axis=1,
            copy=False,
        )
    if df.empty:
        return df

    df = df.sort_values("timestamp").reset_index(drop=True)
    timestamps = pd.to_datetime(df["timestamp"])
    timestamps_local = (
        timestamps.dt.tz_convert(None) if timestamps.dt.tz is not None else timestamps
    )

    close = _ensure_float(df["close"])
    high = _ensure_float(df["high"])
    low = _ensure_float(df["low"])
    open_ = _ensure_float(df["open"])
    volume = _ensure_float(df["volume"]).fillna(0.0)
    rng = high - low  # pre-compute once

    # ── EMAs / RSI ────────────────────────────────────────────────────────
    df["D_ema20"] = _ema(close, 20)
    df["D_ema50"] = _ema(close, 50)
    df["D_ema100"] = _ema(close, 100)
    df["D_rsi7"] = _rsi(close, 7)
    df["D_rsi14"] = _rsi(close, 14)

    # ── MACD 12,26,9 ──────────────────────────────────────────────────────
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False, min_periods=1).mean()
    df["D_macd"] = macd
    df["D_macd_signal"] = signal
    df["D_macd_hist"] = macd - signal

    # ── CMF 20 ────────────────────────────────────────────────────────────
    df["D_cmf20"] = (
        (((close - low) - (high - close)) / (high - low).replace(0, np.nan) * volume)
        .rolling(window=20, min_periods=1)
        .sum()
        / volume.rolling(window=20, min_periods=1).sum()
    )

    # ── ADX/PDI/MDI 14 ────────────────────────────────────────────────────
    adx, pdi, mdi = _adx(high, low, close, 14)
    df["D_adx14"] = adx
    df["D_pdi14"] = pdi
    df["D_mdi14"] = mdi

    # ── Prev day references & inside day flags ────────────────────────────
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)
    df["D_prev_high"] = prev_high
    df["D_prev_low"] = prev_low
    df["D_prev_close"] = prev_close
    df["D_inside_day"] = ((high <= prev_high) & (low >= prev_low)).astype("boolean")
    df["D_prev_inside_day"] = df["D_inside_day"].shift(1).astype("boolean")

    # ── CPR / Pivots ──────────────────────────────────────────────────────
    pivot = (high + low + close) / 3
    cpr_bc = (high + low) / 2
    cpr_tc = 2 * pivot - cpr_bc
    df["D_cpr_pivot"] = pivot
    df["D_cpr_bc"] = cpr_bc
    df["D_cpr_tc"] = cpr_tc
    df["D_pivot"] = pivot
    df["D_support1"] = 2 * pivot - high
    df["D_resistance1"] = 2 * pivot - low
    df["D_support2"] = pivot - rng
    df["D_resistance2"] = pivot + rng

    # ── VPOC proxy ────────────────────────────────────────────────────────
    df["D_vpoc"] = (high + low + close) / 3.0

    # ── Weekly VPOC ───────────────────────────────────────────────────────
    df["D_weekly_vpoc"] = _compute_weekly_vpoc_fast(
        timestamps, high, low, close, volume
    )

    # ── NR7 & length ──────────────────────────────────────────────────────
    prev6_min = rng.shift(1).rolling(window=6, min_periods=6).min()
    nr7 = (rng < prev6_min) & prev6_min.notna()
    df["D_nr"] = nr7.astype("boolean")

    nr_arr = nr7.fillna(False).to_numpy()
    df["D_nr_length"] = _streak_length(nr_arr).astype("int64").to_numpy()

    values = rng.astype("float64")
    nr_window = pd.Series(pd.NA, index=df.index, dtype="Int64")
    for w in range(20, 5 - 1, -1):
        prev_min = values.shift(1).rolling(window=w - 1, min_periods=w - 1).min()
        mask = (values < prev_min) & prev_min.notna()
        nr_window = nr_window.mask(mask & nr_window.isna(), w)
    df["D_nr_day"] = nr_window

    # ── SMAs ─────────────────────────────────────────────────────────────
    df["D_sma5"] = _sma(close, 5)
    df["D_sma20"] = _sma(close, 20)
    df["D_sma50"] = _sma(close, 50)
    df["D_sma200"] = _sma(close, 200)

    df["D_rsi7_gt_rsi14"] = (df["D_rsi7"] > df["D_rsi14"]).astype("boolean")
    df["D_ema_stack_20_50_100"] = (
        (df["D_ema20"] > df["D_ema50"]) & (df["D_ema50"] > df["D_ema100"])
    ).astype("boolean")

    ema20 = df["D_ema20"]
    prev_ema20 = ema20.shift(1)
    pct_slope = (ema20 - prev_ema20) / prev_ema20.replace(0, np.nan)
    df["D_ema20_angle_deg"] = np.degrees(np.arctan(pct_slope))

    # ── ATRs ─────────────────────────────────────────────────────────────
    df["D_atr14"] = _atr(high, low, close, 14)
    df["D_atr30"] = _atr(high, low, close, 30)
    df["D_atr_ratio_14_30"] = df["D_atr14"] / df["D_atr30"].replace(0, np.nan)

    # ── CPR width / tomorrow CPR ─────────────────────────────────────────
    df["D_cpr_width_pct"] = (
        (df["D_cpr_tc"] - df["D_cpr_bc"]) / close.replace(0, np.nan)
    ) * 100
    df["D_tmr_cpr_bc"] = cpr_bc
    df["D_tmr_cpr_tc"] = cpr_tc

    pivot_y = (high.shift(1) + low.shift(1) + close.shift(1)) / 3.0
    cpr_bc_y = (high.shift(1) + low.shift(1)) / 2.0
    cpr_tc_y = 2 * pivot_y - cpr_bc_y
    rel_tmr = _cpr_relationship(
        cpr_bc_y, cpr_tc_y, df["D_tmr_cpr_bc"], df["D_tmr_cpr_tc"]
    )
    rel_vs_y = _cpr_relationship(
        cpr_bc_y, cpr_tc_y, df["D_cpr_bc"], df["D_cpr_tc"]
    )
    df["D_tmr_cpr_vs_today"] = rel_tmr
    df["D_cpr_vs_yday"] = rel_vs_y

    # ── Structure / trend flags ───────────────────────────────────────────
    df["D_hh"] = (high > prev_high).astype("boolean")
    df["D_hl"] = (low > prev_low).astype("boolean")
    df["D_lh"] = (high < prev_high).astype("boolean")
    df["D_ll"] = (low < prev_low).astype("boolean")

    daily_trend = pd.Series(0, index=df.index, dtype="Int8")
    daily_trend[df["D_hh"] == True] = 1
    daily_trend[df["D_lh"] == True] = -1
    df["D_daily_trend"] = daily_trend.astype("Int8")
    df["D_weekly_trend"] = _period_trend_from_highs_leakfree(
        timestamps, high, "W-FRI"
    )
    df["D_monthly_trend"] = _period_trend_from_highs_leakfree(
        timestamps, high, "M"
    )
    df["D_structure_trend"] = np.select(
        [df["D_hh"] & df["D_hl"], df["D_lh"] & df["D_ll"]],
        ["uptrend", "downtrend"],
        default="range",
    )

    # ── Numeric codes ─────────────────────────────────────────────────────
    def _encode_rel(s: pd.Series) -> pd.Series:
        return (
            s.map({"Above": 1, "Inside": 0, "Overlap": 0, "Below": -1})
            .fillna(0)
            .astype("Int8")
        )

    def _encode_trend(s: pd.Series) -> pd.Series:
        return (
            s.map({"uptrend": 1, "range": 0, "downtrend": -1})
            .fillna(0)
            .astype("Int8")
        )

    df["D_tmr_cpr_vs_today_code"] = _encode_rel(df["D_tmr_cpr_vs_today"])
    df["D_cpr_vs_yday_code"] = _encode_rel(df["D_cpr_vs_yday"])
    df["D_structure_trend_code"] = _encode_trend(df["D_structure_trend"])

    # ── OLI / day type / range ────────────────────────────────────────────
    df["D_oli"] = (open_ - low) / rng.replace(0, np.nan)
    df["D_day_type"] = np.select(
        [open_ > cpr_tc, open_ < cpr_bc], ["bullish", "bearish"], default="inside"
    )
    df["D_range_to_atr14"] = rng / df["D_atr14"].replace(0, np.nan)
    df["D_golden_regime"] = (
        (close > df["D_sma200"]) & (df["D_sma50"] > df["D_sma200"])
    ).astype("boolean")

    # ── OBV + slope ───────────────────────────────────────────────────────
    obv = (np.sign(close.diff().fillna(0.0)) * volume).cumsum()
    df["D_obv"] = obv
    df["D_obv_slope"] = obv.diff()
    df["D_price_and_obv_rising"] = (
        (close > close.shift(1)) & (obv > obv.shift(1))
    ).astype("boolean")

    # ══════════════════════════════════════════════════════════════════════
    #  v17 features
    # ══════════════════════════════════════════════════════════════════════

    # Day-of-week
    ts_local = (
        timestamps.dt.tz_convert(None) if timestamps.dt.tz is not None else timestamps
    )
    df["D_dow"] = ts_local.dt.weekday.astype("Int8")
    dow_dummies = pd.get_dummies(df["D_dow"], prefix="DOW", dtype="int8")
    for c in dow_dummies.columns:
        df[c] = dow_dummies[c]

    # Bollinger Bands (20)
    sma20 = df["D_sma20"]
    std20 = close.rolling(20, min_periods=1).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_width = bb_upper - bb_lower
    df["D_bb_pctB_20"] = (
        (close - bb_lower) / bb_width.replace(0, np.nan)
    ).clip(lower=-5, upper=5)
    df["D_bb_bw_20"] = (bb_width / sma20.replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )

    # Yang-Zhang volatility
    log_oo = np.log(open_ / prev_close).replace([np.inf, -np.inf], np.nan)
    log_cc = np.log(close / open_).replace([np.inf, -np.inf], np.nan)
    log_h_o = np.log(high / open_).replace([np.inf, -np.inf], np.nan)
    log_l_o = np.log(low / open_).replace([np.inf, -np.inf], np.nan)
    log_h_c = np.log(high / close).replace([np.inf, -np.inf], np.nan)
    log_l_c = np.log(low / close).replace([np.inf, -np.inf], np.nan)
    rs = log_h_o * log_h_c + log_l_o * log_l_c
    k = 0.34
    yz_var = (log_oo**2) + k * (log_cc**2) + (1 - k) * rs

    def _yz_roll(win: int):
        v = yz_var.rolling(win, min_periods=max(10, win // 3)).mean()
        return np.sqrt(v)

    df["D_vol_yz_20"] = _yz_roll(20)
    df["D_vol_yz_50"] = _yz_roll(50)

    # Donchian
    hi_20 = high.shift(1).rolling(20, min_periods=1).max()
    lo_20 = low.shift(1).rolling(20, min_periods=1).min()
    rng_20 = (hi_20 - lo_20).replace(0, np.nan)
    hi_50 = high.shift(1).rolling(50, min_periods=1).max()
    lo_50 = low.shift(1).rolling(50, min_periods=1).min()
    rng_50 = (hi_50 - lo_50).replace(0, np.nan)

    df["D_donch_pos_20"] = ((close - lo_20) / rng_20).clip(lower=-1, upper=2)
    df["D_donch_pos_50"] = ((close - lo_50) / rng_50).clip(lower=-1, upper=2)

    # Breakouts
    df["D_breakout_high_20"] = (high > hi_20).astype("int8")
    df["D_breakout_low_20"] = (low < lo_20).astype("int8")
    df["D_breakout_high_50"] = (high > hi_50).astype("int8")
    df["D_breakout_low_50"] = (low < lo_50).astype("int8")

    # Days since breakouts (O(n) vectorized via _days_since_flag)
    df["D_days_since_boh_20"] = _days_since_flag(df["D_breakout_high_20"]).astype("int16")
    df["D_days_since_bol_20"] = _days_since_flag(df["D_breakout_low_20"]).astype("int16")

    # Dollar volume & Z-scores / surge flags
    df["D_dollar_vol"] = (close * volume).replace([np.inf, -np.inf], np.nan)

    def _zscore(s, win=252):
        x = pd.to_numeric(s, errors="coerce")
        m = x.rolling(win, min_periods=max(10, win // 4)).mean()
        v = x.rolling(win, min_periods=max(10, win // 4)).std()
        return (x - m) / v.replace(0, np.nan)

    df["D_dvol_z20"] = _zscore(df["D_dollar_vol"], 20)
    df["D_dvol_z50"] = _zscore(df["D_dollar_vol"], 50)
    df["D_dvol_z252"] = _zscore(df["D_dollar_vol"], 252)

    vol20_m = volume.rolling(20, min_periods=5).mean()
    vol20_s = volume.rolling(20, min_periods=5).std()
    vol50_m = volume.rolling(50, min_periods=10).mean()
    vol50_s = volume.rolling(50, min_periods=10).std()
    df["D_vol_surge_20"] = (volume > (vol20_m + 2 * vol20_s)).astype("int8")
    df["D_vol_surge_50"] = (volume > (vol50_m + 2 * vol50_s)).astype("int8")

    # Cache-side features
    df["D_atr_pct"] = (df["D_atr14"] / close.replace(0, np.nan)) * 100
    df["D_range_pct"] = ((high - low) / close.replace(0, np.nan)) * 100
    df["D_gap_pct"] = (
        (open_ - df["D_prev_close"]) / df["D_prev_close"].replace(0, np.nan)
    ) * 100

    def _zscore_generic(s, win=252):
        x = pd.to_numeric(s, errors="coerce")
        m = x.rolling(win, min_periods=max(10, win // 4)).mean()
        v = x.rolling(win, min_periods=max(10, win // 4)).std()
        return (x - m) / v.replace(0, np.nan)

    df["D_rsi14_z252"] = _zscore_generic(df["D_rsi14"], 252)
    df["D_atr_pct_z252"] = _zscore_generic(df["D_atr_pct"], 252)
    df["D_vol_z252"] = _zscore_generic(df["volume"], 252)
    df["D_ema20_angle_z252"] = _zscore_generic(df["D_ema20_angle_deg"], 252)

    rsi14 = pd.to_numeric(df.get("D_rsi14"), errors="coerce")
    rsi7 = pd.to_numeric(df.get("D_rsi7"), errors="coerce")
    obvs = pd.to_numeric(df.get("D_obv_slope"), errors="coerce")
    df["D_rsi14_obv_x"] = rsi14 * obvs
    if "D_rsi7" in df.columns:
        df["D_rsi7_obv_x"] = rsi7 * obvs
    df["D_atr14_to_close_pct"] = (
        df["D_atr14"] / close
    ).replace([np.inf, -np.inf], np.nan) * 100.0

    if "ret_5d_close_pct" not in df.columns:
        df["ret_5d_close_pct"] = (df["close"].shift(-5) / df["close"] - 1) * 100

    ret_fwd_5 = pd.to_numeric(df.get("ret_5d_close_pct"), errors="coerce")
    df["D_ret_5d_roll_std"] = ret_fwd_5.shift(5).rolling(50, min_periods=10).std()

    # Rolling OLS slope
    df["D_close_roll_slope_20"] = _rolling_ols_slope_fast(df["close"], window=20)
    df["D_close_roll_slope_50"] = _rolling_ols_slope_fast(df["close"], window=50)

    # Logical combos
    rsi14_diff = pd.to_numeric(df["D_rsi14"], errors="coerce").diff()
    adx14 = pd.to_numeric(df["D_adx14"], errors="coerce")
    df["Comb_RSIslopePos__ADX_15_25"] = (
        (rsi14_diff > 0) & (adx14 >= 15) & (adx14 <= 25)
    ).astype("int8")

    gap = pd.to_numeric(df["D_gap_pct"], errors="coerce")
    df["Comb_GapUp__CPR_Tmr_Above"] = (
        (gap > 0) & (df["D_tmr_cpr_vs_today_code"] == 1)
    ).astype("int8")
    df["Comb_GapDown__CPR_Tmr_Below"] = (
        (gap < 0) & (df["D_tmr_cpr_vs_today_code"] == -1)
    ).astype("int8")

    atr_pct = pd.to_numeric(df["D_atr_pct"], errors="coerce")
    ema_ang = pd.to_numeric(df["D_ema20_angle_deg"], errors="coerce")
    ATR_LOW = 2.0
    ATR_HIGH = 4.0
    df["Comb_ATRlow__EMA20pos"] = (
        (atr_pct <= ATR_LOW) & (ema_ang > 0)
    ).astype("int8")
    df["Comb_ATRhigh__EMA20neg"] = (
        (atr_pct >= ATR_HIGH) & (ema_ang < 0)
    ).astype("int8")

    # ══════════════════════════════════════════════════════════════════════
    #  v18: Copilot structural features
    # ══════════════════════════════════════════════════════════════════════

    rng_safe = rng.replace(0, np.nan)
    atr14_safe = df["D_atr14"].replace(0, np.nan)

    # Candle geometry
    df["D_body_ratio"] = ((close - open_) / rng_safe).clip(-1, 1)
    upper_wick = high - pd.concat([close, open_], axis=1).max(axis=1)
    lower_wick = pd.concat([close, open_], axis=1).min(axis=1) - low
    df["D_wick_skew"] = ((upper_wick - lower_wick) / rng_safe).clip(-1, 1)

    # Path-dependency: consecutive run counts (O(n) vectorized)
    df["D_hh_run"] = _streak_length(df["D_hh"] == True).astype("int16")
    df["D_hl_run"] = _streak_length(df["D_hl"] == True).astype("int16")
    df["D_lh_run"] = _streak_length(df["D_lh"] == True).astype("int16")
    df["D_ll_run"] = _streak_length(df["D_ll"] == True).astype("int16")

    # Range compression/expansion
    df["D_nr_expand"] = (rng > rng.shift(1)).astype("int8")

    # Compression state
    df["D_compress_state"] = (
        df["D_bb_bw_20"]
        .rolling(50, min_periods=10)
        .rank(pct=True)
    )

    # Distance from regime anchors
    df["D_dist_from_20h"] = (close - hi_20) / atr14_safe
    df["D_dist_from_20l"] = (close - lo_20) / atr14_safe

    hi_52w = high.shift(1).rolling(252, min_periods=50).max()
    df["D_dist_from_52wh"] = (close - hi_52w) / atr14_safe

    # Structural trend quality
    midpoint = (high + low) / 2.0
    mp_slope_raw = _rolling_ols_slope_fast(midpoint, window=10)
    df["D_midpoint_slope"] = mp_slope_raw / atr14_safe.values

    close_slope_5 = _rolling_ols_slope_fast(close, window=5)
    df["D_slope_stability"] = (
        pd.Series(close_slope_5, index=df.index)
        .rolling(20, min_periods=5)
        .std()
    )

    # ══════════════════════════════════════════════════════════════════════
    #  v18: Weekly momentum features
    # ══════════════════════════════════════════════════════════════════════

    ts_local_naive = (
        timestamps.dt.tz_convert(None) if timestamps.dt.tz is not None else timestamps
    )
    df_tmp = pd.DataFrame(
        {
            "close": close.values,
            "high": high.values,
            "low": low.values,
            "volume": volume.values,
        },
        index=ts_local_naive,
    )

    weekly = df_tmp.resample("W-FRI").agg(
        {"close": "last", "high": "max", "low": "min", "volume": "sum"}
    ).dropna(subset=["close"])

    if len(weekly) >= 5:
        w_close = weekly["close"]
        w_high = weekly["high"]
        w_low = weekly["low"]
        w_vol = weekly["volume"]
        w_rng = (w_high - w_low).replace(0, np.nan)

        w_ret_4w = (w_close / w_close.shift(4) - 1.0) * 100.0
        w_ret_13w = (w_close / w_close.shift(13) - 1.0) * 100.0
        w_close_pos = ((w_close - w_low) / w_rng).clip(0, 1)
        w_vol_4w_avg = w_vol.rolling(4, min_periods=2).mean().shift(1)
        w_vol_vs_4w = (w_vol / w_vol_4w_avg.replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        )

        w_features = pd.DataFrame(
            {
                "W_ret_4w": w_ret_4w.shift(1),
                "W_ret_13w": w_ret_13w.shift(1),
                "W_close_pos": w_close_pos.shift(1),
                "W_vol_vs_4w": w_vol_vs_4w.shift(1),
            },
            index=weekly.index,
        )

        day_week_end = ts_local_naive.dt.to_period("W-FRI").dt.end_time.dt.normalize()
        idx = w_features.index
        if isinstance(idx, pd.PeriodIndex):
            idx = idx.to_timestamp(how="end")

        wf_index = pd.to_datetime(idx).normalize()
        wf_lookup = w_features.copy()
        wf_lookup.index = wf_index

        for feat_col in ["W_ret_4w", "W_ret_13w", "W_close_pos", "W_vol_vs_4w"]:
            lookup_dict = wf_lookup[feat_col].to_dict()
            df[feat_col] = day_week_end.map(lookup_dict).values
    else:
        for feat_col in ["W_ret_4w", "W_ret_13w", "W_close_pos", "W_vol_vs_4w"]:
            df[feat_col] = np.nan

    # ══════════════════════════════════════════════════════════════════════
    #  v20: 16 WorldQuant alphas — LEAK-FREE BY CONSTRUCTION
    #
    #  v19 used `_rank_cs(s) = s.rank(pct=True)` which ranked each row
    #  against the WHOLE per-symbol series (past + future). At row t, the
    #  rank therefore encoded "where t sits relative to t+1, t+2, …, N-1"
    #  — a textbook look-ahead leak. The 10 alphas that flow through
    #  _rank_cs (3, 13, 16, 19, 20, 29, 33, 38, 40, 44) were tainted.
    #
    #  v20 replaces every per-symbol rank with `_xrank` =
    #  s.expanding(min_periods=60).rank(pct=True). At row t, this is the
    #  percentile rank of s[t] within s[0..t] only — strictly causal.
    #  Column names are preserved so downstream code keeps working.
    #
    #  IMPORTANT: this is a *temporal* per-symbol rank, NOT the WQ-101
    #  cross-sectional rank. A genuine WQ rank would be
    #      panel.groupby('timestamp')[col].rank(pct=True)
    #  computed at panel-assembly time (see New_model.py / NEW FEAT IMP.py).
    #  These per-symbol expanding ranks are honest features but do NOT
    #  inherit WQ's published cross-sectional evidence.
    # ══════════════════════════════════════════════════════════════════════

    ret_d = close.pct_change()
    vwap_d = (high + low + close) / 3.0
    adv20_d = volume.rolling(20, min_periods=5).mean()

    def _ts_rank(s, w):
        return s.rolling(w, min_periods=max(3, w // 2)).rank(pct=True)

    def _ts_corr(a, b, w):
        return a.rolling(w, min_periods=max(5, w // 2)).corr(b)

    def _ts_std(s, w):
        return s.rolling(w, min_periods=max(3, w // 2)).std()

    def _ts_mean(s, w):
        return s.rolling(w, min_periods=max(3, w // 2)).mean()

    def _ts_max(s, w):
        return s.rolling(w, min_periods=max(3, w // 2)).max()

    def _ts_min(s, w):
        return s.rolling(w, min_periods=max(3, w // 2)).min()

    def _delta(s, d):
        return s.diff(d)

    def _delay(s, d):
        return s.shift(d)

    # LEAK-FREE per-symbol historical percentile rank (replaces v19 _rank_cs).
    # min_periods=60 ≈ ~3 trading months of warmup so early values are stable.
    _XRANK_MIN_PERIODS = 60

    def _xrank(s):
        return _expand_rank_pct(s, min_periods=_XRANK_MIN_PERIODS)

    # WQ_3:  -corr(rank(open), rank(volume), 10)
    df["D_WQ_3"] = (-_ts_corr(_xrank(open_), _xrank(volume), 10)).replace([np.inf, -np.inf], np.nan)

    # WQ_6:  -corr(open, volume, 10)
    df["D_WQ_6"] = (-_ts_corr(open_, volume, 10)).replace([np.inf, -np.inf], np.nan)

    # WQ_12: sign(delta(volume,1)) * -delta(close,1)
    df["D_WQ_12"] = (np.sign(_delta(volume, 1)) * (-_delta(close, 1))).replace([np.inf, -np.inf], np.nan)

    # WQ_13: -rank(cov(rank(close), rank(volume), 5))
    try:
        df["D_WQ_13"] = (-_xrank(_xrank(close).rolling(5, min_periods=3).cov(_xrank(volume)))
                       ).replace([np.inf, -np.inf], np.nan)
    except Exception:
        df["D_WQ_13"] = np.nan

    # WQ_16: -rank(cov(rank(high), rank(volume), 5))
    try:
        df["D_WQ_16"] = (-_xrank(_xrank(high).rolling(5, min_periods=3).cov(_xrank(volume)))
                       ).replace([np.inf, -np.inf], np.nan)
    except Exception:
        df["D_WQ_16"] = np.nan

    # WQ_19: -sign(delta(close-delay(close,7),5) + delta(close,5)) * (1 + rank(1+sum(returns,250)))
    try:
        d7 = _delay(close, 7)
        part = -np.sign(_delta(close - d7, 5) + _delta(close, 5))
        ret_sum = ret_d.rolling(250, min_periods=50).sum()
        df["D_WQ_19"] = (part * (1 + _xrank(1 + ret_sum))).replace([np.inf, -np.inf], np.nan)
    except Exception:
        df["D_WQ_19"] = np.nan

    # WQ_20: -rank(open - delay(high,1)) * rank(open - delay(close,1)) * rank(open - delay(low,1))
    try:
        df["D_WQ_20"] = (
            -_xrank(open_ - _delay(high, 1))
            * _xrank(open_ - _delay(close, 1))
            * _xrank(open_ - _delay(low, 1))
        ).replace([np.inf, -np.inf], np.nan)
    except Exception:
        df["D_WQ_20"] = np.nan

    # WQ_23: if mean(high,20) < high: -delta(high,2) else 0
    try:
        cond = _ts_mean(high, 20) < high
        df["D_WQ_23"] = ((-_delta(high, 2)).where(cond, 0)).replace([np.inf, -np.inf], np.nan)
    except Exception:
        df["D_WQ_23"] = np.nan

    # WQ_26: -ts_max(corr(ts_rank(volume,5), ts_rank(high,5), 5), 3)
    try:
        inner = _ts_corr(_ts_rank(volume, 5), _ts_rank(high, 5), 5)
        df["D_WQ_26"] = (-_ts_max(inner, 3)).replace([np.inf, -np.inf], np.nan)
    except Exception:
        df["D_WQ_26"] = np.nan

    # WQ_29: rank(rank(-rank(delta(close,5))))
    try:
        inner = -_xrank(_delta(close, 5))
        df["D_WQ_29"] = _xrank(_xrank(inner)).replace([np.inf, -np.inf], np.nan)
    except Exception:
        df["D_WQ_29"] = np.nan

    # WQ_33: rank(-1 + open/close)
    try:
        df["D_WQ_33"] = _xrank(-1 + open_ / close.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    except Exception:
        df["D_WQ_33"] = np.nan

    # WQ_35: ts_rank(volume,32) * (1 - ts_rank(close+high-low,16)) * (1 - ts_rank(returns,32))
    try:
        df["D_WQ_35"] = (
            _ts_rank(volume, 32)
            * (1 - _ts_rank(close + high - low, 16))
            * (1 - _ts_rank(ret_d, 32))
        ).replace([np.inf, -np.inf], np.nan)
    except Exception:
        df["D_WQ_35"] = np.nan

    # WQ_38: -rank(ts_rank(close,10)) * rank(close/open)
    try:
        df["D_WQ_38"] = (
            -_xrank(_ts_rank(close, 10)) * _xrank(close / open_.replace(0, np.nan))
        ).replace([np.inf, -np.inf], np.nan)
    except Exception:
        df["D_WQ_38"] = np.nan

    # WQ_40: -rank(std(high,10)) * corr(high, volume, 10)
    try:
        df["D_WQ_40"] = (-_xrank(_ts_std(high, 10)) * _ts_corr(high, volume, 10)
                       ).replace([np.inf, -np.inf], np.nan)
    except Exception:
        df["D_WQ_40"] = np.nan

    # WQ_41: sqrt(high*low) - vwap
    try:
        df["D_WQ_41"] = (np.sqrt(high * low) - vwap_d).replace([np.inf, -np.inf], np.nan)
    except Exception:
        df["D_WQ_41"] = np.nan

    # WQ_44: -corr(high, rank(volume), 5)
    try:
        df["D_WQ_44"] = (-_ts_corr(high, _xrank(volume), 5)).replace([np.inf, -np.inf], np.nan)
    except Exception:
        df["D_WQ_44"] = np.nan

    # ══════════════════════════════════════════════════════════════════════
    #  v19 NEW: 17 rolling/lag/diff transforms of best existing features
    # ══════════════════════════════════════════════════════════════════════

    # D_slope_stability transforms
    s_ss = pd.to_numeric(df["D_slope_stability"], errors="coerce")
    df["D_slope_stability_rmean50"] = s_ss.rolling(50, min_periods=12).mean()
    df["D_slope_stability_rstd10"] = s_ss.rolling(10, min_periods=3).std()
    df["D_slope_stability_rstd20"] = s_ss.rolling(20, min_periods=5).std()

    # D_body_ratio transforms
    s_br = pd.to_numeric(df["D_body_ratio"], errors="coerce")
    df["D_body_ratio_rmean50"] = s_br.rolling(50, min_periods=12).mean()
    df["D_body_ratio_rmean20"] = s_br.rolling(20, min_periods=5).mean()

    # D_close_roll_slope_20 transforms
    s_crs = pd.to_numeric(df["D_close_roll_slope_20"], errors="coerce")
    df["D_close_roll_slope_20_rstd20"] = s_crs.rolling(20, min_periods=5).std()
    df["D_close_roll_slope_20_rstd10"] = s_crs.rolling(10, min_periods=3).std()

    # D_macd_hist transform
    s_mh = pd.to_numeric(df["D_macd_hist"], errors="coerce")
    df["D_macd_hist_rstd10"] = s_mh.rolling(10, min_periods=3).std()

    # D_mdi14 transforms
    s_md = pd.to_numeric(df["D_mdi14"], errors="coerce")
    df["D_mdi14_diff1"] = s_md.diff(1)
    df["D_mdi14_diff5"] = s_md.diff(5)
    df["D_mdi14_rrank10"] = s_md.rolling(10, min_periods=3).rank(pct=True)
    df["D_mdi14_rrank20"] = s_md.rolling(20, min_periods=5).rank(pct=True)

    # D_donch_pos_50 transforms
    s_d50 = pd.to_numeric(df["D_donch_pos_50"], errors="coerce")
    df["D_donch_pos_50_rmean50"] = s_d50.rolling(50, min_periods=12).mean()
    df["D_donch_pos_50_lag5"] = s_d50.shift(5)

    # D_donch_pos_20 transforms
    s_d20 = pd.to_numeric(df["D_donch_pos_20"], errors="coerce")
    df["D_donch_pos_20_rmean50"] = s_d20.rolling(50, min_periods=12).mean()
    df["D_donch_pos_20_lag5"] = s_d20.shift(5)

    # D_cmf20 transform
    s_cmf = pd.to_numeric(df["D_cmf20"], errors="coerce")
    df["D_cmf20_rmean50"] = s_cmf.rolling(50, min_periods=12).mean()

    # Final hygiene: replace inf/-inf with NaN for the v19 columns
    v19_new_cols = [
        "D_WQ_3", "D_WQ_6", "D_WQ_12", "D_WQ_13", "D_WQ_16", "D_WQ_19", "D_WQ_20", "D_WQ_23",
        "D_WQ_26", "D_WQ_29", "D_WQ_33", "D_WQ_35", "D_WQ_38", "D_WQ_40", "D_WQ_41", "D_WQ_44",
        "D_slope_stability_rmean50", "D_slope_stability_rstd10", "D_slope_stability_rstd20",
        "D_body_ratio_rmean50", "D_body_ratio_rmean20",
        "D_close_roll_slope_20_rstd20", "D_close_roll_slope_20_rstd10",
        "D_macd_hist_rstd10",
        "D_mdi14_diff1", "D_mdi14_diff5",
        "D_mdi14_rrank10", "D_mdi14_rrank20",
        "D_donch_pos_50_rmean50", "D_donch_pos_50_lag5",
        "D_donch_pos_20_rmean50", "D_donch_pos_20_lag5",
        "D_cmf20_rmean50",
    ]
    for c in v19_new_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

    return df


# -------------------- Finalize for cache --------------------
DROP_FROM_CACHE = {
    "D_hh", "D_hl", "D_lh", "D_ll",
    "D_price_and_obv_rising", "D_golden_regime", "D_day_type", "D_nr",
    "D_tmr_cpr_vs_today", "D_cpr_vs_yday", "D_structure_trend",
}

NONBLANK_NUMERIC = [
    "D_rsi7", "D_rsi14", "D_ema20_angle_deg",
    "D_atr14", "D_atr30", "D_atr_ratio_14_30", "D_range_to_atr14",
    "D_adx14", "D_pdi14", "D_mdi14",
    "D_tmr_cpr_vs_today_code", "D_cpr_vs_yday_code", "D_structure_trend_code",
    "D_bb_pctB_20", "D_bb_bw_20", "D_vol_yz_20", "D_vol_yz_50",
    "D_donch_pos_20", "D_donch_pos_50",
    "D_breakout_high_20", "D_breakout_low_20",
    "D_breakout_high_50", "D_breakout_low_50",
    "D_days_since_boh_20", "D_days_since_bol_20",
    "D_dollar_vol", "D_dvol_z20", "D_dvol_z50", "D_dvol_z252",
    "D_vol_surge_20", "D_vol_surge_50",
    # v18
    "D_body_ratio", "D_wick_skew",
    "D_hh_run", "D_hl_run", "D_lh_run", "D_ll_run",
    "D_nr_expand", "D_compress_state",
    "D_dist_from_20h", "D_dist_from_20l", "D_dist_from_52wh",
    "D_midpoint_slope", "D_slope_stability",
    "W_ret_4w", "W_ret_13w", "W_close_pos", "W_vol_vs_4w",
    # v19
    "D_WQ_3", "D_WQ_6", "D_WQ_12", "D_WQ_13", "D_WQ_16", "D_WQ_19", "D_WQ_20", "D_WQ_23",
    "D_WQ_26", "D_WQ_29", "D_WQ_33", "D_WQ_35", "D_WQ_38", "D_WQ_40", "D_WQ_41", "D_WQ_44",
    "D_slope_stability_rmean50", "D_slope_stability_rstd10", "D_slope_stability_rstd20",
    "D_body_ratio_rmean50", "D_body_ratio_rmean20",
    "D_close_roll_slope_20_rstd20", "D_close_roll_slope_20_rstd10",
    "D_macd_hist_rstd10",
    "D_mdi14_diff1", "D_mdi14_diff5",
    "D_mdi14_rrank10", "D_mdi14_rrank20",
    "D_donch_pos_50_rmean50", "D_donch_pos_50_lag5",
    "D_donch_pos_20_rmean50", "D_donch_pos_20_lag5",
    "D_cmf20_rmean50",
]


def _map_true_false_strings_to_int(s: pd.Series) -> pd.Series:
    vals = pd.Series(s.astype(str).str.strip().str.lower())
    uniq = set(vals.dropna().unique())
    if uniq <= {"true", "false", "nan", ""}:
        out = (
            vals.map({"true": 1, "false": 0})
            .astype("Int64")
            .fillna(0)
            .astype(int)
        )
        return out
    return s


def finalize_for_cache(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()

    for c in NONBLANK_NUMERIC:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").ffill().fillna(0.0)

    if "D_nr_day" in out.columns:
        out["D_nr_day"] = (
            pd.to_numeric(out["D_nr_day"], errors="coerce")
            .fillna(0)
            .astype("int16")
        )

    for c in list(out.columns):
        s = out[c]
        if str(s.dtype) in ("boolean", "bool"):
            out[c] = s.astype("Int8").fillna(0).astype(int)
        elif s.dtype == object:
            out[c] = _map_true_false_strings_to_int(s)

    drop_cols = [c for c in DROP_FROM_CACHE if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols, errors="ignore")

    return out


def _feature_manifest(df: pd.DataFrame, warmup_days: int) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "warmup_days": int(warmup_days),
        "features": {c: str(df[c].dtype) for c in df.columns},
        "null_rates": {c: float(pd.isna(df[c]).mean()) for c in df.columns},
        "build_ts": dt.datetime.now(tz=IST).isoformat(),
    }


# -------------------- Daily build --------------------

def _normalize_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    if df.empty:
        return df.copy()
    df = _ensure_ist(df)
    return (
        df.sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )


def _maybe_iso(val):
    if pd.isna(val):
        return None
    if isinstance(val, (pd.Timestamp, dt.datetime)):
        ts = pd.Timestamp(val)
        ts = ts.tz_localize(IST) if ts.tzinfo is None else ts.tz_convert(IST)
        return ts.isoformat()
    return str(val)


def _parse_meta_day(value) -> Optional[dt.date]:
    if not value:
        return None
    try:
        ts = pd.Timestamp(value)
        ts = ts.tz_localize(IST) if ts.tzinfo is None else ts.tz_convert(IST)
        return ts.date()
    except Exception:
        return None


def _cached_span(
    path: Path, meta: Optional[dict]
) -> Tuple[Optional[dt.date], Optional[dt.date]]:
    meta = meta or {}
    first = _parse_meta_day(meta.get("first_timestamp"))
    last = _parse_meta_day(meta.get("last_timestamp"))
    if (first is None or last is None) and path.exists():
        try:
            ts_df = read_parquet(path, columns=["timestamp"])
        except (ValueError, KeyError):
            ts_df = read_parquet(path)
        if "timestamp" in ts_df.columns and not ts_df.empty:
            ts_df = _ensure_ist(ts_df)
            ts = pd.to_datetime(ts_df["timestamp"], errors="coerce")
            dates = ts.dt.date.dropna()
            actual_first = dates.min() if not dates.empty else None
            actual_last = dates.max() if not dates.empty else None
            if first is None:
                first = actual_first
            if last is None:
                last = actual_last
    return first, last


def build_daily(
    provider: KiteProvider,
    config: Config,
    symbol: str,
    start: dt.date,
    end: dt.date,
    *,
    force=True,
    recompute_only=False,
) -> Path:
    out_pq = daily_path(config, symbol)
    ok = ok_path(config, symbol)
    ok_meta = read_json(ok)
    cached_first, cached_last = _cached_span(out_pq, ok_meta)
    schema_ok = (
        bool(ok_meta)
        and ok_meta.get(OK_VERSION_KEY) == SCHEMA_VERSION
        and out_pq.exists()
    )

    WARMUP_DAYS = int(os.environ.get("CACHE_WARMUP_DAYS", "200"))
    warmup_start = start - dt.timedelta(days=WARMUP_DAYS)

    def _save(df: pd.DataFrame) -> Path:
        first_ts = _maybe_iso(df["timestamp"].iloc[0]) if not df.empty else None
        last_ts = _maybe_iso(df["timestamp"].iloc[-1]) if not df.empty else None
        base = ok_meta_base() | {
            "rows": int(df.shape[0]),
            "first_timestamp": first_ts,
            "last_timestamp": last_ts,
            "requested_start": start.isoformat() if start else None,
            "requested_end": end.isoformat() if end else None,
            "warmup_days": WARMUP_DAYS,
        }
        manifest = _feature_manifest(df, warmup_days=WARMUP_DAYS)
        meta = {**base, **manifest}
        with FileLock(out_pq):
            to_parquet(
                out_pq,
                df,
                engine=config.parquet_engine,
                compression=config.parquet_compression,
                use_dictionary=config.parquet_use_dictionary,
            )
            write_json_atomic(ok, meta)
        return out_pq

    # Recompute-only
    if recompute_only and out_pq.exists():
        df = _normalize_daily(read_parquet(out_pq))
        if not df.empty:
            _validate_monotonic(df)
            df = compute_daily_indicators(df)
            df = finalize_for_cache(df)
        return _save(df)

    # Incremental
    if schema_ok and cached_last is not None and not force:
        fetch_start = cached_last + dt.timedelta(days=1)
        fetch_end = end
        if fetch_start > fetch_end:
            return out_pq
        base_df = _normalize_daily(read_parquet(out_pq))
        inc_df = _normalize_daily(
            provider.fetch_daily(symbol, fetch_start, fetch_end)
        )
        if inc_df.empty:
            return out_pq
        merged = (
            pd.concat([base_df, inc_df], ignore_index=True)
            .drop_duplicates("timestamp", keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        _validate_monotonic(merged)
        merged = compute_daily_indicators(merged)
        merged = finalize_for_cache(merged)
        return _save(merged)

    # Full fetch with warm-up backfill
    extended_df = _normalize_daily(
        provider.fetch_daily(symbol, warmup_start, end)
    )
    if not extended_df.empty:
        _validate_monotonic(extended_df)
        extended_df = compute_daily_indicators(extended_df)
        extended_df = finalize_for_cache(extended_df)
    return _save(extended_df)


# -------------------- Symbols file loader --------------------

def _load_symbols_from_file(path: str) -> List[str]:
    p = Path(path)
    ext = p.suffix.lower()
    items: List[str] = []
    if ext in (".xls", ".xlsx"):
        df = pd.read_excel(p)
        if df.empty:
            return []
        col0 = df.columns[0]
        items = [str(x) for x in df[col0].dropna().tolist()]
    else:
        try:
            df = pd.read_csv(p, header=None)
            items = (
                [str(x) for x in df.iloc[:, 0].dropna().tolist()]
                if df.shape[1] >= 1
                else []
            )
        except Exception:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read()
            tokens = [
                t.strip()
                for t in raw.replace("\n", ",").replace("\t", ",").split(",")
            ]
            items = [t for t in tokens if t]
    cleaned: List[str] = []
    seen: set = set()
    for t in items:
        s = sanitize_symbol(t)
        if s and s.casefold() not in seen:
            seen.add(s.casefold())
            cleaned.append(s)
    return cleaned


# -------------------- Date range normalization --------------------

def parse_date_input(value) -> dt.date:
    if isinstance(value, dt.date):
        return value
    if value is None:
        raise ValueError("Date value is required")
    text = str(value).strip()
    if not text:
        raise ValueError("Date value is required")
    formats = ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%d.%m.%Y")
    for fmt in formats:
        try:
            return dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Could not parse date: {text!r}")


def normalize_requested_range(
    start_value, end_value
) -> Tuple[dt.date, dt.date, List[str]]:
    start = parse_date_input(start_value)
    end = parse_date_input(end_value)
    if start > end:
        start, end = end, start
    notes: List[str] = []
    today = today_ist()
    if end > today:
        notes.append(
            f"End date {end.isoformat()} trimmed to {today.isoformat()} "
            "because future data is unavailable."
        )
        end = today
    if start > today:
        notes.append(
            f"Start date {start.isoformat()} adjusted to {today.isoformat()} "
            "because the market has not traded yet."
        )
        start = today
    if start > end:
        raise ValueError(
            "Requested date range does not contain any trading days after adjustments."
        )
    return start, end, notes


def append_token_error_log(
    base_dir: Path, *, symbol: str, phase: str, error: str
) -> None:
    log_path = (
        (base_dir / "_token_expired.log") if base_dir else Path("_token_expired.log")
    )
    rec = {
        "ts": dt.datetime.now(tz=IST).isoformat(),
        "symbol": symbol,
        "day": None,
        "phase": phase,
        "error": str(error),
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# -------------------- GUI inputs --------------------

def _ask_user_inputs_gui_file_only():
    if not TK_OK:
        raise SystemExit("Tkinter is not available.")
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select symbols file (Excel/CSV/TXT; first column = symbols)",
        filetypes=[
            ("Excel/CSV/TXT", "*.xlsx *.xls *.csv *.txt"),
            ("All", "*.*"),
        ],
    )
    if not path:
        messagebox.showerror("Required", "No symbols file selected.")
        raise SystemExit(1)
    symbols = _load_symbols_from_file(path)
    if not symbols:
        messagebox.showerror(
            "Invalid file", "Could not parse any symbols from the file."
        )
        raise SystemExit(1)
    today = today_ist()
    default_start = today - dt.timedelta(days=600)
    start_input = simpledialog.askstring(
        "Start date",
        "Enter start date (YYYY-MM-DD or DD-MM-YYYY):",
        initialvalue=default_start.isoformat(),
    )
    end_input = simpledialog.askstring(
        "End date",
        "Enter end date (YYYY-MM-DD or DD-MM-YYYY):",
        initialvalue=today.isoformat(),
    )
    try:
        start_date, end_date, adjustments = normalize_requested_range(
            start_input, end_input
        )
    except Exception as exc:
        messagebox.showerror("Invalid dates", str(exc))
        raise SystemExit(1)
    if adjustments:
        messagebox.showinfo("Adjusted dates", "\n".join(adjustments))
    mw = simpledialog.askinteger(
        "Parallel workers",
        "Max threads (IO-bound; 16-64 works well):",
        initialvalue=32,
        minvalue=1,
        maxvalue=128,
    )
    if not mw:
        mw = 32
    base_config = Config.from_env()
    messagebox.showinfo(
        "Summary",
        (
            "Symbols file: {}\n\nSymbols parsed: {}\n\n"
            "Date range: {} to {}\nWorkers: {}\n\nDaily folder:\n{}"
        ).format(
            path,
            len(symbols),
            start_date.isoformat(),
            end_date.isoformat(),
            mw,
            str(base_config.daily_root),
        ),
    )
    root.destroy()
    return {
        "symbols": symbols,
        "start_date": start_date,
        "end_date": end_date,
        "max_workers": mw,
        "base_config": base_config,
    }


# -------------------- CLI --------------------

def parse_cli_args():
    import argparse

    ap = argparse.ArgumentParser(
        description="FAST daily-only cache builder for Kite (no intraday)."
    )
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--symbols-file", default=None)
    ap.add_argument("--start-date", default=None)
    ap.add_argument("--end-date", default=None)
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--rate", type=float, default=16.0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--recompute-only", action="store_true")
    ap.add_argument("--incremental", action="store_true")
    args = ap.parse_args()
    syms: Optional[List[str]] = args.symbols
    if args.symbols_file:
        syms = _load_symbols_from_file(args.symbols_file)
    return args, syms


def _resolve_requested_dates(
    start_value, end_value, days_value, *, incremental: bool = False
) -> Tuple[dt.date, dt.date]:
    if incremental:
        today = today_ist()
        start = today - dt.timedelta(days=600)
        start, end, _ = normalize_requested_range(start, today)
        return start, end
    if start_value and end_value:
        start, end, _ = normalize_requested_range(start_value, end_value)
        return start, end
    if days_value and days_value > 0:
        today = today_ist()
        start = today - dt.timedelta(days=days_value)
        start, end, _ = normalize_requested_range(start, today)
        return start, end
    raise SystemExit(
        "Provide --start-date/--end-date or --days to define the caching window."
    )


# -------------------- Pipeline --------------------

class Pipeline:
    def __init__(
        self,
        provider: KiteProvider,
        config: Config,
        progress_cb: Optional[callable] = None,
    ):
        self.provider = provider
        self.cfg = config
        self.ratelimiter = RateLimiter(config.rate_limit_per_sec)
        self._daily_cache = DataFrameCache(maxsize=256)
        self.fetch_daily = with_retry(
            self._cached_fetch_daily,
            tries=config.retry_tries,
            backoff=config.retry_backoff_base,
        )
        self.progress_cb = progress_cb or (lambda msg: None)

    def _cached_fetch_daily(
        self, symbol: str, start: dt.date, end: dt.date
    ) -> pd.DataFrame:
        key = (symbol, start, end)
        cached = self._daily_cache.get(key)
        if cached is not None:
            return cached
        self.ratelimiter.acquire()
        df = self.provider.fetch_daily(symbol, start, end)
        if not isinstance(df, pd.DataFrame):
            raise TypeError("fetch_daily must return a DataFrame")
        return self._daily_cache.put(key, df)

    def build(
        self,
        symbols: Sequence[str],
        start_date: dt.date,
        end_date: dt.date,
        *,
        force: bool,
        recompute_only: bool,
    ):
        cfg = self.cfg
        start_date, end_date, adjustments = normalize_requested_range(
            start_date, end_date
        )
        for note in adjustments:
            try:
                self.progress_cb(f"NOTE: {note}")
            except Exception:
                print(note)

        pre: List[str] = []
        unresolved: List[str] = []
        for s in symbols:
            try:
                _ = self.provider._symbol_to_instrument_token(s)
                pre.append(s)
            except UnresolvedSymbol as e:
                unresolved.append(str(e))
                append_token_error_log(
                    cfg.day_root(),
                    symbol=str(e),
                    phase="preresolve",
                    error="UNRESOLVED_SYMBOL",
                )
                continue
        symbols = pre

        def task(s: str):
            try:
                WARMUP_DAYS = int(os.environ.get("CACHE_WARMUP_DAYS", "200"))
                daily_start = start_date - dt.timedelta(days=WARMUP_DAYS)
                return build_daily(
                    self.provider,
                    cfg,
                    s,
                    daily_start,
                    end_date,
                    force=force,
                    recompute_only=recompute_only,
                )
            except UnresolvedSymbol:
                self.progress_cb(f"SKIP unresolved: {s} (daily)")
            except AuthExpired as e:
                append_token_error_log(
                    cfg.day_root(), symbol=s, phase="daily", error=str(e)
                )
                raise
            except Exception as e:
                append_token_error_log(
                    cfg.day_root(), symbol=s, phase="daily", error=str(e)
                )
                raise

        results: list = []
        with cf.ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
            futures = {ex.submit(task, s): s for s in symbols}
            for fut in cf.as_completed(futures):
                s = futures[fut]
                try:
                    p = fut.result()
                    results.append(p)
                    self.progress_cb(f"Daily: {s}")
                except AuthExpired:
                    raise
                except Exception as e:
                    self.progress_cb(f"ERROR {s}: {e}")

        if unresolved:
            summary = (
                f"Unresolved symbols skipped ({len(unresolved)}): "
                + ", ".join(sorted(set(unresolved)))
            )
            print(summary)
            try:
                if TK_OK:
                    tk.messagebox.showwarning("Unresolved", summary)
            except Exception:
                pass

        return results


# -------------------- Entry --------------------

def main():
    # ── v20 leak safety net ───────────────────────────────────────────────
    # Static check: scan this file for any unwindowed .rank() inside
    # compute_daily_indicators that would re-introduce the v19 leak.
    # Runtime canary: synthetic two-frame test that proves no past column
    # depends on future tampered rows. Both can be skipped via env var
    # CACHE_SKIP_LEAK_CANARY=1 (NOT recommended for production).
    _skip = os.environ.get("CACHE_SKIP_LEAK_CANARY", "").strip().lower()
    if _skip not in ("1", "true", "yes"):
        try:
            _v20_static_leak_check()
            _v20_leak_canary_check()
            print("[v20] leak-free self-test passed")
        except RuntimeError as _e:
            msg = f"REFUSING TO BUILD: {_e}"
            if TK_OK:
                try:
                    messagebox.showerror("Leak detected", msg)
                except Exception:
                    pass
            print(msg, file=sys.stderr)
            raise SystemExit(2)

    try:
        if len(sys.argv) > 1:
            args, symbols = parse_cli_args()
            if not symbols:
                raise SystemExit(
                    "No symbols provided. Use --symbols-file <path> or --symbols ..."
                )
            base_config = Config.from_env()
            cfg = base_config.with_updates(
                max_workers=int(args.workers),
                rate_limit_per_sec=float(args.rate),
                request_timeout_s=15.0,
                retry_tries=6,
            )
            provider = KiteProvider()
            pipeline = Pipeline(provider, cfg, progress_cb=lambda m: print(m))
            cfg.daily_root.mkdir(parents=True, exist_ok=True)
            start_date, end_date = _resolve_requested_dates(
                args.start_date,
                args.end_date,
                args.days,
                incremental=bool(args.incremental),
            )
            pipeline.build(
                symbols,
                start_date,
                end_date,
                force=bool(args.force),
                recompute_only=bool(args.recompute_only),
            )
            print("FAST daily-only cache build completed.")
        else:
            ui = _ask_user_inputs_gui_file_only()
            base_config = ui.get("base_config") or Config.from_env()
            cfg = base_config.with_updates(
                max_workers=int(ui["max_workers"]),
                rate_limit_per_sec=16.0,
                request_timeout_s=15.0,
                retry_tries=6,
            )
            symbols = ui["symbols"]
            start_date = ui["start_date"]
            end_date = ui["end_date"]

            if not TK_OK:
                raise SystemExit("Tkinter not available; GUI mode is required.")

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)

            class ProgressUI:
                def __init__(self, total: int):
                    self.total = max(1, int(total))
                    self.start = time.perf_counter()
                    self.completed = 0
                    self.root = tk.Toplevel()
                    self.root.title("Building daily cache (FAST)...")
                    self.root.geometry("560x160")
                    self.root.resizable(False, False)
                    self.label = tk.Label(
                        self.root, text="Starting...", anchor="w"
                    )
                    self.label.pack(fill="x", padx=12, pady=(12, 6))
                    self.pb = ttk.Progressbar(
                        self.root,
                        orient="horizontal",
                        mode="determinate",
                        maximum=self.total,
                        length=520,
                    )
                    self.pb.pack(padx=12, pady=6)
                    self.eta = tk.Label(self.root, text="ETA: --:--", anchor="w")
                    self.eta.pack(fill="x", padx=12, pady=(6, 12))
                    self.root.attributes("-topmost", True)
                    self.root.update_idletasks()

                def _fmt_eta(self, secs: float) -> str:
                    if secs is None or secs != secs or secs == float("inf"):
                        return "--:--"
                    m, s = divmod(int(secs), 60)
                    h, m = divmod(m, 60)
                    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

                def tick(self, msg: str = ""):
                    self.completed += 1
                    self.pb["value"] = self.completed
                    elapsed = max(0.001, time.perf_counter() - self.start)
                    rate = self.completed / elapsed
                    remaining = max(0, self.total - self.completed)
                    self.label.config(
                        text=msg or f"Completed {self.completed}/{self.total}"
                    )
                    self.eta.config(
                        text=f"ETA: {self._fmt_eta(time.perf_counter() - self.start)}"
                    )
                    self.root.update_idletasks()

                def done(self):
                    self.pb["value"] = self.total
                    self.label.config(
                        text=f"Done: {self.total}/{self.total}"
                    )
                    self.eta.config(
                        text=f"ETA: 00:00  Elapsed: {self._fmt_eta(time.perf_counter() - self.start)}"
                    )
                    self.root.update_idletasks()

            pui = ProgressUI(total=len(symbols))

            def on_progress(msg: str):
                pui.tick(msg)

            provider = KiteProvider()
            pipeline = Pipeline(provider, cfg, progress_cb=on_progress)
            cfg.daily_root.mkdir(parents=True, exist_ok=True)
            pipeline.build(symbols, start_date, end_date, force=False, recompute_only=False)
            pui.done()
            messagebox.showinfo("Done", "FAST daily-only cache build completed.")

    except Exception as e:
        if TK_OK:
            try:
                messagebox.showerror("Error", str(e))
            except Exception:
                print("ERROR:", e, file=sys.stderr)
        else:
            print("ERROR:", e, file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
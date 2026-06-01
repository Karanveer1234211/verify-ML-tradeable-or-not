#!/usr/bin/env python3
"""
=============================================================================
INTRADAY 5-MIN CACHER v2 - Full Universe, Seamless Kite Auth
=============================================================================

Reuses your existing Kite authentication from kite_token.json.
Caches 5-min OHLCV for ALL symbols in your universe (~2,500 symbols).

What it does:
  1. Reads access_token from your existing kite_token.json
  2. Loads symbol universe from your daily cache folder structure
  3. Fetches 5-min bars from Kite for each symbol over the last 5 years
  4. Saves one parquet per symbol, resumable
  5. Skips symbols where access fails (logged for review)

Storage:
  - Output: ~40-60 GB for full universe
  - First run: 6-10 hours
  - Incremental: ~15-20 minutes/day

Token refresh:
  - If access_token is expired, prompts you to run your auth script first
  - Does NOT do auth itself (your separate code handles that cleanly)

Usage:
  python intraday_cacher.py                  # cache all from daily cache folder
  python intraday_cacher.py --limit 10       # test with 10 symbols
  python intraday_cacher.py --symbols A,B,C  # specific symbols
  python intraday_cacher.py --full-rebuild   # force refetch
  python intraday_cacher.py --from-file path # custom symbol list
=============================================================================
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG - YOUR PATHS
# =============================================================================

def clean_symbol(sym: str) -> str:
    sym = sym.upper().strip()

    # Remove suffixes
    if sym.endswith("_DAILY"):
        sym = sym[:-6]
    if sym.endswith("_INTRADAY"):
        sym = sym[:-9]

    # Remove file extensions
    for ext in [".PKL", ".PARQUET", ".CSV"]:
        if sym.endswith(ext):
            sym = sym[:-len(ext)]

    return sym

# Kite auth (from your separate auth script)
KITE_TOKEN_FILE = Path(r"C:\Users\karanvsi\PyCharmMiscProject\kite_token.json")

# Master symbol list file (one symbol per line)
# Adjust this path to your master file
MASTER_SYMBOL_FILE = Path(r"C:\Users\karanvsi\PyCharmMiscProject\symbols_master.txt")

# Fallback: daily cache folder (if master file not found)
DAILY_CACHE_DIR = Path(r"C:\Users\karanvsi\Desktop\Pycharm\Cache\cache_daily_new")

# Where to store 5-min parquet files
INTRADAY_DIR = Path(r"C:\Users\karanvsi\Desktop\Pycharm\Cache\intraday_5min")
INTRADAY_DIR.mkdir(parents=True, exist_ok=True)

# Internal cache files
INSTRUMENT_MAP_PATH = INTRADAY_DIR / "_instrument_map.parquet"
CACHE_METADATA_PATH = INTRADAY_DIR / "_cache_metadata.parquet"
FAILURE_LOG_PATH = INTRADAY_DIR / "_failures.csv"

# Fetch config
HISTORY_YEARS = 5
INTERVAL = "5minute"

# Kite API limits
KITE_RATE_LIMIT_PER_SEC = 3
KITE_MAX_DAYS_PER_REQUEST = 60
KITE_BATCH_SIZE = 50  # save metadata every N symbols

IST = "Asia/Kolkata"


# =============================================================================
# KITE CLIENT (using your existing token file)
# =============================================================================

def get_kite_client():
    """
    Load Kite client using your existing token file.
    No new auth logic - if token expired, instruct user to refresh.
    """
    if not KITE_TOKEN_FILE.exists():
        print("\n" + "=" * 70)
        print("KITE TOKEN FILE NOT FOUND")
        print("=" * 70)
        print(f"\nExpected: {KITE_TOKEN_FILE}")
        print("\nRun your Kite auth script first to create the token file.")
        print("=" * 70)
        sys.exit(1)

    try:
        with open(KITE_TOKEN_FILE, "r") as f:
            token_data = json.load(f)
    except Exception as e:
        print(f"FATAL: Failed to read token file: {e}")
        sys.exit(1)

    api_key = token_data.get("api_key")
    access_token = token_data.get("access_token")

    if not api_key or not access_token:
        print("\nFATAL: api_key or access_token missing in token file.")
        print(f"Token file: {KITE_TOKEN_FILE}")
        print("\nRun your Kite auth script to populate it.")
        sys.exit(1)

    try:
        from kiteconnect import KiteConnect
    except ImportError:
        print("\nFATAL: kiteconnect not installed. Run: pip install kiteconnect")
        sys.exit(1)

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)

    # Verify connection - if token is stale, this fails
    try:
        profile = kite.profile()
        print(f"[Kite] Authenticated as: {profile['user_name']} ({profile['user_id']})")
    except Exception as e:
        err_str = str(e)
        if "TokenException" in err_str or "expired" in err_str.lower() or "invalid" in err_str.lower():
            print("\n" + "=" * 70)
            print("ACCESS TOKEN EXPIRED")
            print("=" * 70)
            print(f"\nYour Kite access token has expired (they expire daily ~6 AM IST).")
            print("\nRun your Kite auth script to refresh it, then re-run this cacher.")
            print(f"\nToken file: {KITE_TOKEN_FILE}")
            print("=" * 70)
        else:
            print(f"\nFATAL: Kite auth failed: {e}")
        sys.exit(1)

    return kite


# =============================================================================
# INSTRUMENT MAPPING
# =============================================================================

def load_or_fetch_instrument_map(kite) -> pd.DataFrame:
    """Symbol -> instrument_token mapping. Cached daily."""
    if INSTRUMENT_MAP_PATH.exists():
        age_hours = (datetime.now().timestamp() - INSTRUMENT_MAP_PATH.stat().st_mtime) / 3600
        if age_hours < 24:
            print(f"[Instruments] Using cached map ({age_hours:.1f}h old)")
            return pd.read_parquet(INSTRUMENT_MAP_PATH)

    print("[Instruments] Fetching from Kite...")
    instruments = kite.instruments("NSE")
    df = pd.DataFrame(instruments)
    df = df[df["segment"] == "NSE"].copy()
    df = df[df["instrument_type"] == "EQ"].copy()
    df = df[["tradingsymbol", "instrument_token", "name", "exchange", "lot_size", "tick_size"]]
    df.columns = ["symbol", "instrument_token", "name", "exchange", "lot_size", "tick_size"]
    df.to_parquet(INSTRUMENT_MAP_PATH, index=False)
    print(f"[Instruments] Cached {len(df):,} NSE EQ instruments")
    return df


# =============================================================================
# SYMBOL UNIVERSE FROM DAILY CACHE
# =============================================================================

def get_universe_symbols() -> List[str]:
    """
    Read the symbol universe from your master symbol list file.
    Falls back to daily cache folder if master file doesn't exist.

    Master file format: one symbol per line, comments start with #
    """
    # Primary: master symbol list file
    if MASTER_SYMBOL_FILE.exists():
        print(f"\n[Universe] Reading symbol list from {MASTER_SYMBOL_FILE}")
        symbols = []
        with open(MASTER_SYMBOL_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Handle CSV-style lines (take first column)
                if "," in line:
                    line = line.split(",")[0].strip()
                symbols.append(clean_symbol(line))
        symbols = sorted(set(symbols))
        print(f"[Universe] Loaded {len(symbols):,} symbols from master file")
        if len(symbols) > 0:
            print(f"  Examples: {', '.join(symbols[:5])}")
        return symbols

    # Fallback: daily cache folder
    print(f"\n[Universe] Master file not found at {MASTER_SYMBOL_FILE}")
    print(f"[Universe] Falling back to daily cache folder: {DAILY_CACHE_DIR}")

    if not DAILY_CACHE_DIR.exists():
        raise SystemExit(f"FATAL: Neither master file nor daily cache dir found")

    parquet_files = list(DAILY_CACHE_DIR.glob("*.parquet"))
    csv_files = list(DAILY_CACHE_DIR.glob("*.csv"))

    if len(parquet_files) >= len(csv_files):
        files = parquet_files
        ext = ".parquet"
    else:
        files = csv_files
        ext = ".csv"

    symbols = []
    for f in files:
        name = f.stem
        if name.startswith("_"):
            continue
        symbols.append(clean_symbol(name))

    symbols = sorted(set(symbols))
    print(f"[Universe] Found {len(symbols):,} symbols from cache folder (using {ext})")
    if len(symbols) > 0:
        print(f"  Examples: {', '.join(symbols[:5])}")

    return symbols


# =============================================================================
# FETCH HELPERS
# =============================================================================

def load_metadata() -> pd.DataFrame:
    if CACHE_METADATA_PATH.exists():
        return pd.read_parquet(CACHE_METADATA_PATH)
    return pd.DataFrame(columns=["symbol", "last_cached_date", "n_bars", "first_date"])


def save_metadata(meta: pd.DataFrame):
    meta.to_parquet(CACHE_METADATA_PATH, index=False)


def log_failure(symbol: str, reason: str):
    row = pd.DataFrame([{
        "timestamp": pd.Timestamp.now(),
        "symbol": symbol,
        "reason": reason,
    }])
    if FAILURE_LOG_PATH.exists():
        row.to_csv(FAILURE_LOG_PATH, mode="a", header=False, index=False)
    else:
        row.to_csv(FAILURE_LOG_PATH, index=False)


def fetch_symbol_bars(
        kite,
        symbol: str,
        instrument_token: int,
        from_date: datetime,
        to_date: datetime,
        rate_limiter: dict,
) -> Optional[pd.DataFrame]:
    """
    Fetch 5-min bars in 60-day chunks.
    Handles rate limiting and retries.
    """
    all_chunks = []
    chunk_start = from_date

    while chunk_start < to_date:
        chunk_end = min(chunk_start + timedelta(days=KITE_MAX_DAYS_PER_REQUEST - 1), to_date)

        # Rate limit
        now = time.time()
        elapsed = now - rate_limiter["last_call"]
        min_interval = 1.0 / KITE_RATE_LIMIT_PER_SEC
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        rate_limiter["last_call"] = time.time()

        try:
            data = kite.historical_data(
                instrument_token=instrument_token,
                from_date=chunk_start,
                to_date=chunk_end,
                interval=INTERVAL,
            )
        except Exception as e:
            err_str = str(e)
            if "Too many requests" in err_str or "rate" in err_str.lower():
                time.sleep(3)  # back off
                try:
                    data = kite.historical_data(
                        instrument_token=instrument_token,
                        from_date=chunk_start,
                        to_date=chunk_end,
                        interval=INTERVAL,
                    )
                except Exception as e2:
                    log_failure(symbol, f"API error after retry: {e2}")
                    return None
            elif "TokenException" in err_str:
                print(f"\n[FATAL] Token expired during run. Refresh and resume.")
                sys.exit(1)
            else:
                log_failure(symbol, f"API error: {e}")
                return None

        if data:
            chunk_df = pd.DataFrame(data)
            all_chunks.append(chunk_df)

        chunk_start = chunk_end + timedelta(days=1)

    if not all_chunks:
        return None

    df = pd.concat(all_chunks, ignore_index=True)
    if len(df) == 0:
        return None

    df = df.rename(columns={"date": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(IST)
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert(IST)

    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    return df


# =============================================================================
# CACHE A SINGLE SYMBOL
# =============================================================================

def cache_symbol(
        kite,
        symbol: str,
        instrument_token: int,
        rate_limiter: dict,
        force_full: bool = False,
) -> Tuple[bool, int]:
    """
    Cache one symbol. Incremental if existing parquet found.
    Returns (success, n_bars).
    """
    file_path = INTRADAY_DIR / f"{symbol}.parquet"
    end_date = datetime.now()

    existing = None
    if file_path.exists() and not force_full:
        try:
            existing = pd.read_parquet(file_path)
            existing["timestamp"] = pd.to_datetime(existing["timestamp"])
            if existing["timestamp"].dt.tz is None:
                existing["timestamp"] = existing["timestamp"].dt.tz_localize(IST)
            else:
                existing["timestamp"] = existing["timestamp"].dt.tz_convert(IST)

            last_ts = existing["timestamp"].max()
            start_date = (last_ts + pd.Timedelta(days=1)).to_pydatetime()
            if start_date.tzinfo is not None:
                start_date = start_date.replace(tzinfo=None)

            if start_date.date() >= end_date.date():
                # Already up to date
                return True, len(existing)
        except Exception as e:
            log_failure(symbol, f"Failed to read existing cache, will refetch: {e}")
            start_date = end_date - timedelta(days=int(HISTORY_YEARS * 365.25))
            existing = None
    else:
        start_date = end_date - timedelta(days=int(HISTORY_YEARS * 365.25))

    new_df = fetch_symbol_bars(
        kite=kite,
        symbol=symbol,
        instrument_token=instrument_token,
        from_date=start_date,
        to_date=end_date,
        rate_limiter=rate_limiter,
    )

    if new_df is None or len(new_df) == 0:
        if existing is not None:
            return True, len(existing)  # have old data, no new data is OK
        return False, 0

    # Merge with existing
    if existing is not None and len(existing) > 0:
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    else:
        combined = new_df

    combined.to_parquet(file_path, index=False)
    return True, len(combined)


# =============================================================================
# MAIN BUILD LOOP
# =============================================================================

def run_cache_build(target_symbols: List[str], force_full: bool = False):
    print("\n" + "=" * 80)
    print("INTRADAY 5-MIN CACHE BUILD")
    print("=" * 80)
    print(f"Symbols to cache: {len(target_symbols)}")
    print(f"History years: {HISTORY_YEARS}")
    print(f"Force full refetch: {force_full}")
    print(f"Output: {INTRADAY_DIR}")
    print(f"Token file: {KITE_TOKEN_FILE}")

    # Auth
    kite = get_kite_client()
    instrument_map = load_or_fetch_instrument_map(kite)

    # Symbol -> token
    token_map = dict(zip(instrument_map["symbol"], instrument_map["instrument_token"]))
    missing = [s for s in target_symbols if s not in token_map]
    if missing:
        print(f"\n[WARN] {len(missing)} symbols not found in Kite NSE EQ list:")
        print(f"  Examples: {missing[:10]}")
        for s in missing:
            log_failure(s, "Not in Kite NSE EQ instrument list")

    valid_targets = [s for s in target_symbols if s in token_map]
    print(f"\nWill cache {len(valid_targets)} symbols.\n")

    if len(valid_targets) == 0:
        print("No symbols to cache.")
        return

    # Initialize state
    meta = load_metadata()
    rate_limiter = {"last_call": 0.0}
    t_start = time.time()
    n_success = 0
    n_failed = 0

    # Check disk space
    try:
        import shutil
        free_gb = shutil.disk_usage(INTRADAY_DIR).free / (1024 ** 3)
        print(f"Free disk space: {free_gb:.1f} GB")
        if free_gb < 50:
            print(f"WARNING: Low disk space. Full cache may need 40-60 GB.")
    except Exception:
        pass

    print()

    for i, symbol in enumerate(valid_targets):
        # Periodic progress report
        if i % 25 == 0 and i > 0:
            elapsed = time.time() - t_start
            rate = i / elapsed
            eta = (len(valid_targets) - i) / rate if rate > 0 else 0
            try:
                total_size_gb = sum(p.stat().st_size for p in INTRADAY_DIR.glob("*.parquet")) / (1024 ** 3)
            except Exception:
                total_size_gb = 0
            print(f"\n[Progress] {i}/{len(valid_targets)} ({100 * i / len(valid_targets):.1f}%) "
                  f"elapsed={elapsed / 60:.1f}m  ETA={eta / 60:.1f}m  "
                  f"success={n_success}  failed={n_failed}  size={total_size_gb:.1f}GB\n")

        token = token_map[symbol]

        try:
            ok, n_bars = cache_symbol(
                kite=kite,
                symbol=symbol,
                instrument_token=token,
                rate_limiter=rate_limiter,
                force_full=force_full,
            )
            if ok:
                n_success += 1
                print(f"  [OK] {symbol:<15} bars={n_bars:>7,}")

                meta = meta[meta["symbol"] != symbol]
                meta = pd.concat([meta, pd.DataFrame([{
                    "symbol": symbol,
                    "last_cached_date": pd.Timestamp.now(),
                    "n_bars": n_bars,
                    "first_date": pd.Timestamp.now() - pd.Timedelta(days=HISTORY_YEARS * 365),
                }])], ignore_index=True)
            else:
                n_failed += 1
                print(f"  [FAIL] {symbol}")
        except KeyboardInterrupt:
            print(f"\n\n[Cancelled] Saving progress and exiting...")
            save_metadata(meta)
            print(f"  Progress saved. Re-run to resume.")
            sys.exit(0)
        except Exception as e:
            n_failed += 1
            print(f"  [ERROR] {symbol}: {e}")
            log_failure(symbol, str(e))

        if (i + 1) % KITE_BATCH_SIZE == 0:
            save_metadata(meta)

    save_metadata(meta)

    # Final summary
    elapsed = time.time() - t_start
    try:
        total_size_gb = sum(p.stat().st_size for p in INTRADAY_DIR.glob("*.parquet")) / (1024 ** 3)
    except Exception:
        total_size_gb = 0

    print("\n" + "=" * 80)
    print("CACHE BUILD COMPLETE")
    print("=" * 80)
    print(f"Elapsed: {elapsed / 60:.1f} minutes ({elapsed / 3600:.1f} hours)")
    print(f"Success: {n_success}")
    print(f"Failed:  {n_failed}")
    print(f"Output:  {INTRADAY_DIR}")
    print(f"Total cache size: {total_size_gb:.2f} GB")
    if FAILURE_LOG_PATH.exists():
        print(f"Failure log: {FAILURE_LOG_PATH}")


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    p = argparse.ArgumentParser(description="Cache 5-min OHLCV via Kite Connect (full universe)")
    p.add_argument("--full-rebuild", action="store_true",
                   help="Refetch all data even if cached")
    p.add_argument("--symbols", type=str, default=None,
                   help="Comma-separated symbols (overrides universe)")
    p.add_argument("--limit", type=int, default=None,
                   help="Cache only first N symbols (for testing)")
    p.add_argument("--from-file", type=str, default=None,
                   help="Path to file with one symbol per line")
    args = p.parse_args()

    # Determine target symbols
    if args.symbols:
        target_symbols = [s.strip().upper() for s in args.symbols.split(",")]
        print(f"Using {len(target_symbols)} symbols from --symbols")
    elif args.from_file:
        with open(args.from_file, "r") as f:
            target_symbols = [line.strip().upper() for line in f if line.strip()]
        print(f"Loaded {len(target_symbols)} symbols from {args.from_file}")
    else:
        target_symbols = get_universe_symbols()

    if args.limit:
        target_symbols = target_symbols[:args.limit]
        print(f"Limited to first {len(target_symbols)} symbols")

    run_cache_build(target_symbols, force_full=args.full_rebuild)


if __name__ == "__main__":
    main()
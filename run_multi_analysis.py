"""
QuantAgent Multi-Ticker CLI Script
Fetches stock data for multiple tickers using yfinance_cache with aggressive rate limiting.

Rate Limit Avoidance Strategy (priority order):
1. Cache thoroughly using yfinance_cache
2. Space out requests with global rate limiter (min spacing + per-minute cap)
3. Minimize parallelism (default max-workers=1)
4. Exponential backoff with 429/Retry-After awareness
5. Stable User-Agent via shared requests.Session
"""
import argparse
import sys
import time
import json
import csv
import random
import threading
import email.utils as email_utils
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import pandas as pd
import requests

from run_analysis import (
    fetch_stock_data,
    prepare_data_for_analysis,
    run_analysis,
    format_timeframe_display,
    print_results
)


class YahooRateLimiter:
    """
    Global rate limiter for Yahoo Finance API to avoid 429 errors.

    Implements:
    - Minimum spacing between requests (with jitter)
    - Per-minute request cap
    - Global cooldown on 429 errors
    """
    def __init__(self, min_interval: float = 2.0, per_minute: int = 20):
        """
        Args:
            min_interval: Minimum seconds between requests (default: 2.0s)
            per_minute: Maximum requests per minute (default: 20)
        """
        self.min_interval = min_interval
        self.per_minute = per_minute
        self.lock = threading.Lock()
        self.next_ok = 0.0
        self.calls = deque()
        self.cooldown_until = 0.0

    def wait(self):
        """Block until it's safe to make next request."""
        with self.lock:
            now = time.monotonic()

            # Global cooldown (after 429 error)
            if now < self.cooldown_until:
                wait_time = self.cooldown_until - now
                print(f"⏸ Rate limit cooldown: waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                now = time.monotonic()

            # Per-minute window cleanup
            while self.calls and now - self.calls[0] > 60:
                self.calls.popleft()

            # Enforce per-minute cap
            if len(self.calls) >= self.per_minute:
                sleep_for = 60 - (now - self.calls[0])
                if sleep_for > 0:
                    print(f"⏸ Per-minute limit reached: waiting {sleep_for:.1f}s...")
                    time.sleep(sleep_for)
                    now = time.monotonic()

            # Min spacing with jitter
            if now < self.next_ok:
                time.sleep(self.next_ok - now)
                now = time.monotonic()

            # Add jitter to avoid lockstep requests
            jitter = random.uniform(0.0, 0.5)
            self.next_ok = now + self.min_interval + jitter
            self.calls.append(time.monotonic())

    def set_cooldown(self, seconds: float):
        """Set global cooldown period (e.g., after 429 error)."""
        with self.lock:
            self.cooldown_until = max(self.cooldown_until, time.monotonic() + seconds)
            print(f"⚠ Setting cooldown for {seconds:.1f}s due to rate limit")


# Global rate limiter instance
_rate_limiter = YahooRateLimiter(min_interval=2.0, per_minute=20)

# Global shared session with stable User-Agent
_shared_session = None

# Request de-duplication: avoid concurrent identical requests
_inflight_requests = {}
_inflight_lock = threading.Lock()


def get_shared_session() -> requests.Session:
    """Get or create shared requests.Session with proper headers."""
    global _shared_session
    if _shared_session is None:
        _shared_session = requests.Session()
        _shared_session.headers.update({
            "User-Agent": "QuantAgent/1.0 (Educational Trading Analysis Tool)",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
        })
    return _shared_session


def deduplicate_request(key: str, fetch_fn):
    """
    Ensure only one thread fetches data for a given key at a time.
    Other threads with the same key will wait for the first request to complete.

    Args:
        key: Unique identifier for the request (e.g., "AAPL_1d_3mo")
        fetch_fn: Function to call if this is the first request for this key

    Returns:
        Result from fetch_fn
    """
    with _inflight_lock:
        event = _inflight_requests.get(key)
        if event is None:
            # First request for this key - create event and mark as leader
            event = threading.Event()
            _inflight_requests[key] = event
            is_leader = True
        else:
            # Another thread is already fetching this - wait for it
            is_leader = False

    if is_leader:
        try:
            # Leader performs the actual fetch
            result = fetch_fn()
            # Store result for potential future use (in cache)
            return result
        finally:
            # Signal waiting threads and cleanup
            with _inflight_lock:
                event.set()
                _inflight_requests.pop(key, None)
    else:
        # Wait for leader to complete
        event.wait()
        # Leader has completed - call fetch_fn again (should hit cache now)
        return fetch_fn()


@dataclass
class TickerResult:
    """Result container for each ticker analysis."""
    ticker: str
    status: str  # "success" | "error"
    error_message: Optional[str]
    runtime_seconds: float
    final_trade_decision: Optional[str] = None
    indicator_report: Optional[str] = None
    pattern_report: Optional[str] = None
    trend_report: Optional[str] = None


def parse_retry_after(header_value: Optional[str]) -> Optional[float]:
    """
    Parse Retry-After header value.

    Args:
        header_value: Value from Retry-After header (seconds or HTTP-date)

    Returns:
        Seconds to wait, or None if parsing failed
    """
    if not header_value:
        return None

    # Try parsing as integer (seconds)
    try:
        return max(0.0, float(header_value))
    except ValueError:
        pass

    # Try parsing as HTTP-date
    try:
        dt = email_utils.parsedate_to_datetime(header_value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = (dt - datetime.now(timezone.utc)).total_seconds()
        return max(0.0, delta)
    except Exception:
        return None


def is_rate_limit_error(exception: Exception) -> bool:
    """Check if exception indicates a rate limit error."""
    error_str = str(exception).lower()
    # Common rate limit indicators
    return any(indicator in error_str for indicator in [
        '429', 'too many requests', 'rate limit', 'quota exceeded',
        '999', 'request denied'  # Yahoo specific
    ])


def is_transient_error(exception: Exception) -> bool:
    """Check if exception indicates a transient error worth retrying."""
    error_str = str(exception).lower()
    # Transient errors: timeouts, server errors, connection issues
    return any(indicator in error_str for indicator in [
        'timeout', 'timed out', '502', '503', '504',
        'connection', 'temporary', 'unavailable'
    ])


def process_single_ticker(
    ticker: str,
    interval: str,
    period: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
    limit: int,
    max_retries: int = 7
) -> TickerResult:
    """
    Process a single ticker with advanced retry mechanism.

    Args:
        ticker: Ticker symbol
        interval: Data interval
        period: Period to fetch (mutually exclusive with start_date/end_date)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        limit: Number of most recent data points to analyze
        max_retries: Maximum number of retry attempts (default: 7)

    Returns:
        TickerResult object with analysis results or error information
    """
    start_time = time.perf_counter()

    print(f"\n{'='*80}")
    print(f"Processing: {ticker}")
    print(f"{'='*80}")

    for attempt in range(max_retries):
        try:
            # Create unique key for this request (for de-duplication)
            if start_date and end_date:
                request_key = f"{ticker}_{interval}_{start_date}_{end_date}"
            else:
                request_key = f"{ticker}_{interval}_{period}"

            # Deduplicate and fetch data with rate limiting
            def fetch_with_rate_limit():
                _rate_limiter.wait()
                if start_date and end_date:
                    return fetch_stock_data(
                        ticker=ticker,
                        interval=interval,
                        start_date=start_date,
                        end_date=end_date
                    )
                else:
                    return fetch_stock_data(
                        ticker=ticker,
                        interval=interval,
                        period=period
                    )

            df = deduplicate_request(request_key, fetch_with_rate_limit)

            # Prepare data for analysis
            data_dict = prepare_data_for_analysis(df, limit=limit)

            # Run analysis
            final_state = run_analysis(ticker, interval, data_dict)

            # Extract results
            runtime = time.perf_counter() - start_time

            result = TickerResult(
                ticker=ticker,
                status="success",
                error_message=None,
                runtime_seconds=runtime,
                final_trade_decision=final_state.get("final_trade_decision"),
                indicator_report=final_state.get("indicator_report"),
                pattern_report=final_state.get("pattern_report"),
                trend_report=final_state.get("trend_report")
            )

            print(f"\n✓ {ticker}: Analysis completed successfully ({runtime:.2f}s)")
            return result

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            is_rate_limit = is_rate_limit_error(e)
            is_transient = is_transient_error(e)

            # Don't retry on permanent client errors (4xx except 429)
            if not is_rate_limit and not is_transient and '4' in str(type(e).__name__):
                runtime = time.perf_counter() - start_time
                print(f"\n✗ {ticker}: Permanent error (no retry): {error_msg}")
                return TickerResult(
                    ticker=ticker,
                    status="error",
                    error_message=error_msg,
                    runtime_seconds=runtime
                )

            if attempt < max_retries - 1:
                # Calculate backoff time
                if is_rate_limit:
                    # For rate limits: longer backoff, exponential with cap at 120s
                    base_wait = min(120.0, 2 ** (attempt + 3))  # 8, 16, 32, 64, 120, 120...
                    jitter_range = base_wait * 0.5
                    wait_time = base_wait + random.uniform(0, jitter_range)

                    # Set global cooldown for all threads
                    _rate_limiter.set_cooldown(wait_time)

                    print(f"⚠ {ticker}: Rate limit hit (attempt {attempt + 1}/{max_retries})")
                    print(f"  Error: {error_msg}")
                    print(f"  Waiting {wait_time:.1f}s before retry...")
                else:
                    # For transient errors: standard exponential backoff
                    base_wait = min(60.0, 2 ** attempt)  # 1, 2, 4, 8, 16, 32, 60...
                    jitter_range = base_wait * 0.5
                    wait_time = base_wait + random.uniform(0, jitter_range)

                    print(f"⚠ {ticker}: Transient error (attempt {attempt + 1}/{max_retries})")
                    print(f"  Error: {error_msg}")
                    print(f"  Retrying in {wait_time:.1f}s...")

                time.sleep(wait_time)
            else:
                runtime = time.perf_counter() - start_time
                print(f"\n✗ {ticker}: Analysis failed after {max_retries} attempts: {error_msg}")

                return TickerResult(
                    ticker=ticker,
                    status="error",
                    error_message=error_msg,
                    runtime_seconds=runtime
                )

    # This should not be reached, but handle it just in case
    runtime = time.perf_counter() - start_time
    return TickerResult(
        ticker=ticker,
        status="error",
        error_message="Unknown error: max retries exhausted",
        runtime_seconds=runtime
    )


def print_summary(results: List[TickerResult]):
    """
    Print summary of all ticker analyses.

    Args:
        results: List of TickerResult objects
    """
    print("\n" + "="*80)
    print("MULTI-TICKER ANALYSIS SUMMARY")
    print("="*80)

    successful = [r for r in results if r.status == "success"]
    failed = [r for r in results if r.status == "error"]

    print(f"\nTotal Tickers: {len(results)}")
    print(f"✓ Successful: {len(successful)}")
    print(f"✗ Failed: {len(failed)}")

    if successful:
        total_time = sum(r.runtime_seconds for r in successful)
        avg_time = total_time / len(successful)
        print(f"Average Runtime: {avg_time:.2f}s")

    # Print table header
    print(f"\n{'Ticker':<10} {'Status':<10} {'Runtime':<12} {'Notes':<50}")
    print("-" * 82)

    # Print each result
    for result in results:
        status_icon = "✓" if result.status == "success" else "✗"
        runtime_str = f"{result.runtime_seconds:.2f}s"
        notes = ""

        if result.status == "error":
            # Type-safe truncation: coalesce None first
            msg = result.error_message or ""
            notes = msg[:47] + "..." if len(msg) > 50 else msg
        elif result.final_trade_decision:
            # Extract first line of decision
            first_line = result.final_trade_decision.split('\n')[0]
            notes = first_line[:47] + "..." if len(first_line) > 50 else first_line

        print(f"{result.ticker:<10} {status_icon} {result.status:<8} {runtime_str:<12} {notes}")

    if failed:
        print("\n" + "="*80)
        print("FAILED TICKERS DETAILS")
        print("="*80)
        for result in failed:
            print(f"\n{result.ticker}:")
            print(f"  Error: {result.error_message}")

    print("\n" + "="*80)


def print_detailed_results(results: List[TickerResult]):
    """
    Print detailed analysis results for successful tickers.

    Args:
        results: List of TickerResult objects
    """
    successful = [r for r in results if r.status == "success"]

    if not successful:
        print("\nNo successful analyses to display.")
        return

    for result in successful:
        print("\n" + "="*80)
        print(f"DETAILED RESULTS: {result.ticker}")
        print("="*80)

        if result.final_trade_decision:
            print("\n📊 FINAL TRADE DECISION:")
            print("-" * 80)
            print(result.final_trade_decision)

        if result.indicator_report:
            print("\n📈 INDICATOR ANALYSIS:")
            print("-" * 80)
            print(result.indicator_report)

        if result.pattern_report:
            print("\n🔍 PATTERN ANALYSIS:")
            print("-" * 80)
            print(result.pattern_report)

        if result.trend_report:
            print("\n📉 TREND ANALYSIS:")
            print("-" * 80)
            print(result.trend_report)

        print("\n" + "="*80)


def save_results_json(results: List[TickerResult], output_file: str):
    """
    Save results to JSON file.

    Args:
        results: List of TickerResult objects
        output_file: Output file path
    """
    data = [asdict(r) for r in results]

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")


def save_results_csv(results: List[TickerResult], output_file: str):
    """
    Save results to CSV file.

    Args:
        results: List of TickerResult objects
        output_file: Output file path
    """
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'Ticker', 'Status', 'Runtime (seconds)', 'Error Message',
            'Final Trade Decision', 'Indicator Report', 'Pattern Report', 'Trend Report'
        ])

        # Data rows
        for r in results:
            writer.writerow([
                r.ticker,
                r.status,
                f"{r.runtime_seconds:.2f}",
                r.error_message or '',
                r.final_trade_decision or '',
                r.indicator_report or '',
                r.pattern_report or '',
                r.trend_report or ''
            ])

    print(f"\nResults saved to: {output_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="QuantAgent Multi-Ticker CLI: Analyze multiple stocks concurrently using multi-agent trading system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze multiple stocks with 1-year daily data (safe defaults)
  python run_multi_analysis.py --tickers AAPL TSLA MSFT --period 1y --interval 1d

  # Analyze crypto and stocks with 6-month 4-hour data
  python run_multi_analysis.py --tickers BTC-USD ETH-USD AAPL --period 6mo --interval 4h

  # Analyze with specific date range and save results
  python run_multi_analysis.py --tickers AAPL GOOGL --start 2024-01-01 --end 2024-12-31 --output results.json

  # View detailed analysis for each ticker
  python run_multi_analysis.py --tickers AAPL TSLA --period 3mo --detailed

Rate Limiting Notes:
  - Default max-workers=1 (sequential processing) minimizes 429 errors
  - Global rate limiter enforces 2s min spacing + 20 requests/minute cap
  - Automatic exponential backoff on rate limit errors (up to 120s)
  - yfinance_cache provides persistent caching to reduce API calls

Supported intervals: 1m, 5m, 15m, 30m, 1h, 4h, 1d
Supported periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        """
    )

    # Required arguments
    parser.add_argument(
        "--tickers",
        nargs="+",
        required=True,
        help="Ticker symbols (e.g., AAPL TSLA BTC-USD ^GSPC)"
    )

    # Optional arguments
    parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        choices=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
        help="Data interval (default: 1d)"
    )

    # Period or date range
    date_group = parser.add_mutually_exclusive_group()
    date_group.add_argument(
        "--period",
        type=str,
        default="3mo",
        help="Period to fetch (default: 3mo). Options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max. Note: MACD requires at least 50 data points for accurate calculations."
    )
    date_group.add_argument(
        "--start",
        type=str,
        help="Start date in YYYY-MM-DD format (use with --end)"
    )

    parser.add_argument(
        "--end",
        type=str,
        help="End date in YYYY-MM-DD format (use with --start)"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=45,
        help="Number of most recent data points to analyze (default: 45)"
    )

    # Concurrency settings
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum number of concurrent workers (default: 1, recommended to avoid rate limits). WARNING: Higher values increase risk of 429 errors."
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=7,
        help="Maximum number of retry attempts for failed tickers (default: 7)"
    )

    # Output settings
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (e.g., results.json or results.csv)"
    )

    parser.add_argument(
        "--output-format",
        type=str,
        choices=["json", "csv"],
        help="Output format (default: inferred from file extension)"
    )

    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Print detailed analysis results for each ticker"
    )

    args = parser.parse_args()

    # Validate date range arguments
    if args.start and not args.end:
        parser.error("--start requires --end")
    if args.end and not args.start:
        parser.error("--end requires --start")

    # Remove duplicates and empty strings from tickers
    tickers = []
    seen = set()
    for ticker in args.tickers:
        ticker = ticker.strip().upper()
        if ticker and ticker not in seen:
            tickers.append(ticker)
            seen.add(ticker)

    if not tickers:
        parser.error("No valid tickers provided")

    print(f"\nStarting analysis for {len(tickers)} ticker(s): {', '.join(tickers)}")
    print(f"Max workers: {args.max_workers}")
    print(f"Max retries per ticker: {args.max_retries}")

    # Process tickers concurrently
    results = []
    start_time = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        future_to_ticker = {
            executor.submit(
                process_single_ticker,
                ticker=ticker,
                interval=args.interval,
                period=args.period if not args.start else None,
                start_date=args.start,
                end_date=args.end,
                limit=args.limit,
                max_retries=args.max_retries
            ): ticker
            for ticker in tickers
        }

        # Collect results as they complete
        for future in as_completed(future_to_ticker):
            result = future.result()
            results.append(result)

    # Sort results to match input order
    ticker_order = {t: i for i, t in enumerate(tickers)}
    results.sort(key=lambda r: ticker_order.get(r.ticker, 999))

    total_time = time.perf_counter() - start_time

    # Print summary
    print_summary(results)
    print(f"\nTotal execution time: {total_time:.2f}s")

    # Print detailed results if requested
    if args.detailed:
        print_detailed_results(results)

    # Save results to file if requested
    if args.output:
        # Infer format from file extension if not specified
        output_format = args.output_format
        if not output_format:
            if args.output.endswith('.json'):
                output_format = 'json'
            elif args.output.endswith('.csv'):
                output_format = 'csv'
            else:
                print("\n⚠ Warning: Could not infer output format from file extension. Using JSON.")
                output_format = 'json'

        if output_format == 'json':
            save_results_json(results, args.output)
        else:
            save_results_csv(results, args.output)

    # Exit with error code if any ticker failed
    failed_count = sum(1 for r in results if r.status == "error")
    if failed_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
QuantAgent Multi-Ticker CLI Script
Fetches stock data for multiple tickers sequentially with simple rate limiting.

Processing Strategy:
1. Sequential processing (one ticker at a time)
2. Simple rate limiting (2 second minimum interval with jitter)
3. Skip tickers on download failure (no retry)
4. Cache thoroughly using yfinance
5. Stable User-Agent via shared requests.Session
"""
import argparse
import sys
import time
import json
import csv
import random
import re
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import requests
from dotenv import load_dotenv
import yfinance as yf

# Load environment variables from .env file
load_dotenv()


def configure_logging(
    log_file: Optional[Path] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> None:
    """
    Configure root logger with RotatingFileHandler and StreamHandler.

    Args:
        log_file: Path to log file (default: multi_analysis.log in current directory)
        console_level: Minimum log level for console output (default: INFO)
        file_level: Minimum log level for file output (default: DEBUG)
        max_bytes: Maximum log file size before rotation (default: 10MB)
        backup_count: Number of backup files to keep (default: 5)
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Avoid duplicating handlers
    if getattr(root, "_configured", False):
        return

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler: INFO+
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # File handler: DEBUG+ with rotation
    if log_file is None:
        log_file = Path("multi_analysis.log")

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8"
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Quiet down noisy third-party libraries
    logging.captureWarnings(True)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    root._configured = True

from run_analysis import (
    fetch_stock_data,
    prepare_data_for_analysis,
    run_analysis,
    format_timeframe_display,
    print_results
)


# Simple rate limiting for sequential processing
_last_request_time = 0.0
_min_request_interval = 2.0  # Minimum seconds between requests


def wait_for_rate_limit():
    """
    Simple rate limiter for sequential yfinance requests.
    Enforces minimum interval between requests with jitter.
    """
    global _last_request_time

    now = time.time()
    elapsed = now - _last_request_time

    if elapsed < _min_request_interval:
        # Add jitter (10-20% of interval) to avoid predictable patterns
        jitter = random.uniform(0.1, 0.2) * _min_request_interval
        wait_time = (_min_request_interval - elapsed) + jitter
        logging.debug("⏸ Rate limit: waiting %.2fs before next request", wait_time)
        time.sleep(wait_time)

    _last_request_time = time.time()

# Global shared session with stable User-Agent
_shared_session = None

# Ticker name cache for yfinance lookups
_ticker_name_cache = {}

# Regex for extracting JSON from markdown code blocks
_fenced_json_re = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


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




def extract_json_from_markdown(text: str) -> Optional[str]:
    """
    Extract JSON content from markdown code blocks.

    Args:
        text: Text containing JSON (potentially in ```json...``` blocks)

    Returns:
        Extracted JSON string, or None if no JSON found
    """
    if not text:
        return None

    # Try fenced code block first
    match = _fenced_json_re.search(text)
    if match:
        return match.group(1).strip()

    # Fallback: find balanced JSON object
    start = text.find("{")
    if start == -1:
        return None

    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i+1].strip()
    return None


def parse_final_trade_decision(raw_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Parse Final Trade Decision field to extract structured data.

    Args:
        raw_text: Raw text containing JSON (potentially in markdown code blocks)

    Returns:
        Tuple of (parsed_dict, error_message)
        - parsed_dict: Dict with keys: forecast_horizon, decision, justification, risk_reward_ratio
        - error_message: Error message if parsing failed, None otherwise
    """
    snippet = extract_json_from_markdown(raw_text or "")
    if not snippet:
        return None, "no_json_found"

    try:
        data = json.loads(snippet)
        # Validate required keys
        required_keys = ["forecast_horizon", "decision", "justification", "risk_reward_ratio"]
        if not all(key in data for key in required_keys):
            missing = [k for k in required_keys if k not in data]
            return data, f"missing_keys: {', '.join(missing)}"
        return data, None
    except json.JSONDecodeError as e:
        logging.warning(f"JSON parse failed: {e}; snippet: {snippet[:200]}")
        return None, f"json_decode_error: {e.msg}"


def get_ticker_name(ticker: str) -> str:
    """
    Get ticker name from yfinance with caching.

    Args:
        ticker: Ticker symbol

    Returns:
        Ticker name (longName or shortName), or ticker symbol if not found
    """
    ticker_upper = ticker.upper()

    # Check cache first
    if ticker_upper in _ticker_name_cache:
        return _ticker_name_cache[ticker_upper]

    # Fetch from yfinance
    try:
        ticker_obj = yf.Ticker(ticker_upper)
        info = ticker_obj.info
        name = info.get('longName') or info.get('shortName') or ticker_upper

        # Cache result
        _ticker_name_cache[ticker_upper] = name

        return name
    except Exception as e:
        logging.warning(f"Could not fetch name for {ticker_upper}: {e}")
        # Cache the ticker symbol itself to avoid repeated failures
        _ticker_name_cache[ticker_upper] = ticker_upper
        return ticker_upper


@dataclass
class TickerResult:
    """Result container for each ticker analysis."""
    ticker: str
    status: str  # "success" | "skipped" | "error"
    error_message: Optional[str]
    runtime_seconds: float
    final_trade_decision: Optional[str] = None
    indicator_report: Optional[str] = None
    pattern_report: Optional[str] = None
    trend_report: Optional[str] = None




def process_single_ticker(
    ticker: str,
    interval: str,
    period: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
    limit: int,
    provider: str = "openai",
    agent_model: Optional[str] = None,
    graph_model: Optional[str] = None
) -> TickerResult:
    """
    Process a single ticker without retry - skip on failure.

    Args:
        ticker: Ticker symbol
        interval: Data interval
        period: Period to fetch (mutually exclusive with start_date/end_date)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        limit: Number of most recent data points to analyze
        provider: LLM provider ("openai", "claude_api", "claude_cli")
        agent_model: Model name for agent LLMs (optional)
        graph_model: Model name for graph LLM (optional)

    Returns:
        TickerResult object with analysis results or skip information
    """
    start_time = time.perf_counter()

    logging.info("%s", "="*80)
    logging.info("Processing: %s", ticker)
    logging.info("%s", "="*80)

    try:
        # Fetch data with rate limiting
        wait_for_rate_limit()
        if start_date and end_date:
            df = fetch_stock_data(
                ticker=ticker,
                interval=interval,
                start_date=start_date,
                end_date=end_date
            )
        else:
            df = fetch_stock_data(
                ticker=ticker,
                interval=interval,
                period=period
            )

        # Check for empty data (yfinance "soft fail")
        if df is None or df.empty:
            runtime = time.perf_counter() - start_time
            error_msg = "No data returned from yfinance (empty DataFrame)"
            logging.warning("⊘ %s: Skipped - %s (%.2fs)", ticker, error_msg, runtime)
            return TickerResult(
                ticker=ticker,
                status="skipped",
                error_message=error_msg,
                runtime_seconds=runtime
            )

        # Check for missing required columns
        if "Close" not in df.columns:
            runtime = time.perf_counter() - start_time
            error_msg = "Missing required 'Close' column in data"
            logging.warning("⊘ %s: Skipped - %s (%.2fs)", ticker, error_msg, runtime)
            return TickerResult(
                ticker=ticker,
                status="skipped",
                error_message=error_msg,
                runtime_seconds=runtime
            )

        # Check for all-NaN Close column
        if df["Close"].dropna().empty:
            runtime = time.perf_counter() - start_time
            error_msg = "All-NaN Close column (no valid price data)"
            logging.warning("⊘ %s: Skipped - %s (%.2fs)", ticker, error_msg, runtime)
            return TickerResult(
                ticker=ticker,
                status="skipped",
                error_message=error_msg,
                runtime_seconds=runtime
            )

        # Prepare data for analysis
        data_dict = prepare_data_for_analysis(df, limit=limit)

        # Run analysis with provider settings
        final_state = run_analysis(
            ticker=ticker,
            interval=interval,
            data_dict=data_dict,
            provider=provider,
            agent_model=agent_model,
            graph_model=graph_model
        )

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

        logging.info("✓ %s: Analysis completed successfully (%.2fs)", ticker, runtime)
        return result

    except Exception as e:
        runtime = time.perf_counter() - start_time
        error_msg = f"{type(e).__name__}: {str(e)}"
        logging.warning("⊘ %s: Skipped due to error - %s (%.2fs)", ticker, error_msg, runtime)

        return TickerResult(
            ticker=ticker,
            status="skipped",
            error_message=error_msg,
            runtime_seconds=runtime
        )


def print_summary(results: List[TickerResult]):
    """
    Print summary of all ticker analyses.

    Args:
        results: List of TickerResult objects
    """
    logging.info("%s", "="*80)
    logging.info("MULTI-TICKER ANALYSIS SUMMARY")
    logging.info("%s", "="*80)

    successful = [r for r in results if r.status == "success"]
    skipped = [r for r in results if r.status == "skipped"]
    failed = [r for r in results if r.status == "error"]

    logging.info("Total Tickers: %d", len(results))
    logging.info("✓ Successful: %d", len(successful))
    logging.info("⊘ Skipped: %d", len(skipped))
    logging.info("✗ Failed: %d", len(failed))

    if successful:
        total_time = sum(r.runtime_seconds for r in successful)
        avg_time = total_time / len(successful)
        logging.info("Average Runtime: %.2fs", avg_time)

    # Print table header
    logging.info("%-10s %-10s %-12s %-50s", "Ticker", "Status", "Runtime", "Notes")
    logging.info("%s", "-" * 82)

    # Print each result
    for result in results:
        if result.status == "success":
            status_icon = "✓"
        elif result.status == "skipped":
            status_icon = "⊘"
        else:
            status_icon = "✗"

        runtime_str = f"{result.runtime_seconds:.2f}s"
        notes = ""

        if result.status in ("skipped", "error"):
            # Type-safe truncation: coalesce None first
            msg = result.error_message or ""
            notes = msg[:47] + "..." if len(msg) > 50 else msg
        elif result.final_trade_decision:
            # Extract first line of decision
            first_line = result.final_trade_decision.split('\n')[0]
            notes = first_line[:47] + "..." if len(first_line) > 50 else first_line

        logging.info("%-10s %s %-8s %-12s %s", result.ticker, status_icon, result.status, runtime_str, notes)

    if skipped:
        logging.info("%s", "="*80)
        logging.info("SKIPPED TICKERS DETAILS")
        logging.info("%s", "="*80)
        for result in skipped:
            logging.info("%s:", result.ticker)
            logging.info("  Reason: %s", result.error_message)

    if failed:
        logging.info("%s", "="*80)
        logging.info("FAILED TICKERS DETAILS")
        logging.info("%s", "="*80)
        for result in failed:
            logging.info("%s:", result.ticker)
            logging.info("  Error: %s", result.error_message)

    logging.info("%s", "="*80)


def print_detailed_results(results: List[TickerResult]):
    """
    Print detailed analysis results for successful tickers.

    Args:
        results: List of TickerResult objects
    """
    successful = [r for r in results if r.status == "success"]

    if not successful:
        logging.info("No successful analyses to display.")
        return

    for result in successful:
        logging.info("%s", "="*80)
        logging.info("DETAILED RESULTS: %s", result.ticker)
        logging.info("%s", "="*80)

        if result.final_trade_decision:
            logging.info("📊 FINAL TRADE DECISION:")
            logging.info("%s", "-" * 80)
            logging.info("%s", result.final_trade_decision)

        if result.indicator_report:
            logging.info("📈 INDICATOR ANALYSIS:")
            logging.info("%s", "-" * 80)
            logging.info("%s", result.indicator_report)

        if result.pattern_report:
            logging.info("🔍 PATTERN ANALYSIS:")
            logging.info("%s", "-" * 80)
            logging.info("%s", result.pattern_report)

        if result.trend_report:
            logging.info("📉 TREND ANALYSIS:")
            logging.info("%s", "-" * 80)
            logging.info("%s", result.trend_report)

        logging.info("%s", "="*80)


def save_results_json(results: List[TickerResult], output_path: Path, detailed: bool = False):
    """
    Save results to JSON file.

    Args:
        results: List of TickerResult objects
        output_path: Output file Path object
        detailed: If True, save detailed format. If False, save simple format (default).
    """
    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if detailed:
        # Detailed format: save all fields
        data = [asdict(r) for r in results]
    else:
        # Simple format: extract structured fields from Final Trade Decision
        data = []
        for r in results:
            if r.status == "skipped":
                # Skipped ticker: only include basic info and error message
                simple_record = {
                    "ticker": r.ticker,
                    "status": r.status,
                    "name": get_ticker_name(r.ticker),
                    "error_message": r.error_message
                }
            else:
                # Success or error: parse decision data
                parsed_data, parse_error = parse_final_trade_decision(r.final_trade_decision or "")

                simple_record = {
                    "ticker": r.ticker,
                    "status": r.status,
                    "name": get_ticker_name(r.ticker),
                    "forecast_horizon": parsed_data.get("forecast_horizon") if parsed_data else None,
                    "decision": parsed_data.get("decision") if parsed_data else None,
                    "justification": parsed_data.get("justification") if parsed_data else None,
                    "risk_reward_ratio": parsed_data.get("risk_reward_ratio") if parsed_data else None,
                }

                # Add error message if present
                if r.error_message:
                    simple_record["error_message"] = r.error_message

                # Add parse error for failed cases (optional, for debugging)
                if parse_error:
                    simple_record["parse_error"] = parse_error

            data.append(simple_record)

    with output_path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logging.info("Results saved to: %s", output_path)


def save_results_csv(results: List[TickerResult], output_path: Path, append: bool = False, detailed: bool = False):
    """
    Save results to CSV file.

    Args:
        results: List of TickerResult objects
        output_path: Output file Path object
        append: If True, append to existing file without header. If False, create new file with header.
        detailed: If True, save detailed format. If False, save simple format (default).
    """
    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = 'a' if append else 'w'
    with output_path.open(mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        if detailed:
            # Detailed format: all fields
            if not append:
                writer.writerow([
                    'Ticker', 'Status', 'Runtime (seconds)', 'Error Message',
                    'Final Trade Decision', 'Indicator Report', 'Pattern Report', 'Trend Report'
                ])

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
        else:
            # Simple format: extract structured fields from Final Trade Decision
            if not append:
                writer.writerow([
                    'Ticker', 'Status', 'Name', 'Forecast Horizon', 'Decision', 'Justification', 'Risk/Reward Ratio', 'Error Message'
                ])

            for r in results:
                if r.status == "skipped":
                    # Skipped ticker: only populate ticker, status, and error message
                    writer.writerow([
                        r.ticker,
                        r.status,
                        get_ticker_name(r.ticker),
                        '',  # forecast_horizon
                        '',  # decision
                        '',  # justification
                        '',  # risk_reward_ratio
                        r.error_message or ''
                    ])
                else:
                    # Success or error: parse decision data
                    parsed_data, parse_error = parse_final_trade_decision(r.final_trade_decision or "")

                    writer.writerow([
                        r.ticker,
                        r.status,
                        get_ticker_name(r.ticker),
                        parsed_data.get("forecast_horizon") if parsed_data else '',
                        parsed_data.get("decision") if parsed_data else '',
                        parsed_data.get("justification") if parsed_data else '',
                        parsed_data.get("risk_reward_ratio") if parsed_data else '',
                        r.error_message or ''
                    ])

    if not append:
        logging.info("Results saved to: %s", output_path)


def load_tickers_from_csv(file_path: str, signal_filter: str = 'buy') -> List[str]:
    """
    Load tickers from CSV file filtering by signal column.

    Args:
        file_path: Path to CSV file
        signal_filter: Signal filter ('buy', 'sell', or 'all'). Default: 'buy'

    Returns:
        List of ticker symbols matching the signal filter

    CSV Format:
        ティッカー,銘柄名,シグナル,現在価格,100株価格,200日MA,MA比率(%),割高,割安,2日変動率(%),エラー
    """
    csv_path = Path(file_path)
    if not csv_path.exists():
        logging.error("✗ Error: CSV file not found: %s", file_path)
        sys.exit(1)

    tickers = []
    seen = set()

    # Try UTF-8 first, then CP932 (Shift_JIS for Japanese Windows)
    encodings = ['utf-8-sig', 'utf-8', 'cp932', 'shift_jis']
    last_error = None

    for encoding in encodings:
        try:
            with csv_path.open(mode='r', encoding=encoding, newline='') as f:
                reader = csv.DictReader(f)

                # Normalize headers (strip BOM and whitespace)
                if reader.fieldnames:
                    normalized_headers = {h.strip().replace('\ufeff', ''): h for h in reader.fieldnames}
                else:
                    logging.error("✗ Error: CSV file has no headers: %s", file_path)
                    sys.exit(1)

                # Check required columns
                required_cols = ['ティッカー', 'シグナル']
                missing = [col for col in required_cols if col not in normalized_headers]
                if missing:
                    logging.error("✗ Error: CSV missing required columns: %s", missing)
                    logging.error("Found columns: %s", list(normalized_headers.keys()))
                    sys.exit(1)

                ticker_col = normalized_headers['ティッカー']
                signal_col = normalized_headers['シグナル']
                # 銘柄名 is optional
                name_col = normalized_headers.get('銘柄名')

                # Parse rows
                row_num = 1
                for row in reader:
                    row_num += 1
                    try:
                        signal = row.get(signal_col, '').strip()
                        ticker_raw = row.get(ticker_col, '').strip()
                        name = row.get(name_col, '').strip() if name_col else ''

                        # Filter by signal
                        signal_matches = False
                        if signal_filter == 'all':
                            signal_matches = signal in ('買い', '売り')
                        elif signal_filter == 'buy':
                            signal_matches = signal == '買い'
                        elif signal_filter == 'sell':
                            signal_matches = signal == '売り'

                        if signal_matches:
                            ticker = ticker_raw.upper()
                            if ticker and ticker not in seen:
                                tickers.append(ticker)
                                seen.add(ticker)
                                # Display ticker with name if available
                                if name:
                                    logging.info("  Row %d: %s - %s (signal: %s)", row_num, ticker, name, signal)
                                else:
                                    logging.info("  Row %d: %s (signal: %s)", row_num, ticker, signal)
                    except Exception as e:
                        logging.warning("⚠ Warning: Skipping malformed row %d: %s", row_num, e)
                        continue

            # Success - exit encoding loop
            logging.info("✓ Loaded %d ticker(s) from CSV (encoding: %s)", len(tickers), encoding)
            return tickers

        except UnicodeDecodeError as e:
            last_error = e
            continue
        except Exception as e:
            logging.error("✗ Error reading CSV file: %s", e)
            sys.exit(1)

    # All encodings failed
    logging.error("✗ Error: Could not decode CSV file with any encoding: %s", encodings)
    logging.error("Last error: %s", last_error)
    sys.exit(1)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="QuantAgent Multi-Ticker CLI: Analyze multiple stocks concurrently using multi-agent trading system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze multiple stocks with 1-year daily data (default: OpenAI)
  python run_multi_analysis.py --tickers AAPL TSLA MSFT --period 1y --interval 1d

  # Analyze with Claude CLI
  python run_multi_analysis.py --tickers BTC-USD ETH-USD --provider claude_cli --period 6mo

  # Analyze with Claude API and save results
  python run_multi_analysis.py --tickers AAPL GOOGL --provider claude_api --output results.json

  # View detailed analysis with custom models
  python run_multi_analysis.py --tickers AAPL TSLA --provider openai --agent-model gpt-4o-mini --detailed

  # Analyze from CSV file (default: filters by シグナル="買い")
  python run_multi_analysis.py --csv-file signals.csv --period 1y --interval 1d

  # Analyze from CSV file with different signal filters
  python run_multi_analysis.py --csv-file signals.csv --signal sell --period 1y
  python run_multi_analysis.py --csv-file signals.csv --signal all --period 6mo

Processing Notes:
  - Tickers are processed sequentially (one at a time)
  - Simple rate limiting: 2 second minimum interval with 10-20% jitter
  - Failed data downloads will skip that ticker (no retry)
  - Skipped tickers are recorded in output with error message
  - yfinance caching reduces redundant API calls

Supported intervals: 1m, 5m, 15m, 30m, 1h, 4h, 1d
Supported periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
Supported providers: openai (default), claude_api, claude_cli
        """
    )

    # Ticker input: mutually exclusive (either --tickers or --csv-file)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--tickers",
        nargs="+",
        help="Ticker symbols (e.g., AAPL TSLA BTC-USD ^GSPC)"
    )
    input_group.add_argument(
        "--csv-file",
        type=str,
        help='Path to CSV file with "ティッカー" and "シグナル" columns. Use --signal to filter by signal value.'
    )

    # Optional arguments
    parser.add_argument(
        "--signal",
        type=str,
        default="buy",
        choices=["buy", "sell", "all"],
        help="Signal filter for CSV file (default: buy). Options: buy, sell, all"
    )

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

    # Output settings
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path or directory. If directory is specified, saves as 'quant_agent_result_YYYYMMDD_HHMMSS.{format}' (e.g., results.json or results.csv or ./output_dir/)"
    )

    parser.add_argument(
        "--output-format",
        type=str,
        default="csv",
        choices=["json", "csv"],
        help="Output format (default: csv). Used when --output specifies a directory or has no extension."
    )

    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Print detailed analysis results for each ticker"
    )

    parser.add_argument(
        "--detailed-output",
        action="store_true",
        help="Save detailed format to output files (default: simple format with ticker, name, forecast_horizon, decision, justification, risk_reward_ratio)"
    )

    # LLM Provider settings
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "claude_api", "claude_cli"],
        help="LLM provider to use (default: openai)"
    )

    parser.add_argument(
        "--agent-model",
        type=str,
        help="Model name for agent LLMs (optional, uses provider default)"
    )

    parser.add_argument(
        "--graph-model",
        type=str,
        help="Model name for graph LLM (optional, uses provider default)"
    )

    args = parser.parse_args()

    # Validate date range arguments
    if args.start and not args.end:
        parser.error("--start requires --end")
    if args.end and not args.start:
        parser.error("--end requires --start")

    # Configure logging first
    configure_logging()

    # Load tickers from CSV file or command line
    tickers = []
    if args.csv_file:
        logging.info("Loading tickers from CSV file: %s (filter: %s)", args.csv_file, args.signal)
        tickers = load_tickers_from_csv(args.csv_file, signal_filter=args.signal)
    elif args.tickers:
        # Remove duplicates and empty strings from tickers
        seen = set()
        for ticker in args.tickers:
            ticker = ticker.strip().upper()
            if ticker and ticker not in seen:
                tickers.append(ticker)
                seen.add(ticker)

    if not tickers:
        parser.error("No valid tickers to analyze. Check your input source (CSV or --tickers).")

    logging.info("Starting analysis for %d ticker(s): %s", len(tickers), ', '.join(tickers))
    logging.info("Provider: %s", args.provider)
    logging.info("Processing mode: Sequential")

    # Process tickers sequentially
    results = []
    start_time = time.perf_counter()

    # Determine output path and format early if output is requested
    output_path = None
    output_format = None
    if args.output:
        output_path_obj = Path(args.output)

        # Determine if output is a directory or file
        if output_path_obj.is_dir():
            # Directory: generate timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"quant_agent_result_{timestamp}.{args.output_format}"
            output_path = output_path_obj / filename
            output_format = args.output_format
        else:
            # File path: infer format from extension if not explicitly set
            output_path = output_path_obj

            # Infer format from file extension if --output-format not explicitly provided by user
            if args.output_format == "csv" and not any(arg.startswith("--output-format") for arg in sys.argv):
                if output_path_obj.suffix.lower() == '.json':
                    output_format = 'json'
                elif output_path_obj.suffix.lower() == '.csv':
                    output_format = 'csv'
                else:
                    output_format = args.output_format
            else:
                output_format = args.output_format

        # Initialize CSV file with header if CSV format
        if output_format == 'csv':
            save_results_csv([], output_path, append=False, detailed=args.detailed_output)

    # Process tickers sequentially
    for ticker in tickers:
        result = process_single_ticker(
            ticker=ticker,
            interval=args.interval,
            period=args.period if not args.start else None,
            start_date=args.start,
            end_date=args.end,
            limit=args.limit,
            provider=args.provider,
            agent_model=args.agent_model,
            graph_model=args.graph_model
        )
        results.append(result)

        # Immediately write to CSV if output is CSV format
        if output_path and output_format == 'csv':
            save_results_csv([result], output_path, append=True, detailed=args.detailed_output)

    total_time = time.perf_counter() - start_time

    # Print summary
    print_summary(results)
    logging.info("Total execution time: %.2fs", total_time)

    # Print detailed results if requested
    if args.detailed:
        print_detailed_results(results)

    # Save results to file if requested
    if args.output:
        # Only save JSON format here; CSV was already written incrementally
        if output_format == 'json':
            save_results_json(results, output_path, detailed=args.detailed_output)
        elif output_format == 'csv':
            # CSV already written incrementally during processing
            logging.info("Results saved to: %s", output_path)

    # Exit with error code if any ticker failed or was skipped
    failed_or_skipped_count = sum(1 for r in results if r.status in ("error", "skipped"))
    if failed_or_skipped_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

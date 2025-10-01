"""
QuantAgent CLI Script
Fetches stock data using yfinance and runs TradingGraph analysis.
"""
import argparse
import sys
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import yfinance as yf
from trading_graph import TradingGraph


def fetch_stock_data(
    ticker: str,
    interval: str = "1d",
    period: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Yahoo Finance.

    Args:
        ticker: Ticker symbol (e.g., 'AAPL', 'BTC-USD')
        interval: Data interval ('1m', '5m', '15m', '30m', '1h', '4h', '1d')
        period: Period to fetch (e.g., '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        DataFrame with OHLCV data
    """
    try:
        print(f"Fetching data for {ticker}...")

        # Fetch data
        if period:
            df = yf.download(
                tickers=ticker,
                period=period,
                interval=interval,
                auto_adjust=True,
                prepost=False
            )
        else:
            df = yf.download(
                tickers=ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                prepost=False
            )

        if df is None or df.empty:
            raise ValueError(f"No data returned for {ticker}")

        # Ensure df is a DataFrame
        if isinstance(df, pd.Series):
            df = df.to_frame()

        # Reset index
        df = df.reset_index()

        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Rename columns
        column_mapping = {
            'Date': 'Datetime',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        }
        existing_columns = {old: new for old, new in column_mapping.items() if old in df.columns}
        df = df.rename(columns=existing_columns)

        # Ensure required columns
        required_columns = ["Datetime", "Open", "High", "Low", "Close"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Missing required columns. Available: {list(df.columns)}")

        # Select only required columns
        df = df[required_columns]
        df['Datetime'] = pd.to_datetime(df['Datetime'])

        print(f"Successfully fetched {len(df)} data points")
        print(f"Date range: {df['Datetime'].min()} to {df['Datetime'].max()}")

        return df

    except Exception as e:
        print(f"Error fetching data: {e}")
        sys.exit(1)


def prepare_data_for_analysis(df: pd.DataFrame, limit: int = 45) -> dict:
    """
    Prepare DataFrame for TradingGraph analysis.

    Args:
        df: DataFrame with OHLCV data
        limit: Number of most recent data points to use (default: 45)

    Returns:
        Dictionary formatted for TradingGraph
    """
    # Validate minimum data requirements for technical indicators
    # MACD requires: slowperiod (26) + signalperiod (9) - 1 = 34 data points minimum
    # Recommended: at least 50 data points for stable calculations
    MINIMUM_DATA_POINTS = 50

    if len(df) < MINIMUM_DATA_POINTS:
        print(f"\n⚠️  WARNING: Only {len(df)} data points available.")
        print(f"   Technical indicators (especially MACD) require at least {MINIMUM_DATA_POINTS} data points for accurate calculations.")
        print(f"   Results may be incomplete or inaccurate. Consider using a longer period (e.g., --period 3mo or 6mo).\n")

    # Get the most recent data points
    if len(df) > limit + 3:
        df_slice = df.tail(limit + 3).iloc[:-3]
    else:
        df_slice = df.tail(limit)

    # Reset index
    df_slice = df_slice.reset_index(drop=True)

    # Convert to dict
    df_slice_dict = {}
    required_columns = ["Datetime", "Open", "High", "Low", "Close"]

    for col in required_columns:
        if col == 'Datetime':
            # Convert datetime to string
            df_slice_dict[col] = df_slice[col].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
        else:
            df_slice_dict[col] = df_slice[col].tolist()

    return df_slice_dict


def format_timeframe_display(interval: str) -> str:
    """
    Format interval for display.

    Args:
        interval: Interval string (e.g., '1h', '4h', '1d')

    Returns:
        Formatted display string
    """
    if interval.endswith('h'):
        return interval + 'our'
    elif interval.endswith('m'):
        return interval + 'in'
    elif interval.endswith('d'):
        return interval + 'ay'
    return interval


def run_analysis(ticker: str, interval: str, data_dict: dict) -> dict:
    """
    Run TradingGraph analysis.

    Args:
        ticker: Ticker symbol
        interval: Time interval
        data_dict: Dictionary with kline data

    Returns:
        Analysis results
    """
    print("\nRunning analysis...")

    # Create initial state
    initial_state = {
        "kline_data": data_dict,
        "analysis_results": None,
        "messages": [],
        "time_frame": format_timeframe_display(interval),
        "stock_name": ticker
    }

    # Initialize TradingGraph
    trading_graph = TradingGraph()

    # Run analysis
    try:
        final_state = trading_graph.graph.invoke(initial_state)
        return final_state
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


def print_results(final_state: dict):
    """
    Print analysis results.

    Args:
        final_state: Final state from TradingGraph
    """
    print("\n" + "="*80)
    print("ANALYSIS RESULTS")
    print("="*80)

    # Print final trade decision
    if "final_trade_decision" in final_state and final_state["final_trade_decision"]:
        print("\n📊 FINAL TRADE DECISION:")
        print("-" * 80)
        print(final_state["final_trade_decision"])

    # Print indicator report
    if "indicator_report" in final_state and final_state["indicator_report"]:
        print("\n📈 INDICATOR ANALYSIS:")
        print("-" * 80)
        print(final_state["indicator_report"])

    # Print pattern report
    if "pattern_report" in final_state and final_state["pattern_report"]:
        print("\n🔍 PATTERN ANALYSIS:")
        print("-" * 80)
        print(final_state["pattern_report"])

    # Print trend report
    if "trend_report" in final_state and final_state["trend_report"]:
        print("\n📉 TREND ANALYSIS:")
        print("-" * 80)
        print(final_state["trend_report"])

    print("\n" + "="*80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="QuantAgent CLI: Analyze stock data using multi-agent trading system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze Apple stock with 1-year daily data
  python run_analysis.py AAPL --period 1y --interval 1d

  # Analyze Bitcoin with 6-month 4-hour data
  python run_analysis.py BTC-USD --period 6mo --interval 4h

  # Analyze Tesla with specific date range
  python run_analysis.py TSLA --start 2024-01-01 --end 2024-12-31 --interval 1d

  # Analyze S&P 500 index
  python run_analysis.py ^GSPC --period 3mo --interval 1d

Supported intervals: 1m, 5m, 15m, 30m, 1h, 4h, 1d
Supported periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        """
    )

    # Required arguments
    parser.add_argument(
        "ticker",
        type=str,
        help="Ticker symbol (e.g., AAPL, BTC-USD, ^GSPC)"
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

    args = parser.parse_args()

    # Validate date range arguments
    if args.start and not args.end:
        parser.error("--start requires --end")
    if args.end and not args.start:
        parser.error("--end requires --start")

    # Fetch data
    if args.start and args.end:
        df = fetch_stock_data(
            ticker=args.ticker,
            interval=args.interval,
            start_date=args.start,
            end_date=args.end
        )
    else:
        df = fetch_stock_data(
            ticker=args.ticker,
            interval=args.interval,
            period=args.period
        )

    # Prepare data for analysis
    data_dict = prepare_data_for_analysis(df, limit=args.limit)

    # Run analysis
    final_state = run_analysis(args.ticker, args.interval, data_dict)

    # Print results
    print_results(final_state)


if __name__ == "__main__":
    main()

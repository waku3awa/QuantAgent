# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QuantAgent is a sophisticated multi-agent trading analysis system that combines technical indicators, pattern recognition, and trend analysis using LangChain and LangGraph. The system provides both a web interface and programmatic access for comprehensive market analysis.

## Key Commands

### Development Environment Setup
```bash
# Create and activate conda environment
conda create -n quantagents python=3.10
conda activate quantagents

# Install dependencies
pip install -r requirements.txt

# Alternative TA-Lib installation if needed
conda install -c conda-forge ta-lib
```

### Running the Application
```bash
# Start the web interface
python web_interface.py
```
The web application will be available at `http://127.0.0.1:5000`

### Testing and Validation
The project doesn't include explicit test commands. For validation:
- Run the web interface and test with real market data
- Use the programmatic API with sample data to verify agent functionality

## Architecture Overview

### Multi-Agent System Structure
The system consists of four specialized agents coordinated through LangGraph:

1. **Indicator Agent** (`indicator_agent.py`)
   - Computes technical indicators: RSI, MACD, Stochastic Oscillator, ROC, Williams %R
   - Uses TA-Lib for calculations
   - Provides momentum and signal analysis

2. **Pattern Agent** (`pattern_agent.py`)
   - Generates and analyzes candlestick charts
   - Identifies chart patterns using visual analysis
   - Returns pattern descriptions and trading implications

3. **Trend Agent** (`trend_agent.py`)
   - Creates trend-annotated charts with support/resistance lines
   - Analyzes market direction and channel formations
   - Provides trend strength and direction analysis

4. **Decision Agent** (`decision_agent.py`)
   - Synthesizes outputs from all other agents
   - Makes final LONG/SHORT trading decisions
   - Provides entry/exit points and risk management

### Core Components

- **TradingGraph** (`trading_graph.py`): Main orchestrator that initializes LLMs and coordinates agent execution
- **Agent State** (`agent_state.py`): TypedDict defining shared state structure across all agents
- **Graph Setup** (`graph_setup.py`): Configures LangGraph workflow and agent connections
- **Graph Utilities** (`graph_util.py`): Technical analysis tools and chart generation utilities
- **Web Interface** (`web_interface.py`): Flask-based web application for real-time analysis

### Data Flow
1. Market data ingestion via yfinance
2. Parallel processing by indicator, pattern, and trend agents
3. State aggregation in IndicatorAgentState
4. Final decision synthesis by decision agent
5. Results presentation through web interface or API

## Configuration

### LLM Configuration
Default configuration in `default_config.py`:
- `agent_llm_model`: "gpt-4o-mini" (for individual agents)
- `graph_llm_model`: "gpt-4o" (for graph logic and decision making)
- `agent_llm_temperature`: 0.1
- `graph_llm_temperature`: 0.1

### API Key Setup
Set OpenAI API key either:
- Through web interface API key management
- Environment variable: `export OPENAI_API_KEY="your_api_key_here"`

## Key Dependencies

### Core Libraries
- **LangChain/LangGraph**: Multi-agent orchestration and LLM integration
- **OpenAI**: LLM API access
- **yfinance**: Real-time market data
- **TA-Lib**: Technical analysis calculations
- **pandas/numpy**: Data manipulation
- **matplotlib/mplfinance**: Chart generation
- **Flask**: Web interface

### Important Notes
- Requires LLM with image input capability for visual chart analysis
- Uses matplotlib with 'Agg' backend for headless chart generation
- All chart outputs are base64-encoded for web compatibility

## Development Guidelines

### Code Structure
- Each agent is implemented as a separate module with create_*_agent() factory functions
- State management uses TypedDict for type safety
- Tools are defined in graph_util.py using LangChain @tool decorator
- Configuration is centralized in default_config.py

### Visual Analysis Requirements
The system generates and analyzes charts extensively:
- K-line (candlestick) charts for pattern recognition
- Trend-annotated charts with support/resistance lines
- Base64 encoding for web display and LLM consumption

### Asset Support
Supports multiple asset classes through yfinance:
- Stocks (AAPL, TSLA)
- Crypto (BTC-USD)
- Indices (^GSPC, ^DJI)
- Futures (GC=F, CL=F, NQ=F, ES=F)
- Other instruments (VIX, DXY)

## Programmatic Usage

```python
from trading_graph import TradingGraph

# Initialize with default or custom config
trading_graph = TradingGraph()

# Prepare state with market data
initial_state = {
    "kline_data": your_dataframe_dict,
    "analysis_results": None,
    "messages": [],
    "time_frame": "4hour",
    "stock_name": "BTC"
}

# Execute analysis
final_state = trading_graph.graph.invoke(initial_state)

# Access results
print(final_state.get("final_trade_decision"))
print(final_state.get("indicator_report"))
print(final_state.get("pattern_report"))
print(final_state.get("trend_report"))
```

## Important Considerations

- System is designed for educational and research purposes only
- All analysis is based on historical data patterns
- Requires stable internet connection for real-time data
- LLM costs can accumulate with frequent usage
- Visual chart analysis requires models with image processing capabilities
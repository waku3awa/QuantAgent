from typing import Annotated, Sequence, TypedDict, List, Optional
from typing_extensions import NotRequired
from langgraph.graph import MessagesState
from langchain_core.messages import BaseMessage


class IndicatorAgentState(TypedDict):
    """State type for the Indicator Agent including messages, input data, and analysis result."""
    kline_data: Annotated[dict, "OHLCV dictionary used for computing technical indicators"]
    time_frame: Annotated[str, "time period for k line data provided"]
    stock_name: Annotated[str, "stock name for prompt"]
    
    # Indicator Agent Tools output values (explicitly added per indicator)
    rsi: NotRequired[Annotated[List[float], "Relative Strength Index values"]]
    macd: NotRequired[Annotated[List[float], "MACD line values"]]
    macd_signal: NotRequired[Annotated[List[float], "MACD signal line values"]]
    macd_hist: NotRequired[Annotated[List[float], "MACD histogram values"]]
    stoch_k: NotRequired[Annotated[List[float], "Stochastic Oscillator %K values"]]
    stoch_d: NotRequired[Annotated[List[float], "Stochastic Oscillator %D values"]]
    roc: NotRequired[Annotated[List[float], "Rate of Change values"]]
    willr: NotRequired[Annotated[List[float], "Williams %R values"]]
    indicator_report: NotRequired[Annotated[str, "Final indicator agent summary report to be used by downstream agents"]]


    # Pattern Agent
    pattern_image: NotRequired[Annotated[str, "Base64-encoded K-line chart for pattern recognition agent use"]]
    pattern_image_filename: NotRequired[Annotated[str, "Local file path to saved K-line chart image"]]
    pattern_image_description: NotRequired[Annotated[str, "Brief description of the generated K-line image"]]
    pattern_report: NotRequired[Annotated[str, "Final pattern agent summary report to be used by downstream agents"]]

    # Trend Agent
    trend_image: NotRequired[Annotated[str, "Base64-encoded trend-annotated candlestick (K-line) chart for trend recognition agent use"]]
    trend_image_filename: NotRequired[Annotated[str, "Local file path to saved trendline-enhanced K-line chart image"]]
    trend_image_description: NotRequired[Annotated[str, "Brief description of the chart, including presence of support/resistance lines and visual characteristics"]]
    trend_report: NotRequired[Annotated[str, "Final trend analysis summary, describing structure, directional bias, and technical observations for downstream agents"]]

    # Final analysis and messaging context
    analysis_results: NotRequired[Annotated[Optional[str], "Computed result of the analysis or decision"]]
    messages: Annotated[List[BaseMessage], "List of chat messages used in LLM prompt construction"]
    decision_prompt: NotRequired[Annotated[Optional[str], "decision prompt for reflection"]]
    final_trade_decision: NotRequired[Annotated[Optional[str], "Final BUY or SELL decision made after analyzing indicators"]]

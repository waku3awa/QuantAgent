"""
TradingGraph: Orchestrates the multi-agent trading system using LangChain and LangGraph.
Initializes LLMs, toolkits, and agent nodes for indicator, pattern, and trend analysis.
"""
from typing import Dict, Optional
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from default_config import DEFAULT_CONFIG
from graph_util import TechnicalTools
from graph_setup import SetGraph
from llm_factory import get_chat_model

class TradingGraph:
    """
    Main orchestrator for the multi-agent trading system.
    Sets up LLMs, toolkits, and agent nodes for indicator, pattern, and trend analysis.

    Supports multiple LLM providers:
    - openai: OpenAI ChatGPT models (default)
    - claude_api: Anthropic Claude API
    - claude_cli: Claude CLI wrapper (requires claude command in PATH)
    """
    def __init__(
        self,
        config: Optional[Dict] = None,
        provider: str = "openai",
        agent_model: Optional[str] = None,
        graph_model: Optional[str] = None
    ):
        """
        Initialize TradingGraph with configurable LLM provider.

        Args:
            config: Configuration dictionary (uses DEFAULT_CONFIG if None)
            provider: LLM provider ("openai", "claude_api", "claude_cli")
            agent_model: Model name for agent LLMs (uses provider default if None)
            graph_model: Model name for graph LLM (uses provider default if None)

        Examples:
            >>> # Use OpenAI (default)
            >>> graph = TradingGraph()

            >>> # Use Claude API
            >>> graph = TradingGraph(provider="claude_api")

            >>> # Use Claude CLI
            >>> graph = TradingGraph(provider="claude_cli")

            >>> # Use specific models
            >>> graph = TradingGraph(
            ...     provider="openai",
            ...     agent_model="gpt-4o-mini",
            ...     graph_model="gpt-4o"
            ... )
        """
        # --- Configuration and LLMs ---
        self.config = config if config is not None else DEFAULT_CONFIG.copy()
        self.provider = provider

        # Initialize LLMs with factory function
        self.agent_llm = get_chat_model(
            provider=provider,
            model=agent_model or self.config.get("agent_llm_model"),
            temperature=self.config.get("agent_llm_temperature", 0.1)
        )
        self.graph_llm = get_chat_model(
            provider=provider,
            model=graph_model or self.config.get("graph_llm_model"),
            temperature=self.config.get("graph_llm_temperature", 0.1)
        )
        self.toolkit = TechnicalTools()

        # --- Create tool nodes for each agent ---
        self.tool_nodes = self._set_tool_nodes()

        # --- Graph logic and setup ---
        self.graph_setup = SetGraph(
            self.agent_llm,
            self.graph_llm,
            self.toolkit,
            self.tool_nodes,
            self.config,
        )
        
        # --- The main LangGraph graph object ---
        self.graph = self.graph_setup.set_graph()
    
    def _set_tool_nodes(self) -> Dict[str, ToolNode]:
        """
        Define tool nodes for each agent type (indicator, pattern, trend).
        """
        return {
            "indicator": ToolNode(
                [
                    self.toolkit.compute_macd,
                    self.toolkit.compute_roc,
                    self.toolkit.compute_rsi,
                    self.toolkit.compute_stoch,
                    self.toolkit.compute_willr,
                ]
            ),
            "pattern": ToolNode(
                [
                    self.toolkit.generate_kline_image,
                ]
            ),
            "trend": ToolNode(
                [
                    self.toolkit.generate_trend_image
                ]
            )
        }
    
    def refresh_llms(self, provider: Optional[str] = None):
        """
        Refresh the LLM objects with the current API key from environment.
        This is called when the API key is updated.

        Args:
            provider: Optional new provider to switch to (keeps current if None)
        """
        if provider:
            self.provider = provider

        # Recreate LLM objects with current environment API key and config values
        self.agent_llm = get_chat_model(
            provider=self.provider,
            model=self.config.get("agent_llm_model"),
            temperature=self.config.get("agent_llm_temperature", 0.1)
        )
        self.graph_llm = get_chat_model(
            provider=self.provider,
            model=self.config.get("graph_llm_model"),
            temperature=self.config.get("graph_llm_temperature", 0.1)
        )

        # Recreate the graph setup with new LLMs
        self.graph_setup = SetGraph(
            self.agent_llm,
            self.graph_llm,
            self.toolkit,
            self.tool_nodes,
            self.config,
        )

        # Recreate the main graph
        self.graph = self.graph_setup.set_graph()
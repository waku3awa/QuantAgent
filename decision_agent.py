"""
Agent for making final trade decisions in high-frequency trading (HFT) context.
Combines indicator, pattern, and trend reports to issue a LONG or SHORT order.
"""

from prompt_provider import PromptProvider


def create_final_trade_decider(llm, config=None):
    """
    Create a trade decision agent node. The agent uses LLM to synthesize indicator, pattern, and trend reports
    and outputs a final trade decision (LONG or SHORT) with justification and risk-reward ratio.
    """
    # プロンプトプロバイダーを初期化
    provider = PromptProvider(config)

    def trade_decision_node(state) -> dict:
        indicator_report = state["indicator_report"]
        pattern_report = state['pattern_report']
        trend_report = state['trend_report']
        time_frame = state['time_frame']
        stock_name = state['stock_name']

        # --- 設定ベースでプロンプトを取得・フォーマット ---
        prompt = provider.format_prompt(
            "decision_agent",
            "system",
            time_frame=time_frame,
            stock_name=stock_name,
            indicator_report=indicator_report,
            pattern_report=pattern_report,
            trend_report=trend_report
        )

        # --- LLM call for decision ---
        response = llm.invoke(prompt)

        return {
            "final_trade_decision": response.content,
            "messages": [response],
            "decision_prompt": prompt,
        }

    return trade_decision_node

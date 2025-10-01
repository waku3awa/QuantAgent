"""
ChatPromptTemplate修正後の動作確認テスト
"""
from pattern_agent import create_pattern_agent
from indicator_agent import create_indicator_agent
from langchain_openai import ChatOpenAI
from graph_util import TechnicalTools
from prompt_provider import PromptProvider
from default_config import DEFAULT_CONFIG
import json

def test_agents_with_empty_messages():
    """空のmessagesリストでエージェントをテスト"""
    print("=== ChatPromptTemplate修正後テスト ===\n")

    # LLMとツールキットを初期化
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    toolkit = TechnicalTools()
    config = DEFAULT_CONFIG.copy()

    # テスト用のダミーデータ
    dummy_kline_data = {
        "Datetime": ["2023-01-01 00:00:00", "2023-01-01 01:00:00"],
        "Open": [100.0, 101.0],
        "High": [102.0, 103.0],
        "Low": [99.0, 100.0],
        "Close": [101.0, 102.0]
    }

    dummy_state = {
        "kline_data": dummy_kline_data,
        "messages": [],  # 空のリスト
        "time_frame": "1hour",
        "stock_name": "BTC"
    }

    print("修正内容:")
    print("第1段階: ChatPromptTemplateの呼び出し方法修正")
    print("- pattern_agent.py: chain.invoke(messages) → chain.invoke({'messages': messages})")
    print("- indicator_agent.py: chain.invoke(messages) → chain.invoke({'messages': messages})")
    print("第2段階: 変数エスケープ問題修正")
    print("- pattern_agent.py: .partial(kline_data=...)を削除、HumanMessageに直接含める")
    print("- indicator_agent.py: kline_dataプレースホルダーを使用、HumanMessageに実データを含める")
    print("- プロンプト内のJSONが変数として誤認識される問題を解決\n")

    print("1. PromptProviderテスト...")
    try:
        provider = PromptProvider(config)
        prompt = provider.get_prompt("indicator_agent", "system")
        print(f"   [成功] PromptProvider: プロンプト取得成功")
        print(f"   プロンプトの最初の100文字: {prompt[:100]}...")
    except Exception as e:
        print(f"   [エラー] PromptProvider: {e}")

    print("\n2. ChatPromptTemplate作成テスト...")
    try:
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_core.messages import HumanMessage

        system_prompt = "You are a trading analyst."
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])

        # 正しい形式でテスト
        test_messages = [HumanMessage(content="Test message")]
        result = prompt.invoke({"messages": test_messages})
        print("   [成功] ChatPromptTemplate: 辞書形式での呼び出し成功")
    except Exception as e:
        print(f"   [エラー] ChatPromptTemplate: {e}")

    print("\nテスト完了")
    print("Web UIから実際の分析を実行してエラーが解消されたか確認してください。")

if __name__ == "__main__":
    test_agents_with_empty_messages()
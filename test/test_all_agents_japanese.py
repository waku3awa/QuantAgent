"""
全エージェントの日本語出力機能統合テストスクリプト
各エージェントのプロンプトが正常に日本語化されているかテスト
"""

from prompt_provider import PromptProvider
from default_config import DEFAULT_CONFIG


def test_all_agents_prompts():
    """全エージェントのプロンプト取得をテスト"""
    print("=== 全エージェント日本語プロンプトテスト ===")

    # 日本語設定でテスト
    config_ja = DEFAULT_CONFIG.copy()
    config_ja["output_language"] = "ja"
    provider_ja = PromptProvider(config_ja)

    agents = ["indicator_agent", "pattern_agent", "trend_agent", "decision_agent"]

    for agent in agents:
        print(f"\n--- {agent} テスト ---")

        try:
            # 基本的なシステムプロンプト取得
            system_prompt = provider_ja.get_prompt(agent, "system")
            print(f"[成功] {agent} システムプロンプト取得成功")
            print(f"プロンプトの最初の100文字: {system_prompt[:100]}...")

            # 利用可能な言語チェック
            available_langs = provider_ja.get_available_languages(agent, "system")
            print(f"利用可能な言語: {available_langs}")

        except Exception as e:
            print(f"[エラー] {agent} プロンプト取得エラー: {e}")


def test_pattern_agent_specific():
    """パターンエージェントの特別なプロンプトをテスト"""
    print("\n=== パターンエージェント特別プロンプトテスト ===")

    provider = PromptProvider()

    try:
        # 画像分析プロンプト
        image_analysis = provider.format_prompt(
            "pattern_agent",
            "image_analysis",
            time_frame="4時間",
            pattern_descriptions="テストパターン説明"
        )
        print("[成功] パターンエージェント画像分析プロンプト生成成功")

        # 画像システムプロンプト
        image_system = provider.get_prompt("pattern_agent", "image_system")
        print(f"[成功] 画像システムプロンプト: {image_system}")

    except Exception as e:
        print(f"[エラー] パターンエージェント特別プロンプトエラー: {e}")


def test_trend_agent_specific():
    """トレンドエージェントの特別なプロンプトをテスト"""
    print("\n=== トレンドエージェント特別プロンプトテスト ===")

    provider = PromptProvider()

    try:
        # 画像分析プロンプト
        image_analysis = provider.format_prompt(
            "trend_agent",
            "image_analysis",
            time_frame="1時間"
        )
        print("[成功] トレンドエージェント画像分析プロンプト生成成功")

        # 画像システムプロンプト
        image_system = provider.get_prompt("trend_agent", "image_system")
        print(f"[成功] 画像システムプロンプト: {image_system}")

    except Exception as e:
        print(f"[エラー] トレンドエージェント特別プロンプトエラー: {e}")


def test_indicator_agent_specific():
    """インディケーターエージェントのプロンプトをテスト"""
    print("\n=== インディケーターエージェント特別プロンプトテスト ===")

    provider = PromptProvider()

    try:
        # フォーマット済みプロンプト
        formatted_prompt = provider.format_prompt(
            "indicator_agent",
            "system",
            time_frame="15分",
            kline_data='{"test": "data"}'
        )
        print("[成功] インディケーターエージェントフォーマット済みプロンプト生成成功")
        print(f"フォーマット結果の最初の200文字: {formatted_prompt[:200]}...")

    except Exception as e:
        print(f"[エラー] インディケーターエージェント特別プロンプトエラー: {e}")


def test_decision_agent_specific():
    """決定エージェントのプロンプトをテスト"""
    print("\n=== 決定エージェント特別プロンプトテスト ===")

    provider = PromptProvider()

    try:
        # フォーマット済みプロンプト
        formatted_prompt = provider.format_prompt(
            "decision_agent",
            "system",
            time_frame="4時間",
            stock_name="BTC-USD",
            indicator_report="RSI: 65, MACD: 上昇",
            pattern_report="フラッグパターン",
            trend_report="上昇トレンド"
        )
        print("[成功] 決定エージェントフォーマット済みプロンプト生成成功")
        print(f"フォーマット結果の最初の200文字: {formatted_prompt[:200]}...")

    except Exception as e:
        print(f"[エラー] 決定エージェント特別プロンプトエラー: {e}")


def test_language_switching():
    """言語切り替え機能の統合テスト"""
    print("\n=== 言語切り替え統合テスト ===")

    provider = PromptProvider()

    # 各エージェントで言語切り替えテスト
    agents = ["indicator_agent", "pattern_agent", "trend_agent", "decision_agent"]

    for agent in agents:
        print(f"\n{agent} 言語切り替えテスト:")

        try:
            # 日本語
            provider.set_language("ja")
            ja_prompt = provider.get_prompt(agent, "system")
            print(f"  日本語: {ja_prompt[:50]}...")

            # 英語
            provider.set_language("en")
            en_prompt = provider.get_prompt(agent, "system")
            print(f"  英語: {en_prompt[:50]}...")

            print(f"  [成功] {agent} 言語切り替え正常")

        except Exception as e:
            print(f"  [エラー] {agent} 言語切り替えエラー: {e}")


if __name__ == "__main__":
    print("全エージェント日本語化統合テスト開始\n")

    test_all_agents_prompts()
    test_indicator_agent_specific()
    test_pattern_agent_specific()
    test_trend_agent_specific()
    test_decision_agent_specific()
    test_language_switching()

    print("\n統合テスト完了")
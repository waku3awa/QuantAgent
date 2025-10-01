"""
日本語出力機能のテストスクリプト
プロンプトプロバイダーと設定ベースのシステムが正常に動作するかテスト
"""

from prompt_provider import PromptProvider
from default_config import DEFAULT_CONFIG


def test_prompt_provider():
    """プロンプトプロバイダーの基本機能をテスト"""
    print("=== プロンプトプロバイダーテスト ===")

    # 日本語設定でテスト
    config_ja = DEFAULT_CONFIG.copy()
    config_ja["output_language"] = "ja"
    provider_ja = PromptProvider(config_ja)

    print(f"設定された言語: {provider_ja.language}")

    # 利用可能な言語をチェック
    available_langs = provider_ja.get_available_languages("decision_agent")
    print(f"利用可能な言語: {available_langs}")

    # 日本語プロンプトを取得
    try:
        ja_prompt = provider_ja.format_prompt(
            "decision_agent",
            "system",
            time_frame="4時間",
            stock_name="BTC-USD",
            indicator_report="RSI: 65, MACD: 上昇トレンド",
            pattern_report="フラッグパターン形成中",
            trend_report="上昇トレンド継続"
        )
        print("\n[成功] 日本語プロンプト生成成功")
        print(f"プロンプトの最初の200文字: {ja_prompt[:200]}...")

    except Exception as e:
        print(f"[エラー] 日本語プロンプト生成エラー: {e}")

    # 英語設定でテスト
    print("\n--- 英語設定テスト ---")
    config_en = DEFAULT_CONFIG.copy()
    config_en["output_language"] = "en"
    provider_en = PromptProvider(config_en)

    try:
        en_prompt = provider_en.format_prompt(
            "decision_agent",
            "system",
            time_frame="4hour",
            stock_name="BTC-USD",
            indicator_report="RSI: 65, MACD: uptrend",
            pattern_report="Flag pattern forming",
            trend_report="Uptrend continuing"
        )
        print("[成功] 英語プロンプト生成成功")
        print(f"プロンプトの最初の200文字: {en_prompt[:200]}...")

    except Exception as e:
        print(f"[エラー] 英語プロンプト生成エラー: {e}")


def test_language_switching():
    """言語切り替え機能のテスト"""
    print("\n=== 言語切り替えテスト ===")

    provider = PromptProvider()
    print(f"初期言語: {provider.language}")

    # 英語に切り替え
    provider.set_language("en")
    print(f"切り替え後: {provider.language}")

    # 日本語に戻す
    provider.set_language("ja")
    print(f"再切り替え後: {provider.language}")


def test_variable_validation():
    """変数検証機能のテスト"""
    print("\n=== 変数検証テスト ===")

    provider = PromptProvider()

    # 正常なケース
    is_valid, missing, extra = provider.validate_prompt_variables(
        "decision_agent",
        "system",
        time_frame="4時間",
        stock_name="BTC-USD",
        indicator_report="テスト",
        pattern_report="テスト",
        trend_report="テスト"
    )

    print(f"正常ケース - 有効: {is_valid}, 不足: {missing}, 余分: {extra}")

    # 変数不足のケース
    is_valid, missing, extra = provider.validate_prompt_variables(
        "decision_agent",
        "system",
        time_frame="4時間"
        # 他の変数が不足
    )

    print(f"不足ケース - 有効: {is_valid}, 不足: {missing}, 余分: {extra}")


if __name__ == "__main__":
    print("日本語出力機能テスト開始\n")

    test_prompt_provider()
    test_language_switching()
    test_variable_validation()

    print("\nテスト完了")
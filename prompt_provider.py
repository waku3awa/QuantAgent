"""
プロンプトプロバイダークラス
設定ベースでプロンプトを管理し、多言語対応を実現
"""

from prompts import PROMPTS
from default_config import DEFAULT_CONFIG


class PromptProvider:
    """
    プロンプトの取得と管理を行うクラス
    設定に基づいて適切な言語のプロンプトを提供
    """

    def __init__(self, config=None):
        """
        初期化

        Args:
            config (dict): 設定辞書。Noneの場合はデフォルト設定を使用
        """
        self.config = config if config is not None else DEFAULT_CONFIG.copy()
        self.language = self.config.get("output_language", "ja")

    def get_prompt(self, agent_name, prompt_type="system"):
        """
        指定されたエージェントのプロンプトを取得

        Args:
            agent_name (str): エージェント名（例：'decision_agent'）
            prompt_type (str): プロンプトタイプ（例：'system', 'user'）

        Returns:
            str: プロンプトテンプレート

        Raises:
            KeyError: 指定されたエージェントまたは言語が存在しない場合
        """
        try:
            return PROMPTS[agent_name][prompt_type][self.language]
        except KeyError as e:
            # フォールバック：英語版を試す
            if self.language != "en":
                try:
                    fallback_prompt = PROMPTS[agent_name][prompt_type]["en"]
                    print(f"Warning: {self.language} prompt not found for {agent_name}.{prompt_type}, using English fallback")
                    return fallback_prompt
                except KeyError:
                    pass

            raise KeyError(f"Prompt not found: {agent_name}.{prompt_type}.{self.language}") from e

    def format_prompt(self, agent_name, prompt_type="system", **kwargs):
        """
        プロンプトテンプレートに変数を代入してフォーマット

        Args:
            agent_name (str): エージェント名
            prompt_type (str): プロンプトタイプ
            **kwargs: プロンプトテンプレートの変数

        Returns:
            str: フォーマット済みプロンプト
        """
        template = self.get_prompt(agent_name, prompt_type)
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing template variable: {e}") from e

    def set_language(self, language):
        """
        出力言語を動的に変更

        Args:
            language (str): 言語コード（'ja', 'en'など）
        """
        self.language = language
        self.config["output_language"] = language

    def get_available_languages(self, agent_name, prompt_type="system"):
        """
        指定されたエージェントで利用可能な言語リストを取得

        Args:
            agent_name (str): エージェント名
            prompt_type (str): プロンプトタイプ

        Returns:
            list: 利用可能な言語コードのリスト
        """
        try:
            return list(PROMPTS[agent_name][prompt_type].keys())
        except KeyError:
            return []

    def validate_prompt_variables(self, agent_name, prompt_type="system", **kwargs):
        """
        プロンプトテンプレートの変数が正しく提供されているかチェック

        Args:
            agent_name (str): エージェント名
            prompt_type (str): プロンプトタイプ
            **kwargs: 検証する変数

        Returns:
            tuple: (is_valid, missing_vars, extra_vars)
        """
        template = self.get_prompt(agent_name, prompt_type)

        # テンプレートから必要な変数を抽出
        import re
        required_vars = set(re.findall(r'\{(\w+)\}', template))
        provided_vars = set(kwargs.keys())

        missing_vars = required_vars - provided_vars
        extra_vars = provided_vars - required_vars

        is_valid = len(missing_vars) == 0

        return is_valid, list(missing_vars), list(extra_vars)
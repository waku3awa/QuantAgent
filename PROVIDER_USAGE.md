# LLM Provider Selection Guide

QuantAgentは複数のLLMプロバイダをサポートしています：
- **OpenAI** (デフォルト)
- **Claude API** (Anthropic)
- **Claude CLI** (ローカルCLIラッパー)

## クイックスタート

### 1. OpenAI (デフォルト)

```python
from trading_graph import TradingGraph

# OpenAIを使用（デフォルト）
graph = TradingGraph()

# または明示的に指定
graph = TradingGraph(provider="openai")

# 特定のモデルを指定
graph = TradingGraph(
    provider="openai",
    agent_model="gpt-4o-mini",
    graph_model="gpt-4o"
)
```

**必要な設定:**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 2. Claude API (Anthropic)

```python
from trading_graph import TradingGraph

# Claude APIを使用
graph = TradingGraph(provider="claude_api")

# 特定のモデルを指定
graph = TradingGraph(
    provider="claude_api",
    agent_model="claude-3-5-sonnet-20241022",
    graph_model="claude-3-5-sonnet-20241022"
)
```

**必要な設定:**
```bash
# パッケージのインストール
pip install langchain-anthropic

# APIキーの設定
export ANTHROPIC_API_KEY="your-api-key-here"
```

### 3. Claude CLI (ローカル)

```python
from trading_graph import TradingGraph

# Claude CLIを使用
graph = TradingGraph(provider="claude_cli")
```

**必要な設定:**
```bash
# Claude CLIのインストール
npm install -g @anthropics/claude-code

# 認証（初回のみ）
claude auth

# PATHの確認
which claude  # Unix
where claude  # Windows
```

**特徴:**
- ✅ APIキー不要（Claude CLIが管理）
- ✅ ローカルで動作
- ⚠️ モデル名の指定は無効（CLIが自動選択）
- ⚠️ ツール呼び出しは未サポート

## 実行例

### 基本的な使用

```python
from trading_graph import TradingGraph
import yfinance as yf

# プロバイダを選択
graph = TradingGraph(provider="claude_cli")

# データを取得
ticker = yf.Ticker("AAPL")
df = ticker.history(period="1mo", interval="1d")

# 分析を実行
initial_state = {
    "kline_data": df.to_dict(),
    "analysis_results": None,
    "messages": [],
    "time_frame": "1day",
    "stock_name": "AAPL"
}

result = graph.graph.invoke(initial_state)
print(result["final_trade_decision"])
```

### プロバイダの切り替え

```python
# 最初はOpenAIで初期化
graph = TradingGraph(provider="openai")

# 後からClaude CLIに切り替え
graph.refresh_llms(provider="claude_cli")
```

## プロバイダ比較

| 機能 | OpenAI | Claude API | Claude CLI |
|------|--------|------------|------------|
| APIキー必要 | ✅ | ✅ | ❌ |
| オンライン必須 | ✅ | ✅ | ✅ |
| ツールサポート | ✅ | ✅ | ❌ |
| ビジョンサポート | ✅ | ✅ | ✅ |
| コスト | 従量課金 | 従量課金 | サブスク |
| レスポンス速度 | 高速 | 高速 | やや遅い |

## トラブルシューティング

### OpenAI

**エラー:** `AuthenticationError`
```bash
# APIキーを確認
echo $OPENAI_API_KEY

# 再設定
export OPENAI_API_KEY="sk-..."
```

### Claude API

**エラー:** `ImportError: langchain-anthropic is required`
```bash
pip install langchain-anthropic
```

**エラー:** `AuthenticationError`
```bash
# APIキーを確認
echo $ANTHROPIC_API_KEY

# 再設定
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Claude CLI

**エラー:** `Claude CLI not found`
```bash
# インストール確認
claude --version

# パスの確認
which claude  # Unix
where claude  # Windows

# 再インストール
npm install -g @anthropics/claude-code
```

**エラー:** `Session timeout`
```python
# タイムアウトを延長
from llm_factory import get_chat_model

llm = get_chat_model("claude_cli", timeout=120.0)
```

## 設定ファイルでの指定

`default_config.py`を編集してデフォルトプロバイダを変更:

```python
DEFAULT_CONFIG = {
    "llm_provider": "claude_cli",  # 追加
    "agent_llm_model": "gpt-4o-mini",
    "graph_llm_model": "gpt-4o",
    "agent_llm_temperature": 0.1,
    "graph_llm_temperature": 0.1,
}
```

使用時:

```python
from trading_graph import TradingGraph
from default_config import DEFAULT_CONFIG

# 設定からプロバイダを取得
provider = DEFAULT_CONFIG.get("llm_provider", "openai")
graph = TradingGraph(provider=provider)
```

## Web インターフェースでの使用

`web_interface.py`でプロバイダ選択を追加:

```python
# プロバイダ選択のドロップダウンを追加
provider = request.form.get('provider', 'openai')

# TradingGraphを初期化
trading_graph = TradingGraph(provider=provider)
```

## パフォーマンス最適化

### OpenAI
- `gpt-4o-mini`を使用（高速・低コスト）
- バッチ処理で複数リクエストを最適化

### Claude API
- `claude-3-5-sonnet`を使用（バランス型）
- キャッシング機能を活用

### Claude CLI
- タイムアウトを適切に設定
- 並列実行は避ける（CLIの制限）

## 開発ガイド

### 新しいプロバイダの追加

1. `llm_factory.py`に分岐を追加:

```python
def get_chat_model(provider: str = "openai", ...):
    # ...
    elif provider == "new_provider":
        from langchain_new import ChatNew
        return ChatNew(...)
```

2. テスト:

```python
graph = TradingGraph(provider="new_provider")
```

### カスタムラッパーの作成

`BaseChatModel`を継承:

```python
from langchain_core.language_models.chat_models import BaseChatModel

class ChatCustom(BaseChatModel):
    def _generate(self, messages, stop=None, **kwargs):
        # カスタム実装
        pass

    @property
    def _llm_type(self):
        return "chat_custom"
```

## 参考リンク

- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API Reference](https://platform.openai.com/docs)
- [Anthropic Claude API](https://docs.anthropic.com/)
- [Claude CLI Documentation](https://claude.ai/code)

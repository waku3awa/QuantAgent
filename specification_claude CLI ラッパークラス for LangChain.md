## 実装仕様書：`claude` CLI ラッパークラス for LangChain

### 1. 概要

#### 1.1. 目的
本仕様書は、Anthropic社の`claude`コマンドラインインターフェース（CLI）を、LangChainフレームワーク内で`ChatOpenAI`と同様のインターフェースで利用可能にするためのラッパークラス（以下、`ChatClaudeCLI`）の実装仕様を定義する。

#### 1.2. 背景
既存のLangChainアプリケーションにおいて、LLMバックエンドを`ChatOpenAI`から`claude`へ移行する際に、アプリケーション側のコード変更を最小限に抑えることが求められる。`claude` CLIのヘッドレスモード (`-p`フラグ) を活用し、これをLangChainの`BaseChatModel`に準拠したクラスで抽象化することにより、シームレスな差し替えを実現する。

#### 1.3. スコープ

| 項目 | 対象 | 備考 |
| :--- | :--- | :--- |
| **テキスト入出力** | ✅ 対象 | 単純な文字列プロンプトによる同期的な応答取得 |
| **マルチモーダル入力** | ✅ 対象 (拡張仕様) | Base64エンコードされた画像を含むプロンプトへの対応 |
| **ストリーミング応答** | ❌ 対象外 | `stream`メソッドの実装は本仕様の範囲外とする |
| **バッチ処理** | ❌ 対象外 | `batch`メソッドの実装は本仕様の範囲外とする |
| **パラメータ指定** | ❌ 対象外 | 温度設定 (`-t`) などのCLIオプション指定は対象外とする |

---

### 2. 要求仕様

#### 2.1. 機能要求
| ID | 要求事項 |
| :--- | :--- |
| **FR-01** | クラスは`langchain.chat_models.base.SimpleChatModel`を継承し、LangChainの標準的なインターフェース（`invoke`等）で呼び出し可能であること。 |
| **FR-02** | LangChainの`Message`オブジェクトのリストを入力として受け取れること。 |
| **FR-03** | `claude -p "プロンプト" --output-format json` コマンドを内部で実行し、テキストベースの応答を取得できること。 |
| **FR-04** | （拡張）画像を含むマルチモーダルな入力を受け取り、`echo 'JSON' | claude -p --input-format stream-json ...` 形式でコマンドを実行できること。 |
| **FR-05** | `claude` CLIからのJSON形式の応答をパースし、応答本文の文字列を返すこと。`SimpleChatModel`の仕組みにより、これは自動的に`AIMessage`オブジェクトにラップされる。 |
| **FR-06** | `claude` CLIが存在しない、または実行に失敗した場合、適切な例外 (`RuntimeError`など) を送出すること。 |

#### 2.2. 非機能要求
| ID | 要求事項 |
| :--- | :--- |
| **NFR-01** | `claude` CLIが実行環境にインストールされ、PATHが通っていることを前提とする。 |
| **NFR-02** | `subprocess`の利用において、シェルインジェクションのリスクを考慮した実装を行うこと。（特に`shell=True`を使用する場合） |

---

### 3. アーキテクチャ

#### 3.1. クラス設計
* **クラス名**: `ChatClaudeCLI`
* **継承元**: `langchain.chat_models.base.SimpleChatModel`
* **主要メソッド**:
    * `_call(self, messages: List[BaseMessage], stop: List[str] = None, **kwargs) -> str`: LLM呼び出しのコアロジックを実装する。`invoke`から内部的に呼び出される。
    * `_llm_type(self) -> str`: モデルの種別を示す文字列（例: `"claude-cli"`）を返すプロパティ。

#### 3.2. データフロー
1.  **入力**: アプリケーションが`ChatClaudeCLI`インスタンスの`invoke`メソッドを`List[BaseMessage]`を引数として呼び出す。
2.  **プロンプト変換**: `_call`メソッド内で、`messages`リストから`claude` CLIに渡すプロンプト文字列、または`stream-json`形式のJSON文字列を生成する。
3.  **コマンド実行**: Pythonの`subprocess.run`モジュールを用いて`claude` CLIをサブプロセスとして実行する。
    * テキストの場合: `claude -p "<prompt>" --output-format json`
    * マルチモーダルの場合: `echo '<json_payload>' | claude -p --output-format stream-json --input-format stream-json`
4.  **出力取得**: サブプロセスの標準出力（stdout）から、`claude` CLIが返したJSON形式の応答を文字列としてキャプチャする。
5.  **応答解析**: 取得したJSON文字列をパースし、応答メッセージが含まれるキー（例: `"response"`）から本文を抽出する。
6.  **返却**: 抽出した応答文字列を返す。`SimpleChatModel`の仕組みにより、`AIMessage(content=...)`としてアプリケーションに返却される。

---

### 4. 実装詳細

#### 4.1. `_call` メソッド (テキスト入力)
* **引数**: `messages: List[BaseMessage]`
* **処理**:
    1.  `messages`リストの最後の要素の`content`をプロンプト文字列として抽出する。`content`が文字列でない場合は`ValueError`を送出する。
    2.  `command = ["claude", "-p", prompt, "--output-format", "json"]` のように、コマンドと引数をリスト形式で組み立てる。
    3.  `subprocess.run(command, capture_output=True, text=True, check=True)` を`try-except`ブロック内で実行する。
        * `FileNotFoundError`を捕捉した場合、「`claude`コマンドが見つかりません」というメッセージを含む`RuntimeError`を送出する。
        * `subprocess.CalledProcessError`を捕捉した場合、`e.stderr`の内容を含む`RuntimeError`を送出する。
    4.  成功した場合、`result.stdout`を`json.loads()`でパースし、`response`キーの値（文字列）を返す。

#### 4.2. `_call` メソッド (マルチモーダル入力対応)
* **処理**:
    1.  入力`messages`の`content`がリスト形式（マルチモーダル）であるか判定する。
    2.  マルチモーダルの場合、`messages`を`claude` CLIの`stream-json`が要求する仕様のJSONオブジェクトに変換するヘルパーメソッドを呼び出す。
    3.  `command = "echo '" + json_str + "' | claude ..."` のようにコマンド文字列を構築し、`subprocess.run(command, shell=True, ...)`で実行する。
    4.  以降の処理はテキスト入力の場合と同様。

#### 4.3. ヘルパーメソッド: `_convert_messages_to_claude_format`
* **目的**: LangChainの`Message`リストを`claude` CLI用のJSONオブジェクトに変換する。
* **入力**: `messages: List[BaseMessage]`
* **出力**: `Dict[str, Any]`
* **処理**:
    * `SystemMessage`はトップレベルの`system`キーに割り当てる。
    * `HumanMessage`, `AIMessage`は`messages`リスト内の`{"role": "user", ...}`または`{"role": "assistant", ...}`オブジェクトに変換する。
    * `content`がリストの場合、各要素を`{"type": "text", ...}`または`{"type": "image", "source": {...}}`形式に変換する。
    * 画像データはBase64エンコードされた文字列として`data`キーに設定する。

---

### 5. テストケース
| ID | ケース | 期待される結果 |
| :--- | :--- | :--- |
| **TC-01** | 正常系：単純なテキストプロンプト | `claude`からの正常な応答文字列が返却される。 |
| **TC-02** | 正常系：マルチモーダルプロンプト | `claude`からの正常な応答文字列が返却される。 |
| **TC-03** | 異常系：`claude`コマンド未インストール | `RuntimeError`が送出され、「command not found」という趣旨のメッセージが含まれる。 |
| **TC-04** | 異常系：`claude`コマンド実行時エラー（例: APIキー不正） | `RuntimeError`が送出され、`claude` CLIからのエラーメッセージが含まれる。 |
| **TC-05** | 異常系：空のメッセージリスト | `IndexError`または`ValueError`が送出される。 |
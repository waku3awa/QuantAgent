SECOND_LINK_PATH=$(realpath "$1")
THIRD_LINK_DIR="$HOME/stock_data/screening_3rd"
THIRD_LINK_PATH="$THIRD_LINK_DIR/quant_agent_result_latest.csv"

 # === ここから QuantAgent の処理を追加 ===
# QuantAgent ディレクトリへ移動し、仮想環境を有効化して run_multi_analysis.py を実行
QUANT_DIR="$HOME/QuantAgent"
RESULT_DIR="$QUANT_DIR/result"

echo "Running QuantAgent multi analysis (uses $SECOND_LINK_PATH) ..."
if [ -d "$QUANT_DIR" ]; then
    cd "$QUANT_DIR"
else
    echo "ERROR: QuantAgent directory not found: $QUANT_DIR" >&2
    exit 1
fi

if [ -f ".venv/bin/activate" ]; then
    . .venv/bin/activate
else
    echo "ERROR: .venv/bin/activate not found in $QUANT_DIR" >&2
    exit 1
fi

mkdir -p "$RESULT_DIR"
# run_multi_analysis.py を毎回実行
uv run run_multi_analysis.py --csv-file "$SECOND_LINK_PATH" --output ./result/

NEW_QUANT_RESULT=$(ls -1t "$RESULT_DIR"/quant_agent_result_*.csv 2>/dev/null | head -n1 || true)
if [ -z "$NEW_QUANT_RESULT" ]; then
    echo "ERROR: no quant agent result file found in $RESULT_DIR" >&2
    exit 1
fi
mkdir -p "$THIRD_LINK_DIR"
ln -sf "$NEW_QUANT_RESULT" "$THIRD_LINK_PATH"
echo "Updated symlink: $THIRD_LINK_PATH -> $(readlink -f "$THIRD_LINK_PATH")"
# === QuantAgent 処理ここまで ===

echo "== Done: $(date +'%Y-%m-%d %H:%M:%S') =="
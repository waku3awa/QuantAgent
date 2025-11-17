DEFAULT_CONFIG = {
    "agent_llm_model": "gpt-4o-mini",
    "graph_llm_model": "gpt-4o",
    "agent_llm_temperature": 0.1,
    "graph_llm_temperature": 0.1,
    "api_key": "",
    "output_language": "ja",  # 出力言語設定（ja: 日本語, en: 英語）
    # Ollama settings
    "ollama_model": "gemma3:12b",
    "ollama_base_url": "http://localhost:11434",
    # OpenAI Compatible API settings (read from environment variables)
    # Set OPENAI_COMPATIBLE_BASE_URL, OPENAI_COMPATIBLE_API_KEY, OPENAI_COMPATIBLE_MODEL
}
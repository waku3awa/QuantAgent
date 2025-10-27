"""
LLM Factory: Creates LLM instances based on provider name.

Supports:
- OpenAI (via ChatOpenAI)
- Anthropic Claude API (via ChatAnthropic)
- Claude CLI (via ChatClaudeCLI custom wrapper)
- Ollama (via ChatOllama)
"""

from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI


def get_chat_model(
    provider: str = "openai",
    model: Optional[str] = None,
    temperature: float = 0.1,
    **kwargs
):
    """
    Factory function to create a chat model instance based on provider.

    Args:
        provider: Provider name ("openai", "claude_api", "claude_cli", "ollama")
        model: Model name (provider-specific, uses default if None)
        temperature: Temperature setting for the model
        **kwargs: Additional provider-specific parameters

    Returns:
        Chat model instance (ChatOpenAI, ChatAnthropic, ChatClaudeCLI, or ChatOllama)

    Raises:
        ValueError: If provider is not supported
        ImportError: If required provider package is not installed

    Examples:
        >>> # Use OpenAI
        >>> llm = get_chat_model("openai", model="gpt-4")

        >>> # Use Claude API
        >>> llm = get_chat_model("claude_api", model="claude-3-opus-20240229")

        >>> # Use Claude CLI
        >>> llm = get_chat_model("claude_cli", temperature=0.7)

        >>> # Use Ollama
        >>> llm = get_chat_model("ollama", model="gemma3:12b")
    """
    provider = provider.lower()

    if provider == "openai":
        model_name = model or kwargs.get("openai_model", "gpt-4o-mini")
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            **{k: v for k, v in kwargs.items() if k != "openai_model"}
        )

    elif provider == "claude_api" or provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "langchain-anthropic is required for Claude API. "
                "Install it with: pip install langchain-anthropic"
            )

        model_name = model or kwargs.get("claude_model", "claude-3-5-sonnet-20241022")
        return ChatAnthropic(
            model=model_name,
            temperature=temperature,
            **{k: v for k, v in kwargs.items() if k != "claude_model"}
        )

    elif provider == "claude_cli":
        try:
            from claude_cli_wrapper import ChatClaudeCLI
        except ImportError:
            raise ImportError(
                "claude_cli_wrapper is required for Claude CLI. "
                "Make sure claude_cli_wrapper.py is in your project."
            )

        return ChatClaudeCLI(
            temperature=temperature,
            timeout=kwargs.get("timeout", 60.0)
        )

    elif provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError(
                "langchain-ollama is required for Ollama. "
                "Install it with: pip install langchain-ollama"
            )

        model_name = model or kwargs.get("ollama_model", "gemma3:12b")
        base_url = kwargs.get("ollama_base_url", "http://localhost:11434")
        num_predict = kwargs.get("num_predict", 512)

        return ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url=base_url,
            num_predict=num_predict,
            **{k: v for k, v in kwargs.items() if k not in ["ollama_model", "ollama_base_url", "num_predict"]}
        )

    else:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported providers: openai, claude_api, claude_cli, ollama"
        )


def get_provider_config(provider: str) -> Dict[str, Any]:
    """
    Get default configuration for a specific provider.

    Args:
        provider: Provider name

    Returns:
        Dictionary with default configuration

    Examples:
        >>> config = get_provider_config("openai")
        >>> print(config["default_model"])
        gpt-4o-mini
    """
    configs = {
        "openai": {
            "default_model": "gpt-4o-mini",
            "default_temperature": 0.1,
            "supports_vision": True,
            "supports_tools": True,
        },
        "claude_api": {
            "default_model": "claude-3-5-sonnet-20241022",
            "default_temperature": 0.1,
            "supports_vision": True,
            "supports_tools": True,
        },
        "claude_cli": {
            "default_model": "claude-3-5-sonnet-20241022",
            "default_temperature": 0.1,
            "supports_vision": True,
            "supports_tools": False,  # CLI wrapper doesn't support tools directly
        },
        "ollama": {
            "default_model": "gemma3:12b",
            "default_temperature": 0.1,
            "supports_vision": False,  # Vision support depends on model (e.g., llava)
            "supports_tools": True,  # Tool support depends on model
        },
    }

    provider = provider.lower()
    if provider in configs:
        return configs[provider]
    else:
        raise ValueError(f"Unknown provider: {provider}")


# Convenience functions for specific providers
def create_openai_model(model: str = "gpt-4o-mini", temperature: float = 0.1, **kwargs):
    """Create an OpenAI chat model."""
    return get_chat_model("openai", model=model, temperature=temperature, **kwargs)


def create_claude_api_model(model: str = "claude-3-5-sonnet-20241022", temperature: float = 0.1, **kwargs):
    """Create a Claude API chat model."""
    return get_chat_model("claude_api", model=model, temperature=temperature, **kwargs)


def create_claude_cli_model(temperature: float = 0.1, timeout: float = 60.0, **kwargs):
    """Create a Claude CLI chat model."""
    return get_chat_model("claude_cli", temperature=temperature, timeout=timeout, **kwargs)


def create_ollama_model(model: str = "gemma3:12b", temperature: float = 0.1, base_url: str = "http://localhost:11434", **kwargs):
    """Create an Ollama chat model."""
    return get_chat_model("ollama", model=model, temperature=temperature, ollama_base_url=base_url, **kwargs)


if __name__ == "__main__":
    # Example usage
    print("LLM Factory Examples\n")

    # Test OpenAI
    print("1. Creating OpenAI model...")
    try:
        llm_openai = get_chat_model("openai")
        print(f"   Created: {llm_openai._llm_type}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test Claude API
    print("\n2. Creating Claude API model...")
    try:
        llm_claude_api = get_chat_model("claude_api")
        print(f"   Created: {llm_claude_api._llm_type}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test Claude CLI
    print("\n3. Creating Claude CLI model...")
    try:
        llm_claude_cli = get_chat_model("claude_cli")
        print(f"   Created: {llm_claude_cli._llm_type}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test Ollama
    print("\n4. Creating Ollama model...")
    try:
        llm_ollama = get_chat_model("ollama")
        print(f"   Created: {llm_ollama._llm_type}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test provider config
    print("\n5. Getting provider configs...")
    for provider in ["openai", "claude_api", "claude_cli", "ollama"]:
        config = get_provider_config(provider)
        print(f"   {provider}: {config['default_model']}")

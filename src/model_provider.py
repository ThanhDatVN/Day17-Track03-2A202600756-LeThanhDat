from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class ProviderConfig:
    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


_ALIASES: dict[str, str] = {
    "anthorpic": "anthropic",
    "openai-compatible": "custom",
    "google": "gemini",
}


def normalize_provider(value: str) -> str:
    v = value.strip().lower()
    return _ALIASES.get(v, v)


def build_chat_model(config: ProviderConfig):
    provider = normalize_provider(config.provider)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key or os.environ.get("OPENAI_API_KEY"),
        )

    if provider == "custom":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key or os.environ.get("CUSTOM_API_KEY", "none"),
            base_url=config.base_url or os.environ.get("CUSTOM_BASE_URL"),
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            google_api_key=config.api_key or os.environ.get("GEMINI_API_KEY"),
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key or os.environ.get("ANTHROPIC_API_KEY"),
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=config.model_name,
            temperature=config.temperature,
            base_url=config.base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key or os.environ.get("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )

    raise ValueError(f"Unsupported provider: {config.provider!r}")

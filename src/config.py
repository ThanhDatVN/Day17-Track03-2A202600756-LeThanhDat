from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    data_dir = root / "data"
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "profiles").mkdir(parents=True, exist_ok=True)

    provider = normalize_provider(os.environ.get("LLM_PROVIDER", "openai"))
    model_name = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    temperature = float(os.environ.get("LLM_TEMPERATURE", "0.3"))

    api_key: str | None = None
    base_url: str | None = None

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
    elif provider == "custom":
        api_key = os.environ.get("CUSTOM_API_KEY")
        base_url = os.environ.get("CUSTOM_BASE_URL")
    elif provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
    elif provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    elif provider == "ollama":
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    elif provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")

    model_cfg = ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )

    judge_provider = normalize_provider(os.environ.get("JUDGE_LLM_PROVIDER", provider))
    judge_model_name = os.environ.get("JUDGE_LLM_MODEL", model_name)
    judge_cfg = ProviderConfig(
        provider=judge_provider,
        model_name=judge_model_name,
        temperature=0.0,
        api_key=api_key,
        base_url=base_url,
    )

    compact_threshold = int(os.environ.get("COMPACT_THRESHOLD_TOKENS", "800"))
    compact_keep = int(os.environ.get("COMPACT_KEEP_MESSAGES", "4"))

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold,
        compact_keep_messages=compact_keep,
        model=model_cfg,
        judge_model=judge_cfg,
    )

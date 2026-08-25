"""Provider selection. ``LLM_PROVIDER`` picks the backing model.

``anthropic``
    Claude through Apple's Floodgate gateway (or the public API in
    ``ANTHROPIC_AUTH_MODE=default``).

``vllm``
    A local model behind vLLM's OpenAI-compatible server.

The provider is cached per process and rebuilt when the setting changes, so a test
or a config reload can switch backends without a restart.
"""

from __future__ import annotations

import threading

from ..config import settings
from .base import (
    AgentResult,
    CitedResult,
    Emit,
    LLMProvider,
    ModelRefused,
    ProviderError,
    no_emit,
)

__all__ = [
    "AgentResult",
    "CitedResult",
    "Emit",
    "LLMProvider",
    "ModelRefused",
    "ProviderError",
    "no_emit",
    "provider",
    "reset",
]

_lock = threading.Lock()
_provider: LLMProvider | None = None


def provider() -> LLMProvider:
    """The configured provider, constructed once per process."""
    global _provider

    name = settings().llm_provider
    with _lock:
        if _provider is None or _provider.name != name:
            if name == "anthropic":
                from .anthropic_provider import AnthropicProvider

                _provider = AnthropicProvider()
            elif name == "vllm":
                from .vllm_provider import VLLMProvider

                _provider = VLLMProvider()
            else:  # pragma: no cover - config validates this
                raise ProviderError(f"Unknown LLM_PROVIDER {name!r}")
        return _provider


def reset() -> None:
    """Drop the cached provider, along with any client or credential it holds."""
    global _provider

    with _lock:
        current, _provider = _provider, None
    if current is not None:
        current.reset()

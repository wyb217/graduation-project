"""OpenAI-compatible client wrappers for Point 1 baselines."""

from __future__ import annotations

from typing import Protocol

import httpx
from openai import OpenAI

from point1.baselines.config import ProviderConfig


class VisionLanguageClient(Protocol):
    """Protocol used by the baseline runner to support real and fake clients."""

    def complete(
        self,
        *,
        messages: list[dict[str, object]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Return a raw text completion for one multimodal request."""


class OpenAICompatibleVisionClient:
    """Small wrapper around the OpenAI Python SDK for compatible providers."""

    def __init__(self, provider: ProviderConfig) -> None:
        self._client = OpenAI(
            api_key=provider.api_key,
            base_url=provider.base_url,
            http_client=httpx.Client(trust_env=False),
        )

    def complete(
        self,
        *,
        messages: list[dict[str, object]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Submit one chat-completions request and return the raw text content."""
        completion = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = completion.choices[0].message.content
        if not content:
            raise ValueError("Provider returned empty content.")
        if isinstance(content, str):
            return content
        return "".join(part.text for part in content if getattr(part, "type", None) == "text")

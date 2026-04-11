"""Provider configuration for OpenAI-compatible Point 1 baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from common.io.json_io import read_json

DEFAULT_PROVIDER_CONFIG_PATH = Path("configs/system/providers.local.json")


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    """Configuration for one OpenAI-compatible provider."""

    name: str
    base_url: str
    api_key: str
    model: str


@dataclass(frozen=True, slots=True)
class ProviderCatalog:
    """A collection of named provider configurations."""

    default_provider: str
    providers: dict[str, ProviderConfig]

    def get_provider(self, provider_name: str | None = None) -> ProviderConfig:
        """Return the requested provider or the configured default."""
        resolved_name = self.default_provider if provider_name is None else provider_name
        try:
            return self.providers[resolved_name]
        except KeyError as exc:
            raise KeyError(f"Unknown provider: {resolved_name}") from exc


def load_provider_catalog(path: Path = DEFAULT_PROVIDER_CONFIG_PATH) -> ProviderCatalog:
    """Load provider settings from a local JSON file."""
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("Provider config must be a JSON object.")

    default_provider = str(payload["default_provider"])
    raw_providers = payload.get("providers")
    if not isinstance(raw_providers, dict):
        raise ValueError("Provider config must contain a 'providers' object.")

    providers: dict[str, ProviderConfig] = {}
    for name, raw_provider in raw_providers.items():
        if not isinstance(raw_provider, dict):
            raise ValueError(f"Provider '{name}' must be a JSON object.")
        providers[str(name)] = ProviderConfig(
            name=str(name),
            base_url=str(raw_provider["base_url"]),
            api_key=str(raw_provider["api_key"]),
            model=str(raw_provider["model"]),
        )

    return ProviderCatalog(default_provider=default_provider, providers=providers)

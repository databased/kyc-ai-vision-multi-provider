"""Provider configuration manager — YAML-driven with env var resolution.

Loads provider definitions from ``providers.yaml`` and resolves API keys
from environment variables.  Uses the Singleton pattern so configuration
is loaded once and shared across the application.

Author: Greg Hamer (https://github.com/databased)
License: MIT
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from dotenv import find_dotenv, load_dotenv

DEFAULT_CONFIG_FILE = "providers.yaml"
DEFAULT_PROVIDER_ENV_VAR = "KYC_PROVIDER"


class ProviderType(Enum):
    """Supported LLM provider identifiers."""

    PARASAIL = "parasail"
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    FIREWORKS = "fireworks"
    TOGETHER = "together"
    GOOGLE = "google"


@dataclass
class ProviderFeatures:
    """Provider capability flags."""

    vision: bool = False
    streaming: bool = False
    function_calling: bool = False
    auto_fallback: bool = False


@dataclass
class ProviderPricing:
    """Cost per million tokens."""

    input_tokens_per_million: float = 0.0
    output_tokens_per_million: float = 0.0

    @property
    def is_free(self) -> bool:
        return (
            self.input_tokens_per_million == 0.0
            and self.output_tokens_per_million == 0.0
        )


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider."""

    provider_id: str
    name: str
    base_url: str
    api_key: str | None = None
    default_model: str = ""
    supported_models: list[str] = field(default_factory=list)
    features: ProviderFeatures = field(default_factory=ProviderFeatures)
    pricing: ProviderPricing = field(default_factory=ProviderPricing)
    additional_headers: dict[str, str] = field(default_factory=dict)
    notes: str = ""
    api_compatible: bool = True
    native_sdk: str | None = None  # e.g. "google-genai" for native client

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.base_url)

    @property
    def supports_vision(self) -> bool:
        return self.features.vision

    @property
    def uses_native_sdk(self) -> bool:
        return self.native_sdk is not None

    def validate(self) -> tuple[bool, str | None]:
        """Validate required fields.

        Returns:
            Tuple of (is_valid, error_message).
        """
        if not self.api_key:
            return False, f"API key not configured for {self.name}"
        if not self.base_url and not self.uses_native_sdk:
            return False, f"Base URL not configured for {self.name}"
        if not self.default_model:
            return False, f"No default model specified for {self.name}"
        return True, None


class ProviderConfigManager:
    """Loads and manages provider configurations from YAML.

    Singleton — call ``reset()`` between tests.
    """

    _instance: "ProviderConfigManager | None" = None

    def __new__(cls, config_file: Path | None = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_file: Path | None = None):
        if self._initialized:
            return

        self._load_environment()
        self.config_file_path = self._resolve_config_file(config_file)
        self.providers: dict[str, ProviderConfig] = {}
        self.default_provider: str = ""
        self._raw_config: dict[str, Any] = {}
        self._load_configurations()
        self._initialized = True

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        cls._instance = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_environment() -> None:
        env_file = find_dotenv()
        if env_file:
            load_dotenv(env_file)

    def _resolve_config_file(self, config_file: Path | None) -> Path:
        if config_file:
            if not config_file.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_file}")
            return config_file

        for candidate in [
            Path.cwd() / DEFAULT_CONFIG_FILE,
            Path(__file__).parent.parent / DEFAULT_CONFIG_FILE,
        ]:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"'{DEFAULT_CONFIG_FILE}' not found in cwd or project root"
        )

    def _load_configurations(self) -> None:
        try:
            with open(self.config_file_path, encoding="utf-8") as fh:
                self._raw_config = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise ValueError(f"Error parsing YAML: {exc}") from exc

        self.default_provider = self._raw_config.get("default_provider", "openrouter")
        for pid, pdata in self._raw_config.get("providers", {}).items():
            self.providers[pid] = self._parse_provider(pid, pdata)

    def _parse_provider(self, provider_id: str, data: dict[str, Any]) -> ProviderConfig:
        api_key_env = data.get("api_key_env")
        api_key = os.getenv(api_key_env) if api_key_env else None

        features_data = data.get("features", {})
        features = ProviderFeatures(
            vision=features_data.get("vision", False),
            streaming=features_data.get("streaming", False),
            function_calling=features_data.get("function_calling", False),
            auto_fallback=features_data.get("auto_fallback", False),
        )

        pricing_data = data.get("pricing", {})
        pricing = ProviderPricing(
            input_tokens_per_million=_parse_numeric(
                pricing_data.get("input_tokens_per_million", 0.0)
            ),
            output_tokens_per_million=_parse_numeric(
                pricing_data.get("output_tokens_per_million", 0.0)
            ),
        )

        additional_headers = _resolve_headers(data.get("additional_headers", {}))

        return ProviderConfig(
            provider_id=provider_id,
            name=data.get("name", provider_id),
            base_url=data.get("base_url", ""),
            api_key=api_key,
            default_model=data.get("default_model", ""),
            supported_models=data.get("supported_models", []),
            features=features,
            pricing=pricing,
            additional_headers=additional_headers,
            notes=data.get("notes", ""),
            api_compatible=data.get("api_compatible", True),
            native_sdk=data.get("native_sdk"),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_provider_config(self, provider_id: str | None = None) -> ProviderConfig:
        """Return configuration for a provider.

        Args:
            provider_id: Provider key, or None for the configured default.

        Raises:
            ValueError: If provider is unknown or misconfigured.
        """
        if provider_id is None:
            provider_id = os.getenv(DEFAULT_PROVIDER_ENV_VAR, self.default_provider)

        if provider_id not in self.providers:
            available = ", ".join(self.providers.keys())
            raise ValueError(
                f"Provider '{provider_id}' not found. Available: {available}"
            )

        config = self.providers[provider_id]
        is_valid, error_msg = config.validate()
        if not is_valid:
            raise ValueError(f"Provider '{provider_id}' invalid: {error_msg}")
        return config

    def list_providers(self) -> list[str]:
        return list(self.providers.keys())

    def list_configured_providers(self) -> list[str]:
        return [pid for pid, cfg in self.providers.items() if cfg.is_configured]

    def get_processing_config(self) -> dict[str, Any]:
        return self._raw_config.get("processing", {})

    def get_documents_config(self) -> dict[str, Any]:
        return self._raw_config.get("documents", {})

    def get_logging_config(self) -> dict[str, Any]:
        return self._raw_config.get("logging", {})


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _parse_numeric(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _resolve_headers(headers_config: dict[str, str]) -> dict[str, str]:
    """Resolve header values from environment variables.

    Keys ending in ``_env`` are treated as env-var references.
    """
    resolved: dict[str, str] = {}
    for key, env_ref in headers_config.items():
        if not isinstance(env_ref, str) or not env_ref.endswith("_env"):
            continue
        env_var = env_ref[:-4]
        value = os.getenv(env_var, "")
        if value:
            clean_key = key[:-4] if key.endswith("_env") else key
            header_name = "-".join(part.capitalize() for part in clean_key.split("_"))
            resolved[header_name] = value
    return resolved

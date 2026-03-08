"""Tests for src.config module.

Author: Greg Hamer (https://github.com/databased)
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import (
    ProviderConfig,
    ProviderConfigManager,
    ProviderFeatures,
    ProviderPricing,
    ProviderType,
    _parse_numeric,
    _resolve_headers,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset ProviderConfigManager singleton between tests."""
    ProviderConfigManager.reset()
    yield
    ProviderConfigManager.reset()


# ------------------------------------------------------------------
# ProviderType
# ------------------------------------------------------------------


class TestProviderType:
    def test_all_providers_defined(self):
        expected = {
            "parasail",
            "openrouter",
            "openai",
            "anthropic",
            "fireworks",
            "together",
            "google",
        }
        actual = {pt.value for pt in ProviderType}
        assert actual == expected


# ------------------------------------------------------------------
# ProviderFeatures
# ------------------------------------------------------------------


class TestProviderFeatures:
    def test_defaults_are_false(self):
        f = ProviderFeatures()
        assert not f.vision
        assert not f.streaming
        assert not f.function_calling
        assert not f.auto_fallback

    def test_custom_values(self):
        f = ProviderFeatures(vision=True, streaming=True)
        assert f.vision
        assert f.streaming


# ------------------------------------------------------------------
# ProviderPricing
# ------------------------------------------------------------------


class TestProviderPricing:
    def test_free_tier(self):
        assert ProviderPricing().is_free

    def test_paid_tier(self):
        p = ProviderPricing(
            input_tokens_per_million=2.50,
            output_tokens_per_million=10.00,
        )
        assert not p.is_free


# ------------------------------------------------------------------
# ProviderConfig
# ------------------------------------------------------------------


class TestProviderConfig:
    def test_is_configured(self):
        cfg = ProviderConfig(
            provider_id="t", name="T", base_url="https://x", api_key="sk"
        )
        assert cfg.is_configured

    def test_not_configured_without_key(self):
        cfg = ProviderConfig(provider_id="t", name="T", base_url="https://x")
        assert not cfg.is_configured

    def test_validate_missing_key(self):
        cfg = ProviderConfig(
            provider_id="t",
            name="T",
            base_url="https://x",
            default_model="m",
        )
        ok, msg = cfg.validate()
        assert not ok
        assert "API key" in msg

    def test_validate_missing_model(self):
        cfg = ProviderConfig(
            provider_id="t", name="T", base_url="https://x", api_key="sk"
        )
        ok, msg = cfg.validate()
        assert not ok
        assert "model" in msg

    def test_validate_success(self):
        cfg = ProviderConfig(
            provider_id="t",
            name="T",
            base_url="https://x",
            api_key="sk",
            default_model="m",
        )
        ok, msg = cfg.validate()
        assert ok
        assert msg is None

    def test_supports_vision(self):
        cfg = ProviderConfig(
            provider_id="t",
            name="T",
            base_url="",
            features=ProviderFeatures(vision=True),
        )
        assert cfg.supports_vision

    def test_uses_native_sdk(self):
        cfg = ProviderConfig(
            provider_id="t",
            name="T",
            base_url="",
            native_sdk="google-genai",
        )
        assert cfg.uses_native_sdk

    def test_no_native_sdk(self):
        cfg = ProviderConfig(provider_id="t", name="T", base_url="")
        assert not cfg.uses_native_sdk

    def test_validate_native_sdk_no_base_url(self):
        """Native SDK providers don't require base_url."""
        cfg = ProviderConfig(
            provider_id="t",
            name="T",
            base_url="",
            api_key="sk",
            default_model="m",
            native_sdk="google-genai",
        )
        ok, _ = cfg.validate()
        assert ok


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


class TestParseNumeric:
    def test_int(self):
        assert _parse_numeric(5) == 5.0

    def test_float(self):
        assert _parse_numeric(0.42) == 0.42

    def test_string_returns_zero(self):
        assert _parse_numeric("free") == 0.0

    def test_none_returns_zero(self):
        assert _parse_numeric(None) == 0.0


class TestResolveHeaders:
    @patch.dict(os.environ, {"OPENROUTER_SITE_URL": "https://example.com"})
    def test_resolves_env_suffix(self):
        headers = _resolve_headers({"http_referer_env": "OPENROUTER_SITE_URL_env"})
        assert headers["Http-Referer"] == "https://example.com"

    def test_skips_missing_env(self):
        headers = _resolve_headers({"http_referer_env": "NONEXISTENT_VAR_env"})
        assert headers == {}

    def test_skips_non_env_keys(self):
        headers = _resolve_headers({"plain_key": "plain_value"})
        assert headers == {}


# ------------------------------------------------------------------
# ProviderConfigManager
# ------------------------------------------------------------------


class TestProviderConfigManager:
    @pytest.fixture(autouse=True)
    def _yaml(self, tmp_path):
        content = """\
default_provider: "test_provider"

providers:
  test_provider:
    name: "Test Provider"
    base_url: "https://api.test.com/v1"
    api_key_env: "TEST_API_KEY"
    default_model: "test-model-v1"
    supported_models:
      - "test-model-v1"
    features:
      vision: true
      streaming: true
    pricing:
      input_tokens_per_million: 1.00
      output_tokens_per_million: 2.00

  native_provider:
    name: "Native Test"
    base_url: ""
    api_key_env: "TEST_API_KEY"
    default_model: "native-model"
    native_sdk: "test-sdk"
    features:
      vision: true

processing:
  max_retries: 2

documents:
  input_directory: "docs"

logging:
  level: "DEBUG"
"""
        self.yaml_path = tmp_path / "providers.yaml"
        self.yaml_path.write_text(content)

    @patch.dict(os.environ, {"TEST_API_KEY": "sk-test-123"})
    def test_loads_provider(self):
        mgr = ProviderConfigManager(config_file=self.yaml_path)
        cfg = mgr.get_provider_config("test_provider")
        assert cfg.name == "Test Provider"
        assert cfg.api_key == "sk-test-123"
        assert cfg.features.vision

    @patch.dict(os.environ, {"TEST_API_KEY": "sk-test-123"})
    def test_native_sdk_field(self):
        mgr = ProviderConfigManager(config_file=self.yaml_path)
        cfg = mgr.providers["native_provider"]
        assert cfg.uses_native_sdk
        assert cfg.native_sdk == "test-sdk"

    @patch.dict(os.environ, {"TEST_API_KEY": "sk-test-123"})
    def test_default_provider(self):
        mgr = ProviderConfigManager(config_file=self.yaml_path)
        assert mgr.default_provider == "test_provider"

    @patch.dict(os.environ, {"TEST_API_KEY": "sk-test-123"})
    def test_list_providers(self):
        mgr = ProviderConfigManager(config_file=self.yaml_path)
        assert "test_provider" in mgr.list_providers()
        assert "native_provider" in mgr.list_providers()

    @patch.dict(os.environ, {"TEST_API_KEY": "sk-test-123"})
    def test_list_configured_providers(self):
        mgr = ProviderConfigManager(config_file=self.yaml_path)
        # test_provider has base_url + key → configured
        assert "test_provider" in mgr.list_configured_providers()

    def test_unknown_provider_raises(self):
        mgr = ProviderConfigManager(config_file=self.yaml_path)
        with pytest.raises(ValueError, match="not found"):
            mgr.get_provider_config("nonexistent")

    @patch.dict(os.environ, {"TEST_API_KEY": "sk-test-123"})
    def test_processing_config(self):
        mgr = ProviderConfigManager(config_file=self.yaml_path)
        assert mgr.get_processing_config()["max_retries"] == 2

    @patch.dict(os.environ, {"TEST_API_KEY": "sk-test-123"})
    def test_documents_config(self):
        mgr = ProviderConfigManager(config_file=self.yaml_path)
        assert mgr.get_documents_config()["input_directory"] == "docs"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            ProviderConfigManager(config_file=Path("/nonexistent/providers.yaml"))

    @patch.dict(os.environ, {"TEST_API_KEY": "sk-test-123"})
    def test_singleton_reset(self):
        m1 = ProviderConfigManager(config_file=self.yaml_path)
        ProviderConfigManager.reset()
        m2 = ProviderConfigManager(config_file=self.yaml_path)
        assert m1 is not m2

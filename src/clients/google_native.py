"""Native Google Gemini vision client for KYC processing.

Uses the ``google-genai`` SDK directly, enabling Gemini-specific
features such as ``response_mime_type="application/json"`` for
structured output enforcement.

This client is **optional** — install with:
    pip install "google-genai>=1.0"

If the SDK is not installed, the factory will fall back to the
OpenAI-compatible endpoint for Google.

Author: Greg Hamer (https://github.com/databased)
License: MIT
"""

import json
import logging
from typing import Any

from src.clients.base import BaseProviderClient
from src.config import ProviderConfig

logger = logging.getLogger(__name__)

try:
    from google import genai
    from google.genai import types

    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


def is_available() -> bool:
    """Check whether the google-genai SDK is installed."""
    return GENAI_AVAILABLE


class GoogleNativeClient(BaseProviderClient):
    """Client using the native Google GenAI SDK.

    Raises:
        ImportError: If ``google-genai`` is not installed.
    """

    def __init__(self, config: ProviderConfig) -> None:
        if not GENAI_AVAILABLE:
            raise ImportError(
                "google-genai package not installed. "
                'Install with: pip install "google-genai>=1.0"'
            )
        self._config = config
        self._client = genai.Client(api_key=config.api_key)
        self._model = config.default_model

    def test_connection(self) -> tuple[bool, str]:
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents="Test connection. Reply with exactly one word: 'OK'",
                config=types.GenerateContentConfig(
                    temperature=0.0, max_output_tokens=10
                ),
            )
            if "OK" in (response.text or "").upper():
                return True, "Connection successful"
            return False, f"Unexpected response: {response.text}"
        except Exception as exc:
            logger.error("Google API connection test failed: %s", exc)
            return False, f"API Error: {exc}"

    def extract_identity_data(
        self,
        image_base64: str,
        mime_type: str,
        system_prompt: str,
    ) -> dict[str, Any]:
        try:
            import base64

            image_bytes = base64.b64decode(image_base64)

            image_part = types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type,
            )

            config = types.GenerateContentConfig(
                temperature=0.0,
                system_instruction=system_prompt,
                response_mime_type="application/json",
            )

            response = self._client.models.generate_content(
                model=self._model,
                contents=[
                    "Analyze this identity document and extract "
                    "the information as a flat JSON object.",
                    image_part,
                ],
                config=config,
            )

            return _parse_json(response.text or "")

        except Exception as exc:
            logger.error("Google API extraction failed: %s", exc)
            raise

    def provider_name(self) -> str:
        return f"{self._config.name} (native SDK)"

    def model_name(self) -> str:
        return self._model


def _parse_json(text: str) -> dict[str, Any]:
    """Extract JSON from a response that may be wrapped in markdown."""
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return json.loads(cleaned.strip())

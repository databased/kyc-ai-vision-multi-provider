"""OpenAI-compatible vision client for KYC processing.

Works with any provider that exposes an OpenAI-compatible
chat completions endpoint: OpenRouter, Parasail, OpenAI,
Google (via compatibility layer), Fireworks, Together AI.

Author: Greg Hamer (https://github.com/databased)
License: MIT
"""

import json
import logging
from typing import Any

from openai import OpenAI

from src.clients.base import BaseProviderClient
from src.config import ProviderConfig

logger = logging.getLogger(__name__)


class OpenAICompatClient(BaseProviderClient):
    """Generic client for OpenAI-compatible vision APIs."""

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        self._model = config.default_model

    def test_connection(self) -> tuple[bool, str]:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Test connection. Reply with exactly one word: 'OK'"
                        ),
                    }
                ],
                max_tokens=10,
                temperature=0.0,
            )
            reply = response.choices[0].message.content or ""
            if "OK" in reply.upper():
                return True, "Connection successful"
            return False, f"Unexpected response: {reply}"
        except Exception as exc:
            logger.error(
                "%s connection test failed: %s",
                self._config.name,
                exc,
            )
            return False, f"API Error: {exc}"

    def extract_identity_data(
        self,
        image_base64: str,
        mime_type: str,
        system_prompt: str,
    ) -> dict[str, Any]:
        image_url = f"data:{mime_type};base64,{image_base64}"

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Analyze this identity document and "
                                    "return a flat JSON object per the "
                                    "system instructions."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            },
                        ],
                    },
                ],
                temperature=0.0,
                max_tokens=2000,
            )
            reply = response.choices[0].message.content or ""
            return _parse_json(reply)

        except Exception as exc:
            logger.error("%s extraction failed: %s", self._config.name, exc)
            raise

    def provider_name(self) -> str:
        return self._config.name

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

"""Factory for instantiating the correct provider client.

Reads provider configuration from YAML and returns the appropriate
``BaseProviderClient`` implementation.  For Google, it will prefer the
native SDK if installed, otherwise fall back to the OpenAI-compatible
endpoint.

Author: Greg Hamer (https://github.com/databased)
License: MIT
"""

import logging

from src.clients.base import BaseProviderClient
from src.config import ProviderConfig

logger = logging.getLogger(__name__)


def get_client(config: ProviderConfig) -> BaseProviderClient:
    """Create a provider client from a ProviderConfig.

    Args:
        config: Validated provider configuration.

    Returns:
        An instantiated client implementing BaseProviderClient.
    """
    # Google with native SDK preference
    if config.native_sdk == "google-genai":
        from src.clients import google_native

        if google_native.is_available():
            logger.info("Using native Google GenAI SDK for %s", config.name)
            return google_native.GoogleNativeClient(config)
        logger.info(
            "google-genai not installed; falling back to "
            "OpenAI-compatible endpoint for %s",
            config.name,
        )

    # Default: OpenAI-compatible client
    from src.clients.openai_compat import OpenAICompatClient

    return OpenAICompatClient(config)

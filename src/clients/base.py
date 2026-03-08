"""Abstract base class for KYC vision provider clients.

Author: Greg Hamer (https://github.com/databased)
License: MIT
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseProviderClient(ABC):
    """Contract that every provider client must implement.

    Ensures a uniform interface for connectivity testing and
    identity-data extraction across all LLM vision providers.
    """

    @abstractmethod
    def test_connection(self) -> tuple[bool, str]:
        """Test API connectivity with a lightweight request.

        Returns:
            Tuple of (success, message).
        """

    @abstractmethod
    def extract_identity_data(
        self,
        image_base64: str,
        mime_type: str,
        system_prompt: str,
    ) -> dict[str, Any]:
        """Send an image to the vision model and return parsed JSON.

        Args:
            image_base64: Base64-encoded image bytes.
            mime_type: MIME type (e.g. ``image/jpeg``).
            system_prompt: Extraction instructions for the model.

        Returns:
            Dictionary of extracted identity fields.

        Raises:
            Exception: On API or JSON-parsing failure.
        """

    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name."""

    @abstractmethod
    def model_name(self) -> str:
        """Model identifier string currently in use."""

"""Document discovery, validation, and image encoding.

Author: Greg Hamer (https://github.com/databased)
License: MIT
"""

import base64
import logging
from pathlib import Path

from PIL import Image

from src.models import DocumentInfo

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
MAX_FILE_SIZE_MB = 10.0


class DocumentLoader:
    """Discovers and validates identity document images."""

    def discover(self, directory: Path) -> list[DocumentInfo]:
        """Find all image files in *directory* and validate each one.

        Args:
            directory: Path to scan for document images.

        Returns:
            List of DocumentInfo objects (valid and invalid).
        """
        if not directory.is_dir():
            logger.error("Documents directory not found: %s", directory)
            return []

        documents: list[DocumentInfo] = []
        for file_path in sorted(directory.iterdir()):
            if file_path.is_file() and not file_path.name.startswith("."):
                documents.append(self._validate(file_path))
        return documents

    def _validate(self, file_path: Path) -> DocumentInfo:
        size_mb = file_path.stat().st_size / (1024 * 1024)
        ext = file_path.suffix.lower()

        doc = DocumentInfo(
            path=file_path,
            filename=file_path.name,
            size_mb=round(size_mb, 2),
            format=ext,
            is_valid=False,
        )

        if ext not in SUPPORTED_FORMATS:
            doc.error_message = (
                f"Unsupported format. Expected: {', '.join(sorted(SUPPORTED_FORMATS))}"
            )
            return doc

        if size_mb > MAX_FILE_SIZE_MB:
            doc.error_message = (
                f"File too large ({size_mb:.1f} MB). Max: {MAX_FILE_SIZE_MB} MB"
            )
            return doc

        try:
            with Image.open(file_path) as img:
                img.verify()
            doc.is_valid = True
        except Exception as exc:
            doc.error_message = f"Corrupt or invalid image: {exc}"

        return doc

    @staticmethod
    def encode_base64(file_path: Path) -> str:
        """Read an image file and return its base64-encoded content."""
        with open(file_path, "rb") as fh:
            return base64.b64encode(fh.read()).decode("utf-8")

"""Vision processing and batch orchestration.

Author: Greg Hamer (https://github.com/databased)
License: MIT
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.clients.base import BaseProviderClient
from src.loader import DocumentLoader
from src.models import (
    BatchSummary,
    DocumentInfo,
    ExtractedData,
    ProcessingResult,
)

logger = logging.getLogger(__name__)

# Standardised extraction prompt shared across all providers.
KYC_SYSTEM_PROMPT = (
    "You are a precise identity-document data-extraction system used "
    "for KYC/AML compliance worldwide.\n\n"
    "Analyze the provided image (passport, national ID, or driver's "
    "license) and extract ALL visible text fields.\n\n"
    "Return ONLY a flat JSON object.  Use null for fields that are not "
    "visible.  Keys:\n"
    "  full_name, first_name, last_name, middle_name,\n"
    "  date_of_birth, gender, nationality,\n"
    "  document_type (PASSPORT | DRIVERS_LICENSE | ID_CARD),\n"
    "  document_number, document_class,\n"
    "  issuing_country, issuing_authority,\n"
    "  date_of_issue, date_of_expiry,\n"
    "  address, city, state_province, postal_code,\n"
    "  mrz_line_1, mrz_line_2, mrz_line_3,\n"
    "  overall_confidence (float 0.0-1.0),\n"
    "  is_expired (boolean).\n\n"
    "Use YYYY-MM-DD for all date fields.  Do NOT include any text "
    "outside the JSON object."
)


class VisionProcessor:
    """Sends document images to a vision model and parses results."""

    def __init__(
        self,
        client: BaseProviderClient,
        processing_config: dict | None = None,
    ) -> None:
        self._client = client
        cfg = processing_config or {}
        self._max_retries: int = cfg.get("max_retries", 3)
        self._retry_delay: float = cfg.get("retry_delay_seconds", 2)

    def process(self, doc: DocumentInfo) -> ProcessingResult:
        """Process a single identity document image.

        Returns a ProcessingResult regardless of success or failure.
        """
        start = time.time()
        logger.info("Processing %s", doc.filename)

        if not doc.is_valid:
            logger.warning(
                "Skipping invalid document %s: %s",
                doc.filename,
                doc.error_message,
            )
            return ProcessingResult(
                success=False,
                document_info=doc,
                error_message=doc.error_message,
            )

        last_error: str | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                image_b64 = DocumentLoader.encode_base64(doc.path)
                mime_type = _mime_type(doc.format)

                raw = self._client.extract_identity_data(
                    image_base64=image_b64,
                    mime_type=mime_type,
                    system_prompt=KYC_SYSTEM_PROMPT,
                )

                elapsed = round(time.time() - start, 2)
                extracted = ExtractedData(
                    **{k: v for k, v in raw.items() if k in ExtractedData.model_fields},
                    processing_time_seconds=elapsed,
                    provider_used=self._client.provider_name(),
                    model_used=self._client.model_name(),
                    raw_extraction=raw,
                )

                logger.info(
                    "Extracted %s in %.2fs (attempt %d)",
                    doc.filename,
                    elapsed,
                    attempt,
                )
                return ProcessingResult(
                    success=True,
                    document_info=doc,
                    extracted_data=extracted,
                    processing_time=elapsed,
                )

            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "Attempt %d/%d failed for %s: %s",
                    attempt,
                    self._max_retries,
                    doc.filename,
                    exc,
                )
                if attempt < self._max_retries:
                    time.sleep(self._retry_delay)

        elapsed = round(time.time() - start, 2)
        return ProcessingResult(
            success=False,
            document_info=doc,
            error_message=f"All {self._max_retries} attempts failed: {last_error}",
            processing_time=elapsed,
        )


class BatchProcessor:
    """Orchestrates parallel processing of multiple documents."""

    def __init__(
        self,
        vision: VisionProcessor,
        documents_config: dict | None = None,
    ) -> None:
        self._vision = vision
        cfg = documents_config or {}
        self._output_dir = Path(cfg.get("output_directory", "outputs"))
        self._individual_dir = Path(
            cfg.get("individual_output_directory", "outputs/individual")
        )

    def run(
        self,
        documents: list[DocumentInfo],
        max_workers: int = 4,
    ) -> BatchSummary:
        """Process *documents* in parallel and write results."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._individual_dir.mkdir(parents=True, exist_ok=True)

        results: list[ProcessingResult] = []

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self._vision.process, doc): doc
                for doc in documents
                if doc.is_valid
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                self._save_individual(result)

        # Add skipped (invalid) documents
        processed_files = {r.document_info.filename for r in results}
        for doc in documents:
            if doc.filename not in processed_files:
                results.append(
                    ProcessingResult(
                        success=False,
                        document_info=doc,
                        error_message=doc.error_message,
                    )
                )

        summary = self._build_summary(results)
        self._save_summary(summary)
        return summary

    def _save_individual(self, result: ProcessingResult) -> None:
        if not result.success or result.extracted_data is None:
            return
        stem = result.document_info.path.stem
        out = self._individual_dir / f"{stem}_result.json"
        out.write_text(result.extracted_data.model_dump_json(indent=2))
        logger.info("Saved %s", out)

    def _save_summary(self, summary: BatchSummary) -> None:
        out = self._output_dir / "batch_summary.json"
        out.write_text(summary.model_dump_json(indent=2))
        logger.info("Batch summary saved to %s", out)

    def _build_summary(self, results: list[ProcessingResult]) -> BatchSummary:
        total = len(results)
        successes = [r for r in results if r.success]
        times = [r.processing_time for r in successes if r.processing_time]

        return BatchSummary(
            total_documents=total,
            successful_extractions=len(successes),
            failed_extractions=total - len(successes),
            success_rate=(len(successes) / total * 100) if total else 0.0,
            average_processing_time=(sum(times) / len(times) if times else 0.0),
            total_processing_time=sum(times) if times else 0.0,
            provider_used=self._vision._client.provider_name(),
            model_used=self._vision._client.model_name(),
        )


def _mime_type(ext: str) -> str:
    """Convert a file extension to a MIME type."""
    ext = ext.lstrip(".").lower()
    mapping = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "bmp": "image/bmp",
        "tiff": "image/tiff",
        "webp": "image/webp",
    }
    return mapping.get(ext, f"image/{ext}")

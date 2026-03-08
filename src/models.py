"""Pydantic data models for KYC document processing.

Author: Greg Hamer (https://github.com/databased)
License: MIT
"""

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ExtractedData(BaseModel):
    """Structured data extracted from an identity document."""

    # Personal information
    full_name: str | None = Field(None, description="Full name as printed")
    first_name: str | None = Field(None, description="Given name(s)")
    last_name: str | None = Field(None, description="Family name / surname")
    middle_name: str | None = Field(None, description="Middle name or initial")
    date_of_birth: str | None = Field(None, description="DOB (YYYY-MM-DD)")
    gender: str | None = Field(None, description="Gender / sex as printed")
    nationality: str | None = Field(None, description="Nationality or citizenship")

    # Document details
    document_type: str | None = Field(
        None, description="PASSPORT, DRIVERS_LICENSE, or ID_CARD"
    )
    document_number: str | None = Field(None, description="Primary document number")
    document_class: str | None = Field(None, description="Document class or category")
    issuing_country: str | None = Field(None, description="Country of issuance")
    issuing_authority: str | None = Field(None, description="Issuing agency or state")
    date_of_issue: str | None = Field(None, description="Issue date (YYYY-MM-DD)")
    date_of_expiry: str | None = Field(None, description="Expiry date (YYYY-MM-DD)")

    # Address (common on driver's licenses)
    address: str | None = Field(None, description="Full address as printed")
    city: str | None = Field(None, description="City or locality")
    state_province: str | None = Field(None, description="State, province, or region")
    postal_code: str | None = Field(None, description="ZIP or postal code")

    # Machine-readable zone
    mrz_line_1: str | None = Field(None, description="MRZ line 1")
    mrz_line_2: str | None = Field(None, description="MRZ line 2")
    mrz_line_3: str | None = Field(None, description="MRZ line 3")

    # Confidence and metadata
    overall_confidence: float | None = Field(
        None, description="0.0–1.0 extraction confidence"
    )
    is_expired: bool | None = Field(None, description="Whether document is expired")

    # Processing metadata (populated by the processor, not the model)
    processing_timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )
    processing_time_seconds: float | None = None
    provider_used: str | None = None
    model_used: str | None = None
    raw_extraction: dict[str, Any] = Field(default_factory=dict)


class DocumentInfo(BaseModel):
    """File metadata and validation status for a document image."""

    path: Path
    filename: str
    size_mb: float
    format: str
    is_valid: bool
    error_message: str | None = None


class ProcessingResult(BaseModel):
    """Result container for a single document processing run."""

    success: bool
    document_info: DocumentInfo
    extracted_data: ExtractedData | None = None
    error_message: str | None = None
    processing_time: float | None = None


class BatchSummary(BaseModel):
    """Aggregate statistics for a batch processing run."""

    total_documents: int
    successful_extractions: int
    failed_extractions: int
    success_rate: float
    average_processing_time: float
    total_processing_time: float
    provider_used: str
    model_used: str
    processing_timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )

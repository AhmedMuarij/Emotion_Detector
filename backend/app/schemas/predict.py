"""Pydantic response schemas for EmotionAI API endpoints.

These models define the exact JSON contract between backend and frontend.
They must stay in sync with the TypeScript types in frontend/lib/types.ts.
"""

from __future__ import annotations

from typing import Annotated, Optional

from pydantic import BaseModel, Field


# ── /health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    model_loaded: bool
    version: str = Field(..., examples=["0.1.0"])


# ── /model-info ───────────────────────────────────────────────────────────────

class ModelInputInfo(BaseModel):
    height: int
    width: int
    channels: int
    color_mode: str
    normalization: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    task: str
    num_classes: int
    supported_expressions: list[str]
    index_to_class: dict[str, str]
    input: ModelInputInfo
    disclaimer: str


# ── /predict ─────────────────────────────────────────────────────────────────

class PredictionDetail(BaseModel):
    label: Annotated[str, Field(examples=["Happy"])]
    confidence: Annotated[float, Field(ge=0.0, le=1.0, examples=[0.87])]


class PredictionResponse(BaseModel):
    """Successful prediction with face detected."""
    success: bool = True
    face_detected: bool
    prediction: Optional[PredictionDetail] = None
    probabilities: Optional[dict[str, float]] = None
    processing_time_ms: float
    message: Optional[str] = None


# ── Error ─────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None

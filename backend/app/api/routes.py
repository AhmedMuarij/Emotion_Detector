"""API route implementations for EmotionAI backend.

Routes:
    GET  /health        — service liveness + model status
    GET  /model-info    — safe model metadata for the frontend
    POST /predict       — image upload → face detection → CNN inference → JSON

Security notes:
    - File type is validated by content-type header AND by checking the image
      decode step (cv2.imdecode returns None for invalid files).
    - File size is capped at settings.max_upload_bytes before reading.
    - No stack traces or internal paths are returned to clients.
    - CORS is configured in main.py, not here.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.ml import inference, loader
from app.schemas.predict import (
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    ModelInputInfo,
    PredictionDetail,
    PredictionResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Accepted MIME types for uploads
_ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/bmp",
    "image/webp",
}


# ── GET /health ───────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    tags=["System"],
)
async def health() -> HealthResponse:
    """Return service status and model availability.

    Does not expose sensitive information.
    Safe to call publicly from monitoring systems.
    """
    meta = loader.get_metadata() if loader.is_loaded() else {}
    version = meta.get("model_version", "unknown") if meta else "unknown"
    return HealthResponse(
        status="ok",
        model_loaded=loader.is_loaded(),
        version=version,
    )


# ── GET /model-info ───────────────────────────────────────────────────────────

@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Safe model metadata",
    tags=["System"],
)
async def model_info() -> ModelInfoResponse:
    """Return safe metadata about the loaded model.

    Includes supported expression labels and input specifications.
    Does not expose file paths, credentials, or internal system information.
    """
    if not loader.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. The service may still be starting up.",
        )

    meta = loader.get_metadata()
    num_classes: int = int(meta["num_classes"])
    index_to_class: dict[str, str] = meta["index_to_class"]
    inp = meta["input"]

    supported = [index_to_class[str(i)] for i in range(num_classes)]

    return ModelInfoResponse(
        model_name=meta.get("model_name", "emotion_cnn"),
        model_version=meta.get("model_version", "0.1.0"),
        task=meta.get("task", "facial_expression_recognition"),
        num_classes=num_classes,
        supported_expressions=supported,
        index_to_class=index_to_class,
        input=ModelInputInfo(
            height=int(inp["height"]),
            width=int(inp["width"]),
            channels=int(inp["channels"]),
            color_mode=inp["color_mode"],
            normalization=inp["normalization"],
        ),
        disclaimer=meta.get(
            "disclaimer",
            "Classifies visible facial expressions. Does not determine internal emotional state.",
        ),
    )


# ── POST /predict ─────────────────────────────────────────────────────────────

@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Predict facial expression from uploaded image",
    tags=["Inference"],
)
async def predict(
    request: Request,
    file: UploadFile = File(..., description="Image file (JPEG, PNG, BMP, WebP)"),
) -> PredictionResponse | JSONResponse:
    """Accept an image upload, detect a face, and return expression probabilities.

    The image should contain exactly one face for best results.
    If multiple faces are present, the largest detected face is used.
    """
    # ── Model availability ────────────────────────────────────────────────────
    if not loader.is_loaded():
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ErrorResponse(
                error="Model not available",
                detail="The inference model is not loaded. Try again in a few seconds.",
            ).model_dump(),
        )

    # ── Content-type validation ───────────────────────────────────────────────
    content_type = (file.content_type or "").lower()
    if content_type not in _ALLOWED_CONTENT_TYPES:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                error="Unsupported file type",
                detail=f"Received content-type '{content_type}'. "
                       f"Allowed: {sorted(_ALLOWED_CONTENT_TYPES)}",
            ).model_dump(),
        )

    # ── Size limit (read with cap) ────────────────────────────────────────────
    image_bytes = await file.read(settings.max_upload_bytes + 1)
    if len(image_bytes) > settings.max_upload_bytes:
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content=ErrorResponse(
                error="File too large",
                detail=f"Maximum upload size is {settings.max_upload_bytes // 1024 // 1024} MB.",
            ).model_dump(),
        )
    if len(image_bytes) == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(error="Empty file", detail="Uploaded file contains no data.").model_dump(),
        )

    # ── Inference ─────────────────────────────────────────────────────────────
    try:
        result = inference.predict(image_bytes)
    except ValueError as exc:
        # Image decode failure — bad file content despite valid content-type
        logger.warning("Image decode error: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                error="Invalid image",
                detail="Could not decode the uploaded file as an image.",
            ).model_dump(),
        )
    except Exception as exc:  # noqa: BLE001
        # Unexpected inference error — log details server-side, return generic message
        logger.exception("Unexpected inference error: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="Inference failed",
                detail="An unexpected error occurred during inference. Please try again.",
            ).model_dump(),
        )

    # ── No face detected ──────────────────────────────────────────────────────
    if not result.face_detected:
        return PredictionResponse(
            face_detected=False,
            processing_time_ms=result.processing_time_ms,
            message="No face detected in the uploaded image.",
        )

    # ── Success ───────────────────────────────────────────────────────────────
    return PredictionResponse(
        face_detected=True,
        prediction=PredictionDetail(
            label=result.label,      # type: ignore[arg-type]
            confidence=result.confidence,  # type: ignore[arg-type]
        ),
        probabilities=result.probabilities,
        processing_time_ms=result.processing_time_ms,
    )

# EmotionAI API Reference

Base URL (local): `http://localhost:8000`  
Base URL (production): Set via `NEXT_PUBLIC_API_BASE_URL`

All endpoints return JSON. Errors use the `ErrorResponse` schema.

---

## GET /health

Returns service status and model availability. Safe to call publicly.

**Response 200**
```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "0.1.0"
}
```

---

## GET /model-info

Returns safe model metadata. No secrets or internal paths exposed.

**Response 200**
```json
{
  "model_name": "emotion_cnn",
  "model_version": "0.1.0",
  "task": "facial_expression_recognition",
  "num_classes": 5,
  "supported_expressions": ["Angry", "Happy", "Sad", "Surprise", "Neutral"],
  "index_to_class": {"0": "Angry", "1": "Happy", "2": "Sad", "3": "Surprise", "4": "Neutral"},
  "input": {
    "height": 48,
    "width": 48,
    "channels": 1,
    "color_mode": "grayscale",
    "normalization": "divide_by_255"
  },
  "disclaimer": "Classifies visible facial expressions. Does not determine internal emotional state."
}
```

---

## POST /predict

Accepts an image upload, detects a face, and returns expression probabilities.

**Request**  
- `Content-Type: multipart/form-data`
- Field: `file` — image file (JPEG, PNG, BMP, WebP)
- Max size: 5 MB (configurable via `MAX_UPLOAD_BYTES`)

**Response 200 — face detected**
```json
{
  "success": true,
  "face_detected": true,
  "prediction": {
    "label": "Happy",
    "confidence": 0.87
  },
  "probabilities": {
    "Angry": 0.02,
    "Happy": 0.87,
    "Sad": 0.03,
    "Surprise": 0.01,
    "Neutral": 0.07
  },
  "processing_time_ms": 85.3,
  "message": null
}
```

**Response 200 — no face detected**
```json
{
  "success": true,
  "face_detected": false,
  "prediction": null,
  "probabilities": null,
  "processing_time_ms": 12.1,
  "message": "No face detected in the uploaded image."
}
```

**Response 400 — invalid input**
```json
{
  "success": false,
  "error": "Unsupported file type",
  "detail": "Received content-type 'text/plain'. Allowed: [...]"
}
```

**Response 413 — file too large**
```json
{
  "success": false,
  "error": "File too large",
  "detail": "Maximum upload size is 5 MB."
}
```

**Response 503 — model not loaded**
```json
{
  "success": false,
  "error": "Model not available",
  "detail": "The inference model is not loaded. Try again in a few seconds."
}
```

---

## Error Schema

All error responses follow this schema:

```json
{
  "success": false,
  "error": "Human-readable error title",
  "detail": "Optional additional detail"
}
```

No Python stack traces are returned in production responses.

---

## CORS

Allowed origins are configured via the `CORS_ORIGINS` environment variable (comma-separated).

Default (local): `http://localhost:3000`  
Production: Set to your Vercel URL before deploying.

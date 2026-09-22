/**
 * TypeScript types for EmotionAI frontend.
 * These mirror the backend Pydantic schemas in backend/app/schemas/predict.py.
 * Update both files when changing the API contract.
 */

// ── API response types ────────────────────────────────────────────────────────

export type ExpressionLabel = "Angry" | "Happy" | "Sad" | "Surprise" | "Neutral";

export interface PredictionDetail {
  label: ExpressionLabel;
  confidence: number; // 0.0 – 1.0
}

export interface PredictionResponse {
  success: boolean;
  face_detected: boolean;
  prediction?: PredictionDetail;
  probabilities?: Record<ExpressionLabel, number>;
  processing_time_ms: number;
  message?: string;
}

export interface ModelInputInfo {
  height: number;
  width: number;
  channels: number;
  color_mode: string;
  normalization: string;
}

export interface ModelInfoResponse {
  model_name: string;
  model_version: string;
  task: string;
  num_classes: number;
  supported_expressions: ExpressionLabel[];
  index_to_class: Record<string, string>;
  input: ModelInputInfo;
  disclaimer: string;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  version: string;
}

export interface ErrorResponse {
  success: false;
  error: string;
  detail?: string;
}

// ── Camera / UI state types ───────────────────────────────────────────────────

export type CameraStatus =
  | "idle"
  | "requesting"
  | "active"
  | "stopped"
  | "error"
  | "denied";

export type InferenceStatus =
  | "idle"
  | "processing"
  | "success"
  | "no_face"
  | "error"
  | "backend_unavailable";

export interface AppState {
  cameraStatus: CameraStatus;
  inferenceStatus: InferenceStatus;
  latestPrediction: PredictionResponse | null;
  errorMessage: string | null;
  backendAvailable: boolean;
  modelInfo: ModelInfoResponse | null;
}

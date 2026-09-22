/**
 * API client for the EmotionAI backend.
 *
 * All fetch calls go through this module. The base URL comes from
 * NEXT_PUBLIC_API_BASE_URL (set in .env.local for local dev, or
 * via Vercel environment variables for production).
 *
 * Design decisions:
 * - Returns typed results with discriminated union (data | error).
 * - Never throws unhandled exceptions — callers receive structured errors.
 * - Uses AbortSignal for request cancellation (prevents stale responses
 *   from late-arriving webcam frames).
 */

import type {
  ErrorResponse,
  HealthResponse,
  ModelInfoResponse,
  PredictionResponse,
} from "./types";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ||
  "http://localhost:8000";

// ── Generic fetch wrapper ─────────────────────────────────────────────────────

type ApiResult<T> =
  | { ok: true; data: T }
  | { ok: false; error: string; detail?: string; status?: number };

async function apiFetch<T>(
  path: string,
  options?: RequestInit
): Promise<ApiResult<T>> {
  try {
    const res = await fetch(`${API_BASE}${path}`, {
      ...options,
      headers: {
        Accept: "application/json",
        ...options?.headers,
      },
    });

    const json = await res.json().catch(() => null);

    if (!res.ok) {
      const errBody = json as ErrorResponse | null;
      return {
        ok: false,
        error: errBody?.error ?? `HTTP ${res.status}`,
        detail: errBody?.detail,
        status: res.status,
      };
    }

    return { ok: true, data: json as T };
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      return { ok: false, error: "Request cancelled", status: 0 };
    }
    const message =
      err instanceof Error ? err.message : "Network request failed";
    return { ok: false, error: message };
  }
}

// ── Public API functions ──────────────────────────────────────────────────────

export async function checkHealth(
  signal?: AbortSignal
): Promise<ApiResult<HealthResponse>> {
  return apiFetch<HealthResponse>("/health", { signal });
}

export async function getModelInfo(
  signal?: AbortSignal
): Promise<ApiResult<ModelInfoResponse>> {
  return apiFetch<ModelInfoResponse>("/model-info", { signal });
}

/**
 * Send a captured webcam frame to the backend for expression prediction.
 *
 * @param imageBlob  JPEG blob from canvas.toBlob() or equivalent.
 * @param signal     AbortSignal to cancel an in-flight request.
 */
export async function predictExpression(
  imageBlob: Blob,
  signal?: AbortSignal
): Promise<ApiResult<PredictionResponse>> {
  const form = new FormData();
  form.append("file", imageBlob, "frame.jpg");

  return apiFetch<PredictionResponse>("/predict", {
    method: "POST",
    body: form,
    signal,
  });
}

export { API_BASE };

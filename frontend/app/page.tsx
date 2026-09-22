"use client";

/**
 * EmotionAI — Main Dashboard Page
 *
 * Layout: two-column on desktop (camera left, prediction right),
 * single-column stack on mobile.
 *
 * State is owned here and passed down to child components.
 * API/webcam logic lives in lib/api.ts and components/WebcamCapture.tsx.
 */

import { useCallback, useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { checkHealth, getModelInfo } from "@/lib/api";
import type {
  AppState,
  CameraStatus,
  InferenceStatus,
  ModelInfoResponse,
  PredictionResponse,
} from "@/lib/types";
import CameraControls from "@/components/CameraControls";
import PredictionDisplay from "@/components/PredictionDisplay";

// WebcamCapture uses browser APIs — must be client-side only (no SSR)
const WebcamCapture = dynamic(() => import("@/components/WebcamCapture"), {
  ssr: false,
  loading: () => (
    <div
      className="w-full rounded-xl flex items-center justify-center"
      style={{ height: 300, background: "rgba(0,0,0,0.4)" }}
    >
      <div className="spinner" />
    </div>
  ),
});

const INITIAL_STATE: AppState = {
  cameraStatus: "idle",
  inferenceStatus: "idle",
  latestPrediction: null,
  errorMessage: null,
  backendAvailable: false,
  modelInfo: null,
};

export default function HomePage() {
  const [state, setState] = useState<AppState>(INITIAL_STATE);
  const [isRunning, setIsRunning] = useState(false);

  // ── Check backend health on mount ──────────────────────────────────────
  useEffect(() => {
    let mounted = true;
    const controller = new AbortController();

    const probe = async () => {
      const healthResult = await checkHealth(controller.signal);
      if (!mounted) return;

      if (healthResult.ok && healthResult.data.status === "ok") {
        setState((s) => ({ ...s, backendAvailable: true }));

        // Load model info
        const infoResult = await getModelInfo(controller.signal);
        if (mounted && infoResult.ok) {
          setState((s) => ({ ...s, modelInfo: infoResult.data }));
        }
      } else {
        setState((s) => ({ ...s, backendAvailable: false }));
      }
    };

    probe();
    // Re-probe every 15 seconds to detect backend coming online
    const interval = setInterval(probe, 15_000);

    return () => {
      mounted = false;
      controller.abort();
      clearInterval(interval);
    };
  }, []);

  // ── Callbacks passed to children ───────────────────────────────────────
  const handleCameraStatus = useCallback((status: CameraStatus) => {
    setState((s) => ({ ...s, cameraStatus: status }));
  }, []);

  const handleInferenceStatus = useCallback((status: InferenceStatus) => {
    setState((s) => ({ ...s, inferenceStatus: status }));
  }, []);

  const handlePrediction = useCallback((result: PredictionResponse) => {
    setState((s) => ({
      ...s,
      latestPrediction: result,
      errorMessage: null,
    }));
  }, []);

  const handleError = useCallback((message: string) => {
    setState((s) => ({ ...s, errorMessage: message }));
  }, []);

  const handleStart = useCallback(() => {
    setState((s) => ({ ...s, errorMessage: null, latestPrediction: null }));
    setIsRunning(true);
  }, []);

  const handleStop = useCallback(() => {
    setIsRunning(false);
  }, []);

  const { cameraStatus, inferenceStatus, latestPrediction, errorMessage, backendAvailable, modelInfo } = state;

  return (
    <main className="relative z-10 min-h-screen">
      {/* ── Header ────────────────────────────────────────────────────── */}
      <header className="border-b" style={{ borderColor: "var(--border-subtle)" }}>
        <div className="max-w-6xl mx-auto px-4 sm:px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            {/* Logo mark */}
            <div
              className="w-9 h-9 rounded-xl flex items-center justify-center text-lg font-bold"
              style={{
                background: "linear-gradient(135deg, #3182ce, #553c9a)",
                boxShadow: "0 0 20px rgba(49,130,206,0.4)",
              }}
            >
              🧠
            </div>
            <div>
              <h1 className="font-display font-bold text-lg leading-none" style={{ color: "var(--text-primary)" }}>
                EmotionAI
              </h1>
              <p className="text-xs" style={{ color: "var(--text-muted)" }}>
                Facial Expression Recognition
              </p>
            </div>
          </div>

          {/* Backend status pill */}
          <div className={`glow-badge ${backendAvailable ? "badge-active" : "badge-error"}`}>
            <span className={`pulse-dot ${backendAvailable ? "pulse-dot-green" : "pulse-dot-red"}`} />
            {backendAvailable ? "API Online" : "API Offline"}
          </div>
        </div>
      </header>

      {/* ── Hero tagline ──────────────────────────────────────────────── */}
      <div className="max-w-6xl mx-auto px-4 sm:px-6 pt-8 pb-2">
        <p className="text-sm" style={{ color: "var(--text-muted)" }}>
          Real-time expression classification using a CNN trained on FER2013 ·{" "}
          <span style={{ color: "var(--accent-blue)" }}>
            Classifies visible facial expressions
          </span>{" "}
          — not internal emotional state
        </p>
      </div>

      {/* ── Main content ─────────────────────────────────────────────── */}
      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-6 grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* ── LEFT: Camera ──────────────────────────────────────────── */}
        <div className="space-y-5">
          {/* Camera card */}
          <div className="glass p-1 overflow-hidden" style={{ borderRadius: 20 }}>
            {/* Camera preview area */}
            <div
              className="relative rounded-2xl overflow-hidden flex items-center justify-center"
              style={{ minHeight: 300, background: "rgba(0,0,0,0.6)" }}
            >
              {isRunning ? (
                <WebcamCapture
                  isRunning={isRunning}
                  onCameraStatusChange={handleCameraStatus}
                  onInferenceStatusChange={handleInferenceStatus}
                  onPrediction={handlePrediction}
                  onError={handleError}
                  frameIntervalMs={1000}
                />
              ) : (
                /* Camera placeholder */
                <div className="flex flex-col items-center gap-4 py-14">
                  <div
                    className="w-20 h-20 rounded-full flex items-center justify-center"
                    style={{ background: "rgba(49,130,206,0.1)", border: "2px dashed rgba(99,179,237,0.3)" }}
                  >
                    <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="rgba(99,179,237,0.6)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M23 7l-7 5 7 5V7z" />
                      <rect x="1" y="5" width="15" height="14" rx="2" ry="2" />
                    </svg>
                  </div>
                  <div className="text-center">
                    <p className="font-semibold" style={{ color: "var(--text-secondary)" }}>
                      Camera is off
                    </p>
                    <p className="text-sm mt-1" style={{ color: "var(--text-muted)" }}>
                      Click Start Camera to begin
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Controls card */}
          <div className="glass p-5" style={{ borderRadius: 20 }}>
            <CameraControls
              cameraStatus={cameraStatus}
              inferenceStatus={inferenceStatus}
              isRunning={isRunning}
              backendAvailable={backendAvailable}
              onStart={handleStart}
              onStop={handleStop}
            />
          </div>

          {/* Error message */}
          {errorMessage && (
            <div
              className="rounded-xl px-4 py-3 text-sm flex items-start gap-3"
              style={{
                background: "rgba(252,129,129,0.08)",
                border: "1px solid rgba(252,129,129,0.2)",
                color: "#fc8181",
              }}
              role="alert"
              aria-live="polite"
            >
              <span>⚠️</span>
              <p>{errorMessage}</p>
            </div>
          )}

          {/* Supported expressions */}
          <div className="glass p-5" style={{ borderRadius: 20 }}>
            <p className="text-xs uppercase tracking-widest font-medium mb-3" style={{ color: "var(--text-muted)" }}>
              Supported Expressions
            </p>
            <div className="flex flex-wrap gap-2">
              {(modelInfo?.supported_expressions ?? ["Angry", "Happy", "Sad", "Surprise", "Neutral"]).map((expr) => {
                const emojiMap: Record<string, string> = {
                  Angry: "😠", Happy: "😊", Sad: "😢", Surprise: "😮", Neutral: "😐",
                };
                return (
                  <span
                    key={expr}
                    className="glow-badge badge-idle"
                    style={{ fontSize: "0.7rem" }}
                  >
                    {emojiMap[expr] ?? "🎭"} {expr}
                  </span>
                );
              })}
            </div>
          </div>
        </div>

        {/* ── RIGHT: Prediction ──────────────────────────────────────── */}
        <div className="space-y-5">
          {/* Prediction card */}
          <div className="glass p-6" style={{ borderRadius: 20, minHeight: 340 }}>
            <p className="text-xs uppercase tracking-widest font-medium mb-5" style={{ color: "var(--text-muted)" }}>
              Live Prediction
            </p>
            <PredictionDisplay
              prediction={latestPrediction}
              inferenceStatus={inferenceStatus}
            />
          </div>

          {/* Model info card */}
          {modelInfo && (
            <div className="glass p-5" style={{ borderRadius: 20 }}>
              <p className="text-xs uppercase tracking-widest font-medium mb-3" style={{ color: "var(--text-muted)" }}>
                Model Information
              </p>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div>
                  <p className="text-xs" style={{ color: "var(--text-muted)" }}>Name</p>
                  <p style={{ color: "var(--text-secondary)" }}>{modelInfo.model_name}</p>
                </div>
                <div>
                  <p className="text-xs" style={{ color: "var(--text-muted)" }}>Version</p>
                  <p style={{ color: "var(--text-secondary)" }}>v{modelInfo.model_version}</p>
                </div>
                <div>
                  <p className="text-xs" style={{ color: "var(--text-muted)" }}>Input</p>
                  <p style={{ color: "var(--text-secondary)" }}>
                    {modelInfo.input.height}×{modelInfo.input.width} grayscale
                  </p>
                </div>
                <div>
                  <p className="text-xs" style={{ color: "var(--text-muted)" }}>Classes</p>
                  <p style={{ color: "var(--text-secondary)" }}>{modelInfo.num_classes}</p>
                </div>
              </div>
            </div>
          )}

          {/* Privacy notice */}
          <div
            className="rounded-2xl p-4 text-xs space-y-1"
            style={{
              background: "rgba(99,179,237,0.05)",
              border: "1px solid rgba(99,179,237,0.1)",
              color: "var(--text-muted)",
            }}
          >
            <p className="font-semibold" style={{ color: "var(--accent-blue)" }}>
              🔒 Privacy
            </p>
            <p>Camera frames are sent to the local inference API only. No images are stored, logged, or transmitted to third parties.</p>
            <p>This system classifies <em>visible facial expressions</em>. It does not determine internal emotional state, mental health, or personality.</p>
          </div>
        </div>
      </div>

      {/* ── Footer ───────────────────────────────────────────────────── */}
      <footer
        className="border-t mt-8 py-6"
        style={{ borderColor: "var(--border-subtle)" }}
      >
        <div className="max-w-6xl mx-auto px-4 sm:px-6 flex flex-col sm:flex-row items-center justify-between gap-2 text-xs" style={{ color: "var(--text-muted)" }}>
          <p>EmotionAI · CNN trained on FER2013 · 5-class facial expression recognition</p>
          <p>Portfolio project · MIT License</p>
        </div>
      </footer>
    </main>
  );
}

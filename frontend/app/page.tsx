"use client";

/**
 * EmotionAI — Main Dashboard Page
 *
 * Clean light-theme layout:
 * - Top header with logo + API status
 * - Two-column on desktop (camera | prediction), single-column on mobile
 * - Soft color palette, no dark backgrounds, no gradients
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

const WebcamCapture = dynamic(() => import("@/components/WebcamCapture"), {
  ssr: false,
  loading: () => (
    <div
      className="w-full flex items-center justify-center"
      style={{ height: 340, background: "var(--bg-subtle)", borderRadius: "var(--radius-md)" }}
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

  useEffect(() => {
    let mounted = true;
    const controller = new AbortController();

    const probe = async () => {
      const healthResult = await checkHealth(controller.signal);
      if (!mounted) return;

      if (healthResult.ok && healthResult.data.status === "ok") {
        setState((s) => ({ ...s, backendAvailable: true }));
        const infoResult = await getModelInfo(controller.signal);
        if (mounted && infoResult.ok) {
          setState((s) => ({ ...s, modelInfo: infoResult.data }));
        }
      } else {
        setState((s) => ({ ...s, backendAvailable: false }));
      }
    };

    probe();
    const interval = setInterval(probe, 15_000);
    return () => { mounted = false; controller.abort(); clearInterval(interval); };
  }, []);

  const handleCameraStatus = useCallback((status: CameraStatus) => {
    setState((s) => ({ ...s, cameraStatus: status }));
  }, []);
  const handleInferenceStatus = useCallback((status: InferenceStatus) => {
    setState((s) => ({ ...s, inferenceStatus: status }));
  }, []);
  const handlePrediction = useCallback((result: PredictionResponse) => {
    setState((s) => ({ ...s, latestPrediction: result, errorMessage: null }));
  }, []);
  const handleError = useCallback((message: string) => {
    setState((s) => ({ ...s, errorMessage: message }));
  }, []);
  const handleStart = useCallback(() => {
    setState((s) => ({ ...s, errorMessage: null, latestPrediction: null }));
    setIsRunning(true);
  }, []);
  const handleStop = useCallback(() => { setIsRunning(false); }, []);

  const { cameraStatus, inferenceStatus, latestPrediction, errorMessage, backendAvailable, modelInfo } = state;

  const EXPRESSIONS = modelInfo?.supported_expressions ?? ["Angry", "Happy", "Sad", "Surprise", "Neutral"];
  const EMOJI_MAP: Record<string, string> = {
    Angry: "😠", Happy: "😊", Sad: "😢", Surprise: "😮", Neutral: "😐",
  };

  return (
    <div style={{ minHeight: "100vh", background: "var(--bg-app)" }}>

      {/* ── Header ──────────────────────────────────────────────────── */}
      <header
        style={{
          background: "var(--bg-surface)",
          borderBottom: "1px solid var(--border-base)",
          position: "sticky",
          top: 0,
          zIndex: 50,
        }}
      >
        <div
          style={{
            maxWidth: 1200,
            margin: "0 auto",
            padding: "0 24px",
            height: 64,
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          {/* Logo */}
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div
              style={{
                width: 38,
                height: 38,
                borderRadius: 10,
                background: "var(--blue-light)",
                border: "1.5px solid var(--blue-ring)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: 18,
                flexShrink: 0,
              }}
            >
              🧠
            </div>
            <div>
              <h1
                className="font-display"
                style={{ fontSize: "1.05rem", fontWeight: 800, color: "var(--text-primary)", lineHeight: 1.1 }}
              >
                EmotionAI
              </h1>
              <p style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginTop: 1 }}>
                Facial Expression Recognition
              </p>
            </div>
          </div>

          {/* Nav links + status */}
          <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
            {/* Model badge */}
            {modelInfo && (
              <span
                className="badge badge-info"
                style={{ display: "flex" }}
              >
                CNN · v{modelInfo.model_version}
              </span>
            )}
            {/* API status */}
            <span className={`badge ${backendAvailable ? "badge-online" : "badge-offline"}`}>
              <span className={`pulse-dot ${backendAvailable ? "dot-green" : "dot-red"}`} />
              {backendAvailable ? "API Online" : "API Offline"}
            </span>
          </div>
        </div>
      </header>

      {/* ── Page body ───────────────────────────────────────────────── */}
      <main style={{ maxWidth: 1200, margin: "0 auto", padding: "32px 24px 48px" }}>

        {/* Sub-header */}
        <div style={{ marginBottom: 24 }}>
          <p style={{ fontSize: "0.85rem", color: "var(--text-muted)", maxWidth: 580 }}>
            Real-time facial expression classification using a CNN trained on FER2013.{" "}
            <span style={{ color: "var(--blue)", fontWeight: 500 }}>
              Classifies visible expressions only
            </span>{" "}
            — not internal emotional state.
          </p>
        </div>

        {/* ── Two-column grid ─────────────────────────────────────── */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
            gap: 24,
            alignItems: "start",
          }}
        >

          {/* ── LEFT: Camera Column ─────────────────────────────── */}
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>

            {/* Camera feed card */}
            <div className="card-elevated" style={{ padding: 20 }}>
              <p className="section-label">Live Camera</p>

              {/* Video area */}
              <div
                style={{
                  background: "var(--bg-subtle)",
                  border: "1.5px solid var(--border-base)",
                  borderRadius: "var(--radius-md)",
                  overflow: "hidden",
                  minHeight: 300,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  position: "relative",
                  aspectRatio: "4/3",
                }}
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
                  <div
                    style={{
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "center",
                      gap: 14,
                      padding: "40px 20px",
                    }}
                  >
                    <div
                      style={{
                        width: 72,
                        height: 72,
                        borderRadius: "50%",
                        background: "var(--blue-light)",
                        border: "2px dashed var(--blue-ring)",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                      }}
                    >
                      <svg
                        width="30" height="30" viewBox="0 0 24 24"
                        fill="none" stroke="var(--blue)" strokeWidth="1.8"
                        strokeLinecap="round" strokeLinejoin="round"
                      >
                        <path d="M23 7l-7 5 7 5V7z" />
                        <rect x="1" y="5" width="15" height="14" rx="2" ry="2" />
                      </svg>
                    </div>
                    <div style={{ textAlign: "center" }}>
                      <p style={{ fontWeight: 600, color: "var(--text-secondary)", fontSize: "0.95rem" }}>
                        Camera is off
                      </p>
                      <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: 4 }}>
                        Press Start Camera below to begin
                      </p>
                    </div>
                  </div>
                )}
              </div>

              {/* Controls */}
              <div style={{ marginTop: 16 }}>
                <CameraControls
                  cameraStatus={cameraStatus}
                  inferenceStatus={inferenceStatus}
                  isRunning={isRunning}
                  backendAvailable={backendAvailable}
                  onStart={handleStart}
                  onStop={handleStop}
                />
              </div>
            </div>

            {/* Error */}
            {errorMessage && (
              <div className="alert alert-error" role="alert" aria-live="polite">
                <span style={{ flexShrink: 0 }}>⚠️</span>
                <p>{errorMessage}</p>
              </div>
            )}

            {/* Supported expressions card */}
            <div className="card" style={{ padding: "18px 20px" }}>
              <p className="section-label">Supported Expressions</p>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                {EXPRESSIONS.map((expr) => (
                  <span key={expr} className="expr-chip">
                    {EMOJI_MAP[expr] ?? "🎭"}{" "}{expr}
                  </span>
                ))}
              </div>
            </div>

            {/* Model info card */}
            {modelInfo && (
              <div className="card" style={{ padding: "18px 20px" }}>
                <p className="section-label">Model Details</p>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr",
                    gap: "12px 20px",
                  }}
                >
                  {[
                    { label: "Architecture", value: modelInfo.model_name },
                    { label: "Version",      value: `v${modelInfo.model_version}` },
                    { label: "Input size",   value: `${modelInfo.input.height}×${modelInfo.input.width} grayscale` },
                    { label: "Classes",      value: String(modelInfo.num_classes) },
                    { label: "Dataset",      value: "FER2013" },
                    { label: "Framework",    value: "TensorFlow/Keras" },
                  ].map(({ label, value }) => (
                    <div key={label}>
                      <p style={{ fontSize: "0.7rem", fontWeight: 600, color: "var(--text-faint)", textTransform: "uppercase", letterSpacing: "0.06em" }}>
                        {label}
                      </p>
                      <p style={{ fontSize: "0.85rem", fontWeight: 500, color: "var(--text-secondary)", marginTop: 2 }}>
                        {value}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* ── RIGHT: Prediction Column ─────────────────────────── */}
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>

            {/* Live prediction card */}
            <div
              className="card-elevated"
              style={{ padding: 24, minHeight: 360 }}
            >
              <p className="section-label">Live Prediction</p>
              <PredictionDisplay
                prediction={latestPrediction}
                inferenceStatus={inferenceStatus}
              />
            </div>

            {/* Privacy notice */}
            <div className="alert alert-info" style={{ alignItems: "flex-start" }}>
              <span style={{ fontSize: "1rem", flexShrink: 0 }}>🔒</span>
              <div>
                <p style={{ fontWeight: 600, marginBottom: 4 }}>Privacy Notice</p>
                <p style={{ fontSize: "0.78rem", lineHeight: 1.55, color: "#3357CC", opacity: 0.85 }}>
                  Frames are sent only to the local inference API. No images are stored, logged, or
                  transmitted externally. This system classifies visible facial expressions — it does{" "}
                  <em>not</em> determine internal emotional state, mental health, or personality.
                </p>
              </div>
            </div>

            {/* Quick stats row */}
            {isRunning && (
              <div
                className="card"
                style={{
                  padding: "16px 20px",
                  display: "flex",
                  justifyContent: "space-around",
                  textAlign: "center",
                }}
              >
                {[
                  { label: "Sample Rate",   value: "1 fps" },
                  { label: "Face Crop",     value: "Haar" },
                  { label: "Input",         value: "48×48 gray" },
                ].map(({ label, value }) => (
                  <div key={label}>
                    <p
                      className="font-display"
                      style={{ fontWeight: 700, fontSize: "1rem", color: "var(--blue)" }}
                    >
                      {value}
                    </p>
                    <p style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginTop: 2 }}>
                      {label}
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </main>

      {/* ── Footer ──────────────────────────────────────────────────── */}
      <footer
        style={{
          borderTop: "1px solid var(--border-base)",
          background: "var(--bg-surface)",
          padding: "18px 24px",
        }}
      >
        <div
          style={{
            maxWidth: 1200,
            margin: "0 auto",
            display: "flex",
            flexWrap: "wrap",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 8,
          }}
        >
          <p style={{ fontSize: "0.78rem", color: "var(--text-muted)" }}>
            EmotionAI · CNN trained on FER2013 · 5-class facial expression recognition
          </p>
          <p style={{ fontSize: "0.78rem", color: "var(--text-faint)" }}>
            Portfolio project · MIT License
          </p>
        </div>
      </footer>
    </div>
  );
}

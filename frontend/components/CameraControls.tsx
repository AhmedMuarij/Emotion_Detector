"use client";

/**
 * CameraControls — Start/Stop buttons with contextual status badges.
 * Light-theme redesign: soft colors, clean card-less layout.
 */

import type { CameraStatus, InferenceStatus } from "@/lib/types";

interface CameraControlsProps {
  cameraStatus: CameraStatus;
  inferenceStatus: InferenceStatus;
  isRunning: boolean;
  backendAvailable: boolean;
  onStart: () => void;
  onStop: () => void;
}

function CameraStatusBadge({ status }: { status: CameraStatus }) {
  const configs: Record<CameraStatus, { label: string; cls: string; dotCls: string }> = {
    idle:       { label: "Camera Off",     cls: "badge-idle",        dotCls: "pulse-dot dot-gray" },
    requesting: { label: "Requesting…",    cls: "badge-processing",  dotCls: "pulse-dot dot-teal" },
    active:     { label: "Camera Active",  cls: "badge-active",      dotCls: "pulse-dot dot-green" },
    stopped:    { label: "Camera Stopped", cls: "badge-idle",        dotCls: "pulse-dot dot-gray" },
    error:      { label: "Camera Error",   cls: "badge-offline",     dotCls: "pulse-dot dot-red" },
    denied:     { label: "Access Denied",  cls: "badge-offline",     dotCls: "pulse-dot dot-red" },
  };
  const cfg = configs[status];
  return (
    <span className={`badge ${cfg.cls}`}>
      <span className={cfg.dotCls} />
      {cfg.label}
    </span>
  );
}

function InferenceStatusBadge({ status }: { status: InferenceStatus }) {
  const configs: Record<InferenceStatus, { label: string; cls: string }> = {
    idle:                { label: "Idle",              cls: "badge-idle" },
    processing:          { label: "Processing…",       cls: "badge-processing" },
    success:             { label: "Face Detected",     cls: "badge-active" },
    no_face:             { label: "No Face",           cls: "badge-warn" },
    error:               { label: "Error",             cls: "badge-offline" },
    backend_unavailable: { label: "Backend Offline",   cls: "badge-offline" },
  };
  const cfg = configs[status];
  return <span className={`badge ${cfg.cls}`}>{cfg.label}</span>;
}

export default function CameraControls({
  cameraStatus,
  inferenceStatus,
  isRunning,
  backendAvailable,
  onStart,
  onStop,
}: CameraControlsProps) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>

      {/* Status badges row */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
        <CameraStatusBadge status={cameraStatus} />
        <InferenceStatusBadge status={inferenceStatus} />
      </div>

      {/* Backend offline alert */}
      {!backendAvailable && (
        <div className="alert alert-error" role="alert">
          <span style={{ flexShrink: 0 }}>⚠️</span>
          <div>
            <p style={{ fontWeight: 600, fontSize: "0.85rem" }}>Backend unavailable</p>
            <p style={{ fontSize: "0.75rem", marginTop: 2, opacity: 0.8 }}>
              Start the FastAPI server on{" "}
              <code
                className="mono"
                style={{
                  background: "rgba(224,122,95,0.12)",
                  padding: "1px 5px",
                  borderRadius: 4,
                }}
              >
                {process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000"}
              </code>
            </p>
          </div>
        </div>
      )}

      {/* Camera permission denied */}
      {cameraStatus === "denied" && (
        <div className="alert alert-warn" role="alert">
          <span style={{ flexShrink: 0 }}>🔒</span>
          <div>
            <p style={{ fontWeight: 600, fontSize: "0.85rem" }}>Camera permission denied</p>
            <p style={{ fontSize: "0.75rem", marginTop: 2, opacity: 0.8 }}>
              Click the camera icon in the browser address bar and allow access, then refresh.
            </p>
          </div>
        </div>
      )}

      {/* Buttons */}
      <div style={{ display: "flex", gap: 12 }}>
        <button
          id="btn-start-camera"
          className="btn btn-primary"
          style={{ flex: 1 }}
          onClick={onStart}
          disabled={isRunning || cameraStatus === "requesting"}
          aria-label="Start camera and begin expression recognition"
        >
          <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor">
            <path d="M8 5v14l11-7z" />
          </svg>
          Start Camera
        </button>

        <button
          id="btn-stop-camera"
          className="btn btn-danger"
          style={{ flex: 1 }}
          onClick={onStop}
          disabled={!isRunning}
          aria-label="Stop camera and expression recognition"
        >
          <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">
            <rect x="6" y="6" width="12" height="12" rx="1.5" />
          </svg>
          Stop Camera
        </button>
      </div>

      {/* Running info */}
      {isRunning && (
        <p style={{ fontSize: "0.75rem", textAlign: "center", color: "var(--text-muted)" }}>
          Sampling 1 frame/sec · Largest detected face is used
        </p>
      )}
    </div>
  );
}

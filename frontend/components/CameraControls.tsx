"use client";

/**
 * CameraControls — Start / Stop buttons with contextual status display.
 * Shows camera permission guidance and backend availability warnings.
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
    idle:       { label: "Camera Off",     cls: "badge-idle",       dotCls: "pulse-dot" },
    requesting: { label: "Requesting…",    cls: "badge-processing", dotCls: "pulse-dot pulse-dot-blue" },
    active:     { label: "Camera Active",  cls: "badge-active",     dotCls: "pulse-dot pulse-dot-green" },
    stopped:    { label: "Camera Stopped", cls: "badge-idle",       dotCls: "pulse-dot" },
    error:      { label: "Camera Error",   cls: "badge-error",      dotCls: "pulse-dot pulse-dot-red" },
    denied:     { label: "Access Denied",  cls: "badge-error",      dotCls: "pulse-dot pulse-dot-red" },
  };
  const cfg = configs[status];
  return (
    <span className={`glow-badge ${cfg.cls}`}>
      <span className={cfg.dotCls} />
      {cfg.label}
    </span>
  );
}

function InferenceStatusBadge({ status }: { status: InferenceStatus }) {
  const configs: Record<InferenceStatus, { label: string; cls: string }> = {
    idle:                { label: "Idle",               cls: "badge-idle" },
    processing:          { label: "Processing…",        cls: "badge-processing" },
    success:             { label: "Face Detected",      cls: "badge-active" },
    no_face:             { label: "No Face",            cls: "badge-warning" },
    error:               { label: "Error",              cls: "badge-error" },
    backend_unavailable: { label: "Backend Offline",    cls: "badge-error" },
  };
  const cfg = configs[status];
  return <span className={`glow-badge ${cfg.cls}`}>{cfg.label}</span>;
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
    <div className="space-y-4">
      {/* Status badges */}
      <div className="flex flex-wrap gap-2">
        <CameraStatusBadge status={cameraStatus} />
        <InferenceStatusBadge status={inferenceStatus} />
      </div>

      {/* Backend offline warning */}
      {!backendAvailable && (
        <div
          className="rounded-xl px-4 py-3 text-sm flex items-start gap-3"
          style={{
            background: "rgba(252,129,129,0.08)",
            border: "1px solid rgba(252,129,129,0.2)",
            color: "#fc8181",
          }}
          role="alert"
        >
          <span className="text-base">⚠️</span>
          <div>
            <p className="font-semibold">Backend unavailable</p>
            <p className="text-xs mt-0.5" style={{ color: "rgba(252,129,129,0.75)" }}>
              Make sure the FastAPI server is running on{" "}
              <code className="font-mono bg-black/20 px-1 rounded">
                {process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000"}
              </code>
            </p>
          </div>
        </div>
      )}

      {/* Permission denied guidance */}
      {cameraStatus === "denied" && (
        <div
          className="rounded-xl px-4 py-3 text-sm flex items-start gap-3"
          style={{
            background: "rgba(246,173,85,0.08)",
            border: "1px solid rgba(246,173,85,0.2)",
            color: "#f6ad55",
          }}
          role="alert"
        >
          <span className="text-base">🔒</span>
          <div>
            <p className="font-semibold">Camera permission denied</p>
            <p className="text-xs mt-0.5" style={{ color: "rgba(246,173,85,0.75)" }}>
              Click the camera icon in your browser address bar and allow access,
              then refresh the page.
            </p>
          </div>
        </div>
      )}

      {/* Buttons */}
      <div className="flex gap-3">
        <button
          id="btn-start-camera"
          className="btn btn-primary flex-1"
          onClick={onStart}
          disabled={isRunning || cameraStatus === "requesting"}
          aria-label="Start camera and begin expression recognition"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
            <path d="M8 5v14l11-7z" />
          </svg>
          Start Camera
        </button>

        <button
          id="btn-stop-camera"
          className="btn btn-danger flex-1"
          onClick={onStop}
          disabled={!isRunning}
          aria-label="Stop camera and expression recognition"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
            <rect x="6" y="6" width="12" height="12" rx="1" />
          </svg>
          Stop Camera
        </button>
      </div>

      {/* Frame rate info */}
      {isRunning && (
        <p className="text-xs text-center" style={{ color: "var(--text-muted)" }}>
          Sampling 1 frame/sec · Largest detected face is used
        </p>
      )}
    </div>
  );
}

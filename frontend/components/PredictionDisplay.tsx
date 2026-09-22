"use client";

/**
 * PredictionDisplay — renders the current expression prediction.
 *
 * Light-theme redesign: clean confidence ring in soft blue,
 * muted color-coded probability bars, no dark or neon colors.
 */

import type { InferenceStatus, PredictionResponse } from "@/lib/types";

interface PredictionDisplayProps {
  prediction: PredictionResponse | null;
  inferenceStatus: InferenceStatus;
}

// Per-expression visual config — all soft, readable colors
const EXPRESSION_CONFIG: Record<string, { color: string; bgColor: string; emoji: string; barColor: string }> = {
  Happy:    { color: "#52B788", bgColor: "#E9F7F0", emoji: "😊", barColor: "#52B788" },
  Neutral:  { color: "#7B8FA8", bgColor: "#EDF1F7", emoji: "😐", barColor: "#7B8FA8" },
  Sad:      { color: "#3AAFA9", bgColor: "#E8F7F6", emoji: "😢", barColor: "#3AAFA9" },
  Angry:    { color: "#E07A5F", bgColor: "#FDF0EC", emoji: "😠", barColor: "#E07A5F" },
  Surprise: { color: "#D4AC0D", bgColor: "#FEF9E7", emoji: "😮", barColor: "#D4AC0D" },
};

const DEFAULT_CONFIG = { color: "#7B8FA8", bgColor: "#EDF1F7", emoji: "🤔", barColor: "#7B8FA8" };

function ConfidenceRing({ confidence, color }: { confidence: number; color: string }) {
  const pct = Math.round(confidence * 100);
  const r = 46;
  const circ = 2 * Math.PI * r;
  const dash = circ * Math.min(confidence, 1);

  return (
    <div
      style={{
        position: "relative",
        width: 120,
        height: 120,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        flexShrink: 0,
      }}
    >
      <svg width="120" height="120" style={{ position: "absolute", inset: 0, transform: "rotate(-90deg)" }}>
        {/* Track */}
        <circle cx="60" cy="60" r={r} fill="none" stroke="var(--bg-subtle)" strokeWidth="9" />
        {/* Progress */}
        <circle
          cx="60" cy="60" r={r}
          fill="none"
          stroke={color}
          strokeWidth="9"
          strokeLinecap="round"
          strokeDasharray={`${dash} ${circ}`}
          style={{ transition: "stroke-dasharray 0.5s cubic-bezier(0.4,0,0.2,1)" }}
        />
      </svg>
      <div style={{ textAlign: "center", lineHeight: 1.1 }}>
        <span
          className="font-display"
          style={{ fontSize: "1.5rem", fontWeight: 800, color }}
        >
          {pct}%
        </span>
        <p style={{ fontSize: "0.62rem", color: "var(--text-muted)", marginTop: 2, fontWeight: 500 }}>
          confidence
        </p>
      </div>
    </div>
  );
}

export default function PredictionDisplay({ prediction, inferenceStatus }: PredictionDisplayProps) {

  // Processing state
  if (inferenceStatus === "processing" && !prediction) {
    return (
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 16, padding: "60px 20px" }}>
        <div className="spinner" style={{ width: 32, height: 32 }} />
        <p style={{ fontSize: "0.85rem", color: "var(--text-muted)", fontWeight: 500 }}>Analyzing frame…</p>
      </div>
    );
  }

  // No face detected
  if (inferenceStatus === "no_face" || (prediction && !prediction.face_detected)) {
    return (
      <div
        className="animate-slide-up"
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: 12,
          padding: "60px 20px",
          textAlign: "center",
        }}
      >
        <div
          style={{
            width: 64,
            height: 64,
            borderRadius: "50%",
            background: "var(--bg-subtle)",
            border: "1.5px solid var(--border-base)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: "1.8rem",
          }}
        >
          👤
        </div>
        <div>
          <p style={{ fontWeight: 600, color: "var(--text-secondary)", fontSize: "0.95rem" }}>No face detected</p>
          <p style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginTop: 4, maxWidth: 220 }}>
            Center your face clearly in the camera frame
          </p>
        </div>
      </div>
    );
  }

  // Idle / not started
  if (!prediction || !prediction.face_detected || !prediction.prediction) {
    return (
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: 14,
          padding: "60px 20px",
          textAlign: "center",
        }}
      >
        <div
          style={{
            width: 64,
            height: 64,
            borderRadius: "50%",
            background: "var(--bg-subtle)",
            border: "1.5px dashed var(--border-soft)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: "1.8rem",
            opacity: 0.7,
          }}
        >
          🎭
        </div>
        <p style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>
          Start the camera to begin recognition
        </p>
      </div>
    );
  }

  const { label, confidence } = prediction.prediction;
  const probs = prediction.probabilities ?? {};
  const config = EXPRESSION_CONFIG[label] ?? DEFAULT_CONFIG;

  const sortedProbs = (Object.entries(probs) as [string, number][]).sort(
    ([, a], [, b]) => b - a
  );

  return (
    <div className="animate-slide-up" style={{ display: "flex", flexDirection: "column", gap: 24 }}>

      {/* ── Main result ───────────────────────────────────────────── */}
      <div
        style={{
          background: config.bgColor,
          borderRadius: "var(--radius-lg)",
          padding: "20px 24px",
          display: "flex",
          alignItems: "center",
          gap: 24,
          border: `1.5px solid ${config.color}28`,
        }}
      >
        <ConfidenceRing confidence={confidence} color={config.color} />

        <div style={{ flex: 1, minWidth: 0 }}>
          <p style={{ fontSize: "0.7rem", fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase", color: "var(--text-faint)", marginBottom: 6 }}>
            Predicted Expression
          </p>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ fontSize: "2.2rem" }}>{config.emoji}</span>
            <span
              className="font-display"
              style={{ fontSize: "1.8rem", fontWeight: 800, color: config.color }}
            >
              {label}
            </span>
          </div>
          <p style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginTop: 6 }}>
            {Math.round(confidence * 100)}% model confidence
          </p>
        </div>
      </div>

      {/* ── Probability breakdown ──────────────────────────────── */}
      <div>
        <p className="section-label">All Expression Probabilities</p>
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {sortedProbs.map(([name, prob]) => {
            const cfg = EXPRESSION_CONFIG[name] ?? DEFAULT_CONFIG;
            const pct = Math.round((prob as number) * 100);
            const isTop = name === label;
            return (
              <div key={name}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 5 }}>
                  <span
                    style={{
                      fontSize: "0.83rem",
                      display: "flex",
                      alignItems: "center",
                      gap: 7,
                      fontWeight: isTop ? 600 : 400,
                      color: isTop ? cfg.color : "var(--text-secondary)",
                    }}
                  >
                    <span style={{ fontSize: "0.95rem" }}>{cfg.emoji}</span>
                    {name}
                  </span>
                  <span
                    className="mono"
                    style={{
                      fontWeight: isTop ? 700 : 400,
                      fontSize: "0.8rem",
                      color: isTop ? cfg.color : "var(--text-muted)",
                    }}
                  >
                    {pct}%
                  </span>
                </div>
                <div className="prob-bar-track">
                  <div
                    className="prob-bar-fill"
                    style={{
                      width: `${pct}%`,
                      background: isTop ? cfg.barColor : "var(--bg-muted)",
                      opacity: isTop ? 1 : 0.7,
                    }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── Latency ────────────────────────────────────────────── */}
      {prediction.processing_time_ms > 0 && (
        <div
          style={{
            display: "flex",
            justifyContent: "flex-end",
          }}
        >
          <span
            className="badge badge-idle"
            style={{ fontSize: "0.68rem" }}
          >
            ⚡ {Math.round(prediction.processing_time_ms)} ms inference
          </span>
        </div>
      )}
    </div>
  );
}

"use client";

/**
 * PredictionDisplay — renders the current expression prediction with
 * confidence percentage and a per-class probability breakdown.
 *
 * Shows three distinct states:
 *  - "no_face"    : face not detected message
 *  - "processing" : animated spinner
 *  - prediction   : expression label + confidence + probability bars
 */

import type { InferenceStatus, PredictionResponse } from "@/lib/types";

interface PredictionDisplayProps {
  prediction: PredictionResponse | null;
  inferenceStatus: InferenceStatus;
}

// Expression-specific colors for the probability bars
const EXPRESSION_CONFIG: Record<
  string,
  { color: string; emoji: string; barColor: string }
> = {
  Happy:    { color: "#68d391", emoji: "😊", barColor: "#68d391" },
  Neutral:  { color: "#a0aec0", emoji: "😐", barColor: "#a0aec0" },
  Sad:      { color: "#76e4f7", emoji: "😢", barColor: "#76e4f7" },
  Angry:    { color: "#fc8181", emoji: "😠", barColor: "#fc8181" },
  Surprise: { color: "#f6ad55", emoji: "😮", barColor: "#f6ad55" },
};

const DEFAULT_CONFIG = { color: "#a0aec0", emoji: "🤔", barColor: "#a0aec0" };

function ConfidenceRing({
  confidence,
  color,
}: {
  confidence: number;
  color: string;
}) {
  const pct = Math.round(confidence * 100);
  const r = 44;
  const circ = 2 * Math.PI * r;
  const dash = circ * confidence;

  return (
    <div className="relative flex items-center justify-center" style={{ width: 108, height: 108 }}>
      <svg width="108" height="108" className="absolute inset-0 -rotate-90">
        {/* Track */}
        <circle
          cx="54" cy="54" r={r}
          fill="none"
          stroke="rgba(255,255,255,0.06)"
          strokeWidth="8"
        />
        {/* Progress */}
        <circle
          cx="54" cy="54" r={r}
          fill="none"
          stroke={color}
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={`${dash} ${circ}`}
          style={{
            filter: `drop-shadow(0 0 6px ${color})`,
            transition: "stroke-dasharray 0.5s cubic-bezier(0.4,0,0.2,1)",
          }}
        />
      </svg>
      <span
        className="font-display font-bold text-2xl"
        style={{ color, lineHeight: 1 }}
      >
        {pct}%
      </span>
    </div>
  );
}

export default function PredictionDisplay({
  prediction,
  inferenceStatus,
}: PredictionDisplayProps) {
  // ── Processing ──────────────────────────────────────────────────────────
  if (inferenceStatus === "processing" && !prediction) {
    return (
      <div className="flex flex-col items-center justify-center py-10 gap-4">
        <div className="spinner" style={{ width: 32, height: 32 }} />
        <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
          Analyzing frame…
        </p>
      </div>
    );
  }

  // ── No face ─────────────────────────────────────────────────────────────
  if (inferenceStatus === "no_face" || (prediction && !prediction.face_detected)) {
    return (
      <div className="flex flex-col items-center justify-center py-10 gap-3 animate-slide-up">
        <span style={{ fontSize: "2.5rem" }}>👤</span>
        <p className="font-semibold" style={{ color: "var(--text-secondary)" }}>
          No face detected
        </p>
        <p className="text-xs text-center max-w-xs" style={{ color: "var(--text-muted)" }}>
          Position your face clearly in the camera frame
        </p>
      </div>
    );
  }

  // ── Idle ────────────────────────────────────────────────────────────────
  if (!prediction || !prediction.face_detected || !prediction.prediction) {
    return (
      <div className="flex flex-col items-center justify-center py-10 gap-3">
        <span style={{ fontSize: "2.5rem", opacity: 0.4 }}>🎭</span>
        <p className="text-sm" style={{ color: "var(--text-muted)" }}>
          Start the camera to begin recognition
        </p>
      </div>
    );
  }

  const { label, confidence } = prediction.prediction;
  const probs = prediction.probabilities ?? {};
  const config = EXPRESSION_CONFIG[label] ?? DEFAULT_CONFIG;

  // Sort probabilities highest first for the bar list
  const sortedProbs = (Object.entries(probs) as [string, number][]).sort(
    ([, a], [, b]) => b - a
  );

  return (
    <div className="animate-slide-up space-y-5">
      {/* ── Main prediction ─────────────────────────────────────────────── */}
      <div className="flex items-center gap-6">
        <ConfidenceRing confidence={confidence} color={config.color} />

        <div className="flex-1 min-w-0">
          <p
            className="text-xs uppercase tracking-widest font-medium mb-1"
            style={{ color: "var(--text-muted)" }}
          >
            Predicted Expression
          </p>
          <div className="flex items-center gap-2 mb-1">
            <span style={{ fontSize: "1.6rem" }}>{config.emoji}</span>
            <span
              className="font-display font-bold text-2xl"
              style={{ color: config.color }}
            >
              {label}
            </span>
          </div>
          <p className="text-xs" style={{ color: "var(--text-muted)" }}>
            {Math.round(confidence * 100)}% model confidence
          </p>
        </div>
      </div>

      {/* ── Probability breakdown ──────────────────────────────────────── */}
      <div>
        <p
          className="text-xs uppercase tracking-widest font-medium mb-3"
          style={{ color: "var(--text-muted)" }}
        >
          All Expression Probabilities
        </p>
        <div className="space-y-2.5">
          {sortedProbs.map(([name, prob]) => {
            const cfg = EXPRESSION_CONFIG[name] ?? DEFAULT_CONFIG;
            const pct = Math.round((prob as number) * 100);
            const isTop = name === label;
            return (
              <div key={name} className="space-y-1">
                <div className="flex justify-between items-center">
                  <span
                    className="text-sm flex items-center gap-1.5"
                    style={{
                      color: isTop ? cfg.color : "var(--text-secondary)",
                      fontWeight: isTop ? 600 : 400,
                    }}
                  >
                    <span style={{ fontSize: "0.85rem" }}>{cfg.emoji}</span>
                    {name}
                  </span>
                  <span
                    className="text-xs font-mono"
                    style={{
                      color: isTop ? cfg.color : "var(--text-muted)",
                      fontWeight: isTop ? 600 : 400,
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
                      background: isTop
                        ? cfg.barColor
                        : "rgba(255,255,255,0.15)",
                      boxShadow: isTop ? `0 0 8px ${cfg.barColor}60` : "none",
                    }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── Timing ─────────────────────────────────────────────────────── */}
      {prediction.processing_time_ms > 0 && (
        <p className="text-xs text-right" style={{ color: "var(--text-muted)" }}>
          Processed in {Math.round(prediction.processing_time_ms)} ms
        </p>
      )}
    </div>
  );
}

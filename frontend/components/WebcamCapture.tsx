"use client";

/**
 * WebcamCapture — handles getUserMedia, frame capture, and sending frames
 * to the backend at a controlled rate (default: 1 frame/sec).
 *
 * Design decisions:
 * - The <video> element is rendered here; parent sees only events/data.
 * - Frames are captured to a hidden <canvas> then converted to JPEG blobs.
 * - An AbortController cancels any in-flight request when a new one starts
 *   or when the camera stops — prevents stale responses from updating UI.
 * - The inference interval only fires when the previous request is done
 *   (via `isProcessing` ref), preventing request pile-up.
 * - Camera tracks are explicitly stopped when the component unmounts or
 *   the user clicks Stop — the browser camera indicator light turns off.
 * - Supports front/back camera toggle on mobile devices.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { predictExpression } from "@/lib/api";
import type { CameraStatus, InferenceStatus, PredictionResponse } from "@/lib/types";

interface WebcamCaptureProps {
  isRunning: boolean;
  onCameraStatusChange: (status: CameraStatus) => void;
  onInferenceStatusChange: (status: InferenceStatus) => void;
  onPrediction: (result: PredictionResponse) => void;
  onError: (message: string) => void;
  frameIntervalMs?: number; // default 1000ms = 1 fps
}

export default function WebcamCapture({
  isRunning,
  onCameraStatusChange,
  onInferenceStatusChange,
  onPrediction,
  onError,
  frameIntervalMs = 1000,
}: WebcamCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const isProcessingRef = useRef(false);

  // "user" = front camera, "environment" = back camera
  const [facingMode, setFacingMode] = useState<"user" | "environment">("user");
  const facingModeRef = useRef(facingMode);
  facingModeRef.current = facingMode;

  const stopCamera = useCallback(() => {
    // Cancel any in-flight request
    abortRef.current?.abort();
    abortRef.current = null;

    // Clear capture interval
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    // Stop all media tracks (turns off camera indicator)
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    // Clear video source
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    isProcessingRef.current = false;
    onCameraStatusChange("stopped");
    onInferenceStatusChange("idle");
  }, [onCameraStatusChange, onInferenceStatusChange]);

  const captureAndSend = useCallback(async () => {
    if (isProcessingRef.current) return; // skip if previous request is still running
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2) return;

    // Draw current video frame to canvas
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    // Convert canvas to JPEG blob (quality 0.8 — good balance for upload size)
    const blob = await new Promise<Blob | null>((resolve) =>
      canvas.toBlob(resolve, "image/jpeg", 0.8)
    );
    if (!blob) return;

    isProcessingRef.current = true;
    onInferenceStatusChange("processing");

    // Cancel previous request if still somehow running
    abortRef.current?.abort();
    abortRef.current = new AbortController();

    const result = await predictExpression(blob, abortRef.current.signal);

    if (!result.ok) {
      if (result.error === "Request cancelled") {
        isProcessingRef.current = false;
        return;
      }
      if (result.status === 503 || result.status === 0) {
        onInferenceStatusChange("backend_unavailable");
        onError("Backend not reachable. Is the API server running?");
      } else {
        onInferenceStatusChange("error");
        onError(result.error);
      }
      isProcessingRef.current = false;
      return;
    }

    onPrediction(result.data);
    onInferenceStatusChange(
      result.data.face_detected ? "success" : "no_face"
    );
    isProcessingRef.current = false;
  }, [onPrediction, onInferenceStatusChange, onError]);

  // Start camera with the given facing mode
  const startCamera = useCallback(async (facing: "user" | "environment") => {
    // Stop existing stream first
    abortRef.current?.abort();
    abortRef.current = null;
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    onCameraStatusChange("requesting");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: facing,
        },
        audio: false,
      });

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      onCameraStatusChange("active");

      // Begin frame capture loop
      intervalRef.current = setInterval(captureAndSend, frameIntervalMs);
    } catch (err) {
      if (err instanceof Error) {
        if (
          err.name === "NotAllowedError" ||
          err.name === "PermissionDeniedError"
        ) {
          onCameraStatusChange("denied");
          onError(
            "Camera permission denied. Please allow camera access in your browser settings."
          );
        } else if (
          err.name === "NotFoundError" ||
          err.name === "DevicesNotFoundError"
        ) {
          onCameraStatusChange("error");
          onError("No camera found. Please connect a camera and try again.");
        } else {
          onCameraStatusChange("error");
          onError(`Camera error: ${err.message}`);
        }
      }
    }
  }, [captureAndSend, frameIntervalMs, onCameraStatusChange, onError]);

  // Start / stop camera when isRunning or facingMode changes
  useEffect(() => {
    if (!isRunning) {
      stopCamera();
      return;
    }

    startCamera(facingMode);

    return () => {
      stopCamera();
    };
  }, [isRunning, facingMode, startCamera, stopCamera]);

  // Toggle between front and back camera
  const handleFlipCamera = useCallback(() => {
    setFacingMode((prev) => (prev === "user" ? "environment" : "user"));
  }, []);

  return (
    <div style={{ position: "relative", width: "100%" }}>
      {/* Live video preview */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="w-full rounded-xl object-cover"
        style={{ maxHeight: "360px", background: "#000" }}
        aria-label="Webcam preview"
      />

      {/* Flip camera button — floating overlay */}
      <button
        id="btn-flip-camera"
        onClick={handleFlipCamera}
        title={facingMode === "user" ? "Switch to back camera" : "Switch to front camera"}
        aria-label="Flip camera"
        style={{
          position: "absolute",
          top: 12,
          right: 12,
          width: 40,
          height: 40,
          borderRadius: "50%",
          background: "rgba(0, 0, 0, 0.5)",
          backdropFilter: "blur(8px)",
          border: "1.5px solid rgba(255, 255, 255, 0.25)",
          color: "#FFFFFF",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          cursor: "pointer",
          transition: "all 0.2s ease",
          zIndex: 10,
        }}
        onMouseEnter={(e) => {
          (e.currentTarget as HTMLButtonElement).style.background = "rgba(0, 0, 0, 0.7)";
          (e.currentTarget as HTMLButtonElement).style.transform = "scale(1.1)";
        }}
        onMouseLeave={(e) => {
          (e.currentTarget as HTMLButtonElement).style.background = "rgba(0, 0, 0, 0.5)";
          (e.currentTarget as HTMLButtonElement).style.transform = "scale(1)";
        }}
      >
        {/* Camera flip icon */}
        <svg
          width="20"
          height="20"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M11 19H4a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h5" />
          <path d="M13 5h7a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2h-5" />
          <circle cx="12" cy="12" r="3" />
          <path d="m18 22-3-3 3-3" />
          <path d="m6 2 3 3-3 3" />
        </svg>
      </button>

      {/* Camera mode indicator */}
      <div
        style={{
          position: "absolute",
          bottom: 12,
          left: 12,
          padding: "4px 10px",
          borderRadius: 999,
          background: "rgba(0, 0, 0, 0.5)",
          backdropFilter: "blur(8px)",
          color: "#FFFFFF",
          fontSize: "0.68rem",
          fontWeight: 600,
          letterSpacing: "0.03em",
          zIndex: 10,
        }}
      >
        {facingMode === "user" ? "📷 Front" : "📷 Back"}
      </div>

      {/* Hidden canvas for frame extraction */}
      <canvas ref={canvasRef} className="hidden" aria-hidden="true" />
    </div>
  );
}

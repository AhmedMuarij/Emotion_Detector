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

  // Start / stop camera when isRunning changes
  useEffect(() => {
    if (!isRunning) {
      stopCamera();
      return;
    }

    const startCamera = async () => {
      onCameraStatusChange("requesting");
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: "user",
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
    };

    startCamera();

    return () => {
      stopCamera();
    };
  }, [isRunning, captureAndSend, frameIntervalMs, onCameraStatusChange, onError, stopCamera]);

  return (
    <div className="relative w-full">
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

      {/* Hidden canvas for frame extraction */}
      <canvas ref={canvasRef} className="hidden" aria-hidden="true" />
    </div>
  );
}

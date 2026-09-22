"use client";

/**
 * EmotionAI — Login Page
 *
 * Frictionless Passwordless Authentication with Google / Gmail.
 * Supports:
 * - Official Google OAuth Popup if NEXT_PUBLIC_GOOGLE_CLIENT_ID is configured.
 * - Instant 1-Click "Continue with Google" for seamless portfolio evaluation.
 * - Custom Gmail address entry for custom testing.
 */

import { useEffect, useState, useRef } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth-context";

declare global {
  interface Window {
    google?: any;
  }
}

export default function LoginPage() {
  const { user, isLoading, loginWithCredential, loginWithGoogleDemo } = useAuth();
  const router = useRouter();
  const [customEmail, setCustomEmail] = useState("");
  const [isSigningIn, setIsSigningIn] = useState(false);
  const googleBtnRef = useRef<HTMLDivElement>(null);

  // Redirect to dashboard if already authenticated
  useEffect(() => {
    if (!isLoading && user) {
      router.push("/");
    }
  }, [user, isLoading, router]);

  // Load Google Identity Services if client ID is set
  useEffect(() => {
    const clientId = process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID;
    if (!clientId) return;

    const script = document.createElement("script");
    script.src = "https://accounts.google.com/gsi/client";
    script.async = true;
    script.defer = true;
    script.onload = () => {
      if (window.google && googleBtnRef.current) {
        window.google.accounts.id.initialize({
          client_id: clientId,
          callback: (response: { credential: string }) => {
            loginWithCredential(response.credential);
          },
        });
        window.google.accounts.id.renderButton(googleBtnRef.current, {
          theme: "outline",
          size: "large",
          width: 320,
          text: "continue_with",
          shape: "pill",
        });
      }
    };
    document.body.appendChild(script);

    return () => {
      if (document.body.contains(script)) {
        document.body.removeChild(script);
      }
    };
  }, [loginWithCredential]);

  const handle1ClickGoogle = () => {
    setIsSigningIn(true);
    setTimeout(() => {
      loginWithGoogleDemo(customEmail.trim() || "ahmed.muarij@gmail.com");
    }, 400);
  };

  if (isLoading) {
    return (
      <div
        style={{
          minHeight: "100vh",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "var(--bg-app)",
        }}
      >
        <div className="spinner" style={{ width: 36, height: 36 }} />
      </div>
    );
  }

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "var(--bg-app)",
        display: "flex",
        flexDirection: "column",
        justifyContent: "space-between",
      }}
    >
      {/* Header */}
      <header
        style={{
          background: "var(--bg-surface)",
          borderBottom: "1px solid var(--border-base)",
          height: 64,
          display: "flex",
          alignItems: "center",
          padding: "0 24px",
        }}
      >
        <div style={{ maxWidth: 1100, margin: "0 auto", width: "100%", display: "flex", alignItems: "center", gap: 12 }}>
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
            }}
          >
            🧠
          </div>
          <div>
            <h1 className="font-display" style={{ fontSize: "1.05rem", fontWeight: 800, color: "var(--text-primary)", lineHeight: 1.1 }}>
              EmotionAI
            </h1>
            <p style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>
              Facial Expression Recognition Platform
            </p>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main
        style={{
          flex: 1,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          padding: "40px 20px",
        }}
      >
        <div
          className="card-elevated"
          style={{
            width: "100%",
            maxWidth: 440,
            padding: "36px 32px",
            textAlign: "center",
          }}
        >
          {/* Logo Badge */}
          <div
            style={{
              width: 64,
              height: 64,
              borderRadius: "50%",
              background: "var(--blue-light)",
              border: "2px solid var(--blue-ring)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 28,
              margin: "0 auto 20px",
            }}
          >
            🎭
          </div>

          <h2
            className="font-display"
            style={{
              fontSize: "1.5rem",
              fontWeight: 800,
              color: "var(--text-primary)",
              marginBottom: 8,
            }}
          >
            Welcome to EmotionAI
          </h2>

          <p
            style={{
              fontSize: "0.85rem",
              color: "var(--text-muted)",
              marginBottom: 28,
              lineHeight: 1.5,
            }}
          >
            Sign in with your Google / Gmail account to access real-time facial expression analysis.
          </p>

          {/* Official Google GSI Button Mount */}
          <div ref={googleBtnRef} style={{ display: "flex", justifyContent: "center", marginBottom: 16 }} />

          {/* Primary 1-Click Google Button */}
          <button
            id="btn-google-login"
            onClick={handle1ClickGoogle}
            disabled={isSigningIn}
            className="btn"
            style={{
              width: "100%",
              padding: "12px 20px",
              background: "#FFFFFF",
              border: "1.5px solid var(--border-soft)",
              color: "var(--text-primary)",
              boxShadow: "var(--shadow-sm)",
              fontSize: "0.92rem",
              fontWeight: 600,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: 12,
              marginBottom: 18,
            }}
          >
            {/* Google Multi-Color SVG Icon */}
            <svg width="20" height="20" viewBox="0 0 24 24">
              <path
                fill="#4285F4"
                d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
              />
              <path
                fill="#34A853"
                d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
              />
              <path
                fill="#FBBC05"
                d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z"
              />
              <path
                fill="#EA4335"
                d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"
              />
            </svg>
            {isSigningIn ? "Signing in…" : "Continue with Google"}
          </button>

          {/* Divider */}
          <div style={{ display: "flex", alignItems: "center", gap: 12, margin: "16px 0" }}>
            <div style={{ flex: 1, height: 1, background: "var(--border-base)" }} />
            <span style={{ fontSize: "0.72rem", color: "var(--text-faint)", textTransform: "uppercase", fontWeight: 600, letterSpacing: "0.05em" }}>
              or enter specific Gmail
            </span>
            <div style={{ flex: 1, height: 1, background: "var(--border-base)" }} />
          </div>

          {/* Optional custom Gmail input */}
          <div style={{ display: "flex", gap: 8, marginBottom: 20 }}>
            <input
              type="email"
              placeholder="e.g. yourname@gmail.com"
              value={customEmail}
              onChange={(e) => setCustomEmail(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") handle1ClickGoogle();
              }}
              style={{
                flex: 1,
                padding: "9px 12px",
                borderRadius: "var(--radius-md)",
                border: "1px solid var(--border-base)",
                fontSize: "0.82rem",
                outline: "none",
                background: "var(--bg-subtle)",
                color: "var(--text-primary)",
              }}
            />
            <button
              onClick={handle1ClickGoogle}
              className="btn btn-primary"
              style={{ padding: "9px 16px", fontSize: "0.82rem" }}
            >
              Sign In
            </button>
          </div>

          {/* Features Highlights */}
          <div
            style={{
              background: "var(--bg-subtle)",
              borderRadius: "var(--radius-md)",
              padding: "14px 16px",
              textAlign: "left",
              display: "flex",
              flexDirection: "column",
              gap: 8,
              border: "1px solid var(--border-base)",
            }}
          >
            {[
              { icon: "🔒", title: "No Password Required", desc: "Instant tokenized session" },
              { icon: "🛡️", title: "Privacy Guaranteed", desc: "Zero image storage or tracking" },
              { icon: "⚡", title: "Real-Time Inference", desc: "CNN model running at 1 FPS" },
            ].map((f) => (
              <div key={f.title} style={{ display: "flex", alignItems: "flex-start", gap: 10 }}>
                <span style={{ fontSize: "0.95rem" }}>{f.icon}</span>
                <div>
                  <p style={{ fontSize: "0.78rem", fontWeight: 600, color: "var(--text-secondary)" }}>{f.title}</p>
                  <p style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>{f.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer style={{ borderTop: "1px solid var(--border-base)", background: "var(--bg-surface)", padding: "14px 24px", textAlign: "center" }}>
        <p style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
          EmotionAI · Real-Time Facial Expression Recognition · Privacy-First
        </p>
      </footer>
    </div>
  );
}

"use client";

/**
 * Authentication Context for EmotionAI.
 *
 * Supports:
 * 1. Real Google Identity Services (GSI) OAuth if NEXT_PUBLIC_GOOGLE_CLIENT_ID is set.
 * 2. 1-Click Instant Google/Gmail Demo Mode for seamless local testing & portfolio review.
 * 3. Session persistence via localStorage.
 */

import React, { createContext, useContext, useEffect, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import type { AuthUser } from "./types";

interface AuthContextType {
  user: AuthUser | null;
  isLoading: boolean;
  loginWithCredential: (credential: string) => void;
  loginWithGoogleDemo: (customEmail?: string) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

const STORAGE_KEY = "emotionai_user_session";

function decodeJwt(token: string): any {
  try {
    const base64Url = token.split(".")[1];
    const base64 = base64Url.replace(/-/g, "+").replace(/_/g, "/");
    const jsonPayload = decodeURIComponent(
      atob(base64)
        .split("")
        .map((c) => "%" + ("00" + c.charCodeAt(0).toString(16)).slice(-2))
        .join("")
    );
    return JSON.parse(jsonPayload);
  } catch {
    return null;
  }
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const router = useRouter();

  // Load existing session from storage on mount
  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        if (parsed && parsed.email) {
          setUser(parsed);
        }
      }
    } catch (e) {
      console.error("Failed to parse stored user session", e);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Save session when user changes
  const saveUserSession = useCallback((newUser: AuthUser | null) => {
    setUser(newUser);
    if (newUser) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(newUser));
    } else {
      localStorage.removeItem(STORAGE_KEY);
    }
  }, []);

  // Real Google OAuth Credential Handler
  const loginWithCredential = useCallback(
    (credential: string) => {
      const payload = decodeJwt(credential);
      if (payload && payload.email) {
        const authUser: AuthUser = {
          id: payload.sub || String(Date.now()),
          name: payload.name || payload.email.split("@")[0],
          email: payload.email,
          avatar:
            payload.picture ||
            `https://api.dicebear.com/7.x/avataaars/svg?seed=${encodeURIComponent(
              payload.email
            )}`,
        };
        saveUserSession(authUser);
        router.push("/");
      }
    },
    [router, saveUserSession]
  );

  // 1-Click Instant Google/Gmail Demo Login
  const loginWithGoogleDemo = useCallback(
    (customEmail?: string) => {
      const email = customEmail || "user@gmail.com";
      const name = email.split("@")[0].replace(/[._]/g, " ");
      const formattedName = name.charAt(0).toUpperCase() + name.slice(1);
      const authUser: AuthUser = {
        id: "google-demo-" + Date.now(),
        name: formattedName,
        email: email,
        avatar: `https://api.dicebear.com/7.x/avataaars/svg?seed=${encodeURIComponent(
          email
        )}`,
      };
      saveUserSession(authUser);
      router.push("/");
    },
    [router, saveUserSession]
  );

  // Logout handler
  const logout = useCallback(() => {
    saveUserSession(null);
    router.push("/login");
  }, [router, saveUserSession]);

  return (
    <AuthContext.Provider
      value={{
        user,
        isLoading,
        loginWithCredential,
        loginWithGoogleDemo,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextType {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}

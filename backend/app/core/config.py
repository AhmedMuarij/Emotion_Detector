"""Environment-driven configuration for the EmotionAI backend.

All values can be overridden via environment variables or a .env file.
Do not put secrets directly in this file.

Usage:
    from app.core.config import settings
    print(settings.model_path)
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# Repo root: backend/app/core/config.py → parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Comma-separated list of allowed CORS origins.
    # Example: "http://localhost:3000,https://emotionai.vercel.app"
    cors_origins: str = "http://localhost:3000"

    # ── Model ─────────────────────────────────────────────────────────────────
    model_path: Path = _REPO_ROOT / "models" / "emotion_cnn.keras"
    model_metadata_path: Path = _REPO_ROOT / "models" / "metadata.json"

    # ── Upload limits ─────────────────────────────────────────────────────────
    # Maximum accepted image upload size in bytes (default 5 MB)
    max_upload_bytes: int = 5 * 1024 * 1024  # 5 MB

    # ── Face detection ────────────────────────────────────────────────────────
    # Haar cascade XML path (None → use OpenCV bundled default)
    haar_cascade_path: str | None = None

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def cors_origins_list(self) -> list[str]:
        """Parse the comma-separated CORS_ORIGINS string into a list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


# Module-level singleton — import this everywhere.
settings = Settings()

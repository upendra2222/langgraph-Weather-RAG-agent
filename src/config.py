from __future__ import annotations

import os
from dotenv import load_dotenv
from dataclasses import dataclass


@dataclass
class Settings:
    """Application configuration loaded from environment variables."""

    openweather_api_key: str | None
    groq_api_key: str | None
    langsmith_api_key: str | None
    langsmith_project: str | None

    @classmethod
    def load(cls) -> "Settings":
        # Load .env file if present (no-op if not)
        try:
            load_dotenv()
        except Exception:
            # If python-dotenv isn't available or load fails, continue using os.environ
            pass

        return cls(
            openweather_api_key=os.getenv("OPENWEATHER_API_KEY"),
            groq_api_key=os.getenv("GROQ_API_KEY"),
            langsmith_api_key=os.getenv("LANGSMITH_API_KEY"),
            langsmith_project=os.getenv("LANGSMITH_PROJECT"),
        )


settings = Settings.load()



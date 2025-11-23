from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import requests

from .config import settings


class WeatherAPIError(Exception):
    """Raised when the weather API call fails."""


@dataclass
class WeatherClient:
    """Simple OpenWeatherMap client."""

    base_url: str = "https://api.openweathermap.org/data/2.5/weather"

    def get_weather(self, location: str) -> Dict[str, Any]:
        if not settings.openweather_api_key:
            raise WeatherAPIError("OPENWEATHER_API_KEY is not set.")

        params = {
            "q": location,
            "appid": settings.openweather_api_key,
            "units": "metric",
        }
        response = requests.get(self.base_url, params=params, timeout=10)
        if response.status_code != 200:
            raise WeatherAPIError(
                f"Weather API error {response.status_code}: {response.text}"
            )
        return response.json()



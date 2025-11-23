from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.weather import WeatherAPIError, WeatherClient


def test_weather_success():
    client = WeatherClient()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"weather": "ok"}

    with patch("src.weather.requests.get", return_value=mock_response):
        with patch("src.weather.settings") as mock_settings:
            mock_settings.openweather_api_key = "test"
            result = client.get_weather("London")
            assert result == {"weather": "ok"}


def test_weather_missing_api_key():
    client = WeatherClient()
    with patch("src.weather.settings") as mock_settings:
        mock_settings.openweather_api_key = None
        try:
            client.get_weather("London")
        except WeatherAPIError as e:
            assert "OPENWEATHER_API_KEY is not set" in str(e)



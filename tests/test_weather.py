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
            # Should not reach here
            assert False, "WeatherAPIError should have been raised"
        except WeatherAPIError as e:
            assert "OPENWEATHER_API_KEY is not set" in str(e)


def test_weather_api_failure():
    # 🌟 NEW TEST: Ensures WeatherAPIError is raised on non-200 status.
    client = WeatherClient()
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.text = "City not found"

    with patch("src.weather.requests.get", return_value=mock_response):
        with patch("src.weather.settings") as mock_settings:
            mock_settings.openweather_api_key = "test"
            try:
                client.get_weather("InvalidCity")
                assert False, "WeatherAPIError should have been raised"
            except WeatherAPIError as e:
                assert "Weather API error 404: City not found" in str(e)
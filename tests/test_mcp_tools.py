"""Tests for MCP tool wrappers.

Tests the weather and currency tool functions.
Note: These tests require valid API keys to test live functionality.
Tests with missing keys verify graceful error handling.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Ensure project root is on the path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# ---- Weather Server Tests ----

class TestWeatherServer:
    """Tests for the weather MCP server tools."""

    def test_format_weather_output(self):
        """Test weather data formatting."""
        from src.mcp_servers.weather_server import _format_weather

        mock_data = {
            "name": "Singapore",
            "sys": {"country": "SG"},
            "weather": [{"main": "Clouds", "description": "scattered clouds"}],
            "main": {
                "temp": 30.5,
                "feels_like": 35.2,
                "humidity": 78,
                "pressure": 1010,
            },
            "wind": {"speed": 3.5},
            "visibility": 10000,
        }

        result = _format_weather(mock_data)
        assert "Singapore" in result
        assert "30.5" in result
        assert "scattered clouds" in result
        assert "78%" in result

    def test_format_forecast_output(self):
        """Test forecast data formatting."""
        from src.mcp_servers.weather_server import _format_forecast

        mock_data = {
            "city": {"name": "Singapore", "country": "SG"},
            "list": [
                {
                    "dt": 1700000000,
                    "weather": [{"main": "Rain", "description": "light rain"}],
                    "main": {"temp": 28.0, "humidity": 85},
                    "pop": 0.75,
                    "rain": {"3h": 1.5},
                }
            ],
        }

        result = _format_forecast(mock_data)
        assert "Singapore" in result
        assert "light rain" in result
        assert "75%" in result


# ---- Currency Server Tests ----

class TestCurrencyServer:
    """Tests for the currency MCP server tools."""

    @pytest.mark.asyncio
    async def test_missing_api_key_handling(self):
        """Test that missing API key raises a clear error."""
        from src.mcp_servers.currency_server import _get_api_key

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="EXCHANGERATE_API_KEY"):
                _get_api_key()


# ---- Integration Test Markers ----

@pytest.mark.skipif(
    not os.environ.get("OPENWEATHER_API_KEY"),
    reason="OPENWEATHER_API_KEY not set",
)
class TestWeatherLive:
    """Live tests requiring a valid OpenWeatherMap API key."""

    @pytest.mark.asyncio
    async def test_get_current_weather_live(self):
        from src.mcp_servers.weather_server import get_current_weather

        result = await get_current_weather(city="Singapore")
        assert "Singapore" in result
        assert "Temperature" in result

    @pytest.mark.asyncio
    async def test_get_forecast_live(self):
        from src.mcp_servers.weather_server import get_weather_forecast

        result = await get_weather_forecast(city="Singapore", days=2)
        assert "Singapore" in result
        assert "Forecast" in result


@pytest.mark.skipif(
    not os.environ.get("EXCHANGERATE_API_KEY"),
    reason="EXCHANGERATE_API_KEY not set",
)
class TestCurrencyLive:
    """Live tests requiring a valid ExchangeRate API key."""

    @pytest.mark.asyncio
    async def test_convert_currency_live(self):
        from src.mcp_servers.currency_server import convert_currency

        result = await convert_currency(
            amount=100.0, from_currency="USD", to_currency="SGD"
        )
        assert "USD" in result
        assert "SGD" in result
        assert "Exchange Rate" in result

    @pytest.mark.asyncio
    async def test_get_exchange_rate_live(self):
        from src.mcp_servers.currency_server import get_exchange_rate

        result = await get_exchange_rate(
            base_currency="SGD", target_currencies="USD,INR"
        )
        assert "SGD" in result
        assert "USD" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

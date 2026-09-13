"""MCP Weather Server.

Provides weather information via the OpenWeatherMap API,
exposed as MCP tools using the FastMCP framework.

Usage (standalone):
    python -m src.mcp_servers.weather_server
"""

import os
import json
from datetime import datetime, timezone

import httpx
from mcp.server.fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP("weather-server")

OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"


def _get_api_key() -> str:
    """Get the OpenWeatherMap API key from environment."""
    key = os.environ.get("OPENWEATHER_API_KEY", "")
    if not key:
        raise ValueError(
            "OPENWEATHER_API_KEY environment variable is not set. "
            "Get a free key at https://openweathermap.org/api"
        )
    return key


def _format_weather(data: dict) -> str:
    """Format weather API response into human-readable text."""
    city = data.get("name", "Unknown")
    country = data.get("sys", {}).get("country", "")
    weather = data.get("weather", [{}])[0]
    main = data.get("main", {})
    wind = data.get("wind", {})

    return (
        f"Current Weather in {city}, {country}:\n"
        f"  Condition: {weather.get('main', 'N/A')} - {weather.get('description', 'N/A')}\n"
        f"  Temperature: {main.get('temp', 'N/A')}\u00b0C "
        f"(Feels like: {main.get('feels_like', 'N/A')}\u00b0C)\n"
        f"  Humidity: {main.get('humidity', 'N/A')}%\n"
        f"  Wind: {wind.get('speed', 'N/A')} m/s\n"
        f"  Pressure: {main.get('pressure', 'N/A')} hPa\n"
        f"  Visibility: {data.get('visibility', 'N/A')} metres"
    )


def _format_forecast(data: dict) -> str:
    """Format 5-day forecast API response into human-readable text."""
    city = data.get("city", {}).get("name", "Unknown")
    country = data.get("city", {}).get("country", "")
    forecasts = data.get("list", [])

    lines = [f"Weather Forecast for {city}, {country}:\n"]

    # Group by day
    current_date = ""
    for entry in forecasts:
        dt = datetime.fromtimestamp(entry["dt"], tz=timezone.utc)
        date_str = dt.strftime("%A, %B %d")
        time_str = dt.strftime("%I:%M %p")

        if date_str != current_date:
            current_date = date_str
            lines.append(f"\n--- {date_str} ---")

        weather = entry.get("weather", [{}])[0]
        main = entry.get("main", {})
        rain = entry.get("rain", {}).get("3h", 0)
        pop = entry.get("pop", 0) * 100  # Probability of precipitation

        rain_info = f", Rain: {rain}mm" if rain > 0 else ""
        lines.append(
            f"  {time_str}: {weather.get('description', 'N/A')}, "
            f"{main.get('temp', 'N/A')}\u00b0C, "
            f"Humidity: {main.get('humidity', 'N/A')}%, "
            f"Rain chance: {pop:.0f}%{rain_info}"
        )

    return "\n".join(lines)


@mcp.tool()
async def get_current_weather(city: str = "Singapore") -> str:
    """Get the current weather conditions for a city.

    Args:
        city: The city name (default: Singapore).

    Returns:
        A formatted string with current weather information.
    """
    api_key = _get_api_key()
    url = f"{OPENWEATHER_BASE_URL}/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric",
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params, timeout=10.0)
        response.raise_for_status()
        data = response.json()

    return _format_weather(data)


@mcp.tool()
async def get_weather_forecast(
    city: str = "Singapore", days: int = 3
) -> str:
    """Get the weather forecast for a city (up to 5 days).

    Uses 3-hour interval forecasts from OpenWeatherMap.

    Args:
        city: The city name (default: Singapore).
        days: Number of days to forecast (1-5, default: 3).

    Returns:
        A formatted string with the weather forecast.
    """
    api_key = _get_api_key()
    days = max(1, min(days, 5))
    cnt = days * 8  # 8 entries per day (3-hour intervals)

    url = f"{OPENWEATHER_BASE_URL}/forecast"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric",
        "cnt": cnt,
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params, timeout=10.0)
        response.raise_for_status()
        data = response.json()

    return _format_forecast(data)


if __name__ == "__main__":
    mcp.run(transport="stdio")

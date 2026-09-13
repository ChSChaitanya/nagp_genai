"""MCP Currency Conversion Server.

Provides currency conversion via the ExchangeRate API,
exposed as MCP tools using the FastMCP framework.

Usage (standalone):
    python -m src.mcp_servers.currency_server
"""

import os
from typing import Optional

import httpx
from mcp.server.fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP("currency-server")

EXCHANGERATE_BASE_URL = "https://v6.exchangerate-api.com/v6"


def _get_api_key() -> str:
    """Get the ExchangeRate API key from environment."""
    key = os.environ.get("EXCHANGERATE_API_KEY", "")
    if not key:
        raise ValueError(
            "EXCHANGERATE_API_KEY environment variable is not set. "
            "Get a free key at https://www.exchangerate-api.com/"
        )
    return key


@mcp.tool()
async def convert_currency(
    amount: float,
    from_currency: str,
    to_currency: str,
) -> str:
    """Convert an amount from one currency to another.

    Uses real-time exchange rates from ExchangeRate API.

    Args:
        amount: The amount to convert.
        from_currency: Source currency code (e.g., 'USD', 'INR', 'SGD').
        to_currency: Target currency code (e.g., 'SGD', 'USD', 'INR').

    Returns:
        A formatted string showing the conversion result.
    """
    api_key = _get_api_key()
    from_code = from_currency.upper().strip()
    to_code = to_currency.upper().strip()

    url = f"{EXCHANGERATE_BASE_URL}/{api_key}/pair/{from_code}/{to_code}/{amount}"

    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=10.0)
        response.raise_for_status()
        data = response.json()

    if data.get("result") != "success":
        error_type = data.get("error-type", "unknown")
        return f"Currency conversion failed: {error_type}"

    rate = data.get("conversion_rate", 0)
    converted = data.get("conversion_result", 0)

    return (
        f"Currency Conversion Result:\n"
        f"  {amount:,.2f} {from_code} = {converted:,.2f} {to_code}\n"
        f"  Exchange Rate: 1 {from_code} = {rate:.4f} {to_code}\n"
        f"  Source: ExchangeRate API (real-time)"
    )


@mcp.tool()
async def get_exchange_rate(
    base_currency: str = "SGD",
    target_currencies: Optional[str] = None,
) -> str:
    """Get current exchange rates for a base currency.

    Args:
        base_currency: The base currency code (default: SGD).
        target_currencies: Comma-separated list of target currencies
            (e.g., 'USD,INR,EUR'). If not provided, shows common
            travel currencies.

    Returns:
        A formatted string with exchange rates.
    """
    api_key = _get_api_key()
    base_code = base_currency.upper().strip()

    url = f"{EXCHANGERATE_BASE_URL}/{api_key}/latest/{base_code}"

    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=10.0)
        response.raise_for_status()
        data = response.json()

    if data.get("result") != "success":
        error_type = data.get("error-type", "unknown")
        return f"Failed to get exchange rates: {error_type}"

    rates = data.get("conversion_rates", {})

    # Determine which currencies to show
    if target_currencies:
        targets = [
            c.strip().upper() for c in target_currencies.split(",")
        ]
    else:
        # Default: common travel currencies
        targets = ["USD", "EUR", "GBP", "INR", "JPY", "AUD", "CNY", "MYR", "THB", "KRW"]

    lines = [f"Exchange Rates (Base: 1 {base_code}):\n"]
    for currency in targets:
        if currency in rates and currency != base_code:
            lines.append(f"  1 {base_code} = {rates[currency]:.4f} {currency}")

    lines.append(f"\n  Source: ExchangeRate API (real-time)")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run(transport="stdio")

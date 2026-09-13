"""Shared utility functions for the AI Travel Planning Assistant."""

import logging
import sys
from pathlib import Path


def setup_logging(level: int = logging.INFO) -> None:
    """Configure application-wide logging.

    Args:
        level: Logging level (default: INFO).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def format_sources_for_display(sources: str) -> list[dict]:
    """Parse a sources string into structured data for UI display.

    Args:
        sources: Semicolon-separated source entries,
                 each like 'Title (URL)'.

    Returns:
        A list of dicts with 'title' and optional 'url' keys.
    """
    if not sources:
        return []

    result = []
    for entry in sources.split(";"):
        entry = entry.strip()
        if not entry:
            continue

        if "(" in entry and entry.endswith(")"):
            title, _, url = entry.rpartition("(")
            result.append({
                "title": title.strip(),
                "url": url.rstrip(")").strip(),
            })
        else:
            result.append({"title": entry, "url": ""})

    return result


def format_tool_calls_for_display(tool_calls: list[dict]) -> str:
    """Format tool call information for the Streamlit UI.

    Args:
        tool_calls: List of tool call dicts from the agent.

    Returns:
        A formatted Markdown string.
    """
    if not tool_calls:
        return ""

    lines = ["**MCP Tools Used:**\n"]
    for tc in tool_calls:
        tool_name = tc.get("tool", "unknown")
        tool_input = tc.get("input", {})
        lines.append(f"- **{tool_name}** with input: `{tool_input}`")

    return "\n".join(lines)


SAMPLE_QUESTIONS = [
    # RAG-only questions
    "What are the must-visit attractions in Singapore?",
    "Which neighbourhoods are suitable for cultural experiences?",
    "How can a tourist travel around Singapore?",
    "Suggest activities for a family with children.",
    "What indoor attractions can I visit on a rainy day?",
    "What are the best hawker centres in Singapore?",
    # MCP-only questions
    "What is the weather in Singapore right now?",
    "What is the forecast for the next 3 days?",
    "Convert INR 50,000 to SGD.",
    "How much is 200 SGD in USD?",
    # Combined RAG + MCP questions
    "Create a three-day Singapore itinerary for next week and adjust it according to the weather forecast.",
    "I have a budget of INR 60,000. Convert it to SGD and suggest a three-day itinerary.",
    "Suggest outdoor attractions and replace them with indoor options if rain is expected.",
    "Plan a family trip and include the latest weather forecast.",
]

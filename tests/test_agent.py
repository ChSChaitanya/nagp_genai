"""Tests for the Travel Planning Agent.

Tests agent initialization, RAG context retrieval, and the
full query pipeline. Some tests require API keys.
"""

import os
import sys
from pathlib import Path

import pytest

# Ensure project root is on the path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.helpers import (
    format_sources_for_display,
    format_tool_calls_for_display,
    SAMPLE_QUESTIONS,
)


# ---- Helper Tests ----

class TestHelpers:
    """Tests for utility helper functions."""

    def test_format_sources_with_url(self):
        sources = "Singapore Guide (https://example.com/sg); Tips (https://tips.com)"
        result = format_sources_for_display(sources)
        assert len(result) == 2
        assert result[0]["title"] == "Singapore Guide"
        assert result[0]["url"] == "https://example.com/sg"

    def test_format_sources_without_url(self):
        sources = "Singapore Guide; Tips"
        result = format_sources_for_display(sources)
        assert len(result) == 2
        assert result[0]["url"] == ""

    def test_format_empty_sources(self):
        assert format_sources_for_display("") == []

    def test_format_tool_calls(self):
        tool_calls = [
            {
                "tool": "get_current_weather",
                "input": {"city": "Singapore"},
                "output": "Sunny, 31C",
            }
        ]
        result = format_tool_calls_for_display(tool_calls)
        assert "get_current_weather" in result
        assert "Singapore" in result

    def test_format_empty_tool_calls(self):
        assert format_tool_calls_for_display([]) == ""

    def test_sample_questions_defined(self):
        assert len(SAMPLE_QUESTIONS) >= 10
        for q in SAMPLE_QUESTIONS:
            assert isinstance(q, str)
            assert len(q) > 10


# ---- Agent Integration Tests ----

@pytest.mark.skipif(
    not os.environ.get("DATABRICKS_TOKEN"),
    reason="DATABRICKS_TOKEN not set",
)
class TestAgentIntegration:
    """Integration tests requiring Databricks token."""

    @pytest.fixture(autouse=True)
    def setup_agent(self):
        """Create and initialize the agent for each test."""
        from src.agents.travel_agent import TravelPlanningAgent

        self.agent = TravelPlanningAgent()
        self.agent.initialize()

    def test_rag_query(self):
        """Test a destination knowledge query (RAG only)."""
        result = self.agent.query(
            "What are the must-visit attractions in Singapore?"
        )
        assert "answer" in result
        assert len(result["answer"]) > 50
        assert "sources" in result

    def test_context_retrieval(self):
        """Test that RAG context is retrieved for destination queries."""
        context, sources = self.agent._retrieve_context(
            "best hawker centres in Singapore"
        )
        assert len(context) > 0
        assert "hawker" in context.lower() or "food" in context.lower()

    def test_conversation_history(self):
        """Test multi-turn conversation context retention."""
        self.agent.query("Tell me about Chinatown in Singapore.")
        assert self.agent.conversation_length == 2  # user + assistant

        result = self.agent.query("What food can I try there?")
        # The agent should use conversation context
        assert len(result["answer"]) > 20
        assert self.agent.conversation_length == 4

    def test_reset_conversation(self):
        """Test conversation reset."""
        self.agent.query("Hello")
        assert self.agent.conversation_length > 0
        self.agent.reset_conversation()
        assert self.agent.conversation_length == 0

    def test_missing_knowledge_handling(self):
        """Test that the agent handles queries outside the knowledge base."""
        result = self.agent.query(
            "What are the best hotels in Antarctica?"
        )
        # Should indicate lack of knowledge rather than fabricating
        assert len(result["answer"]) > 20


@pytest.mark.skipif(
    not all([
        os.environ.get("DATABRICKS_TOKEN"),
        os.environ.get("OPENWEATHER_API_KEY"),
        os.environ.get("EXCHANGERATE_API_KEY"),
    ]),
    reason="All API keys required for full integration tests",
)
class TestFullIntegration:
    """Full integration tests requiring all API keys."""

    @pytest.fixture(autouse=True)
    def setup_agent(self):
        from src.agents.travel_agent import TravelPlanningAgent

        self.agent = TravelPlanningAgent()
        self.agent.initialize()

    def test_combined_rag_and_mcp(self):
        """Test the primary combined scenario from the assignment."""
        result = self.agent.query(
            "Create a three-day Singapore itinerary for next week "
            "and adjust it according to the weather forecast."
        )
        assert len(result["answer"]) > 100
        # Should have used weather tool
        tool_names = [tc["tool"] for tc in result.get("tool_calls", [])]
        assert any("weather" in t or "forecast" in t for t in tool_names)

    def test_budget_conversion_with_itinerary(self):
        """Test budget conversion combined with itinerary."""
        result = self.agent.query(
            "I have a budget of INR 60,000. Convert it to SGD "
            "and suggest a three-day itinerary."
        )
        assert len(result["answer"]) > 100
        tool_names = [tc["tool"] for tc in result.get("tool_calls", [])]
        assert any("currency" in t or "convert" in t for t in tool_names)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

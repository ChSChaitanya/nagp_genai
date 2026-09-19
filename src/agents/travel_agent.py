"""Travel Planning Agent - Core orchestration module.

Combines RAG retrieval with MCP tool execution using LangChain.
Supports multi-turn conversation with context retention.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Optional

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent, create_structured_chat_agent
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from src.rag.document_loader import load_knowledge_base
from src.rag.chunker import chunk_documents
from src.rag.vector_store import TravelVectorStore
from src.prompts.templates import SYSTEM_PROMPT, RAG_CONTEXT_TEMPLATE
from config.settings import get_settings, PROJECT_ROOT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MCP-compatible tool wrappers (LangChain @tool decorator)
# These wrap the MCP server functionality as LangChain tools so the agent
# can call them directly. In production, these would connect to MCP servers
# via the MCP client protocol.
# ---------------------------------------------------------------------------


def _create_weather_tools():
    """Create weather-related LangChain tools backed by MCP server logic."""
    import httpx

    settings = get_settings()

    @tool
    def get_current_weather(city: str = "Singapore") -> str:
        """Get the current weather conditions for a city.
        Use this tool when the user asks about current weather,
        temperature, or conditions in a city.

        Args:
            city: The city name (default: Singapore).
        """
        api_key = settings.openweather_api_key
        if not api_key:
            return (
                "Weather service unavailable: OPENWEATHER_API_KEY not configured. "
                "Unable to retrieve current weather data."
            )

        try:
            url = "https://api.openweathermap.org/data/2.5/weather"
            params = {"q": city, "appid": api_key, "units": "metric"}
            response = httpx.get(url, params=params, timeout=10.0)
            response.raise_for_status()
            data = response.json()

            weather = data.get("weather", [{}])[0]
            main = data.get("main", {})
            wind = data.get("wind", {})

            return (
                f"[Source: OpenWeatherMap - Live Data]\n"
                f"Current Weather in {data.get('name', city)}, "
                f"{data.get('sys', {}).get('country', '')}:\n"
                f"  Condition: {weather.get('main', 'N/A')} - "
                f"{weather.get('description', 'N/A')}\n"
                f"  Temperature: {main.get('temp', 'N/A')}\u00b0C "
                f"(Feels like: {main.get('feels_like', 'N/A')}\u00b0C)\n"
                f"  Humidity: {main.get('humidity', 'N/A')}%\n"
                f"  Wind: {wind.get('speed', 'N/A')} m/s\n"
                f"  Visibility: {data.get('visibility', 'N/A')} metres"
            )
        except httpx.HTTPStatusError as e:
            return f"Weather API error: {e.response.status_code} - {e.response.text}"
        except Exception as e:
            return f"Weather service error: {str(e)}"

    @tool
    def get_weather_forecast(city: str = "Singapore", days: int = 3) -> str:
        """Get the weather forecast for a city for the next few days.
        Use this tool when the user asks about future weather,
        forecasts, or whether to plan indoor/outdoor activities.

        Args:
            city: The city name (default: Singapore).
            days: Number of days to forecast (1-5, default: 3).
        """
        api_key = settings.openweather_api_key
        if not api_key:
            return (
                "Weather service unavailable: OPENWEATHER_API_KEY not configured. "
                "Unable to retrieve forecast data."
            )

        try:
            days = max(1, min(days, 5))
            cnt = days * 8
            url = "https://api.openweathermap.org/data/2.5/forecast"
            params = {
                "q": city, "appid": api_key,
                "units": "metric", "cnt": cnt,
            }
            response = httpx.get(url, params=params, timeout=10.0)
            response.raise_for_status()
            data = response.json()

            city_name = data.get("city", {}).get("name", city)
            country = data.get("city", {}).get("country", "")
            forecasts = data.get("list", [])

            from datetime import datetime, timezone

            lines = [
                f"[Source: OpenWeatherMap - Live Data]",
                f"Weather Forecast for {city_name}, {country}:\n",
            ]
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
                pop = entry.get("pop", 0) * 100
                rain = entry.get("rain", {}).get("3h", 0)
                rain_info = f", Rain: {rain}mm" if rain > 0 else ""

                lines.append(
                    f"  {time_str}: {weather.get('description', 'N/A')}, "
                    f"{main.get('temp', 'N/A')}\u00b0C, "
                    f"Humidity: {main.get('humidity', 'N/A')}%, "
                    f"Rain chance: {pop:.0f}%{rain_info}"
                )

            return "\n".join(lines)
        except httpx.HTTPStatusError as e:
            return f"Forecast API error: {e.response.status_code} - {e.response.text}"
        except Exception as e:
            return f"Forecast service error: {str(e)}"

    return [get_current_weather, get_weather_forecast]


def _create_currency_tools():
    """Create currency-related LangChain tools backed by MCP server logic."""
    import httpx

    settings = get_settings()

    @tool
    def convert_currency(
        amount: float, from_currency: str, to_currency: str
    ) -> str:
        """Convert an amount from one currency to another using live exchange rates.
        Use this tool when the user asks to convert money between currencies
        or wants to know how much their budget is worth in another currency.

        Args:
            amount: The amount to convert.
            from_currency: Source currency code (e.g., 'USD', 'INR', 'SGD').
            to_currency: Target currency code (e.g., 'SGD', 'USD', 'INR').
        """
        api_key = settings.exchangerate_api_key
        if not api_key:
            return (
                "Currency service unavailable: EXCHANGERATE_API_KEY not configured. "
                "Unable to perform conversion."
            )

        try:
            from_code = from_currency.upper().strip()
            to_code = to_currency.upper().strip()
            url = (
                f"https://v6.exchangerate-api.com/v6/{api_key}"
                f"/pair/{from_code}/{to_code}/{amount}"
            )
            response = httpx.get(url, timeout=10.0)
            response.raise_for_status()
            data = response.json()

            if data.get("result") != "success":
                return f"Conversion failed: {data.get('error-type', 'unknown')}"

            rate = data.get("conversion_rate", 0)
            converted = data.get("conversion_result", 0)

            return (
                f"[Source: ExchangeRate API - Live Data]\n"
                f"Currency Conversion Result:\n"
                f"  {amount:,.2f} {from_code} = {converted:,.2f} {to_code}\n"
                f"  Exchange Rate: 1 {from_code} = {rate:.4f} {to_code}"
            )
        except Exception as e:
            return f"Currency conversion error: {str(e)}"

    @tool
    def get_exchange_rate(
        base_currency: str = "SGD",
        target_currencies: str = "USD,INR,EUR,GBP,JPY",
    ) -> str:
        """Get current exchange rates for a base currency against multiple targets.
        Use this tool when the user wants to see exchange rates or compare
        currency values.

        Args:
            base_currency: The base currency code (default: SGD).
            target_currencies: Comma-separated target currency codes.
        """
        api_key = settings.exchangerate_api_key
        if not api_key:
            return (
                "Currency service unavailable: EXCHANGERATE_API_KEY not configured. "
                "Unable to retrieve exchange rates."
            )

        try:
            base_code = base_currency.upper().strip()
            url = (
                f"https://v6.exchangerate-api.com/v6/{api_key}"
                f"/latest/{base_code}"
            )
            response = httpx.get(url, timeout=10.0)
            response.raise_for_status()
            data = response.json()

            if data.get("result") != "success":
                return f"Rate lookup failed: {data.get('error-type', 'unknown')}"

            rates = data.get("conversion_rates", {})
            targets = [c.strip().upper() for c in target_currencies.split(",")]

            lines = [
                f"[Source: ExchangeRate API - Live Data]",
                f"Exchange Rates (Base: 1 {base_code}):\n",
            ]
            for cur in targets:
                if cur in rates and cur != base_code:
                    lines.append(f"  1 {base_code} = {rates[cur]:.4f} {cur}")

            return "\n".join(lines)
        except Exception as e:
            return f"Exchange rate error: {str(e)}"

    return [convert_currency, get_exchange_rate]


class TravelPlanningAgent:
    """Orchestrates RAG retrieval and MCP tool execution for travel planning."""

    def __init__(self, settings=None):
        """Initialize the travel planning agent.

        Args:
            settings: Application settings. Uses defaults if not provided.
        """
        self.settings = settings or get_settings()
        self.vector_store: Optional[TravelVectorStore] = None
        self.agent_executor: Optional[AgentExecutor] = None
        self.chat_history: List = []
        self._initialized = False

    def initialize(self) -> None:
        """Set up the RAG pipeline and agent. Call once before querying."""
        if self._initialized:
            return

        logger.info("Initializing Travel Planning Agent...")

        # --- RAG Setup (Local HuggingFace Embeddings) ---
        self.vector_store = TravelVectorStore(
            embedding_model=self.settings.embedding_model,
            persist_path=self.settings.vector_store_abs_path,
        )

        # Try loading persisted index first
        if not self.vector_store.load_index():
            logger.info("Building new vector index from knowledge base...")
            documents = load_knowledge_base(
                self.settings.knowledge_base_abs_path
            )
            chunks = chunk_documents(
                documents,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )
            self.vector_store.build_index(chunks)
            logger.info("Vector index built with %d chunks.", len(chunks))

        # --- LLM Setup (supports Gemini and OpenAI) ---
        from langchain_openai import ChatOpenAI

        provider = self.settings.llm_provider.lower().strip()
        if provider == "openai":
            logger.info("Using OpenAI provider: %s", self.settings.openai_model)
            llm = ChatOpenAI(
                model=self.settings.openai_model,
                api_key=self.settings.openai_api_key,
                temperature=0.3,
            )
        else:
            logger.info("Using Gemini provider (via OpenAI-compat): %s", self.settings.gemini_model)
            llm = ChatOpenAI(
                model=self.settings.gemini_model,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=self.settings.google_api_key,
                temperature=0.3,
            )

        # --- Tools ---
        tools = []
        tools.extend(_create_weather_tools())
        tools.extend(_create_currency_tools())

        # --- Agent Prompt & Agent ---
        system_prompt = SYSTEM_PROMPT.format(
            destination=self.settings.destination_city
        )

        if provider == "openai":
            # OpenAI: native tool calling works perfectly
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            agent = create_tool_calling_agent(llm, tools, prompt)
        else:
            # Gemini: use structured chat agent (text-based JSON tool calls)
            # to avoid Gemini's thought_signature requirement in native
            # function calling.
            structured_suffix = (
                "\n\n## Available Tools\n\n{tools}\n\n"
                "## Tool Invocation Format\n\n"
                "To use a tool, respond with a markdown JSON code block "
                "containing \"action\" (tool name) and \"action_input\" "
                "(tool arguments).\n\n"
                "Valid \"action\" values: \"Final Answer\" or {tool_names}\n\n"
                "```json\n"
                '{{\n  "action": "<tool_name>",\n'
                '  "action_input": {{"arg1": "value1"}}\n'
                "}}\n```\n\n"
                "When you have the final answer:\n\n"
                "```json\n"
                '{{\n  "action": "Final Answer",\n'
                '  "action_input": "<your complete response>"\n'
                "}}\n```\n\n"
                "Follow this loop:\n"
                "Thought: reason about what to do\n"
                "Action:\n```json\n<json blob>\n```\n"
                "Observation: tool result\n"
                "... (repeat until done)\n\n"
                "ALWAYS respond with exactly one valid JSON blob per turn."
            )

            full_system = system_prompt + structured_suffix
            prompt = ChatPromptTemplate.from_messages([
                ("system", full_system),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}\n\n{agent_scratchpad}"),
            ])
            agent = create_structured_chat_agent(llm, tools, prompt)

        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10,
            return_intermediate_steps=True,
        )

        self._initialized = True
        logger.info("Travel Planning Agent initialized successfully.")

    def _retrieve_context(self, query: str) -> tuple[str, str]:
        """Retrieve relevant knowledge base context for a query.

        Returns:
            A tuple of (context_text, sources_text).
        """
        if self.vector_store is None:
            return "", ""

        results = self.vector_store.similarity_search(
            query, k=self.settings.retriever_top_k
        )

        if not results:
            return "No relevant information found in the knowledge base.", ""

        context_parts = []
        sources = set()
        for doc in results:
            title = doc.metadata.get("source_title", "Unknown")
            url = doc.metadata.get("source_url", "")
            section = doc.metadata.get("section", "")

            header = f"[From: {title}"
            if section:
                header += f" > {section}"
            header += "]"

            context_parts.append(f"{header}\n{doc.page_content}")

            source_entry = title
            if url:
                source_entry += f" ({url})"
            sources.add(source_entry)

        context_text = "\n\n---\n\n".join(context_parts)
        sources_text = "; ".join(sorted(sources))

        return context_text, sources_text

    def query(self, user_input: str) -> dict:
        """Process a user query through the agent.

        Args:
            user_input: The user's question or request.

        Returns:
            A dict with 'answer', 'sources', and 'tool_calls' keys.
        """
        if not self._initialized:
            self.initialize()

        # Retrieve RAG context
        context, sources = self._retrieve_context(user_input)

        # Build the augmented input
        rag_context = RAG_CONTEXT_TEMPLATE.format(
            context=context, sources=sources
        )
        augmented_input = f"{rag_context}\n\nUser Question: {user_input}"

        # Run the agent
        result = self.agent_executor.invoke({
            "input": augmented_input,
            "chat_history": self.chat_history,
        })

        answer = result.get("output", "")

        # Extract tool call information
        tool_calls = []
        for step in result.get("intermediate_steps", []):
            if len(step) >= 2:
                action = step[0]
                tool_output = step[1]
                tool_calls.append({
                    "tool": action.tool,
                    "input": action.tool_input,
                    "output": str(tool_output)[:500],
                })

        # Update conversation history
        self.chat_history.append(HumanMessage(content=user_input))
        self.chat_history.append(AIMessage(content=answer))

        return {
            "answer": answer,
            "sources": sources,
            "tool_calls": tool_calls,
        }

    def reset_conversation(self) -> None:
        """Clear the conversation history."""
        self.chat_history.clear()
        logger.info("Conversation history cleared.")

    @property
    def conversation_length(self) -> int:
        """Number of messages in the current conversation."""
        return len(self.chat_history)

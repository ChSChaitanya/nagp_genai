"""AI Travel Planning Assistant - Streamlit Application.

A context-aware travel assistant that combines RAG-based destination
knowledge with real-time MCP tools for weather and currency information.

Usage:
    streamlit run app.py
"""

import sys
from pathlib import Path

# Ensure project root is on the Python path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv(project_root / ".env")

from src.agents.travel_agent import TravelPlanningAgent
from src.utils.helpers import (
    setup_logging,
    format_sources_for_display,
    format_tool_calls_for_display,
    SAMPLE_QUESTIONS,
)

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Travel Planning Assistant - Singapore",
    page_icon="\u2708\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Logging ---
setup_logging()


# --- Session State Initialization ---
def init_session_state():
    """Initialize Streamlit session state variables."""
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_initialized" not in st.session_state:
        st.session_state.agent_initialized = False


init_session_state()


# --- Sidebar ---
with st.sidebar:
    st.title("\u2708\ufe0f Travel Assistant")
    st.markdown("---")

    st.subheader("Configuration")
    st.caption(
        "Set your API keys in the `.env` file or enter them below. "
        "Keys entered here are not persisted."
    )

    import os

    llm_provider = st.selectbox(
        "LLM Provider",
        ["gemini", "openai"],
        index=0,
        help="Choose Google Gemini or OpenAI as the chat model.",
    )
    os.environ["LLM_PROVIDER"] = llm_provider

    if llm_provider == "gemini":
        llm_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            help="Get one at https://aistudio.google.com/apikey",
        )
        if llm_key:
            os.environ["GOOGLE_API_KEY"] = llm_key
    else:
        llm_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Get one at https://platform.openai.com/api-keys",
        )
        if llm_key:
            os.environ["OPENAI_API_KEY"] = llm_key

    weather_key = st.text_input(
        "OpenWeatherMap API Key",
        type="password",
        help="Required for weather tools.",
    )
    currency_key = st.text_input(
        "ExchangeRate API Key",
        type="password",
        help="Required for currency tools.",
    )

    # Apply MCP keys to environment
    if weather_key:
        os.environ["OPENWEATHER_API_KEY"] = weather_key
    if currency_key:
        os.environ["EXCHANGERATE_API_KEY"] = currency_key

    st.markdown("---")

    # Initialize / Reset buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Initialize Agent", use_container_width=True):
            key_var = "GOOGLE_API_KEY" if llm_provider == "gemini" else "OPENAI_API_KEY"
            if not os.environ.get(key_var):
                st.error(f"Please provide your {llm_provider.capitalize()} API key.")
            else:
                with st.spinner("Initializing agent..."):
                    try:
                        agent = TravelPlanningAgent()
                        agent.initialize()
                        st.session_state.agent = agent
                        st.session_state.agent_initialized = True
                        st.success("Agent ready!")
                    except Exception as e:
                        st.error(f"Initialization failed: {e}")

    with col2:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.agent:
                st.session_state.agent.reset_conversation()
            st.rerun()

    st.markdown("---")

    # Sample Questions
    st.subheader("Sample Questions")
    st.caption("Click a question to try it:")

    # Group questions by type
    rag_questions = SAMPLE_QUESTIONS[:6]
    mcp_questions = SAMPLE_QUESTIONS[6:10]
    combined_questions = SAMPLE_QUESTIONS[10:]

    with st.expander("Destination Knowledge (RAG)", expanded=False):
        for q in rag_questions:
            if st.button(q, key=f"rag_{q[:30]}", use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()

    with st.expander("Live Data (MCP Tools)", expanded=False):
        for q in mcp_questions:
            if st.button(q, key=f"mcp_{q[:30]}", use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()

    with st.expander("Combined (RAG + MCP)", expanded=True):
        for q in combined_questions:
            if st.button(q, key=f"cmb_{q[:30]}", use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()

    st.markdown("---")
    st.caption(
        "Built with LangChain, FAISS, Google Gemini / OpenAI, MCP, "
        "and Streamlit. Destination: Singapore."
    )


# --- Main Content ---
st.title("\u2708\ufe0f AI Travel Planning Assistant")
st.markdown(
    "Plan your trip to **Singapore** with AI-powered destination "
    "knowledge, real-time weather forecasts, and live currency conversion."
)

# Status indicators
if st.session_state.agent_initialized:
    st.success(
        "Agent is ready. Ask me anything about travelling to Singapore!",
        icon="\u2705",
    )
else:
    st.info(
        "Click **Initialize Agent** in the sidebar to get started. "
        "Select your LLM provider and enter your API key.",
        icon="\u2139\ufe0f",
    )

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show sources and tool calls for assistant messages
        if message["role"] == "assistant":
            if message.get("sources"):
                with st.expander("Sources", expanded=False):
                    sources = format_sources_for_display(
                        message["sources"]
                    )
                    for src in sources:
                        if src["url"]:
                            st.markdown(
                                f"- [{src['title']}]({src['url']})"
                            )
                        else:
                            st.markdown(f"- {src['title']}")

            if message.get("tool_calls"):
                with st.expander("MCP Tools Used", expanded=False):
                    st.markdown(
                        format_tool_calls_for_display(
                            message["tool_calls"]
                        )
                    )


def process_question(question: str):
    """Process a user question through the agent and display results."""
    if not st.session_state.agent_initialized:
        st.warning("Please initialize the agent first.")
        return

    # Add user message
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )
    with st.chat_message("user"):
        st.markdown(question)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.agent.query(question)
                answer = result["answer"]
                sources = result.get("sources", "")
                tool_calls = result.get("tool_calls", [])

                st.markdown(answer)

                # Show sources
                if sources:
                    with st.expander("Sources", expanded=False):
                        parsed = format_sources_for_display(sources)
                        for src in parsed:
                            if src["url"]:
                                st.markdown(
                                    f"- [{src['title']}]({src['url']})"
                                )
                            else:
                                st.markdown(f"- {src['title']}")

                # Show tool calls
                if tool_calls:
                    with st.expander("MCP Tools Used", expanded=False):
                        st.markdown(
                            format_tool_calls_for_display(tool_calls)
                        )

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "tool_calls": tool_calls,
                })

            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": "",
                    "tool_calls": [],
                })


# Handle pending question from sidebar buttons
if "pending_question" in st.session_state:
    question = st.session_state.pending_question
    del st.session_state.pending_question
    process_question(question)

# Chat input
if prompt := st.chat_input(
    "Ask about Singapore travel, weather, or currency..."
):
    process_question(prompt)

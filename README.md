# AI Travel Planning Assistant

A context-aware AI travel assistant for **Singapore** that combines document-based destination knowledge (RAG) with real-time information from MCP tools (weather forecasts and currency conversion).

Built with **LangChain**, **FAISS**, **Google Gemini / OpenAI**, **MCP (Model Context Protocol)**, and **Streamlit**.

---

## Quick Start

```bash
# 1. Clone and enter the project
git clone https://github.com/ChSChaitanya/nagp_genai
cd nagp_genai

# 2. Create virtual environment and install
python -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate        # Windows (CMD)
# .\venv\Scripts\Activate.ps1  # Windows (PowerShell)
pip install -r requirements.txt

# 3. Configure API keys
cp .env.example .env           # Linux/macOS  (Windows: copy .env.example .env)
# Edit .env and fill in your keys (see "API Key Setup" section)

# 4. Run
streamlit run app.py
```

The app opens at **http://localhost:8501**. Select your LLM provider in the sidebar, enter your API key, click **Initialize Agent**, and start chatting.

---

## Architecture Overview

```
+-----------------+      +------------------+      +-------------------+
|   Streamlit UI  | ---> | Travel Planning  | ---> | Google Gemini     |
|   (app.py)      |      |   Agent          |      | or OpenAI (LLM)   |
+-----------------+      +--------+---------+      +-------------------+
                                  |
                    +-------------+-------------+
                    |                           |
           +--------v---------+      +---------v----------+
           |   RAG Pipeline   |      |   MCP Tool Layer   |
           |                  |      |                    |
           | - Doc Loader     |      | - Weather Server   |
           | - Chunker        |      |   (OpenWeatherMap) |
           | - FAISS Vector   |      | - Currency Server  |
           |   Store          |      |   (ExchangeRate)   |
           +------------------+      +--------------------+
                    |
           +--------v---------+
           |  Knowledge Base  |
           |  (3 MD files)    |
           +------------------+
```

### Data Flow

1. **User query** enters through the Streamlit chat interface.
2. The **Travel Planning Agent** retrieves relevant context from the FAISS vector store (RAG).
3. The agent determines if MCP tools are needed (weather/currency) and invokes them.
4. Retrieved context + tool results are combined with the user query.
5. The LLM generates a grounded response with source attribution.
6. The response is displayed with expandable source and tool-usage details.

### LLM Provider Architecture

The app supports two LLM providers, selectable via `LLM_PROVIDER` in `.env` or the sidebar dropdown:

| Provider | How It Works | Agent Type |
| --- | --- | --- |
| **Google Gemini** (default) | Uses Google's OpenAI-compatible endpoint via `ChatOpenAI` | ReAct agent (text-based tool invocation) |
| **OpenAI** | Direct OpenAI API via `ChatOpenAI` | Native tool-calling agent |

Both providers use **local HuggingFace embeddings** (`sentence-transformers/all-MiniLM-L6-v2`) for RAG — no cloud embedding API is needed.

---

## Project Structure

```
nagp_genai/
├── app.py                              # Streamlit UI entry point
├── requirements.txt                    # Python dependencies
├── .env                                # Environment variables (create from .env.example)
├── README.md                           # This file
├── .gitignore
│
├── config/
│   ├── __init__.py
│   └── settings.py                     # Centralized configuration (pydantic-settings)
│
├── data/
│   └── knowledge_base/                 # Travel resource documents
│       ├── singapore_wikivoyage_guide.md
│       ├── singapore_essential_travel_info.md
│       └── singapore_itineraries_and_activities.md
│
├── src/
│   ├── __init__.py
│   ├── rag/                            # RAG pipeline
│   │   ├── __init__.py
│   │   ├── document_loader.py          # Load & parse Markdown with metadata
│   │   ├── chunker.py                  # Markdown-aware text splitting
│   │   └── vector_store.py             # FAISS index management
│   │
│   ├── mcp_servers/                    # MCP tool servers
│   │   ├── __init__.py
│   │   ├── weather_server.py           # Weather tool (OpenWeatherMap)
│   │   └── currency_server.py          # Currency tool (ExchangeRate API)
│   │
│   ├── agents/                         # LangChain agent orchestration
│   │   ├── __init__.py
│   │   └── travel_agent.py             # Main travel planning agent
│   │
│   ├── prompts/                        # Prompt engineering
│   │   ├── __init__.py
│   │   └── templates.py                # System & context prompts
│   │
│   └── utils/                          # Shared utilities
│       ├── __init__.py
│       └── helpers.py                  # Logging, formatting, sample questions
│
└── tests/
    ├── __init__.py
    ├── test_rag.py                     # RAG pipeline tests
    ├── test_mcp_tools.py               # MCP tool tests
    └── test_agent.py                   # Agent integration tests
```

---

## Knowledge Base Sources

The knowledge base contains three curated Markdown documents with YAML front-matter for source tracking:

| Document | Content | Original Source |
| --- | --- | --- |
| `singapore_wikivoyage_guide.md` | Districts, attractions, transport, food, culture | [Wikivoyage Singapore](https://en.wikivoyage.org/wiki/Singapore) |
| `singapore_essential_travel_info.md` | Practical info, climate, money, connectivity, etiquette | [Visit Singapore - Essential Info](https://www.visitsingapore.com/travel-guide-tips/essential-information/) |
| `singapore_itineraries_and_activities.md` | Sample itineraries (3-day, 5-day, family, cultural), indoor/outdoor activities | [Visit Singapore - Itineraries](https://www.visitsingapore.com/editorials/four-day-itinerary/) |

Documents cover: major attractions, neighbourhoods, local transportation, cultural and practical tips, food and hawker centres, sample itineraries, and indoor/outdoor activity suggestions.

---

## RAG Workflow

1. **Document Loading** (`document_loader.py`): Reads Markdown files, parses YAML front-matter to extract `title`, `source`, and `url` metadata for citations.

2. **Chunking** (`chunker.py`): Two-stage splitting:
   - Stage 1: `MarkdownHeaderTextSplitter` splits by `#`, `##`, `###` headers to preserve section context.
   - Stage 2: `RecursiveCharacterTextSplitter` further splits oversized sections (default: 1000 chars, 200 overlap).
   - Each chunk retains original metadata plus section breadcrumbs.

3. **Embedding & Indexing** (`vector_store.py`): Generates embeddings using a local HuggingFace model (`sentence-transformers/all-MiniLM-L6-v2`, downloaded on first run) and stores them in a FAISS vector index. The index is persisted to disk for fast reload.

4. **Retrieval**: Top-K similarity search (default K=5) retrieves the most relevant chunks for each query. Retrieved chunks are formatted with source citations and injected into the LLM prompt.

---

## MCP Tools

### Weather Tool (OpenWeatherMap)
- `get_current_weather(city)`: Returns current conditions (temperature, humidity, wind, description).
- `get_weather_forecast(city, days)`: Returns 3-hour interval forecasts for up to 5 days.
- Server: `src/mcp_servers/weather_server.py` (FastMCP with stdio transport).

### Currency Tool (ExchangeRate API)
- `convert_currency(amount, from_currency, to_currency)`: Converts between any two currencies.
- `get_exchange_rate(base_currency, target_currencies)`: Shows exchange rates against multiple targets.
- Server: `src/mcp_servers/currency_server.py` (FastMCP with stdio transport).

Both tools are also wrapped as LangChain `@tool` functions in the agent for seamless integration with the LangChain agent executor.

---

## Prompt and Context Strategy

1. **System Prompt**: Establishes the assistant persona (Singapore travel expert), defines grounding rules, source-attribution requirements, and response types (RAG-only, MCP-only, combined).

2. **Smart RAG Injection**: Context is only retrieved and injected when the query needs destination knowledge. Pure weather/currency queries skip retrieval for concise, tool-only answers.

3. **Tool Result Labeling**: Weather and currency outputs include explicit source labels (`[Source: OpenWeatherMap]`, `[Source: ExchangeRate API]`).

4. **Honesty Guardrails**: The model states when information is missing rather than fabricating, and marks its own suggestions as `[AI Recommendation]`.

5. **Multi-turn Context**: Chat history is maintained via LangChain's `MessagesPlaceholder`, allowing the agent to reference previous messages and user preferences.

---

## API Key Setup

You need **three free API keys**:

| Key | Where to Get It | Sign-Up Steps |
| --- | --- | --- |
| **Google Gemini API Key** | https://aistudio.google.com/apikey | Sign in with Google → Click "Create API Key" → Copy |
| **OpenWeatherMap API Key** | https://openweathermap.org/api | Sign up free → Go to "API keys" tab in profile → Copy |
| **ExchangeRate API Key** | https://www.exchangerate-api.com/ | Sign up with email → Key shown on dashboard → Copy |

> **Using OpenAI instead?** Get a key at https://platform.openai.com/api-keys and set `LLM_PROVIDER=openai` in `.env`.

### Configuring `.env`

Open `.env` and replace **only** the placeholder values:

```dotenv
# Choose your provider: 'gemini' (default) or 'openai'
LLM_PROVIDER=gemini

# Fill in ONE of these (based on LLM_PROVIDER)
GOOGLE_API_KEY=<paste your Gemini key here>
# OPENAI_API_KEY=<paste your OpenAI key here>   # uncomment if using OpenAI

# Fill in BOTH of these (required for weather and currency tools)
OPENWEATHER_API_KEY=<paste your OpenWeatherMap key here>
EXCHANGERATE_API_KEY=<paste your ExchangeRate key here>
```

Leave all other values (model names, chunk sizes, paths) at their defaults.

---

## Setup Instructions

### Prerequisites

- **Python 3.11+** (verify: `python --version`)
- **pip** (comes with Python)
- Internet access (for API calls and first-run embedding model download)

### Step-by-Step Installation

#### 1. Clone the Repository

```bash
git clone <repository-url>
cd nagp_genai
```

#### 2. Create a Virtual Environment

```bash
# Linux / macOS
python -m venv venv
source venv/bin/activate

# Windows (Command Prompt)
python -m venv venv
venv\Scripts\activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **First-run note**: The embedding model (`sentence-transformers/all-MiniLM-L6-v2`, ~90 MB) downloads automatically when you initialize the agent. Subsequent runs load it from cache.

#### 4. Configure Environment Variables

Edit `.env` and fill in your API keys (see **API Key Setup** above).

#### 5. Run the Application

```bash
streamlit run app.py
```

The app opens in your browser at **http://localhost:8501**.

#### 6. Initialize and Use

1. In the sidebar, select your **LLM Provider** (Gemini or OpenAI).
2. Enter your LLM API key (if not already in `.env`).
3. Optionally enter Weather and Currency API keys in the sidebar.
4. Click **Initialize Agent** and wait for "Agent ready!".
5. Start chatting!

### Running Tests

```bash
# Unit tests (no API keys required)
pytest tests/ -v

# Full integration tests (API keys must be set in .env or environment)
pytest tests/ -v
```

---

## Sample Questions and Expected Behaviour

### RAG-Only (Destination Knowledge)
| Question | Expected Response |
| --- | --- |
| What are the must-visit attractions in Singapore? | Lists Gardens by the Bay, Marina Bay Sands, Merlion Park, etc. with source citations |
| Which neighbourhoods are suitable for cultural experiences? | Describes Chinatown, Little India, Kampong Glam with details from the knowledge base |
| What indoor attractions can I visit? | Lists National Gallery, ArtScience Museum, S.E.A. Aquarium, etc. |

### MCP-Only (Live Data)
| Question | Expected Response |
| --- | --- |
| What is the weather in Singapore? | Current conditions from OpenWeatherMap with temperature, humidity, wind |
| Convert INR 50,000 to SGD | Live conversion with exchange rate from ExchangeRate API |

### Combined (RAG + MCP)
| Question | Expected Response |
| --- | --- |
| Create a three-day itinerary and adjust for weather | Day-by-day itinerary from KB + live forecast, with indoor alternatives for rainy days |
| Budget of INR 60,000 - convert to SGD and suggest itinerary | Currency conversion + itinerary with cost estimates |

---

## Troubleshooting

| Problem | Solution |
| --- | --- |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` inside your activated virtual environment |
| `OPENWEATHER_API_KEY not configured` | Add your key to `.env` or enter it in the Streamlit sidebar |
| Agent initialization slow on first run | The embedding model (~90 MB) is downloading; subsequent runs are instant |
| `streamlit: command not found` | Activate your virtual environment first |
| Empty responses or LLM errors | Verify your `GOOGLE_API_KEY` or `OPENAI_API_KEY` is valid and has quota |
| `Connection refused` on weather/currency | Check internet connection and verify API keys are correct |
| PowerShell blocks `Activate.ps1` | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` first |

---

## Technology Stack

| Component | Technology |
| --- | --- |
| LLM | Google Gemini (`gemini-3.6-flash`) or OpenAI (`gpt-4o-mini`) via `langchain-openai` |
| LLM API | Google OpenAI-compatible endpoint (Gemini) / Direct OpenAI API |
| Embeddings | Local HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`) — no API key needed |
| Vector Store | FAISS (`faiss-cpu`) |
| Agent Framework | LangChain (`AgentExecutor` + `create_react_agent` / `create_tool_calling_agent`) |
| MCP Framework | FastMCP (`mcp` package) |
| Weather API | OpenWeatherMap (free tier) |
| Currency API | ExchangeRate API (free tier) |
| UI | Streamlit |
| Configuration | `pydantic-settings`, `python-dotenv` |
| Testing | `pytest`, `pytest-asyncio` |
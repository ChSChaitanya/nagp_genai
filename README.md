# AI Travel Planning Assistant

A context-aware AI travel assistant for **Singapore** that combines document-based destination knowledge (RAG) with real-time information from MCP tools (weather forecasts and currency conversion).

Built with **LangChain**, **FAISS**, **Google Gemini / OpenAI**, **MCP (Model Context Protocol)**, and **Streamlit**.

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

---

## Project Structure

```
nagp_genai/
├── app.py                              # Streamlit UI entry point
├── requirements.txt                    # Python dependencies
├── .env.example                        # Environment variable template
├── README.md                           # This file
├── LICENSE
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

The prompting approach uses a **layered architecture**:

1. **System Prompt**: Establishes the assistant persona (Singapore travel expert), defines grounding rules, source-attribution requirements, and the three response types (RAG-only, MCP-only, combined).

2. **RAG Context Injection**: Retrieved knowledge-base chunks are inserted as a clearly delineated `KNOWLEDGE BASE CONTEXT` section with source citations. The model is instructed to use only this content for destination facts.

3. **Tool Result Labeling**: Weather and currency tool outputs include explicit source labels (`[Source: OpenWeatherMap - Live Data]`, `[Source: ExchangeRate API - Live Data]`) so the model can distinguish real-time data from static content.

4. **Honesty Guardrails**: The prompt explicitly instructs the model to:
   - State when the knowledge base lacks information instead of fabricating
   - Mark its own suggestions as `[AI Recommendation]`
   - Preserve user preferences from the conversation context

5. **Multi-turn Context**: Chat history is maintained via LangChain's `MessagesPlaceholder`, allowing the agent to reference previous messages and user preferences.

---

## Setup Instructions

### Prerequisites
- Python 3.11+
- An API key for one of the supported LLM providers:
  - [Google Gemini](https://aistudio.google.com/apikey) (default, free tier available)
  - [OpenAI](https://platform.openai.com/api-keys) (alternative)
- API keys for MCP tools:
  - [OpenWeatherMap](https://openweathermap.org/api) (free tier, for weather tools)
  - [ExchangeRate API](https://www.exchangerate-api.com/) (free tier, for currency tools)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd nagp_genai

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and add your API keys
```

### Running the Application

```bash
# Start the Streamlit app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

1. Select your LLM provider (Gemini or OpenAI) and enter the API key in the sidebar (or configure in `.env`).
2. Click **Initialize Agent** to build the vector index and load tools.
3. Start asking questions!

### Running Tests

```bash
# Run all tests (unit tests work without API keys)
pytest tests/ -v

# Run with API keys for integration tests
# For Gemini:
export LLM_PROVIDER=gemini
export GOOGLE_API_KEY=your_key
# Or for OpenAI:
# export LLM_PROVIDER=openai
# export OPENAI_API_KEY=your_key
export OPENWEATHER_API_KEY=your_key
export EXCHANGERATE_API_KEY=your_key
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

## Minimum Acceptance Criteria Checklist

- [x] Knowledge base created from at least three travel resources
- [x] Embedding-based semantic retrieval (FAISS + local HuggingFace embeddings)
- [x] Grounded answers with source references
- [x] Weather information through an MCP tool (OpenWeatherMap)
- [x] Currency conversion through an MCP tool (ExchangeRate API)
- [x] At least one response combining RAG and MCP
- [x] Multi-turn conversation with retained context
- [x] Appropriate tool selection based on user intent
- [x] Clear handling of missing knowledge and tool failures
- [x] A simple, usable interface (Streamlit)

---

## Technology Stack

| Component | Technology |
| --- | --- |
| LLM | Google Gemini (gemini-2.0-flash) or OpenAI (gpt-4o-mini) |
| Embeddings | Local HuggingFace (sentence-transformers/all-MiniLM-L6-v2) |
| Vector Store | FAISS (faiss-cpu) |
| Orchestration | LangChain |
| MCP Framework | FastMCP (mcp package) |
| Weather API | OpenWeatherMap |
| Currency API | ExchangeRate API |
| UI | Streamlit |
| Configuration | pydantic-settings, python-dotenv |
| Testing | pytest, pytest-asyncio |
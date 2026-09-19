"""Centralized configuration for the AI Travel Planning Assistant.

Loads settings from environment variables with sensible defaults.
Uses pydantic-settings for type-safe configuration management.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field

# Project root directory (two levels up from config/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings loaded from .env file and environment variables."""

    # --- LLM Provider Selection ---
    llm_provider: str = Field(
        default="gemini",
        description="LLM provider: 'gemini' or 'openai'",
    )

    # --- Google Gemini ---
    google_api_key: str = Field(
        default="", description="Google Gemini API key"
    )
    gemini_model: str = Field(
        default="gemini-3.6-flash",
        description="Google Gemini model name",
    )

    # --- OpenAI ---
    openai_api_key: str = Field(
        default="", description="OpenAI API key"
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model name",
    )

    # --- Embeddings (local) ---
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace model name for local embeddings",
    )

    # --- MCP: Weather ---
    openweather_api_key: str = Field(
        default="", description="OpenWeatherMap API key"
    )

    # --- MCP: Currency ---
    exchangerate_api_key: str = Field(
        default="", description="ExchangeRate API key"
    )

    # --- Application ---
    destination_city: str = Field(
        default="Singapore", description="Target travel destination"
    )
    knowledge_base_path: str = Field(
        default="data/knowledge_base",
        description="Relative path to knowledge base docs",
    )
    vector_store_path: str = Field(
        default="data/vector_store",
        description="Relative path to persisted vector store",
    )
    chunk_size: int = Field(default=1000, description="Document chunk size")
    chunk_overlap: int = Field(default=200, description="Chunk overlap")
    retriever_top_k: int = Field(
        default=5, description="Number of retrieved chunks"
    )

    # --- Streamlit ---
    streamlit_port: int = Field(default=8501, description="Streamlit port")

    model_config = {
        "env_file": str(PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @property
    def knowledge_base_abs_path(self) -> Path:
        """Absolute path to the knowledge base directory."""
        return PROJECT_ROOT / self.knowledge_base_path

    @property
    def vector_store_abs_path(self) -> Path:
        """Absolute path to the vector store directory."""
        return PROJECT_ROOT / self.vector_store_path


def get_settings() -> Settings:
    """Factory that returns a cached Settings instance."""
    return Settings()

"""Centralized configuration for the AI Travel Planning Assistant.

Loads settings from environment variables with sensible defaults.
Uses pydantic-settings for type-safe configuration management.
"""

import os
import re
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator

# Project root directory (two levels up from config/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings loaded from .env file and environment variables."""

    # --- Databricks Model Serving ---
    databricks_host: str = Field(
        default="https://adb-6192355565634015.15.azuredatabricks.net",
        description="Databricks workspace URL",
    )
    databricks_token: str = Field(
        default="", description="Databricks personal access token"
    )
    databricks_llm_endpoint: str = Field(
        default="databricks-claude-sonnet-4-6",
        description="Databricks serving endpoint name for chat LLM",
    )

    @field_validator("databricks_llm_endpoint", mode="after")
    @classmethod
    def _strip_endpoint_url(cls, v: str) -> str:
        """Extract just the endpoint name if a full URL was provided."""
        # Handles: https://.../serving-endpoints/<name>/invocations
        match = re.search(r"/serving-endpoints/([^/]+)", v)
        return match.group(1) if match else v.strip()
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

"""Vector store management for the RAG pipeline.

Handles embedding generation, FAISS index creation, persistence,
and semantic retrieval of travel knowledge chunks.
"""

import logging
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_databricks import DatabricksEmbeddings
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)


class TravelVectorStore:
    """Manages the FAISS vector store for the travel knowledge base."""

    def __init__(
        self,
        embedding_endpoint: str = "databricks-bge-large-en",
        databricks_host: str = "",
        databricks_token: str = "",
        persist_path: Optional[str | Path] = None,
    ):
        """Initialize the vector store manager.

        Args:
            embedding_endpoint: Databricks serving endpoint for embeddings.
            databricks_host: Databricks workspace URL.
            databricks_token: Databricks personal access token.
            persist_path: Directory to save/load the FAISS index.
        """
        self.embeddings = DatabricksEmbeddings(
            endpoint=embedding_endpoint,
        )
        self.persist_path = Path(persist_path) if persist_path else None
        self._store: Optional[FAISS] = None

    @property
    def store(self) -> FAISS:
        """Access the underlying FAISS store."""
        if self._store is None:
            raise RuntimeError(
                "Vector store not initialized. "
                "Call build_index() or load_index() first."
            )
        return self._store

    def build_index(self, chunks: List[Document]) -> None:
        """Build a FAISS index from document chunks.

        Args:
            chunks: List of Document chunks with embeddings-ready text.
        """
        if not chunks:
            raise ValueError("No chunks provided to build the index.")

        logger.info("Building FAISS index from %d chunks...", len(chunks))
        self._store = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings,
        )
        logger.info("FAISS index built successfully.")

        # Persist if a path is configured
        if self.persist_path:
            self.save_index()

    def save_index(self) -> None:
        """Save the FAISS index to disk."""
        if self._store is None:
            raise RuntimeError("No index to save.")
        if self.persist_path is None:
            raise ValueError("No persist path configured.")

        self.persist_path.mkdir(parents=True, exist_ok=True)
        self._store.save_local(str(self.persist_path))
        logger.info("FAISS index saved to %s", self.persist_path)

    def load_index(self) -> bool:
        """Load a previously saved FAISS index.

        Returns:
            True if the index was loaded, False if no saved index exists.
        """
        if self.persist_path is None:
            return False

        index_file = self.persist_path / "index.faiss"
        if not index_file.exists():
            logger.info("No saved index found at %s", self.persist_path)
            return False

        logger.info("Loading FAISS index from %s", self.persist_path)
        self._store = FAISS.load_local(
            str(self.persist_path),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("FAISS index loaded successfully.")
        return True

    def get_retriever(self, top_k: int = 5):
        """Create a retriever from the vector store.

        Args:
            top_k: Number of documents to retrieve.

        Returns:
            A LangChain retriever instance.
        """
        return self.store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        )

    def similarity_search(
        self, query: str, k: int = 5
    ) -> List[Document]:
        """Perform a direct similarity search.

        Args:
            query: The search query.
            k: Number of results.

        Returns:
            A list of relevant Document objects.
        """
        return self.store.similarity_search(query, k=k)

    def similarity_search_with_score(
        self, query: str, k: int = 5
    ) -> List[tuple[Document, float]]:
        """Perform similarity search and return scores.

        Args:
            query: The search query.
            k: Number of results.

        Returns:
            A list of (Document, score) tuples.
        """
        return self.store.similarity_search_with_score(query, k=k)

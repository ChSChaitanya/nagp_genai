"""RAG (Retrieval-Augmented Generation) pipeline package.

Provides document loading, chunking, embedding, and retrieval
for the travel knowledge base.
"""

from src.rag.document_loader import load_knowledge_base
from src.rag.chunker import chunk_documents
from src.rag.vector_store import TravelVectorStore

__all__ = ["load_knowledge_base", "chunk_documents", "TravelVectorStore"]

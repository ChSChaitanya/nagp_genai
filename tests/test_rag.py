"""Tests for the RAG pipeline components.

Tests document loading, chunking, and vector store operations.
"""

import sys
from pathlib import Path

import pytest

# Ensure project root is on the path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.rag.document_loader import load_knowledge_base, _parse_frontmatter
from src.rag.chunker import chunk_documents


# ---- Fixtures ----

@pytest.fixture
def knowledge_base_path():
    """Path to the test knowledge base directory."""
    return project_root / "data" / "knowledge_base"


@pytest.fixture
def loaded_documents(knowledge_base_path):
    """Load documents from the knowledge base."""
    return load_knowledge_base(knowledge_base_path)


# ---- Document Loader Tests ----

class TestFrontmatterParser:
    """Tests for YAML front-matter parsing."""

    def test_parse_valid_frontmatter(self):
        text = '---\ntitle: "Test Title"\nurl: "https://example.com"\n---\n\n# Content'
        metadata, body = _parse_frontmatter(text)
        assert metadata["title"] == "Test Title"
        assert metadata["url"] == "https://example.com"
        assert body.strip() == "# Content"

    def test_parse_no_frontmatter(self):
        text = "# Just a heading\n\nSome content."
        metadata, body = _parse_frontmatter(text)
        assert metadata == {}
        assert body == text

    def test_parse_empty_frontmatter(self):
        text = "---\n---\n\nContent here."
        metadata, body = _parse_frontmatter(text)
        assert metadata == {}
        assert "Content here." in body


class TestDocumentLoader:
    """Tests for loading knowledge base documents."""

    def test_load_returns_documents(self, loaded_documents):
        assert len(loaded_documents) >= 3, (
            "Expected at least 3 knowledge base documents."
        )

    def test_documents_have_content(self, loaded_documents):
        for doc in loaded_documents:
            assert len(doc.page_content) > 100, (
                f"Document {doc.metadata.get('source_file')} has too little content."
            )

    def test_documents_have_metadata(self, loaded_documents):
        for doc in loaded_documents:
            assert "source_title" in doc.metadata
            assert "source_file" in doc.metadata
            assert doc.metadata["source_title"] != ""

    def test_documents_have_urls(self, loaded_documents):
        for doc in loaded_documents:
            assert "source_url" in doc.metadata
            # All our docs should have URLs
            assert doc.metadata["source_url"].startswith("http")

    def test_load_nonexistent_path_raises(self):
        with pytest.raises(FileNotFoundError):
            load_knowledge_base("/nonexistent/path")


# ---- Chunker Tests ----

class TestChunker:
    """Tests for document chunking."""

    def test_chunks_created(self, loaded_documents):
        chunks = chunk_documents(loaded_documents, chunk_size=500, chunk_overlap=100)
        assert len(chunks) > len(loaded_documents), (
            "Chunking should produce more chunks than input documents."
        )

    def test_chunk_size_respected(self, loaded_documents):
        chunk_size = 600
        chunks = chunk_documents(
            loaded_documents, chunk_size=chunk_size, chunk_overlap=100
        )
        # Allow some tolerance for header inclusion
        oversized = [c for c in chunks if len(c.page_content) > chunk_size * 1.5]
        assert len(oversized) == 0, (
            f"{len(oversized)} chunks exceed 1.5x the target chunk size."
        )

    def test_chunks_retain_metadata(self, loaded_documents):
        chunks = chunk_documents(loaded_documents)
        for chunk in chunks:
            assert "source_title" in chunk.metadata
            assert "source_file" in chunk.metadata

    def test_chunks_have_section_context(self, loaded_documents):
        chunks = chunk_documents(loaded_documents)
        chunks_with_section = [
            c for c in chunks if "section" in c.metadata
        ]
        assert len(chunks_with_section) > 0, (
            "At least some chunks should have section context."
        )

    def test_empty_documents_list(self):
        chunks = chunk_documents([])
        assert chunks == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

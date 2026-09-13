"""Text chunking utilities for the RAG pipeline.

Splits documents into semantically meaningful chunks using
Markdown-aware splitting, preserving section context and metadata.
"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """Split documents into chunks suitable for embedding.

    Uses a two-stage approach:
    1. Split by Markdown headers to preserve section context.
    2. Further split large sections with RecursiveCharacterTextSplitter.

    Each chunk retains the original document metadata plus
    the section headers it belongs to.

    Args:
        documents: List of LangChain Document objects.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        A list of chunked Document objects with enriched metadata.
    """
    # Stage 1: Split by Markdown headers
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "heading_1"),
            ("##", "heading_2"),
            ("###", "heading_3"),
        ],
        strip_headers=False,
    )

    # Stage 2: Further split oversized chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    all_chunks: List[Document] = []

    for doc in documents:
        # Split by headers first
        header_chunks = md_splitter.split_text(doc.page_content)

        for hc in header_chunks:
            # Merge original doc metadata with header metadata
            merged_metadata = {**doc.metadata, **hc.metadata}

            # Build a section context string for better retrieval
            section_parts = []
            for key in ["heading_1", "heading_2", "heading_3"]:
                if key in hc.metadata:
                    section_parts.append(hc.metadata[key])
            if section_parts:
                merged_metadata["section"] = " > ".join(section_parts)

            # Further split if the chunk is too large
            if len(hc.page_content) > chunk_size:
                sub_chunks = text_splitter.create_documents(
                    texts=[hc.page_content],
                    metadatas=[merged_metadata],
                )
                all_chunks.extend(sub_chunks)
            else:
                all_chunks.append(
                    Document(
                        page_content=hc.page_content,
                        metadata=merged_metadata,
                    )
                )

    return all_chunks

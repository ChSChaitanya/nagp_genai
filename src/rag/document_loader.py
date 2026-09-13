"""Document loader for the travel knowledge base.

Loads Markdown files from the knowledge base directory, extracting
front-matter metadata (title, source, url) for citation tracking.
"""

import re
from pathlib import Path
from typing import List

from langchain_core.documents import Document


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Extract YAML front-matter from a Markdown file.

    Returns:
        A tuple of (metadata_dict, remaining_text).
    """
    metadata = {}
    body = text

    frontmatter_pattern = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
    match = frontmatter_pattern.match(text)

    if match:
        raw = match.group(1)
        body = text[match.end():]
        for line in raw.strip().splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                metadata[key.strip()] = value.strip().strip('"').strip("'")

    return metadata, body


def load_knowledge_base(knowledge_base_path: str | Path) -> List[Document]:
    """Load all Markdown files from the knowledge base directory.

    Each file is converted to a LangChain Document with metadata
    containing the source title, URL, and file name.

    Args:
        knowledge_base_path: Path to the directory containing .md files.

    Returns:
        A list of LangChain Document objects.
    """
    kb_path = Path(knowledge_base_path)
    if not kb_path.exists():
        raise FileNotFoundError(
            f"Knowledge base directory not found: {kb_path}"
        )

    documents: List[Document] = []
    md_files = sorted(kb_path.glob("*.md"))

    if not md_files:
        raise ValueError(
            f"No Markdown files found in {kb_path}. "
            "Please add travel resource documents."
        )

    for md_file in md_files:
        raw_text = md_file.read_text(encoding="utf-8")
        frontmatter, body = _parse_frontmatter(raw_text)

        metadata = {
            "source_file": md_file.name,
            "source_title": frontmatter.get("title", md_file.stem),
            "source_url": frontmatter.get("url", ""),
            "source": frontmatter.get("source", md_file.stem),
        }

        doc = Document(page_content=body.strip(), metadata=metadata)
        documents.append(doc)

    return documents

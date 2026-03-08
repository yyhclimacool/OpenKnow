"""MCP Server for Knowledge Database - integrates with Cursor and Claude Code."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config
from src.searcher import KnowledgeBase

from src.logging_config import setup_logging

setup_logging(level=logging.INFO, console=False)
logger = logging.getLogger(__name__)

mcp = FastMCP(
    "knowledge-db",
    instructions=(
        "Knowledge database search tool. Use search_knowledge to find relevant "
        "information from indexed internal documents (markdown and images). "
        "The knowledge base contains internal documentation that has been "
        "semantically indexed for retrieval."
    ),
)

_kb: KnowledgeBase | None = None


def _get_kb() -> KnowledgeBase:
    global _kb
    if _kb is None:
        _kb = KnowledgeBase(load_config())
    return _kb


@mcp.tool()
def search_knowledge(query: str, top_k: int = 5, source_filter: str | None = None) -> str:
    """Search the knowledge database for relevant document fragments.

    Args:
        query: Natural language search query describing what you're looking for.
        top_k: Maximum number of results to return (default 5).
        source_filter: Optional path substring to filter results by source directory.

    Returns:
        Formatted search results with content, source path, and relevance score.
    """
    kb = _get_kb()
    results = kb.search(query, top_k=top_k, source_filter=source_filter)

    if not results:
        return "No relevant documents found for the given query."

    output_parts: list[str] = []
    for i, r in enumerate(results, 1):
        output_parts.append(
            f"--- Result {i} (relevance: {r.relevance_score:.2%}) ---\n"
            f"Source: {r.source_path}\n"
            f"Section: {r.heading_chain}\n"
            f"\n{r.content}\n"
        )

    return "\n".join(output_parts)


@mcp.tool()
def add_directory(directory: str) -> str:
    """Add a new directory to the knowledge database and index all its documents.

    Scans the directory recursively for markdown (.md) and image files,
    generates embeddings, and stores them for semantic search.

    Args:
        directory: Absolute path to the directory to add and index.

    Returns:
        Summary of the indexing operation.
    """
    kb = _get_kb()
    try:
        result = kb.add_directory(directory)
        return (
            f"Successfully indexed directory: {result['directory']}\n"
            f"Documents found: {result['documents']}\n"
            f"Chunks created: {result['chunks']}"
        )
    except Exception as e:
        return f"Error indexing directory: {e}"


@mcp.tool()
def add_content(
    content: str,
    title: str | None = None,
    tags: list[str] | None = None,
) -> str:
    """Add user-provided content to the knowledge database.

    The content is saved as a markdown file (for future re-indexing) and
    immediately indexed for semantic search.

    Args:
        content: The text content to add (markdown format recommended).
        title: Optional title for the document. Auto-detected from content if omitted.
        tags: Optional list of tags for categorization.

    Returns:
        Summary of the saved file and indexing result.
    """
    kb = _get_kb()
    try:
        result = kb.add_content(content, title=title, tags=tags)
        return (
            f"Content saved and indexed successfully.\n"
            f"File: {result['file']}\n"
            f"Title: {result.get('title', 'N/A')}\n"
            f"Chunks created: {result['chunks']}"
        )
    except Exception as e:
        return f"Error adding content: {e}"


@mcp.tool()
def list_sources() -> str:
    """List all indexed document sources in the knowledge database.

    Returns:
        List of configured source directories and index statistics.
    """
    kb = _get_kb()
    stats = kb.get_stats()
    sources = kb.list_sources()

    lines = ["=== Knowledge Database Status ==="]
    lines.append(f"Total chunks indexed: {stats['total_chunks']}")
    lines.append(f"Total files: {stats['total_files']}")
    lines.append(f"\nConfigured sources ({len(sources)}):")
    for s in sources:
        lines.append(f"  - {s}")

    if not sources:
        lines.append("  (no sources configured)")

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Knowledge DB MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport protocol (default: stdio)",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="HTTP host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=8787, help="HTTP port (default: 8787)"
    )
    args = parser.parse_args()

    if args.transport in ("sse", "streamable-http"):
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        setup_logging(level=logging.INFO, console=True)
        logger.info(
            "Starting MCP server on http://%s:%d (%s)",
            args.host, args.port, args.transport,
        )

    mcp.run(transport=args.transport)

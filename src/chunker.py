from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from .parser import ParsedDocument

logger = logging.getLogger(__name__)

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

DEFAULT_MAX_CHUNK_TOKENS = 800
DEFAULT_OVERLAP_TOKENS = 100
APPROX_CHARS_PER_TOKEN = 3.5


@dataclass
class Chunk:
    content: str
    source_path: str
    heading_chain: str = ""
    chunk_index: int = 0
    doc_type: str = "markdown"
    metadata: dict = field(default_factory=dict)

    @property
    def display_text(self) -> str:
        prefix = f"[{self.heading_chain}]\n\n" if self.heading_chain else ""
        return prefix + self.content


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / APPROX_CHARS_PER_TOKEN))


def _split_by_paragraphs(
    text: str,
    max_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[str]:
    """Split long text into smaller pieces along paragraph boundaries."""
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_tokens = _estimate_tokens(para)

        if current_tokens + para_tokens > max_tokens and current:
            chunks.append("\n\n".join(current))
            overlap_chars = int(overlap_tokens * APPROX_CHARS_PER_TOKEN)
            full_text = "\n\n".join(current)
            if len(full_text) > overlap_chars:
                tail = full_text[-overlap_chars:]
                boundary = tail.find("\n\n")
                overlap_text = tail[boundary + 2 :] if boundary != -1 else tail
                current = [overlap_text] if overlap_text.strip() else []
                current_tokens = _estimate_tokens(overlap_text)
            else:
                current = []
                current_tokens = 0

        current.append(para)
        current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current))

    return chunks if chunks else [text]


def _build_heading_chain(headings: list[tuple[int, str]], current_level: int, current_title: str) -> str:
    """Build a heading chain like '# Overview > ## Installation'."""
    chain_parts: list[str] = []
    for level, title in headings:
        if level < current_level:
            chain_parts.append(f"{'#' * level} {title}")
    chain_parts.append(f"{'#' * current_level} {current_title}")
    return " > ".join(chain_parts)


def chunk_markdown(
    doc: ParsedDocument,
    max_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Split a markdown document into chunks by heading sections."""
    content = doc.content
    headings_found = list(HEADING_RE.finditer(content))

    if not headings_found:
        pieces = _split_by_paragraphs(content, max_tokens, overlap_tokens)
        return [
            Chunk(
                content=piece,
                source_path=doc.source_path,
                heading_chain=doc.title,
                chunk_index=i,
                doc_type=doc.doc_type,
                metadata=doc.metadata,
            )
            for i, piece in enumerate(pieces)
        ]

    sections: list[tuple[int, str, str]] = []
    heading_stack: list[tuple[int, str]] = []

    for idx, match in enumerate(headings_found):
        level = len(match.group(1))
        title = match.group(2).strip()
        start = match.end()
        end = headings_found[idx + 1].start() if idx + 1 < len(headings_found) else len(content)
        section_text = content[start:end].strip()

        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()

        chain = _build_heading_chain(heading_stack, level, title)
        heading_stack.append((level, title))

        if section_text:
            sections.append((level, chain, section_text))

    preamble_end = headings_found[0].start() if headings_found else len(content)
    preamble = content[:preamble_end].strip()
    if preamble:
        sections.insert(0, (0, doc.title, preamble))

    chunks: list[Chunk] = []
    chunk_idx = 0
    for _level, chain, text in sections:
        if _estimate_tokens(text) <= max_tokens:
            chunks.append(
                Chunk(
                    content=text,
                    source_path=doc.source_path,
                    heading_chain=chain,
                    chunk_index=chunk_idx,
                    doc_type=doc.doc_type,
                    metadata=doc.metadata,
                )
            )
            chunk_idx += 1
        else:
            for piece in _split_by_paragraphs(text, max_tokens, overlap_tokens):
                chunks.append(
                    Chunk(
                        content=piece,
                        source_path=doc.source_path,
                        heading_chain=chain,
                        chunk_index=chunk_idx,
                        doc_type=doc.doc_type,
                        metadata=doc.metadata,
                    )
                )
                chunk_idx += 1

    return chunks


def chunk_image_doc(doc: ParsedDocument) -> list[Chunk]:
    """Wrap an image document description as a single chunk."""
    return [
        Chunk(
            content=doc.content,
            source_path=doc.source_path,
            heading_chain=doc.title,
            chunk_index=0,
            doc_type="image",
            metadata=doc.metadata,
        )
    ]


def chunk_documents(
    docs: list[ParsedDocument],
    max_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Chunk a list of parsed documents."""
    all_chunks: list[Chunk] = []
    for doc in docs:
        if doc.doc_type == "image":
            chks = chunk_image_doc(doc)
            all_chunks.extend(chks)
            logger.debug("Chunked image %s -> %d chunks", doc.source_path, len(chks))
        else:
            chks = chunk_markdown(doc, max_tokens, overlap_tokens)
            all_chunks.extend(chks)
            logger.debug("Chunked %s -> %d chunks", Path(doc.source_path).name, len(chks))
    logger.info("Chunked %d documents into %d chunks total", len(docs), len(all_chunks))
    return all_chunks

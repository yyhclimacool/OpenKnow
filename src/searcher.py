from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .config import AppConfig, load_config
from .embedder import Embedder, create_embedder
from .store import SearchResult, VectorStore
from .parser import ParsedDocument, scan_directory
from .chunker import chunk_documents

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeSearchResult:
    content: str
    source_path: str
    heading_chain: str
    relevance_score: float
    doc_type: str = "markdown"

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "source_path": self.source_path,
            "heading_chain": self.heading_chain,
            "relevance_score": round(self.relevance_score, 4),
            "doc_type": self.doc_type,
        }


class KnowledgeBase:
    """High-level facade for indexing and searching the knowledge database."""

    def __init__(self, config: AppConfig | None = None):
        self.config = config or load_config()
        self._embedder: Embedder | None = None
        self._store = VectorStore(
            persist_dir=self.config.persist_dir_abs,
            collection_name=self.config.storage.collection_name,
        )

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = create_embedder(self.config.embedding)
        return self._embedder

    def add_directory(self, directory: str | Path) -> dict:
        directory = Path(directory).resolve()
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        dir_str = str(directory)
        if dir_str not in self.config.sources:
            self.config.sources.append(dir_str)
            self.config.save()

        logger.info("Indexing directory: %s", directory)
        docs = scan_directory(directory, self.config.vision)
        if not docs:
            raise ValueError(f"No markdown files found in {directory}")
            return {"directory": dir_str, "documents": 0, "chunks": 0}

        logger.info("Parsed %d documents, chunking...", len(docs))
        chunks = chunk_documents(docs)
        if not chunks:
            logger.warning("Chunking produced no chunks for %d documents", len(docs))
            return {"directory": dir_str, "documents": len(docs), "chunks": 0}

        logger.info("Generated %d chunks, computing embeddings (provider=%s)...",
                    len(chunks), self.config.embedding.provider)
        texts = [c.display_text for c in chunks]
        embeddings = self.embedder.embed_texts(texts)
        logger.info("Embeddings ready, upserting to vector store...")
        added = self._store.add_chunks(chunks, embeddings)

        logger.info("Indexed %d documents -> %d chunks from %s", len(docs), added, directory)
        return {"directory": dir_str, "documents": len(docs), "chunks": added}

    def add_content(
        self,
        content: str,
        title: str | None = None,
        tags: list[str] | None = None,
    ) -> dict:
        """Save user-provided content as a markdown file and index it."""
        notes_dir = self.config.notes_dir_abs
        notes_dir.mkdir(parents=True, exist_ok=True)

        notes_dir_str = str(notes_dir)
        if notes_dir_str not in self.config.sources:
            self.config.sources.append(notes_dir_str)
            self.config.save()

        if not title:
            first_line = content.strip().splitlines()[0] if content.strip() else ""
            title = re.sub(r"^#+\s*", "", first_line).strip() or "untitled"

        slug = re.sub(r"[^\w\u4e00-\u9fff-]", "_", title)[:60].strip("_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{slug}.md"
        file_path = notes_dir / filename

        has_h1 = bool(re.match(r"^\s*#\s+", content))
        lines: list[str] = []
        if not has_h1:
            lines.append(f"# {title}\n")
        if tags:
            lines.append(f"Tags: {', '.join(tags)}\n")
        lines.append(content)
        file_path.write_text("\n".join(lines), encoding="utf-8")

        logger.info("Saved note to %s", file_path)

        doc = ParsedDocument(
            source_path=str(file_path),
            content=file_path.read_text(encoding="utf-8"),
            title=title,
            doc_type="markdown",
        )
        chunks = chunk_documents([doc])
        if not chunks:
            return {"file": str(file_path), "chunks": 0}

        texts = [c.display_text for c in chunks]
        embeddings = self.embedder.embed_texts(texts)
        added = self._store.add_chunks(chunks, embeddings)

        logger.info("Indexed note: %s -> %d chunks", file_path.name, added)
        return {"file": str(file_path), "title": title, "chunks": added}

    def remove_directory(self, directory: str | Path) -> int:
        directory = Path(directory).resolve()
        dir_str = str(directory)
        deleted = self._store.delete_by_source(dir_str)
        if dir_str in self.config.sources:
            self.config.sources.remove(dir_str)
            self.config.save()
        return deleted

    def reindex_all(self) -> dict:
        self._store.clear()
        results = {}
        for src in list(self.config.sources):
            src_path = Path(src)
            if src_path.is_dir():
                results[src] = self.add_directory(src_path)
            else:
                logger.warning("Source directory not found, removing: %s", src)
                self.config.sources.remove(src)
                self.config.save()
        return results

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_filter: str | None = None,
    ) -> list[KnowledgeSearchResult]:
        query_embedding = self.embedder.embed_query(query)
        # 多取一些 chunk，按文档去重后仍能返回 top_k 个文档
        fetch_k = max(top_k * 5, 50)
        chunk_results = self._store.search(
            query_embedding, top_k=fetch_k, source_filter=source_filter
        )

        # 按 source_path 分组，取每个文档最佳分数及对应 chunk 内容（图片等无法读文本时用）
        doc_best: dict[str, tuple[float, str]] = {}
        for r in chunk_results:
            score = 1.0 - r.distance
            if r.source_path not in doc_best or score > doc_best[r.source_path][0]:
                doc_best[r.source_path] = (score, r.content)

        # 按分数排序，取 top_k 个文档
        sorted_docs = sorted(
            doc_best.items(), key=lambda x: x[1][0], reverse=True
        )[:top_k]

        out: list[KnowledgeSearchResult] = []
        for source_path, (relevance_score, chunk_content) in sorted_docs:
            try:
                full_content = Path(source_path).read_text(encoding="utf-8")
            except OSError:
                full_content = chunk_content
            out.append(
                KnowledgeSearchResult(
                    content=full_content,
                    source_path=source_path,
                    heading_chain="",
                    relevance_score=relevance_score,
                    doc_type="markdown",
                )
            )
        return out

    def get_stats(self) -> dict:
        stats = self._store.get_stats()
        stats["configured_sources"] = self.config.sources
        return stats

    def list_sources(self) -> list[str]:
        return list(self.config.sources)

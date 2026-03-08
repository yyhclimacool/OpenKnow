from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings

from .chunker import Chunk

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    content: str
    source_path: str
    heading_chain: str
    distance: float
    doc_type: str = "markdown"
    chunk_index: int = 0


class VectorStore:
    """Vector store backed by ChromaDB with cosine similarity search."""

    def __init__(self, persist_dir: str | Path, collection_name: str = "knowledge_base"):
        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(self._persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "VectorStore initialized at %s (%d chunks)",
            self._persist_dir,
            self._collection.count(),
        )

    @staticmethod
    def _chunk_id(chunk: Chunk) -> str:
        raw = f"{chunk.source_path}::{chunk.heading_chain}::{chunk.chunk_index}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        if not chunks:
            return 0

        ids = [self._chunk_id(c) for c in chunks]
        documents = [c.display_text for c in chunks]
        metadatas = [
            {
                "source_path": c.source_path,
                "heading_chain": c.heading_chain,
                "chunk_index": c.chunk_index,
                "doc_type": c.doc_type,
            }
            for c in chunks
        ]

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("Upserted %d chunks", len(chunks))
        return len(chunks)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        source_filter: str | None = None,
    ) -> list[SearchResult]:
        count = self._collection.count()
        if count == 0:
            return []

        # Fetch more candidates when filtering so we still have top_k after pruning
        fetch_k = min(count, top_k * 4 if source_filter else top_k)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_k,
            include=["documents", "metadatas", "distances"],
        )

        search_results: list[SearchResult] = []
        docs = results.get("documents") or []
        metas = results.get("metadatas") or []
        dists = results.get("distances") or []
        if not docs or not docs[0]:
            return search_results

        for doc, meta, dist in zip(docs[0], metas[0], dists[0]):
            src = meta.get("source_path", "")
            if source_filter and source_filter not in src:
                continue
            search_results.append(
                SearchResult(
                    content=doc,
                    source_path=src,
                    heading_chain=meta.get("heading_chain", ""),
                    distance=dist,
                    doc_type=meta.get("doc_type", "markdown"),
                    chunk_index=meta.get("chunk_index", 0),
                )
            )
            if len(search_results) >= top_k:
                break

        return search_results

    def delete_by_source(self, source_path_prefix: str) -> int:
        all_data = self._collection.get(include=["metadatas"])
        ids_to_delete = [
            doc_id
            for doc_id, meta in zip(all_data["ids"], all_data["metadatas"])
            if meta.get("source_path", "").startswith(source_path_prefix)
        ]

        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
            logger.info(
                "Deleted %d chunks matching source: %s",
                len(ids_to_delete),
                source_path_prefix,
            )
        return len(ids_to_delete)

    def get_stats(self) -> dict:
        all_data = self._collection.get(include=["metadatas"])
        sources: dict[str, int] = {}
        for meta in all_data["metadatas"]:
            src = meta.get("source_path", "unknown")
            sources[src] = sources.get(src, 0) + 1

        unique_dirs: set[str] = set()
        for src in sources:
            parts = Path(src).parts
            if len(parts) >= 2:
                unique_dirs.add(str(Path(*parts[:-1])))

        return {
            "total_chunks": self._collection.count(),
            "total_files": len(sources),
            "unique_directories": len(unique_dirs),
            "files": sources,
        }

    def clear(self) -> None:
        name = self._collection.name
        meta = self._collection.metadata
        self._client.delete_collection(name)
        self._collection = self._client.get_or_create_collection(
            name=name,
            metadata=meta,
        )
        logger.info("Collection cleared")

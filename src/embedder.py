from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from .config import EmbeddingConfig

logger = logging.getLogger(__name__)

BATCH_SIZE = 128


class Embedder(ABC):
    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        ...

    @abstractmethod
    def dimension(self) -> int:
        ...


class OpenAIEmbedder(Embedder):
    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI

        self._client = OpenAI()
        self._model = model
        self._dim: int | None = None

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(input=texts, model=self._model)
        embeddings = [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
        if self._dim is None and embeddings:
            self._dim = len(embeddings[0])
        return embeddings

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        all_embeddings: list[list[float]] = []
        total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
        for i in range(0, len(texts), BATCH_SIZE):
            batch_num = i // BATCH_SIZE + 1
            batch = texts[i : i + BATCH_SIZE]
            if total_batches > 1:
                logger.info("Embedding batch %d/%d (%d texts)", batch_num, total_batches, len(batch))
            all_embeddings.extend(self._embed_batch(batch))
        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        return self._embed_batch([query])[0]

    def dimension(self) -> int:
        if self._dim is None:
            probe = self.embed_query("dimension probe")
            self._dim = len(probe)
        return self._dim


class LocalEmbedder(Embedder):
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local embedding. "
                "Install it with: uv pip install 'knowledge-db[local]'"
            )
        logger.info("Loading local embedding model: %s", model_name)
        try:
            self._model = SentenceTransformer(model_name, local_files_only=True)
            logger.info("Loaded model from local cache")
        except Exception:
            logger.info("Local cache miss, downloading model: %s", model_name)
            self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        logger.info("Encoding %d texts with local model...", len(texts))
        embeddings = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]

    def dimension(self) -> int:
        return self._dim


def create_embedder(config: EmbeddingConfig) -> Embedder:
    if config.provider == "local":
        return LocalEmbedder(config.local_model)
    return OpenAIEmbedder(config.openai_model)

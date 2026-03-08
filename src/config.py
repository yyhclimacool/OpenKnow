from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config.json"

load_dotenv(_PROJECT_ROOT / ".env")


@dataclass
class EmbeddingConfig:
    provider: str = "openai"
    openai_model: str = "text-embedding-3-small"
    local_model: str = "BAAI/bge-small-zh-v1.5"


@dataclass
class VisionConfig:
    provider: str = "openai"
    model: str = "gpt-4o"
    api_key: str | None = None
    base_url: str | None = None
    skip: bool = False  # True 时跳过图片描述，不调用 Vision API


@dataclass
class StorageConfig:
    persist_dir: str = "./data/chromadb"
    collection_name: str = "knowledge_base"
    notes_dir: str = "./knowledge/notes"


@dataclass
class AppConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    sources: list[str] = field(default_factory=list)

    @property
    def persist_dir_abs(self) -> Path:
        p = Path(self.storage.persist_dir)
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        return p

    @property
    def notes_dir_abs(self) -> Path:
        p = Path(self.storage.notes_dir)
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        return p

    @property
    def config_path(self) -> Path:
        return self._config_path

    def save(self, path: Path | None = None) -> None:
        path = path or self._config_path
        data = {
            "embedding": {
                "provider": self.embedding.provider,
                "openai_model": self.embedding.openai_model,
                "local_model": self.embedding.local_model,
            },
            "vision": {
                "provider": self.vision.provider,
                "model": self.vision.model,
                "skip": self.vision.skip,
                **({"api_key": self.vision.api_key} if self.vision.api_key else {}),
                **({"base_url": self.vision.base_url} if self.vision.base_url else {}),
            },
            "storage": {
                "persist_dir": self.storage.persist_dir,
                "collection_name": self.storage.collection_name,
                "notes_dir": self.storage.notes_dir,
            },
            "sources": self.sources,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def load_config(path: Path | str | None = None) -> AppConfig:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        cfg = AppConfig()
        cfg._config_path = path  # type: ignore[attr-defined]
        return cfg

    with open(path, encoding="utf-8") as f:
        raw = json.load(f) or {}

    emb = raw.get("embedding", {})
    vis = raw.get("vision", {})
    sto = raw.get("storage", {})

    cfg = AppConfig(
        embedding=EmbeddingConfig(
            provider=emb.get("provider", "openai"),
            openai_model=emb.get("openai_model", "text-embedding-3-small"),
            local_model=emb.get("local_model", "BAAI/bge-small-zh-v1.5"),
        ),
        vision=VisionConfig(
            provider=vis.get("provider", "openai"),
            model=vis.get("model", "gpt-4o"),
            api_key=vis.get("api_key") or None,
            base_url=vis.get("base_url") or None,
            skip=vis.get("skip", False),
        ),
        storage=StorageConfig(
            persist_dir=sto.get("persist_dir", "./data/chromadb"),
            collection_name=sto.get("collection_name", "knowledge_base"),
            notes_dir=sto.get("notes_dir", "./knowledge/notes"),
        ),
        sources=raw.get("sources") or [],
    )
    cfg._config_path = path  # type: ignore[attr-defined]
    return cfg

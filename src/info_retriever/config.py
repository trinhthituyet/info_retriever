"""Runtime configuration, loaded from the environment / .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    db_path: Path
    blob_dir: Path

    extract_model: str
    agent_model: str
    agent_effort: str

    embed_model: str
    embed_dim: int
    embed_query_prefix: str
    embed_passage_prefix: str


@lru_cache(maxsize=1)
def settings() -> Settings:
    load_dotenv()

    data_dir = Path(os.getenv("DATA_DIR", "./data")).expanduser().resolve()
    blob_dir = data_dir / "blobs"
    data_dir.mkdir(parents=True, exist_ok=True)
    blob_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        data_dir=data_dir,
        db_path=data_dir / "documents.db",
        blob_dir=blob_dir,
        extract_model=os.getenv("EXTRACT_MODEL", "claude-opus-5"),
        agent_model=os.getenv("AGENT_MODEL", "claude-opus-5"),
        agent_effort=os.getenv("AGENT_EFFORT", "medium"),
        embed_model=os.getenv("EMBED_MODEL", "BAAI/bge-m3"),
        embed_dim=int(os.getenv("EMBED_DIM", "1024")),
        embed_query_prefix=os.getenv("EMBED_QUERY_PREFIX", ""),
        embed_passage_prefix=os.getenv("EMBED_PASSAGE_PREFIX", ""),
    )

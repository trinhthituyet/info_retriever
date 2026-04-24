"""Local embedding model. Nothing here touches the network after first download."""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from .config import settings


@lru_cache(maxsize=1)
def _model():
    # Imported lazily: sentence-transformers pulls in torch, which is slow to
    # import and unnecessary for commands that never embed (list, delete, ...).
    from sentence_transformers import SentenceTransformer

    cfg = settings()
    model = SentenceTransformer(cfg.embed_model)
    actual = model.get_sentence_embedding_dimension()
    if actual != cfg.embed_dim:
        raise RuntimeError(
            f"EMBED_DIM is {cfg.embed_dim} but {cfg.embed_model} produces {actual}-dim vectors. "
            f"Set EMBED_DIM={actual} in .env, then delete {cfg.db_path.name} and re-ingest "
            f"(the vector table's dimension is fixed at creation)."
        )
    return model


def embed_passages(texts: Sequence[str]) -> list[list[float]]:
    prefix = settings().embed_passage_prefix
    payload = [f"{prefix}{text}" for text in texts] if prefix else list(texts)
    vectors = _model().encode(payload, normalize_embeddings=True, show_progress_bar=False)
    return [vector.tolist() for vector in vectors]


def embed_query(text: str) -> list[float]:
    prefix = settings().embed_query_prefix
    vector = _model().encode(
        f"{prefix}{text}" if prefix else text,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vector.tolist()

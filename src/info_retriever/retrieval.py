"""Hybrid retrieval: dense vectors + BM25 keyword, fused with Reciprocal Rank Fusion.

RRF is used rather than score normalisation because cosine distance and BM25 live
on incomparable scales; ranks are all we can safely combine.
"""

from __future__ import annotations

from typing import Any

from . import db, embed

RRF_K = 60


def hybrid_search(
    query: str, *, limit: int = 8, doc_type: str | None = None
) -> list[dict[str, Any]]:
    pool = max(limit * 3, 20)
    dense = db.vector_search(embed.embed_query(query), limit=pool, doc_type=doc_type)
    sparse = db.keyword_search(query, limit=pool, doc_type=doc_type)

    scores: dict[int, float] = {}
    records: dict[int, dict[str, Any]] = {}

    for ranked in (dense, sparse):
        for rank, hit in enumerate(ranked):
            chunk_id = hit["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
            records.setdefault(chunk_id, hit)

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]

    results = []
    for chunk_id, score in ordered:
        hit = records[chunk_id]
        results.append(
            {
                "document_id": hit["document_id"],
                "document_title": hit["title"],
                "doc_type": hit["doc_type"],
                "page": hit["page"],
                "heading": hit["heading"],
                "content": hit["content"],
                "score": round(score, 5),
            }
        )
    return results

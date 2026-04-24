"""Hybrid retrieval: dense vectors + BM25 keyword, fused with Reciprocal Rank Fusion.

RRF is used rather than score normalisation because cosine distance and BM25 live
on incomparable scales; ranks are all we can safely combine. The same property makes
it the natural way to fuse *several* queries — a translated question, a
contract-vocabulary rewrite, and a corpus-language variant each produce their own
ranking, and RRF merges them without needing their scores to be comparable.
"""

from __future__ import annotations

from typing import Any, Sequence

from . import db, embed

RRF_K = 60


def hybrid_search(
    query: str | Sequence[str], *, limit: int = 8, doc_type: str | None = None
) -> list[dict[str, Any]]:
    """Rank chunks against one or more queries.

    Passing several queries is how a rewritten or translated question is used: each
    contributes a ranking and RRF fuses them, so a chunk that several phrasings agree
    on outranks one that only matched a single wording.
    """
    queries = [query] if isinstance(query, str) else [q for q in query if q and q.strip()]
    if not queries:
        return []

    pool = max(limit * 3, 20)
    scores: dict[int, float] = {}
    records: dict[int, dict[str, Any]] = {}
    matched_by: dict[int, set[str]] = {}

    for text in queries:
        dense = db.vector_search(embed.embed_query(text), limit=pool, doc_type=doc_type)
        sparse = db.keyword_search(text, limit=pool, doc_type=doc_type)

        for ranked in (dense, sparse):
            for rank, hit in enumerate(ranked):
                chunk_id = hit["chunk_id"]
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
                records.setdefault(chunk_id, hit)
                matched_by.setdefault(chunk_id, set()).add(text)

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
                # Which phrasings found this chunk — useful when diagnosing whether a
                # rewrite or a translation is what actually earned the hit.
                "matched_queries": sorted(matched_by[chunk_id]),
            }
        )
    return results

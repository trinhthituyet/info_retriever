"""Storage backend interface, and everything portable between the two.

The split is narrow on purpose. Almost all of this app's SQL is ordinary — the parts
that genuinely differ between SQLite and Postgres are:

* **DDL** — ``vec0``/``fts5`` virtual tables versus a ``vector`` column with an HNSW
  index and a ``tsvector`` column with a GIN index.
* **Vector search** — ``embedding match ? and k = ?`` versus the ``<=>`` distance
  operator with ``order by``.
* **Keyword search** — ``bm25(chunk_fts)`` versus ``ts_rank_cd``.
* **Autoincrement, JSON columns and parameter markers.**

Everything else is shared here, so a change to (say) the conversation queries happens
once and applies to both. A backend supplies the four divergent pieces and the
connection; it does not reimplement the app's queries.

Portable SQL is written with ``?`` markers and translated per backend. That is safe
only because no query in this app contains a literal ``?`` inside a string — worth
remembering before adding one.
"""

from __future__ import annotations

import json
import re
import uuid
from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Any, Iterator, Sequence

TITLE_MAX = 70


def now() -> str:
    """Microsecond-precision ISO timestamp.

    Not ``timespec="seconds"``: two turns in the same second would tie, and
    conversation ordering is by ``updated_at``. Ties made the most-recent
    conversation land arbitrarily in the list.
    """
    return datetime.now().isoformat()


def today() -> str:
    return date.today().isoformat()


def title_from(question: str) -> str:
    collapsed = " ".join(question.split())
    if len(collapsed) <= TITLE_MAX:
        return collapsed
    return collapsed[: TITLE_MAX - 1].rstrip() + "…"


def normalise_date(raw: str | None) -> str | None:
    """Best-effort ISO-8601 normalisation of a model-supplied date string.

    Returns ``None`` rather than raising when the value is unparseable, so a
    sloppy date in one scanned contract never blocks ingestion.
    """
    if not raw:
        return None
    text = raw.strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d.%m.%Y", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    if re.fullmatch(r"\d{4}-\d{2}", text):
        return f"{text}-01"
    if re.fullmatch(r"\d{4}", text):
        return f"{text}-01-01"
    return None


_FTS_SPECIAL = re.compile(r'["*():^&|!<>]')


def keyword_terms(query: str) -> list[str]:
    """Split a user query into literal terms, stripped of operator characters.

    Both backends need this: FTS5 and ``to_tsquery`` each treat punctuation as syntax,
    so user input has to be reduced to bare terms before it is quoted or joined.
    """
    return [term for term in _FTS_SPECIAL.sub(" ", query).split() if term]


class Backend(ABC):
    """One storage engine. Supplies a connection, the DDL, and the two search queries."""

    #: Short identifier surfaced in /api/stats.
    name: str = "unknown"

    # ------------------------------------------------------------- connection --

    @abstractmethod
    def connect(self) -> Any:
        """A new connection whose rows behave like mappings."""

    @abstractmethod
    def session(self) -> Iterator[Any]:
        """Context manager yielding a connection, committing on success."""

    @abstractmethod
    def execute(self, conn: Any, sql: str, params: Sequence[Any] = ()) -> Any:
        """Run portable ``?``-marker SQL, returning a cursor-like object."""

    @abstractmethod
    def init_schema(self) -> None:
        """Create tables and indexes. Safe to call repeatedly."""

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Non-secret backend state for /api/stats."""

    # ----------------------------------------------------------------- writes --

    @abstractmethod
    def insert_chunk(self, conn: Any, document_id: str, ordinal: int, chunk: dict[str, Any],
                     vector: Sequence[float]) -> None:
        """Insert one chunk plus its vector and keyword index entries."""

    @abstractmethod
    def delete_chunks(self, conn: Any, document_id: str) -> None:
        """Remove a document's chunks and any satellite index rows."""

    # ---------------------------------------------------------------- searches --

    @abstractmethod
    def vector_search(self, embedding: Sequence[float], *, limit: int,
                      doc_type: str | None) -> list[dict[str, Any]]:
        """Nearest chunks by cosine distance."""

    @abstractmethod
    def keyword_search(self, query: str, *, limit: int,
                       doc_type: str | None) -> list[dict[str, Any]]:
        """Best chunks by lexical relevance."""

    # ----------------------------------------------------- JSON round-tripping --

    def dump_json(self, value: Any) -> Any:
        """Encode for storage. SQLite stores text; Postgres stores jsonb natively."""
        return json.dumps(value, ensure_ascii=False)

    def load_json(self, value: Any) -> Any:
        """Decode from storage, tolerating a driver that already parsed it."""
        if value is None:
            return None
        if isinstance(value, (dict, list)):
            return value
        return json.loads(value)


def new_id() -> str:
    return str(uuid.uuid4())

"""Provider-agnostic interface for the five model operations this app needs.

The seam is deliberately at the *semantic* level — transcribe, classify, extract,
run the agent, cite — not at "send a chat message". Anthropic and an
OpenAI-compatible vLLM server disagree about almost everything below that line
(structured outputs, tool-call plumbing, how documents are attached, whether
citations exist at all), so a lower seam would leak one provider's shape into the
other. At this level each implementation owns its own wire format and the callers
stay identical.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ..loaders import LoadedFile
from ..schemas import BaseContract, Classification, QueryPlan

#: ``emit(kind, payload)`` — progress channel. Kinds: ``stage``, ``tool``,
#: ``delta`` (streamed answer text), ``draft``.
Emit = Callable[[str, dict[str, Any]], None]


def no_emit(kind: str, payload: dict[str, Any]) -> None:
    pass


@dataclass
class Turn:
    """One completed exchange, as replayed into a later request.

    Carries the *outcome* of a turn, never its tool transcript. A single
    ``read_document`` result can be 60 KB, so replaying transcripts would exhaust
    the context window within a few turns; the documents remain in the corpus and
    the agent re-reads them when a follow-up needs them.
    """

    question: str
    answer: str
    sources: list[str] = field(default_factory=list)


@dataclass
class AgentResult:
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CitedResult:
    text: str
    citations: list[dict[str, Any]] = field(default_factory=list)


def render_history(
    turns: Sequence[Turn], *, max_turns: int = 12, max_chars: int = 12_000
) -> list[tuple[str, str]]:
    """``[(role, content), ...]`` oldest-first, trimmed to fit.

    Trims from the **oldest** end: a follow-up almost always refers to the last
    thing said, so recent turns are the ones worth keeping. Cited sources are
    appended to each answer because that is what lets "what about that one?"
    resolve — without them an elliptical follow-up has no referent.
    """
    recent = list(turns)[-max_turns:] if max_turns > 0 else []

    rendered: list[tuple[str, str]] = []
    budget = max_chars
    for turn in reversed(recent):
        answer = turn.answer
        if turn.sources:
            answer = f"{answer}\n\n[Answered from: {', '.join(turn.sources)}]"
        cost = len(turn.question) + len(answer)
        if rendered and budget - cost < 0:
            break
        budget -= cost
        rendered.append(("assistant", answer))
        rendered.append(("user", turn.question))

    rendered.reverse()
    return rendered


class ProviderError(RuntimeError):
    """The provider could not complete the operation."""


class ModelRefused(ProviderError):
    """The model declined the request on safety grounds."""


class LLMProvider(ABC):
    """One backing model, and everything needed to talk to it."""

    #: Short identifier surfaced in /api/stats.
    name: str = "unknown"

    #: False when the provider cannot produce span-accurate citations natively and
    #: relies on locating quotes in the source text instead. Purely informational —
    #: both paths return the same ``CitedResult`` shape.
    native_citations: bool = False

    @abstractmethod
    def transcribe(self, loaded: LoadedFile) -> str:
        """Vision transcription of a scan or photo, with ``@@PAGE:N@@`` markers."""

    @abstractmethod
    def classify(self, text: str) -> Classification:
        """Decide the document type and write a short summary."""

    @abstractmethod
    def extract_fields(self, text: str, doc_type: str) -> BaseContract:
        """Fill the per-document-type schema from the document text."""

    @abstractmethod
    def plan_query(self, question: str) -> QueryPlan:
        """Detect the question's language, render it in English, and derive search terms.

        Documents are assumed to be English, so the search queries are English too.
        """

    @abstractmethod
    def run_agent(
        self,
        *,
        question: str,
        instructions: str,
        catalogue: str,
        tools: Sequence[Any],
        history: Sequence[Turn] = (),
        emit: Emit = no_emit,
    ) -> AgentResult:
        """Answer using the tools, returning the draft and the calls it made.

        ``instructions`` and ``catalogue`` are passed separately rather than as one
        prompt because Anthropic wants them as two cacheable system blocks while an
        OpenAI-compatible server wants a single system string.

        ``history`` is prior turns in this conversation, oldest first.
        """

    @abstractmethod
    def cite(
        self,
        *,
        question: str,
        draft: str,
        documents: Sequence[Mapping[str, Any]],
        history: Sequence[Turn] = (),
        language: str = "en",
        emit: Emit = no_emit,
    ) -> CitedResult:
        """Re-answer from the source documents, attaching verbatim citations.

        ``history`` matters here too: an elliptical follow-up ("and the deposit?")
        is uninterpretable without it.

        ``language`` is the ISO 639-1 code the answer must be written in. This pass
        produces the text the user reads, so the rule has to arrive here — a language
        instruction given only to :meth:`run_agent` governs the draft, not the output.
        """

    def describe(self) -> dict[str, Any]:
        return {"llm_provider": self.name, "native_citations": self.native_citations}

    def reset(self) -> None:
        """Drop any cached client or credential."""

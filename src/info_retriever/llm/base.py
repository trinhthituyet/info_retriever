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
from ..schemas import BaseContract, Classification

#: ``emit(kind, payload)`` — progress channel. Kinds: ``stage``, ``tool``,
#: ``delta`` (streamed answer text), ``draft``.
Emit = Callable[[str, dict[str, Any]], None]


def no_emit(kind: str, payload: dict[str, Any]) -> None:
    pass


@dataclass
class AgentResult:
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CitedResult:
    text: str
    citations: list[dict[str, Any]] = field(default_factory=list)


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
    def run_agent(
        self,
        *,
        question: str,
        instructions: str,
        catalogue: str,
        tools: Sequence[Any],
        emit: Emit = no_emit,
    ) -> AgentResult:
        """Answer using the tools, returning the draft and the calls it made.

        ``instructions`` and ``catalogue`` are passed separately rather than as one
        prompt because Anthropic wants them as two cacheable system blocks while an
        OpenAI-compatible server wants a single system string.
        """

    @abstractmethod
    def cite(
        self,
        *,
        question: str,
        draft: str,
        documents: Sequence[Mapping[str, Any]],
        emit: Emit = no_emit,
    ) -> CitedResult:
        """Re-answer from the source documents, attaching verbatim citations."""

    def describe(self) -> dict[str, Any]:
        return {"llm_provider": self.name, "native_citations": self.native_citations}

    def reset(self) -> None:
        """Drop any cached client or credential."""

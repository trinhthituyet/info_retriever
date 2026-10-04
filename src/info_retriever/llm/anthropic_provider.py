"""Anthropic provider: Claude via Apple's Floodgate gateway, or the public API.

Uses the features that only exist here — native PDF document blocks, structured
outputs, the beta tool runner, and real citation spans.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Mapping, Sequence

import anthropic

from .. import auth, loaders
from ..config import settings
from ..loaders import LoadedFile
from ..schemas import (
    BaseContract,
    Classification,
    DraftAssessment,
    QueryPlan,
    extraction_model_for,
)
from . import prompts
from .base import (
    AgentResult,
    CitedResult,
    Emit,
    LLMProvider,
    ModelRefused,
    Turn,
    no_emit,
    render_history,
)


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    native_citations = True

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._client: anthropic.Anthropic | None = None
        self._token: str | None = None

    # ------------------------------------------------------------------ client --

    def client(self) -> anthropic.Anthropic:
        """A client for the configured gateway, rebuilt when the token rotates.

        The SDK takes ``auth_token`` as a fixed string at construction and an
        appleconnect token is short-lived relative to how long this server runs, so
        the client is replaced whenever the credential changes. Callers always go
        through this method, which is what makes the rotation invisible to them.
        """
        cfg = settings()

        # Floodgate's certificate chains to an Apple-internal root that the stock
        # certifi bundle does not carry, so point httpx at the apple-certifi one.
        # An SSLContext, not a path string: httpx deprecated `verify=<str>`.
        http_client = None
        if cfg.ca_bundle:
            import ssl

            http_client = anthropic.DefaultHttpxClient(
                verify=ssl.create_default_context(cafile=cfg.ca_bundle)
            )

        if cfg.auth_mode == "default":
            with self._lock:
                if self._client is None:
                    # Resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an
                    # `ant auth login` profile — in that order.
                    self._client = anthropic.Anthropic(
                        base_url=cfg.base_url, http_client=http_client
                    )
                return self._client

        token = auth.auth_token()
        with self._lock:
            if self._client is None or self._token != token:
                self._client = anthropic.Anthropic(
                    auth_token=token, base_url=cfg.base_url, http_client=http_client
                )
                self._token = token
            return self._client

    def reset(self) -> None:
        with self._lock:
            self._client = None
            self._token = None
        auth.invalidate()

    def describe(self) -> dict[str, Any]:
        cfg = settings()
        return {
            **super().describe(),
            "model": cfg.agent_model,
            "effort": cfg.agent_effort,
            "ca_bundle": cfg.ca_bundle,
            **auth.describe(),
        }

    @staticmethod
    def _check(response: Any) -> Any:
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise ModelRefused(
                f"Claude declined this request (category: {category or 'unspecified'})."
            )
        return response

    # ------------------------------------------------------------- extraction --

    def transcribe(self, loaded: LoadedFile) -> str:
        with self.client().messages.stream(
            model=settings().extract_model,
            max_tokens=64000,
            system=prompts.TRANSCRIBE_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": [
                        *loaded.content_blocks,
                        {"type": "text", "text": prompts.transcribe_user_prompt()},
                    ],
                }
            ],
        ) as stream:
            response = self._check(stream.get_final_message())

        return "\n".join(block.text for block in response.content if block.type == "text").strip()

    def _parse(self, system: str, prompt: str, output_format: type, *, effort: str | None = None) -> Any:
        kwargs: dict[str, Any] = {}
        if effort:
            kwargs["output_config"] = {"effort": effort}
        response = self._check(
            self.client().messages.parse(
                model=settings().extract_model,
                max_tokens=16000,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_format=output_format,
                **kwargs,
            )
        )
        if response.parsed_output is None:
            raise ModelRefused(f"{output_format.__name__} extraction returned no parsed output.")
        return response.parsed_output

    def classify(self, text: str) -> Classification:
        return self._parse(
            prompts.CLASSIFY_SYSTEM, prompts.classify_user_prompt(text), Classification
        )

    def extract_fields(self, text: str, doc_type: str) -> BaseContract:
        return self._parse(
            prompts.EXTRACT_SYSTEM,
            prompts.extract_user_prompt(text, doc_type),
            extraction_model_for(doc_type),
        )

    def plan_query(self, question: str) -> QueryPlan:
        # Translation and keyword extraction are shallow work that sits in front of
        # every question, so run it at low effort rather than the agent's setting.
        return self._parse(
            prompts.QUERY_PLAN_SYSTEM,
            prompts.query_plan_user_prompt(question),
            QueryPlan,
            effort="low",
        )

    def assess_draft(
        self,
        *,
        question: str,
        draft: str,
        documents_read: Sequence[tuple[str, str]],
        catalogue: str,
    ) -> DraftAssessment:
        # A bookkeeping judgement — which claims rest on unread documents — not
        # reasoning about the contracts themselves, so low effort is enough.
        return self._parse(
            prompts.ASSESS_DRAFT_SYSTEM,
            prompts.assess_draft_user_prompt(question, draft, list(documents_read), catalogue),
            DraftAssessment,
            effort="low",
        )

    # ------------------------------------------------------------------ agent --

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
        cfg = settings()
        emit("stage", {"stage": "searching", "detail": "reading the document catalogue"})

        messages = [
            {"role": role, "content": content}
            for role, content in render_history(
                history,
                max_turns=cfg.history_max_turns,
                max_chars=cfg.history_max_chars,
            )
        ]
        messages.append({"role": "user", "content": question})

        runner = self.client().beta.messages.tool_runner(
            model=cfg.agent_model,
            max_tokens=16000,
            system=[
                {"type": "text", "text": instructions},
                # Stable content first: the catalogue changes only on ingest, so the
                # breakpoint here caches instructions and catalogue together. History
                # lives in `messages`, after the breakpoint, so it grows without
                # invalidating the cached prefix.
                {"type": "text", "text": catalogue, "cache_control": {"type": "ephemeral"}},
            ],
            tools=list(tools),
            output_config={"effort": cfg.agent_effort},
            messages=messages,
        )

        tool_calls: list[dict[str, Any]] = []
        final_text = ""

        for message in runner:
            self._check(message)
            texts = [block.text for block in message.content if block.type == "text"]
            if texts:
                final_text = "\n".join(texts).strip()
            for block in message.content:
                if block.type == "tool_use":
                    call = {"name": block.name, "input": block.input}
                    tool_calls.append(call)
                    emit("tool", call)

        emit("draft", {"text": final_text})
        return AgentResult(text=final_text, tool_calls=tool_calls)

    # ---------------------------------------------------------------- citation --

    @staticmethod
    def _citable_block(document: Mapping[str, Any]) -> dict[str, Any] | None:
        """Prefer the original PDF, which yields page-level citations. Fall back to
        the extracted text, which yields character offsets. Images are not citable."""
        title = document.get("title") or document.get("original_name")

        if document.get("mime_type") == "application/pdf":
            blob = Path(str(document.get("file_path", "")))
            if blob.is_file():
                try:
                    loaded = loaders.load(blob)
                except loaders.UnsupportedFile:
                    loaded = None
                if loaded is not None:
                    block = dict(loaded.content_blocks[0])
                    block["title"] = title
                    block["citations"] = {"enabled": True}
                    return block

        text = document.get("full_text") or ""
        if not text.strip():
            return None
        return {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": text},
            "title": title,
            "citations": {"enabled": True},
        }

    def answer_from_excerpts(
        self,
        *,
        question: str,
        excerpts: Sequence[Mapping[str, Any]],
        history: Sequence[Turn] = (),
        language: str = "en",
        emit: Emit = no_emit,
    ) -> str:
        from .. import db

        cfg = settings()
        messages: list[dict[str, Any]] = [
            {"role": role, "content": content}
            for role, content in render_history(
                history,
                max_turns=cfg.history_max_turns,
                max_chars=cfg.history_max_chars,
            )
        ]
        messages.append(
            {
                "role": "user",
                "content": prompts.excerpt_answer_user_prompt(
                    question, [dict(e) for e in excerpts], db.today(), language
                ),
            }
        )
        with self.client().messages.stream(
            model=cfg.agent_model,
            max_tokens=16000,
            output_config={"effort": cfg.agent_effort},
            system=prompts.EXCERPT_ANSWER_SYSTEM,
            messages=messages,
        ) as stream:
            for delta in stream.text_stream:
                emit("delta", {"text": delta})
            response = self._check(stream.get_final_message())
        return "".join(b.text for b in response.content if b.type == "text").strip()

    def cite(
        self,
        *,
        question: str,
        draft: str,
        documents: Sequence[Mapping[str, Any]],
        history: Sequence[Turn] = (),
        language: str = "en",
        unattached: Sequence[str] = (),
        emit: Emit = no_emit,
    ) -> CitedResult:
        from .. import db

        blocks = [block for block in (self._citable_block(d) for d in documents) if block]
        if not blocks:
            return CitedResult(text=draft)

        emit(
            "stage",
            {
                "stage": "citing",
                "detail": f"verifying against {len(blocks)} document{'s' if len(blocks) != 1 else ''}",
            },
        )

        cfg = settings()
        # Prior turns go before the documents so an elliptical follow-up
        # ("and the deposit?") has a referent by the time the question arrives.
        messages: list[dict[str, Any]] = [
            {"role": role, "content": content}
            for role, content in render_history(
                history,
                max_turns=cfg.history_max_turns,
                max_chars=cfg.history_max_chars,
            )
        ]
        messages.append(
            {
                "role": "user",
                "content": [
                    *blocks,
                    {
                        "type": "text",
                        "text": prompts.cite_user_prompt(
                            question, draft, db.today(), language, unattached
                        ),
                    },
                ],
            }
        )

        with self.client().messages.stream(
            model=cfg.agent_model,
            max_tokens=16000,
            output_config={"effort": cfg.agent_effort},
            messages=messages,
        ) as stream:
            for delta in stream.text_stream:
                emit("delta", {"text": delta})
            response = stream.get_final_message()

        if response.stop_reason == "refusal":
            return CitedResult(text=draft)

        parts: list[str] = []
        citations: list[dict[str, Any]] = []
        for block in response.content:
            if block.type != "text":
                continue
            parts.append(block.text)
            for citation in getattr(block, "citations", None) or []:
                entry: dict[str, Any] = {
                    "document_title": getattr(citation, "document_title", None),
                    "cited_text": getattr(citation, "cited_text", ""),
                    "located": True,
                }
                if getattr(citation, "type", "") == "page_location":
                    entry["page"] = citation.start_page_number
                elif getattr(citation, "type", "") == "char_location":
                    entry["char_start"] = citation.start_char_index
                citations.append(entry)

        return CitedResult(text="".join(parts).strip(), citations=citations)

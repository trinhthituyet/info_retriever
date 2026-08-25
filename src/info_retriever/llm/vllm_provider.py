"""vLLM provider, over vLLM's OpenAI-compatible server.

Three things Anthropic gives for free have to be built by hand here:

* **Structured output** — via ``response_format`` json_schema (vLLM's guided
  decoding), then validated with Pydantic on our side rather than trusted.
* **The tool loop** — there is no tool runner, so the request/execute/feed-back
  cycle is written out. Requires the server to be started with
  ``--enable-auto-tool-choice`` and a ``--tool-call-parser``.
* **Citations** — no such feature exists, so the model is asked for the quotes it
  relied on and :mod:`info_retriever.llm.citations` locates them in the source text.
  That still yields real page numbers, because ingested text carries page markers.

PDFs have no native representation either: pages are rasterised (see
``loaders.as_image_parts``), so transcription needs a vision-capable served model.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Mapping, Sequence

from ..config import settings
from ..loaders import LoadedFile, as_image_parts
from ..schemas import BaseContract, Classification, extraction_model_for
from . import citations as citation_tools
from . import prompts
from .base import AgentResult, CitedResult, Emit, LLMProvider, ProviderError, no_emit

MAX_TOOL_ITERATIONS = 8


class VLLMProvider(LLMProvider):
    name = "vllm"
    native_citations = False

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._client: Any | None = None

    # ------------------------------------------------------------------ client --

    def client(self) -> Any:
        from openai import OpenAI

        cfg = settings()
        if not cfg.vllm_model:
            raise ProviderError(
                "VLLM_MODEL is not set. Set it to the model name your vLLM server "
                "serves — `curl $VLLM_BASE_URL/models` lists them."
            )

        with self._lock:
            if self._client is None:
                self._client = OpenAI(
                    base_url=cfg.vllm_base_url,
                    # vLLM ignores the key unless started with --api-key, but the
                    # SDK refuses to construct without one.
                    api_key=cfg.vllm_api_key or "not-needed",
                )
            return self._client

    def reset(self) -> None:
        with self._lock:
            self._client = None

    def describe(self) -> dict[str, Any]:
        cfg = settings()
        return {
            **super().describe(),
            "model": cfg.vllm_model or "(unset — VLLM_MODEL)",
            "base_url": cfg.vllm_base_url,
        }

    def _complete(self, **kwargs: Any) -> Any:
        cfg = settings()
        kwargs.setdefault("model", cfg.vllm_model)
        kwargs.setdefault("max_tokens", cfg.vllm_max_tokens)
        kwargs.setdefault("temperature", cfg.vllm_temperature)
        try:
            return self.client().chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - normalise into our error type
            import openai

            if isinstance(exc, openai.APIConnectionError):
                raise ProviderError(
                    f"Could not reach the vLLM server at {cfg.vllm_base_url}. "
                    "Check it is running and VLLM_BASE_URL is correct."
                ) from exc
            if isinstance(exc, openai.NotFoundError):
                raise ProviderError(
                    f"The vLLM server does not serve a model named {cfg.vllm_model!r}. "
                    f"`curl {cfg.vllm_base_url}/models` lists what it has."
                ) from exc
            if isinstance(exc, openai.BadRequestError):
                # Usually guided decoding or tool calling not enabled on the server.
                raise ProviderError(
                    f"The vLLM server rejected the request ({exc}). Structured output "
                    "needs a guided-decoding backend, and tool calling needs "
                    "--enable-auto-tool-choice with a --tool-call-parser."
                ) from exc
            if isinstance(exc, openai.APIStatusError):
                # Covers a proxy or load balancer answering instead of vLLM, which
                # otherwise surfaces as a bare 502 with no hint of where to look.
                raise ProviderError(
                    f"HTTP {exc.status_code} from {cfg.vllm_base_url} — this may be a "
                    f"proxy answering rather than vLLM itself. ({exc})"
                ) from exc
            raise

    @staticmethod
    def _text_of(response: Any) -> str:
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise ProviderError(
                "The vLLM response hit the token limit. Raise VLLM_MAX_TOKENS."
            )
        return (choice.message.content or "").strip()

    def _json_schema_call(
        self, *, system: str, prompt: str, output_format: type, schema_name: str
    ) -> Any:
        """Guided-decode into a Pydantic model, then validate rather than trust."""
        schema = output_format.model_json_schema()
        response = self._complete(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema, "strict": True},
            },
        )
        body = self._text_of(response)
        try:
            return output_format.model_validate_json(body)
        except Exception as exc:  # noqa: BLE001 - includes pydantic ValidationError
            raise ProviderError(
                f"{settings().vllm_model} returned output that does not match the "
                f"{schema_name} schema. If this recurs, the served model may be too "
                f"small for structured extraction. ({exc})"
            ) from exc

    # ------------------------------------------------------------- extraction --

    def transcribe(self, loaded: LoadedFile) -> str:
        parts = as_image_parts(loaded, dpi=settings().vllm_pdf_dpi)
        content: list[dict[str, Any]] = [
            {"type": "text", "text": prompts.transcribe_user_prompt()}
        ]
        for media_type, encoded in parts:
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{encoded}"}}
            )

        response = self._complete(
            messages=[
                {"role": "system", "content": prompts.TRANSCRIBE_SYSTEM},
                {"role": "user", "content": content},
            ],
            # Transcription output scales with page count, unlike the other calls.
            max_tokens=max(settings().vllm_max_tokens, 4096 * max(len(parts), 1)),
            temperature=0.0,
        )
        text = self._text_of(response)
        if not text:
            raise ProviderError(
                "The vLLM server returned an empty transcription. Is the served model "
                "vision-capable? A text-only model cannot read a scan."
            )
        return text

    def classify(self, text: str) -> Classification:
        return self._json_schema_call(
            system=prompts.CLASSIFY_SYSTEM,
            prompt=prompts.classify_user_prompt(text),
            output_format=Classification,
            schema_name="classification",
        )

    def extract_fields(self, text: str, doc_type: str) -> BaseContract:
        model = extraction_model_for(doc_type)
        return self._json_schema_call(
            system=prompts.EXTRACT_SYSTEM,
            prompt=prompts.extract_user_prompt(text, doc_type),
            output_format=model,
            schema_name=f"{doc_type}_contract",
        )

    # ------------------------------------------------------------------ agent --

    @staticmethod
    def _tool_specs(tools: Sequence[Any]) -> list[dict[str, Any]]:
        """Adapt the shared tool registry to OpenAI's function-calling shape.

        The registry stays Anthropic-decorated so there is one source of truth for
        names, descriptions and schemas; only the envelope differs.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema,
                },
            }
            for tool in tools
        ]

    def run_agent(
        self,
        *,
        question: str,
        instructions: str,
        catalogue: str,
        tools: Sequence[Any],
        emit: Emit = no_emit,
    ) -> AgentResult:
        by_name = {tool.name: tool for tool in tools}
        messages: list[dict[str, Any]] = [
            # One system string, not two cacheable blocks: an OpenAI-compatible
            # server has no prompt-cache breakpoints to place.
            {"role": "system", "content": f"{instructions}\n\n{catalogue}"},
            {"role": "user", "content": question},
        ]

        emit("stage", {"stage": "searching", "detail": "reading the document catalogue"})

        tool_calls: list[dict[str, Any]] = []
        final_text = ""

        for _ in range(MAX_TOOL_ITERATIONS):
            response = self._complete(
                messages=messages,
                tools=self._tool_specs(tools),
                tool_choice="auto",
            )
            message = response.choices[0].message
            requested = list(message.tool_calls or [])

            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    **(
                        {
                            "tool_calls": [
                                {
                                    "id": call.id,
                                    "type": "function",
                                    "function": {
                                        "name": call.function.name,
                                        "arguments": call.function.arguments,
                                    },
                                }
                                for call in requested
                            ]
                        }
                        if requested
                        else {}
                    ),
                }
            )

            if message.content:
                final_text = message.content.strip()

            if not requested:
                break

            for call in requested:
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}

                record = {"name": call.function.name, "input": arguments}
                tool_calls.append(record)
                emit("tool", record)

                tool = by_name.get(call.function.name)
                if tool is None:
                    result = f"No tool named {call.function.name}."
                else:
                    try:
                        result = str(tool.call(arguments))
                    except Exception as exc:  # noqa: BLE001 - report, let it retry
                        result = f"Tool failed: {type(exc).__name__}: {exc}"

                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )
        else:
            emit(
                "stage",
                {
                    "stage": "searching",
                    "detail": f"stopped after {MAX_TOOL_ITERATIONS} tool rounds",
                },
            )

        emit("draft", {"text": final_text})
        return AgentResult(text=final_text, tool_calls=tool_calls)

    # ---------------------------------------------------------------- citation --

    def cite(
        self,
        *,
        question: str,
        draft: str,
        documents: Sequence[Mapping[str, Any]],
        emit: Emit = no_emit,
    ) -> CitedResult:
        from .. import db

        usable = [d for d in documents if (d.get("full_text") or "").strip()]
        if not usable:
            return CitedResult(text=draft)

        emit(
            "stage",
            {
                "stage": "citing",
                "detail": f"verifying against {len(usable)} document{'s' if len(usable) != 1 else ''}",
            },
        )

        attached = "\n\n".join(
            f"<document id=\"{d.get('id')}\" title=\"{d.get('title')}\">\n"
            f"{d.get('full_text')}\n</document>"
            for d in usable
        )
        prompt = (
            f"{prompts.cite_user_prompt(question, draft, db.today())}\n\n{attached}"
        )

        response = self._complete(
            messages=[
                {"role": "system", "content": prompts.QUOTE_CITE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        body = self._text_of(response)

        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            # Not fatal: the prose is still usable, we just have no quotes to locate.
            emit("delta", {"text": body})
            return CitedResult(text=body or draft)

        answer = str(payload.get("answer") or "").strip() or draft
        emit("delta", {"text": answer})

        quotes = payload.get("quotes")
        if not isinstance(quotes, list):
            return CitedResult(text=answer)

        index = {str(d.get("id")): dict(d) for d in usable}
        located = citation_tools.attach(
            [q for q in quotes if isinstance(q, dict)],
            index,
        )
        return CitedResult(text=answer, citations=located)

"""End-to-end vLLM check against a fake OpenAI-compatible server.

Unlike test_providers.py this does not stub `_complete`, so the real `openai` SDK
builds and sends the request. It proves the wire shapes are actually accepted, not
just that our adapter code runs.

Sockets cannot be bound in every environment, so this is skipped when that fails
rather than reported as a failure.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest

RECEIVED: list[dict] = []
REPLIES: list[dict] = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence the default stderr logging
        pass

    def do_GET(self):
        if self.path.endswith("/models"):
            self._json({"object": "list", "data": [{"id": "fake-model"}]})
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        RECEIVED.append(json.loads(self.rfile.read(length) or b"{}"))
        message = REPLIES.pop(0) if REPLIES else {"content": "ok"}
        self._json(
            {
                "id": "cmpl-1",
                "object": "chat.completion",
                "created": 0,
                "model": "fake-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", **message},
                        "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
                    }
                ],
            }
        )

    def _json(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def fake_server(tmp_path, monkeypatch):
    RECEIVED.clear()
    REPLIES.clear()
    try:
        server = HTTPServer(("127.0.0.1", 0), Handler)
    except (PermissionError, OSError) as exc:
        pytest.skip(f"cannot bind a local socket here: {exc}")

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_PROVIDER", "vllm")
    monkeypatch.setenv("VLLM_MODEL", "fake-model")
    monkeypatch.setenv("VLLM_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")

    from info_retriever import llm
    from info_retriever.config import settings

    settings.cache_clear()
    llm.reset()
    try:
        yield llm
    finally:
        llm.reset()
        settings.cache_clear()
        server.shutdown()
        server.server_close()


def test_classify_round_trips_over_real_http(fake_server):
    REPLIES.append(
        {
            "content": json.dumps(
                {
                    "doc_type": "rental",
                    "title": "Lease",
                    "language": "en",
                    "summary": "A lease agreement.",
                }
            )
        }
    )

    result = fake_server.provider().classify("3. RENT\n1500 USD per month.")
    assert result.doc_type == "rental"
    assert result.title == "Lease"

    sent = RECEIVED[0]
    assert sent["model"] == "fake-model"
    assert sent["response_format"]["type"] == "json_schema"
    assert sent["messages"][0]["role"] == "system"


def test_tool_call_round_trips_over_real_http(fake_server):
    REPLIES.extend(
        [
            {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "read_document",
                            "arguments": '{"document_id": "doc-1"}',
                        },
                    }
                ],
            },
            {"content": "Sixty days notice."},
        ]
    )

    class Tool:
        name = "read_document"
        description = "Read a document."
        input_schema = {
            "type": "object",
            "properties": {"document_id": {"type": "string"}},
            "additionalProperties": False,
        }

        def call(self, arguments):
            return "document body"

    result = fake_server.provider().run_agent(
        question="notice period?", instructions="i", catalogue="c", tools=[Tool()]
    )

    assert result.text == "Sixty days notice."
    assert result.tool_calls == [{"name": "read_document", "input": {"document_id": "doc-1"}}]

    # The follow-up request must carry a well-formed tool result the server accepted.
    followup = RECEIVED[1]["messages"]
    assert followup[-1] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "document body",
    }
    assistant = followup[-2]
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["function"]["name"] == "read_document"


@pytest.mark.parametrize(
    ("exception_name", "expected"),
    [
        ("APIConnectionError", "VLLM_BASE_URL"),
        ("NotFoundError", "does not serve a model"),
        ("BadRequestError", "enable-auto-tool-choice"),
    ],
)
def test_transport_errors_are_translated_into_actionable_messages(
    tmp_path, monkeypatch, exception_name, expected
):
    """Injected rather than provoked over the network: an intervening proxy can turn
    an unreachable host into an unrelated status code, which made this flaky."""
    import httpx
    import openai

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_PROVIDER", "vllm")
    monkeypatch.setenv("VLLM_MODEL", "fake-model")

    from info_retriever import llm
    from info_retriever.config import settings

    settings.cache_clear()
    llm.reset()
    provider = llm.provider()

    request = httpx.Request("POST", "http://127.0.0.1:8001/v1/chat/completions")
    if exception_name == "APIConnectionError":
        error: Exception = openai.APIConnectionError(request=request)
    else:
        status = 404 if exception_name == "NotFoundError" else 400
        error = getattr(openai, exception_name)(
            message="boom",
            response=httpx.Response(status, request=request),
            body=None,
        )

    class Failing:
        def create(self, **kwargs):
            raise error

    monkeypatch.setattr(
        provider, "client", lambda: SimpleNamespace(chat=SimpleNamespace(completions=Failing()))
    )

    with pytest.raises(llm.ProviderError, match=expected):
        provider.classify("text")

    llm.reset()
    settings.cache_clear()

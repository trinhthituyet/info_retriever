"""Provider-layer tests: selection, the vLLM adapters, and quote locating.

No network and no vLLM server — the OpenAI client is stubbed. The point is to pin
the wire shapes each provider builds and the degradations it makes explicit.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from info_retriever.loaders import PAGE_MARKER

LEASE = (
    f"{PAGE_MARKER.format(page=1)}\n"
    "3. RENT\n"
    "Tenant shall pay rent of 1,500 USD per month, due on the first day of each month.\n"
    f"{PAGE_MARKER.format(page=2)}\n"
    "9. TERMINATION\n"
    "Either party may terminate by giving sixty (60) days written notice.\n"
)


# --------------------------------------------------------------------------- #
# quote locating (the vLLM citation substitute)
# --------------------------------------------------------------------------- #


def test_page_offsets_and_lookup():
    from info_retriever.llm import citations

    offsets = citations.page_offsets(LEASE)
    assert [page for _, page in offsets] == [1, 2]
    assert citations.page_at(offsets, LEASE.index("Tenant shall pay")) == 1
    assert citations.page_at(offsets, LEASE.index("sixty (60) days")) == 2
    assert citations.page_at([], 5) is None


def test_locate_exact_and_whitespace_drift():
    from info_retriever.llm import citations

    assert citations.locate("sixty (60) days written notice", LEASE) is not None
    # A model retyping a quote across a line break collapses the newline.
    assert citations.locate("9. TERMINATION Either party may terminate", LEASE) is not None
    # Case drift.
    assert citations.locate("SIXTY (60) DAYS WRITTEN NOTICE", LEASE) is not None


def test_locate_rejects_paraphrase_and_trivia():
    from info_retriever.llm import citations

    assert citations.locate("the tenant can leave after two months", LEASE) is None
    assert citations.locate("rent", LEASE) is None, "too short to be a meaningful match"


def test_attach_yields_page_numbers_like_the_native_path():
    from info_retriever.llm import citations

    documents = {"doc-1": {"id": "doc-1", "title": "Lease", "full_text": LEASE}}
    attached = citations.attach(
        [{"document_id": "doc-1", "text": "sixty (60) days written notice"}], documents
    )
    assert len(attached) == 1
    assert attached[0]["page"] == 2
    assert attached[0]["located"] is True
    assert attached[0]["document_title"] == "Lease"


def test_attach_flags_an_unfindable_quote_rather_than_inventing_a_page():
    from info_retriever.llm import citations

    documents = {"doc-1": {"id": "doc-1", "title": "Lease", "full_text": LEASE}}
    attached = citations.attach(
        [{"document_id": "doc-1", "text": "tenant may vacate whenever they wish"}], documents
    )
    assert attached[0]["located"] is False
    assert "page" not in attached[0], "an unlocated quote must not claim a page"


def test_attach_searches_other_documents_when_the_id_is_wrong():
    from info_retriever.llm import citations

    documents = {"doc-1": {"id": "doc-1", "title": "Lease", "full_text": LEASE}}
    attached = citations.attach(
        [{"document_id": "nonexistent", "text": "sixty (60) days written notice"}], documents
    )
    assert attached[0]["located"] is True
    assert attached[0]["page"] == 2


# --------------------------------------------------------------------------- #
# provider selection
# --------------------------------------------------------------------------- #


@pytest.fixture
def vllm(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_PROVIDER", "vllm")
    monkeypatch.setenv("VLLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
    monkeypatch.setenv("VLLM_BASE_URL", "http://127.0.0.1:8001/v1")

    from info_retriever import llm
    from info_retriever.config import settings

    settings.cache_clear()
    llm.reset()
    yield llm
    llm.reset()
    settings.cache_clear()


def test_provider_is_selected_by_env(tmp_path, monkeypatch):
    from info_retriever import llm
    from info_retriever.config import settings

    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    settings.cache_clear()
    llm.reset()
    assert llm.provider().name == "anthropic"
    assert llm.provider().native_citations is True

    monkeypatch.setenv("LLM_PROVIDER", "vllm")
    monkeypatch.setenv("VLLM_MODEL", "some-model")
    settings.cache_clear()
    llm.reset()
    assert llm.provider().name == "vllm"
    assert llm.provider().native_citations is False

    llm.reset()
    settings.cache_clear()


def test_unknown_provider_is_rejected_by_config(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_PROVIDER", "ollama")

    from info_retriever.config import settings

    settings.cache_clear()
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        settings()
    settings.cache_clear()


def test_missing_vllm_model_names_the_variable(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_PROVIDER", "vllm")
    monkeypatch.setenv("VLLM_MODEL", "")

    from info_retriever import llm
    from info_retriever.config import settings

    settings.cache_clear()
    llm.reset()
    with pytest.raises(llm.ProviderError, match="VLLM_MODEL"):
        llm.provider().client()
    llm.reset()
    settings.cache_clear()


# --------------------------------------------------------------------------- #
# vLLM adapters
# --------------------------------------------------------------------------- #


def _completion(content: str, *, tool_calls=None, finish_reason="stop"):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)])


def _stub_complete(provider, responses: list, captured: list | None = None):
    queue = list(responses)

    def fake_complete(**kwargs):
        if captured is not None:
            # Deep-copy: the loop appends to the same `messages` list across turns,
            # so a stored reference would show only its final state.
            captured.append(copy.deepcopy(kwargs))
        return queue.pop(0)

    provider._complete = fake_complete  # type: ignore[method-assign]


def test_tool_specs_reuse_the_shared_registry(vllm):
    """One source of truth: the Anthropic-decorated registry supplies names,
    descriptions and schemas; only the envelope differs per provider."""
    from info_retriever.tools import TOOLS

    specs = vllm.provider()._tool_specs(TOOLS)
    assert [spec["function"]["name"] for spec in specs] == [t.name for t in TOOLS]
    for spec in specs:
        assert spec["type"] == "function"
        assert spec["function"]["description"], "description must carry over"
        assert spec["function"]["parameters"]["type"] == "object"


def test_classify_uses_guided_json_and_validates(vllm):
    provider = vllm.provider()
    captured: list[dict] = []
    payload = {
        "doc_type": "rental",
        "title": "Lease",
        "language": "en",
        "summary": "A lease.",
    }
    _stub_complete(provider, [_completion(json.dumps(payload))], captured)

    result = provider.classify(LEASE)
    assert result.doc_type == "rental"

    request = captured[0]
    assert request["response_format"]["type"] == "json_schema"
    schema = request["response_format"]["json_schema"]["schema"]
    assert schema["additionalProperties"] is False, "guided decoding needs a closed schema"


def test_schema_violation_is_a_clear_provider_error(vllm):
    provider = vllm.provider()
    # Valid JSON, wrong shape — what a small model typically returns.
    _stub_complete(provider, [_completion('{"doc_type": "rental"}')])

    with pytest.raises(vllm.ProviderError, match="does not match the classification schema"):
        provider.classify(LEASE)


def test_truncated_output_is_reported_not_silently_accepted(vllm):
    provider = vllm.provider()
    _stub_complete(provider, [_completion('{"answer": "part', finish_reason="length")])

    with pytest.raises(vllm.ProviderError, match="VLLM_MAX_TOKENS"):
        provider.classify(LEASE)


def test_agent_loop_dispatches_tools_and_feeds_results_back(vllm, monkeypatch):
    provider = vllm.provider()

    call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="read_document", arguments='{"document_id": "doc-1"}'),
    )
    captured: list[dict] = []
    _stub_complete(
        provider,
        [_completion("", tool_calls=[call]), _completion("Sixty days notice.")],
        captured,
    )

    executed: list[dict] = []

    class FakeTool:
        name = "read_document"
        description = "Read a document."
        input_schema = {"type": "object", "properties": {}, "additionalProperties": False}

        def call(self, arguments):
            executed.append(arguments)
            return "document body"

    events: list[tuple[str, dict]] = []
    result = provider.run_agent(
        question="notice period?",
        instructions="INSTRUCTIONS",
        catalogue="CATALOGUE",
        tools=[FakeTool()],
        emit=lambda kind, payload: events.append((kind, payload)),
    )

    assert executed == [{"document_id": "doc-1"}], "the tool must actually run"
    assert result.text == "Sixty days notice."
    assert result.tool_calls == [{"name": "read_document", "input": {"document_id": "doc-1"}}]
    assert ("tool", {"name": "read_document", "input": {"document_id": "doc-1"}}) in events

    # Second request must carry the tool result back for the model to use.
    followup = captured[1]["messages"]
    assert followup[-1]["role"] == "tool"
    assert followup[-1]["tool_call_id"] == "call-1"
    assert followup[-1]["content"] == "document body"

    # Instructions and catalogue collapse into one system string for OpenAI-compat.
    system = captured[0]["messages"][0]
    assert system["role"] == "system"
    assert "INSTRUCTIONS" in system["content"] and "CATALOGUE" in system["content"]


def test_agent_loop_survives_a_failing_tool(vllm):
    provider = vllm.provider()
    call = SimpleNamespace(
        id="call-1", function=SimpleNamespace(name="read_document", arguments="{}")
    )
    captured: list[dict] = []
    _stub_complete(
        provider, [_completion("", tool_calls=[call]), _completion("Could not read it.")], captured
    )

    class Exploding:
        name = "read_document"
        description = "Read a document."
        input_schema = {"type": "object", "properties": {}, "additionalProperties": False}

        def call(self, arguments):
            raise RuntimeError("disk on fire")

    result = provider.run_agent(
        question="q", instructions="i", catalogue="c", tools=[Exploding()]
    )
    assert result.text == "Could not read it."
    assert "disk on fire" in captured[1]["messages"][-1]["content"]


def test_agent_loop_tolerates_malformed_tool_arguments(vllm):
    provider = vllm.provider()
    call = SimpleNamespace(
        id="call-1", function=SimpleNamespace(name="t", arguments="{not json")
    )
    _stub_complete(provider, [_completion("", tool_calls=[call]), _completion("done")])

    class Tool:
        name = "t"
        description = "d"
        input_schema = {"type": "object", "properties": {}, "additionalProperties": False}

        def call(self, arguments):
            assert arguments == {}, "unparseable arguments become an empty dict"
            return "ok"

    assert provider.run_agent(
        question="q", instructions="i", catalogue="c", tools=[Tool()]
    ).text == "done"


def test_answer_from_excerpts_numbers_the_excerpts_and_offers_no_tools(vllm):
    provider = vllm.provider()
    captured: list = []
    _stub_complete(provider, [_completion("Sixty days [1].")], captured)

    events: list[tuple[str, dict]] = []
    text = provider.answer_from_excerpts(
        question="notice period?",
        excerpts=[{"document_title": "Lease", "page": 2, "content": "sixty (60) days written notice"}],
        emit=lambda kind, p: events.append((kind, p)),
    )

    assert text == "Sixty days [1]."
    request = captured[0]
    assert "tools" not in request, "single pass: the model gets no tools"
    prompt = request["messages"][-1]["content"]
    assert '<excerpt n="1" document="Lease" page="2">' in prompt
    assert "sixty (60) days written notice" in prompt
    assert any(kind == "delta" for kind, _ in events), "the UI needs the answer text"


def test_cite_locates_quotes_and_streams_the_answer(vllm):
    provider = vllm.provider()
    payload = {
        "answer": "Sixty days written notice.",
        "quotes": [{"document_id": "doc-1", "text": "sixty (60) days written notice"}],
    }
    _stub_complete(provider, [_completion(json.dumps(payload))])

    events: list[tuple[str, dict]] = []
    result = provider.cite(
        question="notice period?",
        draft="about two months",
        documents=[{"id": "doc-1", "title": "Lease", "full_text": LEASE}],
        emit=lambda kind, p: events.append((kind, p)),
    )

    assert result.text == "Sixty days written notice."
    assert result.citations[0]["page"] == 2
    assert result.citations[0]["located"] is True
    assert any(kind == "delta" for kind, _ in events), "UI needs answer text to render"


def test_every_answer_writing_prompt_carries_the_scope_rule():
    """Answer what was asked and stop. The rule must reach each prompt that writes
    text the user reads — the citation pass is that text on the tools path, and
    run_ask's single pass on the other — and the lines that invited padding must not
    come back in any of them."""
    from info_retriever import agent
    from info_retriever.llm import prompts

    writers = {
        "agent pass": agent.INSTRUCTIONS,
        "citation pass": prompts.cite_user_prompt("q", "d", "2026-01-01"),
        "run_ask": prompts.EXCERPT_ANSWER_SYSTEM,
    }
    for name, text in writers.items():
        assert prompts.ANSWER_SCOPE_RULE in text, f"{name} lacks the scope rule"
        for padding in ("say what they do cover", "then the supporting detail",
                        "name the document and clause"):
            assert padding not in text, f"{name} invites padding: {padding!r}"


def test_the_cite_prompt_forbids_narrating_the_draft():
    """The citation pass writes what the user reads, and the user never saw the draft.
    Without this rule the model reports its own verification ("The draft is correct")."""
    from info_retriever.llm import prompts

    prompt = prompts.cite_user_prompt("q", "d", "2026-01-01")
    assert "The user never sees the draft" in prompt
    assert "Do not mention the draft" in prompt


def test_cite_accepts_quotes_encoded_as_a_json_string(vllm):
    """json_object mode constrains only the outer object; a model can return the
    nested list as a string. That must still produce citations, not silently none."""
    provider = vllm.provider()
    quotes = [{"document_id": "doc-1", "text": "sixty (60) days written notice"}]
    payload = {"answer": "Sixty days written notice.", "quotes": json.dumps(quotes)}
    _stub_complete(provider, [_completion(json.dumps(payload))])

    result = provider.cite(
        question="notice period?",
        draft="about two months",
        documents=[{"id": "doc-1", "title": "Lease", "full_text": LEASE}],
    )

    assert result.citations[0]["located"] is True
    assert result.citations[0]["page"] == 2


def test_cite_falls_back_to_the_draft_when_json_is_unparseable(vllm):
    provider = vllm.provider()
    _stub_complete(provider, [_completion("I cannot produce JSON, sorry.")])

    result = provider.cite(
        question="q",
        draft="the draft",
        documents=[{"id": "doc-1", "title": "Lease", "full_text": LEASE}],
    )
    # Prose is still useful; we simply have no quotes to locate.
    assert result.text == "I cannot produce JSON, sorry."
    assert result.citations == []


def test_cite_without_usable_text_returns_the_draft_unchanged(vllm):
    provider = vllm.provider()
    result = provider.cite(
        question="q", draft="the draft", documents=[{"id": "d", "title": "t", "full_text": "  "}]
    )
    assert result.text == "the draft"
    assert result.citations == []


def test_describe_exposes_provider_and_model_but_no_credential(vllm):
    described = vllm.provider().describe()
    assert described["llm_provider"] == "vllm"
    assert described["native_citations"] is False
    assert described["model"] == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert "api_key" not in json.dumps(described).lower()


# --------------------------------------------------------------------------- #
# PDF rasterisation, needed only off the Anthropic path
# --------------------------------------------------------------------------- #


def test_scanned_pdf_without_pymupdf_says_how_to_proceed(tmp_path, monkeypatch):
    import builtins

    from info_retriever import loaders

    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4 not-a-real-pdf")
    loaded = loaders.LoadedFile(
        path=pdf,
        mime_type="application/pdf",
        content_blocks=[],
        text=None,
        page_count=1,
        sha256="x",
    )

    real_import = builtins.__import__

    def no_pymupdf(name, *args, **kwargs):
        if name == "pymupdf":
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pymupdf)

    with pytest.raises(loaders.UnsupportedFile) as excinfo:
        loaders.as_image_parts(loaded)
    message = str(excinfo.value)
    assert "pymupdf" in message
    assert "LLM_PROVIDER=anthropic" in message, "offer the no-install way out"


def test_images_pass_through_without_rasterisation(tmp_path):
    import base64

    from info_retriever import loaders

    png = tmp_path / "page.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    loaded = loaders.load(png)

    parts = loaders.as_image_parts(loaded)
    assert len(parts) == 1
    media_type, encoded = parts[0]
    assert media_type == "image/png"
    assert base64.standard_b64decode(encoded) == png.read_bytes()

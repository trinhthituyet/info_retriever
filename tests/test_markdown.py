"""Inline-Markdown rendering in the answer bubble.

The renderer lives in `app.js` and cannot be executed here, so these tests extract
the literal regex from the source and exercise it with an exact port of
`appendInline`. That keeps the cases honest — editing the pattern in `app.js`
changes what these tests run.

The pattern's job is as much about what it must *not* match: a contract full of
`2 * 3` or `****` separators must not sprout emphasis.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

APP_JS = Path(__file__).resolve().parents[1] / "src" / "info_retriever" / "static" / "app.js"
STYLE_CSS = APP_JS.parent / "style.css"


def _pattern() -> re.Pattern[str]:
    source = APP_JS.read_text()
    match = re.search(r"const INLINE_MARKDOWN =\s*\n?\s*/(.+)/g;", source)
    assert match, "could not find INLINE_MARKDOWN in app.js"
    # The pattern uses only syntax common to JS and Python regex.
    return re.compile(match.group(1))


def render(text: str) -> str:
    """Exact port of `appendInline`: emphasis becomes tags, everything else literal."""
    pattern = _pattern()
    out: list[str] = []
    last = 0
    for match in pattern.finditer(text):
        out.append(text[last : match.start()])
        bold, bold_underscore, code, italic = match.groups()
        if bold is not None or bold_underscore is not None:
            out.append(f"<strong>{bold or bold_underscore}</strong>")
        elif code is not None:
            out.append(f"<code>{code}</code>")
        else:
            out.append(f"<em>{italic}</em>")
        last = match.end()
    out.append(text[last:])
    return "".join(out)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("**payment cycle** is monthly", "<strong>payment cycle</strong> is monthly"),
        ("Rent is **1,500 USD** per month.", "Rent is <strong>1,500 USD</strong> per month."),
        ("**a** and **b**", "<strong>a</strong> and <strong>b</strong>"),
        ("__also bold__", "<strong>also bold</strong>"),
        ("*italic*", "<em>italic</em>"),
        ("use `read_document` now", "use <code>read_document</code> now"),
        ("**Term**\nStarts 2024-03-01", "<strong>Term</strong>\nStarts 2024-03-01"),
        ("**multi word bold here**", "<strong>multi word bold here</strong>"),
    ],
    ids=[
        "bold-phrase",
        "bold-amount",
        "two-bolds",
        "underscore-bold",
        "italic",
        "inline-code",
        "bold-then-newline",
        "multi-word",
    ],
)
def test_emphasis_is_rendered(text, expected):
    assert render(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "2 * 3 * 4",
        "clause 5 * see annex",
        "unclosed **bold",
        "a ** b",
        "**",
        "****",
        "*",
        "_ underscore alone _",
        "plain text with no markers",
        "",
        "5 * 3 = 15 and 4 * 2 = 8",
    ],
    ids=[
        "arithmetic",
        "footnote-asterisk",
        "unclosed-bold",
        "spaced-double",
        "bare-double",
        "quad-separator",
        "single-asterisk",
        "spaced-underscores",
        "plain",
        "empty",
        "two-multiplications",
    ],
)
def test_non_emphasis_is_left_literal(text):
    assert render(text) == text, "a false positive would corrupt document wording"


def test_answer_text_never_goes_through_innerhtml():
    """Answers quote documents we did not author, so emphasis must be built from DOM
    nodes. A single innerHTML assignment here would make prompt-injected markup live."""
    source = APP_JS.read_text()

    # Match property access (`node.innerHTML`), not the word — the comments in app.js
    # discuss innerHTML precisely to forbid it.
    for sink in (".innerHTML", ".outerHTML", ".insertAdjacentHTML", "document.write"):
        assert sink not in source, f"{sink} is an HTML-injection sink; build nodes instead"

    assert "document.createTextNode" in source, "literal runs must be text nodes"
    assert 'el("strong"' in source and 'el("em"' in source and 'el("code"' in source


def test_streaming_stays_plain_and_reconciles_at_the_end():
    """Mid-stream a token like `**pay` has no closing delimiter, so parsing during
    streaming would flicker between literal and bold. Deltas append as text; the final
    `answer` event re-renders the whole body."""
    source = APP_JS.read_text()
    assert "answerBox.textContent += data.text" in source, "deltas append plain text"
    assert "setRichText(answerBox, data.text || answerBox.textContent)" in source, (
        "the terminal answer event must re-render with Markdown applied"
    )


def test_emphasis_has_styling():
    css = STYLE_CSS.read_text()
    for selector in (".bubble.answer strong", ".bubble.answer em", ".bubble.answer code"):
        assert selector in css, f"{selector} renders unstyled without a rule"


# --------------------------------------------------------------------------- #
# paragraph spacing
# --------------------------------------------------------------------------- #


def _paragraphs(text: str) -> list[str]:
    """Port of the paragraph split in `setRichText`."""
    parts = [part.strip("\n") for part in re.split(r"\n{2,}", text)]
    return [part for part in parts if part.strip()]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("one", ["one"]),
        ("one\n\ntwo", ["one", "two"]),
        ("one\n\n\n\ntwo", ["one", "two"]),
        ("one\nstill one\n\ntwo", ["one\nstill one", "two"]),
        ("\n\none\n\n", ["one"]),
        ("a\n\nb\n\nc", ["a", "b", "c"]),
        ("   ", []),
        ("", []),
    ],
    ids=[
        "single",
        "two-paragraphs",
        "extra-blank-lines-collapse",
        "single-newline-stays-inside",
        "leading-trailing-blanks",
        "three-paragraphs",
        "whitespace-only",
        "empty",
    ],
)
def test_blank_lines_split_into_paragraphs(text, expected):
    assert _paragraphs(text) == expected


def test_paragraphs_are_elements_so_the_gap_is_styleable():
    """A literal newline cannot be styled. The gap is only adjustable once paragraph
    breaks are elements, which is why setRichText emits <p>."""
    source = APP_JS.read_text()
    assert 'el("p")' in source, "paragraphs must be real elements"
    assert re.search(r"split\(/\\n\{2,\}/\)", source), "blank lines delimit paragraphs"


def test_the_paragraph_gap_is_two_thirds_of_a_blank_line():
    css = STYLE_CSS.read_text()

    # One source of truth for line-height: the body must consume the variable, or the
    # derived gap silently drifts from the text it is spacing.
    assert "line-height: var(--line-height)" in css
    assert not re.search(r"font:\s*15px/1\.55", css), "the shorthand duplicated 1.55"

    gap = re.search(r"--answer-para-gap:\s*(.+?);", css)
    assert gap, "--answer-para-gap must be defined"
    assert "var(--line-height)" in gap.group(1), "derive the gap, do not hardcode it"
    ratio = re.search(r"0\.6+7?", gap.group(1))
    assert ratio, f"expected a two-thirds factor, got {gap.group(1)!r}"
    assert abs(float(ratio.group(0)) - 2 / 3) < 0.005

    # Applied only between consecutive paragraphs — not above the first one, which
    # would add a gap where the bubble padding already provides one.
    assert ".bubble.answer p + p { margin-top: var(--answer-para-gap); }" in css
    assert re.search(r"\.bubble\.answer p \{\s*\n\s*margin: 0;", css), (
        "paragraphs must start with no margin of their own"
    )
    assert "white-space: pre-wrap" in css, "single newlines inside a paragraph still break"


def test_citation_quotes_are_not_markdown_rendered():
    """Cited text is verbatim contract wording. An asterisk in a contract is an
    asterisk, so quotes stay plain text."""
    source = APP_JS.read_text()
    citations_block = source[source.index("function citationsBlock") :]
    citations_block = citations_block[: citations_block.index("\nfunction ")]
    assert "setRichText" not in citations_block
    assert "quote.textContent" in citations_block

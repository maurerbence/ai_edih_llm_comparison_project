"""Tests for the framework-agnostic rendering module."""

from __future__ import annotations

import pytest

from app.rendering import render_highlighted_html, render_plain_html


def test_render_plain_html_emits_text() -> None:
    html = render_plain_html("Paris is the capital of France.")
    assert "Paris is the capital of France." in html


def test_newlines_become_br() -> None:
    html = render_plain_html("line1\nline2")
    assert "<br>" in html
    assert "\n" not in html


def test_text_is_forced_black() -> None:
    html = render_plain_html("Anything")
    assert "color:#000" in html


@pytest.mark.parametrize("user_text", ["<script>x</script>", "a & b", '"q"'])
def test_html_escape_user_text(user_text: str) -> None:
    html = render_plain_html(user_text)
    assert user_text not in html
    assert "&" in html


def test_highlighted_wraps_matching_quote_with_color() -> None:
    text = "Tim Berners-Lee invented the web in 1989."
    html = render_highlighted_html(text, [("Tim Berners-Lee", "#cfeacc")])
    assert "background-color:#cfeacc" in html
    assert "Tim Berners-Lee" in html
    # The non-quoted tail still appears verbatim outside any span.
    assert " invented the web in 1989." in html


def test_highlighted_is_case_insensitive() -> None:
    html = render_highlighted_html("Paris is nice.", [("paris", "#fbd3d3")])
    assert "background-color:#fbd3d3" in html
    assert ">Paris<" in html  # original casing preserved inside the span


def test_highlighted_skips_non_matching_quote() -> None:
    html = render_highlighted_html("Hello world.", [("not present", "#cfeacc")])
    assert "background-color" not in html
    assert "Hello world." in html


def test_highlighted_handles_overlapping_quotes_longest_first() -> None:
    text = "The quick brown fox."
    html = render_highlighted_html(text, [("quick", "#fbd3d3"), ("quick brown", "#cfeacc")])
    # Longer quote wins; shorter overlapping quote is skipped.
    assert html.count("<span") == 1
    assert "background-color:#cfeacc" in html
    assert "background-color:#fbd3d3" not in html


def test_highlighted_escapes_quote_content() -> None:
    text = "Use <b>bold</b> tags."
    html = render_highlighted_html(text, [("<b>bold</b>", "#cfeacc")])
    assert "<b>bold</b>" not in html  # raw tags must be escaped
    assert "&lt;b&gt;bold&lt;/b&gt;" in html
    assert "background-color:#cfeacc" in html


def test_highlighted_empty_quote_is_ignored() -> None:
    html = render_highlighted_html("Hello.", [("", "#cfeacc")])
    assert "background-color" not in html
    assert "Hello." in html

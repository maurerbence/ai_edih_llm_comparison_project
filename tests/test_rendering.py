"""Tests for the framework-agnostic rendering module."""

from __future__ import annotations

import pytest

from app.rendering import render_plain_html


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

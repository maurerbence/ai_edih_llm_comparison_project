"""Tests for pure helper functions in :mod:`app.runner` (no API access)."""

from __future__ import annotations

import pytest

from app.runner import _dedupe_consecutive_sentences, _looks_like_refusal


@pytest.mark.parametrize(
    "answer",
    [
        "I cannot help with that request.",
        "I can't answer that.",
        "I'm sorry, but I cannot provide that information.",
        "Sorry, I can't comply with this.",
        "Unfortunately, I am unable to assist with that.",
        "As an AI language model, I cannot generate that content.",
    ],
)
def test_looks_like_refusal_recognises_common_decline_phrases(answer: str) -> None:
    assert _looks_like_refusal(answer) is True


@pytest.mark.parametrize(
    "answer",
    [
        "Neil Armstrong was the first man on the Moon.",
        "The Python programming language was created by Guido van Rossum.",
        # "I cannot" appearing mid-sentence is fine — refusals begin the answer.
        "There are many reasons one cannot easily summarize this.",
        "",
        "   ",
    ],
)
def test_looks_like_refusal_does_not_misfire_on_real_answers(answer: str) -> None:
    assert _looks_like_refusal(answer) is False


# ── Consecutive-duplicate-sentence dedup ─────────────────────────────────


def test_dedupe_collapses_adjacent_identical_sentence() -> None:
    text = (
        "Neil Armstrong walked on the Moon."
        "Neil Armstrong walked on the Moon. Buzz Aldrin followed him."
    )
    out = _dedupe_consecutive_sentences(text)
    assert out.count("Neil Armstrong walked on the Moon.") == 1
    assert "Buzz Aldrin followed him." in out


def test_dedupe_collapses_when_duplicates_separated_by_newline() -> None:
    """Models sometimes emit duplicates with a newline between them."""
    text = (
        "Neil Armstrong walked on the Moon.\n"
        "Neil Armstrong walked on the Moon. Buzz Aldrin followed him."
    )
    out = _dedupe_consecutive_sentences(text)
    assert out.count("Neil Armstrong walked on the Moon.") == 1
    assert "Buzz Aldrin followed him." in out


def test_dedupe_collapses_when_duplicates_separated_by_space() -> None:
    text = (
        "Neil Armstrong walked on the Moon. "
        "Neil Armstrong walked on the Moon. Buzz Aldrin followed him."
    )
    out = _dedupe_consecutive_sentences(text)
    assert out.count("Neil Armstrong walked on the Moon.") == 1
    assert "Buzz Aldrin followed him." in out


def test_dedupe_leaves_non_duplicates_alone() -> None:
    text = "Paris is the capital of France. Population is around 2.1 million people."
    assert _dedupe_consecutive_sentences(text) == text


def test_dedupe_does_not_collapse_short_repeats() -> None:
    """Below the min-length threshold, identical short sentences are preserved."""
    text = "Yes. Yes. The water boils at 100 degrees Celsius."
    assert _dedupe_consecutive_sentences(text) == text

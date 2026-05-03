"""Tests for :mod:`app.judge`. Pure-Python helpers only — no API access."""

from __future__ import annotations

from app.judge import JUDGE_SYSTEM_PROMPT


def test_judge_system_prompt_lists_all_four_labels() -> None:
    """Every label the FindingSet schema accepts must be defined."""
    for label in ("agree", "disagree", "unique_to_a", "unique_to_b"):
        assert label in JUDGE_SYSTEM_PROMPT


def test_judge_system_prompt_keeps_specificity_guidance() -> None:
    """Strict-subsumption-as-AGREE — the user's "1909 vs 20th century" bug —
    must remain in the prompt as a worked example."""
    assert "1909" in JUDGE_SYSTEM_PROMPT
    assert "20th" in JUDGE_SYSTEM_PROMPT and "century" in JUDGE_SYSTEM_PROMPT
    assert "STRICT-SUBSUMPTION" in JUDGE_SYSTEM_PROMPT


def test_judge_system_prompt_calls_out_multi_valued_roles() -> None:
    """Multi-valued roles (founders, members) must not be labelled DISAGREE
    when each side names a different filler."""
    assert "MULTI-VALUED" in JUDGE_SYSTEM_PROMPT
    assert "Founded by Gates" in JUDGE_SYSTEM_PROMPT

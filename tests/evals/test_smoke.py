"""End-to-end pipeline evals against the real OpenAI API.

Skipped by default (markers: ``evals``). Run with::

    OPENAI_API_KEY=sk-… uv run pytest -m evals

These cost money. Assertions are deliberately loose — LLMs are non-deterministic.
"""

from __future__ import annotations

import os

import pytest

from app.runner import run_pipeline

pytestmark = pytest.mark.evals

MODEL_A = "gpt-4o-mini"
MODEL_B = "gpt-4o-mini"
MIXED_MODEL_B = "gpt-4o"


def _require_api_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; skipping eval.")


def test_pipeline_completes_end_to_end() -> None:
    _require_api_key()
    result = run_pipeline("What is the capital of France?", MODEL_A, MODEL_B)
    assert result.answer_a.strip()
    assert result.answer_b.strip()
    # Either findings were produced or a refusal was detected.
    has_findings = bool(result.findings.findings)
    refusal = result.metadata.refusal_a or result.metadata.refusal_b
    assert has_findings or refusal


def test_factual_agreement_paris() -> None:
    _require_api_key()
    result = run_pipeline(
        "What is the capital of France? Answer in one short sentence.",
        MODEL_A,
        MODEL_B,
    )
    findings_text = " ".join(
        f"{f.summary} {f.quote_a} {f.quote_b}".lower() for f in result.findings.findings
    )
    assert "paris" in findings_text, f"expected a Paris finding; got: {result.findings.findings}"
    # The judge should label at least one finding as agree on Paris.
    agreed = [f for f in result.findings.findings if f.label == "agree"]
    assert agreed, f"expected at least one 'agree' finding; got: {result.findings.findings}"


def test_population_question_produces_findings() -> None:
    _require_api_key()
    result = run_pipeline(
        "What is the capital of France and what is its approximate population? Give a specific number.",
        MODEL_A,
        MIXED_MODEL_B,
    )
    # We expect Paris agreement plus possibly a population-related finding.
    has_paris_finding = any(
        "paris" in (f.summary + f.quote_a + f.quote_b).lower()
        for f in result.findings.findings
    )
    assert has_paris_finding, "expected at least one finding touching Paris"

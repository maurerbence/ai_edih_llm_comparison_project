"""Pydantic data contracts shared across the pipeline.

The schema is intentionally tiny: workers produce two answers; the
holistic judge reads both raw texts and emits a list of findings; the
UI renders the findings list. There is no claim extraction stage and
no character-level span rendering.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

FindingLabel = Literal["agree", "disagree", "unique_to_a", "unique_to_b"]


class TextFinding(BaseModel):
    """One finding emitted by the holistic judge.

    For ``agree`` / ``disagree``, both ``quote_a`` and ``quote_b`` carry
    text excerpts from the corresponding answer. For ``unique_to_a`` /
    ``unique_to_b``, only the relevant side's quote is populated; the
    other is the empty string. (OpenAI strict-mode structured output
    forbids per-field defaults, so "missing" is encoded as empty string
    rather than ``None``.)
    """

    label: FindingLabel
    summary: str
    quote_a: str
    quote_b: str
    rationale: str


class FindingSet(BaseModel):
    findings: list[TextFinding]


class RenderMetadata(BaseModel):
    judge_model: str
    refusal_a: bool = False
    refusal_b: bool = False
    refusal_reason: str = ""


class PipelineResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_a: str
    model_b: str
    answer_a: str
    answer_b: str
    findings: FindingSet
    metadata: RenderMetadata

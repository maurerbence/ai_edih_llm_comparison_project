"""Pipeline orchestration.

Stages:

1. **Workers** — two parallel ADK agents producing the answers, with
   token-level streaming to the UI.
2. **Holistic judge** — one OpenAI call sees both raw answer texts and
   emits the findings (agreements, disagreements, claims unique to
   each side).

No claim extraction, no embedding pre-filter, no span-based rendering.
The findings list is the unit of UI display; answer panes are plain
text. See ``docs/documentation.md`` for the rationale.

Refusal handling: an answer is treated as a refusal only when its text
matches a refusal phrase (e.g. "I cannot help with that"). Otherwise
the comparison runs against whatever the workers produced.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Literal

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.agents import build_pipeline
from app.config import settings
from app.judge import judge_answers
from app.observability import configure_observability
from app.schema import FindingSet, PipelineResult, RenderMetadata

APP_NAME = "model_diff"
_USER_ID = "user"

configure_observability()


# ── Streaming event types ──────────────────────────────────────────────────


Source = Literal["a", "b"]


@dataclass(frozen=True)
class WorkerChunk:
    source: Source
    delta: str


@dataclass(frozen=True)
class Stage:
    name: str


@dataclass(frozen=True)
class Done:
    result: PipelineResult


StreamEvent = WorkerChunk | Stage | Done


# ── ADK event helpers ──────────────────────────────────────────────────────


def _extract_text(event: object) -> str:
    content = getattr(event, "content", None)
    parts = getattr(content, "parts", None) or []
    return "".join(getattr(p, "text", "") or "" for p in parts)


# ── Refusal detection ──────────────────────────────────────────────────────


_REFUSAL_PATTERNS = (
    re.compile(r"^\s*i (?:can(?:not|'t)|am unable|won't|will not)\b", re.IGNORECASE),
    re.compile(r"^\s*i'?m (?:sorry|unable)\b", re.IGNORECASE),
    re.compile(r"^\s*sorry,? (?:but )?i\b", re.IGNORECASE),
    re.compile(
        r"^\s*as an? (?:ai|language model)\b.*\b(?:cannot|can't|unable|won't)\b",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*unfortunately,? i (?:cannot|can't|am unable)\b", re.IGNORECASE),
)


def _looks_like_refusal(answer: str) -> bool:
    """True iff the answer text matches a known refusal pattern.

    Empty / whitespace-only answers do *not* count as refusals — those
    are upstream worker failures, not model declines.
    """
    text = answer.strip()
    if not text:
        return False
    return any(p.search(text) for p in _REFUSAL_PATTERNS)


# ── Sentence dedup (gpt-4o family glitch) ──────────────────────────────────


_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?\n]")
_DEDUP_MIN_CHARS = 20


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return non-empty (start, end) spans for each sentence in ``text``."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(text):
        m = _SENTENCE_BOUNDARY_RE.search(text, cursor)
        end = m.end() if m else len(text)
        s = cursor
        while s < end and text[s].isspace():
            s += 1
        if s < end:
            spans.append((s, end))
        cursor = end
    return spans


def _dedupe_consecutive_sentences(text: str) -> str:
    """Strip consecutive identical sentences.

    Some models (notably gpt-4o-mini, but also gpt-4o) occasionally emit
    the first sentence twice in a row, e.g. ``"X is true.X is true."`` or
    ``"X is true.\\nX is true."``. Strict policy: only collapse adjacent
    sentences whose stripped text is byte-identical and at least
    ``_DEDUP_MIN_CHARS`` long. Anything more aggressive risks removing
    intentional repetition.
    """
    spans = _sentence_spans(text)
    if len(spans) < 2:
        return text

    dropped_ranges: list[tuple[int, int]] = []
    last_kept_idx: int | None = None

    for i, (start, end) in enumerate(spans):
        sentence = text[start:end].strip()
        if last_kept_idx is not None:
            prev_start, prev_end = spans[last_kept_idx]
            prev_sentence = text[prev_start:prev_end].strip()
            if sentence == prev_sentence and len(sentence) >= _DEDUP_MIN_CHARS:
                dropped_ranges.append((prev_end, end))
                continue
        last_kept_idx = i

    if not dropped_ranges:
        return text

    parts: list[str] = []
    cursor = 0
    for d_start, d_end in dropped_ranges:
        parts.append(text[cursor:d_start])
        cursor = d_end
    parts.append(text[cursor:])
    return "".join(parts)


# ── ADK runner glue ────────────────────────────────────────────────────────


async def _run_adk_pipeline(
    prompt: str, model_a: str, model_b: str
) -> AsyncIterator[tuple[str, object]]:
    """Drive the ADK pipeline; yield ('event', event) for streaming and
    ('done', state_dict) at the end."""
    pipeline = build_pipeline(model_a, model_b)
    sessions = InMemorySessionService()  # type: ignore[no-untyped-call]
    session_id = str(uuid.uuid4())

    await sessions.create_session(
        app_name=APP_NAME, user_id=_USER_ID, session_id=session_id, state={"user_query": prompt}
    )
    runner = Runner(agent=pipeline, app_name=APP_NAME, session_service=sessions)
    message = types.Content(role="user", parts=[types.Part(text=prompt)])

    async for event in runner.run_async(user_id=_USER_ID, session_id=session_id, new_message=message):
        yield ("event", event)

    session = await sessions.get_session(app_name=APP_NAME, user_id=_USER_ID, session_id=session_id)
    if session is None:
        raise RuntimeError(f"session {session_id} disappeared after creation")
    yield ("done", dict(session.state))


# ── Public streaming pipeline ──────────────────────────────────────────────


async def stream_pipeline(
    prompt: str,
    model_a: str,
    model_b: str,
) -> AsyncIterator[StreamEvent]:
    state: dict[str, object] = {}

    async for kind, payload in _run_adk_pipeline(prompt, model_a, model_b):
        if kind == "done":
            assert isinstance(payload, dict)
            state = payload
            break

        event = payload
        author = getattr(event, "author", "") or ""
        if author in ("worker_a", "worker_b") and getattr(event, "partial", False):
            delta = _extract_text(event)
            if delta:
                yield WorkerChunk(source="a" if author == "worker_a" else "b", delta=delta)

    answer_a = _dedupe_consecutive_sentences(str(state.get("answer_a", "")))
    answer_b = _dedupe_consecutive_sentences(str(state.get("answer_b", "")))

    refusal_a = _looks_like_refusal(answer_a)
    refusal_b = _looks_like_refusal(answer_b)
    if refusal_a or refusal_b:
        yield Done(
            result=_build_refusal_result(model_a, model_b, answer_a, answer_b, refusal_a, refusal_b)
        )
        return

    yield Stage(name="judging")
    findings = await judge_answers(answer_a, answer_b)

    yield Done(
        result=PipelineResult(
            model_a=model_a,
            model_b=model_b,
            answer_a=answer_a,
            answer_b=answer_b,
            findings=findings,
            metadata=RenderMetadata(judge_model=settings.judge_model),
        )
    )


def _build_refusal_result(
    model_a: str,
    model_b: str,
    answer_a: str,
    answer_b: str,
    refusal_a: bool,
    refusal_b: bool,
) -> PipelineResult:
    reason_parts: list[str] = []
    if refusal_a:
        reason_parts.append("model A refused")
    if refusal_b:
        reason_parts.append("model B refused")
    return PipelineResult(
        model_a=model_a,
        model_b=model_b,
        answer_a=answer_a,
        answer_b=answer_b,
        findings=FindingSet(findings=[]),
        metadata=RenderMetadata(
            judge_model=settings.judge_model,
            refusal_a=refusal_a,
            refusal_b=refusal_b,
            refusal_reason=" and ".join(reason_parts) or "",
        ),
    )


def to_sync_iterator(agen: AsyncIterator[StreamEvent]) -> Iterator[StreamEvent]:
    """Drive an async generator from sync code (Streamlit callbacks are sync).

    Some libraries we depend on (notably LiteLLM's LoggingWorker) spawn
    background tasks on whatever loop they're first invoked from. If we
    close the loop while those tasks are still pending, asyncio prints
    "Task was destroyed but it is pending!" and "Event loop is closed"
    tracebacks. We cancel and drain pending tasks before shutting down.
    """
    loop = asyncio.new_event_loop()
    try:
        while True:
            try:
                yield loop.run_until_complete(agen.__anext__())
            except StopAsyncIteration:
                return
    finally:
        aclose = getattr(agen, "aclose", None)
        if aclose is not None:
            try:
                loop.run_until_complete(aclose())
            except RuntimeError:
                pass
        pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
        for t in pending:
            t.cancel()
        if pending:
            try:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except RuntimeError:
                pass
        loop.close()


def run_pipeline(prompt: str, model_a: str, model_b: str) -> PipelineResult:
    """Synchronous, non-streaming entry point for tests and CLI use."""
    for event in to_sync_iterator(stream_pipeline(prompt, model_a, model_b)):
        if isinstance(event, Done):
            return event.result
    raise RuntimeError("pipeline ended without a Done event")


__all__ = [
    "Done",
    "Stage",
    "StreamEvent",
    "WorkerChunk",
    "run_pipeline",
    "stream_pipeline",
    "to_sync_iterator",
]

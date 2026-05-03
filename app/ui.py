"""Streamlit entry point.

Run with::

    uv run streamlit run app/ui.py

Thin Streamlit-only wrapper around :func:`app.runner.stream_pipeline`.
Pipeline logic lives in :mod:`app.runner`, :mod:`app.judge`, and
:mod:`app.agents`.
"""

from __future__ import annotations

import traceback
from collections import Counter
from typing import Any

import streamlit as st

from app.config import settings
from app.rendering import render_plain_html
from app.runner import Done, Stage, WorkerChunk, stream_pipeline, to_sync_iterator
from app.schema import FindingSet, PipelineResult, TextFinding

DEFAULT_PROMPT_PLACEHOLDER = (
    "e.g. In 3-4 sentences, who invented the World Wide Web, in what year, and what was the first web browser called?"
)
CURSOR = "▌"
CACHE_KEY = "_pipeline_cache"

AGREE_BG = "#cfeacc"
DISAGREE_BG = "#fbd3d3"

# Curated demo questions. Picked to exercise both the agree path (well-known
# facts both models will state cleanly) and the disagree / one-sided path
# (numbers and dates where models often differ in precision or estimate).
DEMO_QUESTIONS: list[tuple[str, str]] = [
    (
        "Mona Lisa",
        "In 3-4 sentences, who painted the Mona Lisa and where is it displayed today?",
    ),
    (
        "Microsoft founders",
        "In 3-4 sentences, who founded Microsoft and in what year?",
    ),
    (
        "Tokyo population",
        "In 3-4 sentences, Exactly what is the current population of Tokyo and what is its exact land area?",
    ),
    (
        "Roman Empire",
        "In 3-4 sentences, in what year did the Western Roman Empire fall and exactly when did the Roman Empire begin?",
    ),
    (
        "1918 deaths",
        "Exactly how many people died in 1918?",
    ),
    (
        "Hungary students",
        "Exactly how many students are in Hungary?",
    ),
]


# ── Error surfacing ────────────────────────────────────────────────────────


def _surface_error(status: Any, exc: BaseException) -> None:
    leaves = _flatten_exception_group(exc)
    friendly = _friendly_message(leaves)
    if friendly:
        status.error(friendly)
    else:
        headline = "; ".join(f"{type(e).__name__}: {e}" for e in leaves) or f"{type(exc).__name__}: {exc}"
        status.error(f"pipeline failed — {headline}")
    with st.expander("Technical details"):
        formatted = "".join(traceback.format_exception(exc))
        st.code(formatted, language="text")


def _flatten_exception_group(exc: BaseException) -> list[BaseException]:
    if isinstance(exc, BaseExceptionGroup):
        out: list[BaseException] = []
        for sub in exc.exceptions:
            out.extend(_flatten_exception_group(sub))
        return out
    return [exc]


_MODEL_NOT_FOUND_HINTS = (
    "model_not_found",
    "does not exist",
    "do not have access",
    "invalid model",
)
_AUTH_HINTS = ("invalid_api_key", "incorrect api key", "authentication")


def _friendly_message(leaves: list[BaseException]) -> str | None:
    blob = " ".join(str(e) for e in leaves).lower()
    if any(h in blob for h in _MODEL_NOT_FOUND_HINTS):
        return (
            "One of the chosen models isn't available with your API key. "
            "Try a different model from the dropdown."
        )
    if any(h in blob for h in _AUTH_HINTS):
        return "OpenAI rejected the API key. Check OPENAI_API_KEY in your .env."
    return None


# ── Cache ──────────────────────────────────────────────────────────────────


def _cache() -> dict[tuple[str, str, str], PipelineResult]:
    cache: dict[tuple[str, str, str], PipelineResult] = st.session_state.setdefault(
        CACHE_KEY, {}
    )
    return cache


# ── Widgets ────────────────────────────────────────────────────────────────


def _model_pickers() -> tuple[str, str]:
    models = list(settings.available_models)
    col_a, col_b = st.columns(2)
    with col_a:
        model_a = st.selectbox("Model A", models, index=0, key="model_a")
    with col_b:
        default_b = 1 if len(models) > 1 else 0
        model_b = st.selectbox("Model B", models, index=default_b, key="model_b")
    return model_a, model_b


# ── Streaming → placeholders ───────────────────────────────────────────────


def _stream_into_slots(
    slot_a: Any,
    slot_b: Any,
    status: Any,
    prompt: str,
    model_a: str,
    model_b: str,
) -> PipelineResult | None:
    buf_a, buf_b = "", ""
    workers_done = False
    status.info("Workers responding…")

    result: PipelineResult | None = None
    for evt in to_sync_iterator(stream_pipeline(prompt, model_a, model_b)):
        match evt:
            case WorkerChunk(source="a", delta=delta):
                buf_a += delta
                slot_a.markdown(render_plain_html(buf_a + CURSOR), unsafe_allow_html=True)
            case WorkerChunk(source="b", delta=delta):
                buf_b += delta
                slot_b.markdown(render_plain_html(buf_b + CURSOR), unsafe_allow_html=True)
            case Stage(name=name):
                if not workers_done:
                    workers_done = True
                    slot_a.markdown(render_plain_html(buf_a), unsafe_allow_html=True)
                    slot_b.markdown(render_plain_html(buf_b), unsafe_allow_html=True)
                status.info(f"{name}…")
            case Done(result=r):
                result = r
    return result


# ── Findings rendering ─────────────────────────────────────────────────────


def _summary_line(findings: FindingSet) -> str:
    counts = Counter(f.label for f in findings.findings)
    a = counts["agree"]
    d = counts["disagree"]
    o = counts["unique_to_a"] + counts["unique_to_b"]
    if not (a or d or o):
        return "No comparable findings."

    parts: list[str] = []
    if a:
        parts.append(f"**{a}** agreement{'s' if a != 1 else ''}")
    if d:
        parts.append(f"**{d}** disagreement{'s' if d != 1 else ''}")
    if o:
        parts.append(f"**{o}** mentioned by only one side")
    if len(parts) == 1:
        return f"Found {parts[0]}."
    return "Found " + ", ".join(parts[:-1]) + f", and {parts[-1]}."


def _render_summary_metrics(findings: FindingSet) -> None:
    counts = Counter(f.label for f in findings.findings)
    cols = st.columns(3)
    cols[0].metric("Agree", counts["agree"])
    cols[1].metric("Disagree", counts["disagree"])
    cols[2].metric("One-sided", counts["unique_to_a"] + counts["unique_to_b"])


def _render_finding(f: TextFinding, model_a: str, model_b: str) -> None:
    if f.label == "agree":
        body = f"**Both agree** — {f.summary}"
        if f.quote_a or f.quote_b:
            body += (
                f"\n\n- **{model_a}:** {f.quote_a or '_(no excerpt)_'}"
                f"\n- **{model_b}:** {f.quote_b or '_(no excerpt)_'}"
            )
        if f.rationale:
            body += f"\n\n_{f.rationale}_"
        st.success(body)
    elif f.label == "disagree":
        body = (
            f"**Models disagree** — {f.summary}"
            f"\n\n- **{model_a}:** {f.quote_a}"
            f"\n- **{model_b}:** {f.quote_b}"
        )
        if f.rationale:
            body += f"\n\n_{f.rationale}_"
        st.error(body)
    elif f.label == "unique_to_a":
        body = f"**Only {model_a} mentions this** — {f.summary}"
        if f.quote_a:
            body += f"\n\n> {f.quote_a}"
        st.info(body)
    elif f.label == "unique_to_b":
        body = f"**Only {model_b} mentions this** — {f.summary}"
        if f.quote_b:
            body += f"\n\n> {f.quote_b}"
        st.info(body)


def _render_findings_section(result: PipelineResult, prompt: str) -> None:
    st.divider()
    st.markdown(_summary_line(result.findings))
    _render_summary_metrics(result.findings)

    if not result.findings.findings:
        return

    st.subheader("Findings")
    for f in result.findings.findings:
        _render_finding(f, result.model_a, result.model_b)

    st.download_button(
        "Download as Markdown",
        data=_format_as_markdown(prompt, result),
        file_name="model-comparison.md",
        mime="text/markdown",
    )


def _format_as_markdown(prompt: str, result: PipelineResult) -> str:
    """Self-contained Markdown export of the comparison."""
    lines: list[str] = []
    lines.append(f"# Comparison: {result.model_a} vs {result.model_b}\n")
    lines.append(f"**Question:** {prompt}\n")

    lines.append(f"## Answer A — {result.model_a}\n")
    lines.append(result.answer_a.strip() + "\n")

    lines.append(f"## Answer B — {result.model_b}\n")
    lines.append(result.answer_b.strip() + "\n")

    lines.append("## Findings\n")
    if not result.findings.findings:
        lines.append("_No comparable findings._\n")
        return "\n".join(lines)

    for f in result.findings.findings:
        if f.label == "agree":
            lines.append(f"- **Both agree** — {f.summary}")
            if f.quote_a:
                lines.append(f"  - {result.model_a}: {f.quote_a}")
            if f.quote_b:
                lines.append(f"  - {result.model_b}: {f.quote_b}")
        elif f.label == "disagree":
            lines.append(f"- **Models differ** — {f.summary}")
            lines.append(f"  - {result.model_a}: {f.quote_a}")
            lines.append(f"  - {result.model_b}: {f.quote_b}")
        elif f.label == "unique_to_a":
            lines.append(f"- **Only {result.model_a} mentions** — {f.summary}")
            if f.quote_a:
                lines.append(f"  > {f.quote_a}")
        elif f.label == "unique_to_b":
            lines.append(f"- **Only {result.model_b} mentions** — {f.summary}")
            if f.quote_b:
                lines.append(f"  > {f.quote_b}")
    return "\n".join(lines) + "\n"


# ── Result rendering ───────────────────────────────────────────────────────


def _render_final(slot_a: Any, slot_b: Any, status: Any, result: PipelineResult, prompt: str) -> None:
    meta = result.metadata
    if meta.refusal_a or meta.refusal_b:
        status.warning(f"Refusal detected — {meta.refusal_reason}.")
        slot_a.markdown(render_plain_html(result.answer_a), unsafe_allow_html=True)
        slot_b.markdown(render_plain_html(result.answer_b), unsafe_allow_html=True)
        return

    status.empty()
    slot_a.markdown(render_plain_html(result.answer_a), unsafe_allow_html=True)
    slot_b.markdown(render_plain_html(result.answer_b), unsafe_allow_html=True)
    _render_findings_section(result, prompt)


def _show_comparison(prompt: str, model_a: str, model_b: str) -> None:
    key = (prompt, model_a, model_b)
    cached = _cache().get(key)

    status = st.empty()
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader(f"A · {model_a}")
        slot_a = st.empty()
    with col_b:
        st.subheader(f"B · {model_b}")
        slot_b = st.empty()

    if cached is not None:
        status.caption("served from cache")
        result: PipelineResult | None = cached
    else:
        try:
            result = _stream_into_slots(slot_a, slot_b, status, prompt, model_a, model_b)
        except Exception as exc:
            _surface_error(status, exc)
            return
        if result is not None:
            _cache()[key] = result

    if result is None:
        status.error("pipeline ended without producing a result")
        return

    _render_final(slot_a, slot_b, status, result, prompt)


def _render_demo_question_buttons() -> None:
    """One-click pills that autofill the question textbox."""
    st.caption("Or try a demo question:")
    cols = st.columns(len(DEMO_QUESTIONS))
    for i, (label, question) in enumerate(DEMO_QUESTIONS):
        if cols[i].button(label, width="stretch", key=f"demo_{i}"):
            st.session_state["prompt_input"] = question
            st.rerun()


def _legend() -> str:
    return (
        f"Pick two OpenAI models, ask a question, and the holistic judge "
        f"summarises where they "
        f"<span style='background-color:{AGREE_BG}; color:#000; padding:0 4px;'>agree</span> "
        f"and where they "
        f"<span style='background-color:{DISAGREE_BG}; color:#000; padding:0 4px;'>disagree</span>."
    )


def main() -> None:
    st.set_page_config(page_title="Two-Model Diff", layout="wide")
    st.title("Two-Model Comparison Chatbot")
    st.markdown(_legend(), unsafe_allow_html=True)

    model_a, model_b = _model_pickers()
    _render_demo_question_buttons()
    # Wrap the input + submit in a form so Enter inside the textbox triggers
    # the comparison directly — the bare text_input + button combo requires
    # Cmd+Enter then click.
    with st.form("compare_form", clear_on_submit=False, border=False):
        prompt = st.text_input(
            "Your question", placeholder=DEFAULT_PROMPT_PLACEHOLDER, key="prompt_input"
        )
        submitted = st.form_submit_button("Compare", type="primary")

    if submitted and prompt.strip():
        _show_comparison(prompt.strip(), model_a, model_b)


main()

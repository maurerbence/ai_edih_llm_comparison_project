# model-diff

A Streamlit app that asks the same question of two OpenAI models in
parallel, then runs a holistic LLM judge over both raw answers and
renders a side-by-side comparison with a structured findings list:

- 🟩 **agree**       — both models assert the same fact (or one strictly
  subsumes the other without contradicting it)
- 🟥 **disagree**    — the answers contradict on a specific value or polarity
- 🔵 **one-sided**   — a fact only one model mentioned

Each finding includes a one-line summary, the supporting quote from each
answer, and a one-sentence rationale.

Built on [Google ADK](https://google.github.io/adk-docs/) for parallel
worker orchestration; OpenAI models are reached via ADK's LiteLLM
integration.

## Architecture

```
                user prompt
                     │
                     ▼
   ┌──────────────────────────────────────┐
   │            ParallelAgent             │
   │   worker_a            worker_b       │   workers stream tokens
   │                                      │   to the UI in real time
   └──────────────────────────────────────┘
                     │
                     ▼
            sentence dedup        collapse "X.X." duplicates that
                     │             gpt-4o-family models occasionally emit
                     ▼
            refusal check         if either answer matches a refusal
                     │             phrase, render plain & skip the judge
                     ▼
            holistic judge        one OpenAI call sees both raw answer
                     │             texts and emits a FindingSet directly
                     ▼             (agreements, disagreements, one-sided)
            Streamlit UI          plain answer panes + summary metrics
                                  + findings list with quotes
```

**Why no claim extraction?** The previous version decomposed each answer
into atomic claims, embedded them, and asked the judge to label pairwise
relationships. That introduced cross-pairing artifacts and a two-stage
LLM cost without making the user-facing output meaningfully better. The
holistic judge sees both full answers and emits the findings directly —
fewer false positives, less code, and the findings list (the most-useful
artifact per the customer review) is unchanged.

**Caching.** Re-runs of the same `(prompt, model_a, model_b)` are served
from `st.session_state` — no token cost on repeat within a session.

**Streaming.** Worker outputs stream into the UI panes as tokens arrive;
once both workers finish, the judge runs in the background and the
findings render on completion.

**Refusal handling.** A side is treated as a refusal only when its answer
text starts with a recognised refusal phrase (e.g. "I cannot help with
that"). Otherwise the holistic judge handles whatever the workers
produced.

**Observability.** If `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` are
set, every model call (workers, judge) is auto-traced via LiteLLM's
Langfuse callback. No-op without keys.

## Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
# edit .env and set OPENAI_API_KEY (and optionally Langfuse keys)
```

## Run

```bash
uv run streamlit run app/ui.py
```

## Project layout

```
app/
  __init__.py       package init (silences ADK PLUGGABLE_AUTH warning)
  config.py         pydantic-settings — single source of truth for env config
  schema.py         pydantic data contracts (TextFinding, FindingSet, …)
  rendering.py      plain HTML rendering for the answer panes
  agents.py         ADK agent factory (two parallel workers)
  judge.py          holistic judge prompt + single OpenAI call
  runner.py         async streaming runner, refusal detection, sentence dedup
  observability.py  Langfuse / LiteLLM callback wiring (lazy, opt-in)
  ui.py             Streamlit entry point
tests/
  test_rendering.py        pure-function rendering tests
  test_judge.py            judge prompt invariants
  test_runner_helpers.py   refusal detection + sentence dedup
  evals/
    test_smoke.py          end-to-end checks against the OpenAI API (opt-in)
docs/
  documentation.md           full system documentation
  research_plan.md           original design / research log
  system_review.md           internal architecture review (historical)
  client_readiness_review.md handover review (historical)
  customer_review*.md        customer-perspective reviews (historical)
  manual_test_run_*.md       manual UI test runs and findings
  skills/                    original skill specs the design started from
```

The non-UI modules (`config`, `schema`, `rendering`, `agents`, `judge`,
`runner`, `observability`) are framework-agnostic; `ui.py` is the only
Streamlit-aware code.

## Development

```bash
uv sync --all-groups       # install dev tools (ruff, mypy, pytest)
uv run pytest              # unit tests (skips evals)
uv run pytest -m evals     # end-to-end evals — costs OpenAI tokens
uv run ruff check .        # lint
uv run ruff format .       # format
uv run mypy app            # type-check
```

CI runs lint + format check + mypy + pytest on every push and pull request — see `.github/workflows/ci.yml`.

## Configuration

`app/config.py` exposes a frozen `Settings` instance loaded from `.env`.
Override any field via environment variable (case-insensitive).

| Variable              | Default                                                                            | Purpose                                                                  |
| --------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `OPENAI_API_KEY`      | _required_                                                                         | Authenticates all OpenAI calls.                                          |
| `JUDGE_MODEL`         | `gpt-5`                                                                            | Model used for the holistic judge.                                       |
| `AVAILABLE_MODELS`    | `gpt-5,gpt-5-mini,gpt-5-nano,gpt-4o,gpt-4o-mini,gpt-4.1,gpt-4.1-mini,gpt-4.1-nano` | Models offered in the UI dropdowns.                                      |
| `LANGFUSE_PUBLIC_KEY` | _unset_                                                                            | Enables Langfuse tracing if set with secret.                             |
| `LANGFUSE_SECRET_KEY` | _unset_                                                                            | Required alongside the public key.                                       |
| `LANGFUSE_HOST`       | _unset_                                                                            | Optional self-hosted Langfuse URL.                                       |

# model-diff — system documentation

## 1. Executive summary

`model-diff` is a Streamlit app that asks the same question of two
OpenAI models in parallel and produces a structured comparison: a list
of agreements, disagreements, and facts only one side mentioned. Each
finding includes a one-line summary, the supporting quote from each
answer, and a one-sentence rationale.

The pipeline is deliberately small: two parallel workers produce the
answers, then a single holistic LLM judge call sees both raw answer
texts and emits the structured comparison directly.In this PROTOTYPE there is no claim
decomposer, no embedding pre-filter, no character-level span rendering.

The project is built on Google ADK for the parallel worker layer
(streaming, output keys, session state) and reaches OpenAI through ADK's
LiteLLM integration. The non-UI core is framework-agnostic — a FastAPI
or CLI front-end could replace Streamlit without touching the pipeline.
Latency is dominated by the workers and the judge call; on a typical
3-sentence question the end-to-end takes 3-11 seconds depending on the
chosen models.

The project is at Prototype/Pilot quality. Quality gates (ruff, mypy,
pytest) pass on every commit. The shipping checklist for production is in §9.

## 2. System architecture

### 2.1 High-level data flow

```
                user prompt
                     │
                     ▼
   ┌──────────────────────────────────────┐
   │            ParallelAgent             │   workers stream tokens
   │   worker_a            worker_b       │   to the UI in real time
   │                                      │   (output_keys: answer_a / answer_b)
   └──────────────────────────────────────┘
                     │
                     ▼
            sentence dedup        collapse "X.X." duplicates that
                     │             gpt-4o-family models occasionally emit
                     ▼
            refusal check         if either answer matches a refusal
                     │             phrase, skip the judge and render plain
                     ▼
            holistic judge        one OpenAI call sees both raw answers
                     │             and emits a FindingSet directly
                     ▼             (agree / disagree / unique_to_a / unique_to_b)
            Streamlit UI          plain answer panes + summary metrics
                                  + findings list with quotes + tooltips
```

Streaming model: worker tokens forward to the UI as they arrive. The
judge runs after both workers finish. The status line shows the current
stage so the UI never feels frozen.

The earlier iterations (atomic claim extraction → embedding pre-filter →
pairwise judge → span rendering) are documented in §8 — they shipped,
they introduced their own classes of bugs, and they were ultimately
removed.

## 3. Repository / modules

```
app/
  __init__.py       package init — silences ADK PLUGGABLE_AUTH warning
  config.py         pydantic-settings — single source of truth for env config
  schema.py         pydantic data contracts (TextFinding, FindingSet, …)
  agents.py         ADK agent factory (two parallel workers)
  judge.py          holistic judge prompt + single OpenAI call
  runner.py         async streaming runner, refusal detection, sentence dedup
  rendering.py      plain HTML rendering for the answer panes
  observability.py  Langfuse / LiteLLM callback wiring (lazy, opt-in)
  ui.py             Streamlit entry point
tests/
  test_rendering.py        pure-function rendering tests
  test_judge.py            judge prompt invariants
  test_runner_helpers.py   refusal detection + sentence dedup
  evals/                   end-to-end checks against the OpenAI API (opt-in)
docs/
  documentation.md           ← you are here (documentataion)
  skills/                    original skill specs the design started from
```

The non-UI modules (`config`, `schema`, `agents`, `judge`, `runner`,
`rendering`, `observability`) are framework-agnostic. `ui.py` is the
only Streamlit-aware code; `runner.stream_pipeline` returns plain async
events any frontend can consume.

## 4. Detailed description of components

### 4.1 `app/config.py`

A frozen pydantic-settings `Settings` instance. Loads `.env` once at
import time, exposes a single `settings` object, and re-exports
`OPENAI_API_KEY` into `os.environ` so downstream libraries (OpenAI SDK,
LiteLLM, Langfuse) pick it up via their own lookups.

Live fields:

| Field                | Default              | Purpose                                                |
| -------------------- | -------------------- | ------------------------------------------------------ |
| `openai_api_key`     | _required_           | OpenAI auth                                            |
| `judge_model`        | `gpt-5`              | Model used for the holistic judge                      |
| `judge_temperature`  | `0.0`                | Sampling temperature (no-op for gpt-5; gpt-4 family)   |
| `available_models`   | gpt-5 + gpt-4 family | UI dropdown choices for the workers                    |
| `langfuse_*`         | unset                | Optional observability                                 |

Two model-compat helpers live here too:

- `temperature_kwargs(model, value)` returns `{"temperature": value}` for
  gpt-4 family and `{}` for gpt-5 (which only supports the default
  `temperature=1`).
- `reasoning_kwargs(model, effort)` returns `{"reasoning_effort": effort}`
  for gpt-5 family and `{}` for gpt-4 family. The judge passes
  `effort="low"` so reasoning is on, but cheap.

### 4.2 `app/schema.py`

Pydantic v2 contracts. Three types live here, all small:

- `TextFinding` — one alignment between the two answers. Has `label`
  (`agree | disagree | unique_to_a | unique_to_b`), `summary`, `quote_a`,
  `quote_b`, `rationale`. For unique-to-one-side findings, the other
  side's quote is the empty string (OpenAI strict-mode structured
  output forbids per-field defaults, so "missing" is encoded as `""`).
- `FindingSet` — `findings: list[TextFinding]`. The judge's output type.
- `RenderMetadata` — small metadata block (judge model name; refusal
  flags) shown by the UI.
- `PipelineResult` — top-level: model names, both raw answer texts,
  `FindingSet`, `RenderMetadata`.

There is no `Claim`, `JudgedPair`, `JudgmentSet`, `RenderSpan`,
`ResponseRender`, or `RenderPlan` — those went away with the claim
extraction stage.

### 4.3 `app/agents.py`

Factory for the ADK pipeline. Just one ParallelAgent containing two
LlmAgent workers (`worker_a`, `worker_b`) with the user-picked models.
The `WORKER_INSTRUCTION` asks for 2-4 sentences, factual, no hedging,
no disclaimers, no questions back to the user.

`_openai(model, **kwargs)` wraps `LiteLlm(model=f"openai/{model}", …)`
and drops `temperature` from kwargs when the model is gpt-5 family —
gpt-5 only supports `temperature=1` and would otherwise fail with
`UnsupportedParamsError`.

### 4.4 `app/judge.py`

The judge sees both raw answer texts in a single OpenAI call and returns
a `FindingSet`. The system prompt (~70 lines) defines the four labels,
five rules, and explicit examples for the failure modes that matter:

- Strict subsumption is `agree`, not `disagree`. Worked example:
  `"X built in 1909"` agrees with `"X built in the 20th century"` —
  rationale should call out which side is more specific.
- Multi-valued roles (founders, members, ingredients) don't `disagree`
  on different fillers. Worked example: `"Founded by Gates"` and
  `"Founded by Allen"` are both true.
- Each substantive fact in either answer should appear in exactly one
  finding (no double-counting).

`judge_answers(answer_a, answer_b)` is the public entry point. It
returns `FindingSet(findings=[])` when either input is empty. The judge
runs at `reasoning_effort="low"` — enough to apply the rules without
spending the default `medium` effort that adds 10-30s per call.

### 4.5 `app/runner.py`

Pipeline orchestration. Responsibilities:

1. **ADK driver** (`_run_adk_pipeline`) — creates a session, runs the
   `ParallelAgent`, yields events as they arrive, returns the final
   session state.
2. **Streaming events** (`stream_pipeline`) — translates ADK events into
   `WorkerChunk | Stage | Done` events for any UI to consume.
3. **Sentence dedup** (`_dedupe_consecutive_sentences`) — collapses
   `"X.X."` and `"X.\nX."` glitches the gpt-4o family occasionally emits.
   Uses `_sentence_spans` to be robust to whitespace and newlines
   between duplicates.
4. **Refusal detection** (`_looks_like_refusal`) — pattern-matches the
   answer text. When either answer matches, the pipeline yields a
   `Done` with empty findings + a refusal flag in the metadata.
5. **Judge dispatch** — calls `judge_answers(answer_a, answer_b)` and
   wraps the result in a `PipelineResult`.

`run_pipeline` is a synchronous, non-streaming wrapper used by tests and
CLI use. `to_sync_iterator` drives the async generator from sync code
(Streamlit callbacks are sync) and cleanly drains LiteLLM's background
tasks on shutdown so asyncio doesn't print "Event loop is closed"
tracebacks.

### 4.6 `app/rendering.py`

Single function: `render_plain_html(text)`. Wraps the text in a styled
`<div>`, escapes HTML, converts `\n` to `<br>`. Forces text colour to
`#000` so the panes remain legible across Streamlit's light/dark themes.

Earlier versions also produced colour-highlighted span output; that went
away with the claim extraction stage. The two-colour signal now lives
in the findings list (see §4.8).

### 4.7 `app/observability.py`

Lazy, idempotent Langfuse instrumentation through LiteLLM's success /
failure callbacks. Activates only if both `LANGFUSE_PUBLIC_KEY` and
`LANGFUSE_SECRET_KEY` are set; no-op otherwise. We don't instantiate any
tracer ourselves — LiteLLM ships first-class Langfuse support, so we
just register the callback names and let LiteLLM record every model
invocation that flows through ADK's `LiteLlm` wrapper.

### 4.8 `app/ui.py`

Thin Streamlit wrapper. Responsibilities:

- Two model dropdowns (defaults to `gpt-5` vs `gpt-5-mini`).
- Six demo-question pills covering both the agree path (well-known
  facts) and the disagree / one-sided path (numbers, dates, "exactly
  how many" questions where models often differ).
- Form wrapping the question textbox + submit button so Enter triggers
  the comparison.
- Error surfacer that pattern-matches OpenAI errors (`model_not_found`,
  `invalid_api_key`) into user-actionable hints, with the full stack
  trace in an expander.
- Streaming worker output into placeholders, then the findings list
  underneath when the judge finishes.
- Findings rendering — green success box for agreements, red error box
  for disagreements, blue info box for one-sided. Each finding shows
  the summary, both quotes (where applicable), and the rationale.
- Markdown export of the comparison (question + both answers + findings
  list) via a download button.
- Per-session in-memory cache keyed on `(prompt, model_a, model_b)` so
  re-runs of the same triple within a session don't re-spend tokens.

## 5. Deployment

### 5.1 Local

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
# edit .env and set OPENAI_API_KEY (and optionally Langfuse keys)
uv run streamlit run app/ui.py
```

Streamlit binds to `localhost:8501` by default. No persistent storage,
no database, no auth — single-user / demo configuration.

The single environment variable required at runtime is `OPENAI_API_KEY`.
Langfuse keys are optional. For multi-user deployment, see §6.2 — the
session-state cache and lack of auth are the blockers.

### 5.2 CI

GitHub Actions runs ruff (lint + format check), mypy (strict), and
pytest (unit; evals are opt-in) on every push and pull request. See
`.github/workflows/ci.yml`. CI takes ~1 minute on a cold cache.

## 6. Operations & monitoring

### 6.1 Observability

When `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set, every
model call (workers + judge) auto-traces to Langfuse via LiteLLM's
callback. Traces include token counts, latencies, prompts, completions,
and errors. `LANGFUSE_HOST` points at a self-hosted instance if you
have one.

When Langfuse is not configured, the app keeps working with no extra
logging — `configure_observability()` is a no-op.

### 6.2 Known operational limits

- **Cache is per-session, in-memory.** A browser refresh wipes it; two
  users in different sessions don't share the cache.
- **No auth, no rate limit.** A public deployment will burn tokens.
  
- **No cost cap.** The judge call is the dominant cost. 
- **Errors surface as Streamlit toasts.** OpenAI 429s, model-not-found,
  bad keys are pattern-matched into friendly messages; everything else
  shows a stack trace in an expander. 

### 6.3 Quality gates

- `uv run ruff check .` — lint.
- `uv run ruff format --check .` — format check.
- `uv run mypy app` — strict type-check (pydantic plugin enabled).
- `uv run pytest` — unit tests (evals skipped by default).
- `uv run pytest -m evals` — end-to-end evals against the live OpenAI
  API. Costs tokens; opt-in only.

The unit suite covers the rendering helper, the judge prompt invariants
(rules + worked-example fixtures), refusal detection, and the sentence
dedup (including the newline-between-duplicates corner case).

## 7. Known design choices

### 7.1 Holistic judge over raw answers (no claim extraction)

The previous version decomposed each answer into atomic claims with
character offsets, embedded them, and asked the judge to label pairwise
relationships. That made cross-pair labels possible (claim a-1 paired
with claim b-2 instead of b-1) and required a series of prompt rules to
describe when the cross-pair was "real" versus an artifact. It also
made the user's specific complaint — `"X built in 1509"` labelled
DISAGREE against `"X built in the 16th century"` — happen routinely.

The current version sends both raw answer texts to the judge in one
prompt and asks for the alignment + uniques directly. The strict-
subsumption case is in the prompt as the first AGREE example. The
multi-valued role case is its own rule. The judge produces text-quote
findings the user can read directly. Simpler code, fewer false
positives, no offset bookkeeping.

### 7.2 Two-colour UI

The brief asks for two states (agree / disagree). The schema has four
labels (agree, disagree, unique_to_a, unique_to_b). `agree` renders
green, `disagree` red, both `unique_to_*` render blue. The findings
list is the unit of UI display; the answer panes are plain text.

### 7.3 Refusal handling — text patterns, not empty extraction

A natural-but-wrong implementation would say "if the judge returned
zero findings, the model refused." That fires falsely whenever the
judge has nothing substantive to align (short answers; tangential
questions). We pattern-match the answer text instead (`I cannot…`,
`I'm sorry…`, etc.) at the runner layer, before the judge is even
called.

### 7.4 GPT-5 model defaults

`extractor_model` is gone. `judge_model` defaults to `gpt-5`. Workers
default to `gpt-5` vs `gpt-5-mini` in the dropdown. The 4-family models
are still in the dropdown so users can compare across generations or
fall back if their key doesn't have gpt-5 access.

### 7.5 GPT-5 parameter compat

GPT-5 rejects `temperature=0` (only accepts `1`, the default) and
exposes a `reasoning_effort` parameter the 4 family doesn't. Two helpers
in `config.py` (`temperature_kwargs`, `reasoning_kwargs`) return the
appropriate kwargs dict per model and are spread into the OpenAI SDK
call. The judge passes `reasoning_effort="low"` so reasoning runs but
doesn't spend the default `medium` effort that adds 10-30s per call.

### 7.6 Streaming + per-session cache

Workers stream into the UI panes as tokens arrive; the judge runs
invisibly afterwards with a status update. The
`(prompt, model_a, model_b)` triple is cached in `st.session_state`.
Cache is intentionally session-scoped — a browser refresh wipes it,
which is the right behaviour for a demo and the wrong behaviour for
production (see §6.2).

## 8. Experiments tried before this version and findings

This is what got tried, in roughly chronological order, and why it's
not in the current code.

### 8.1 Atomic claim decomposition

Each answer was split into one-fact-per-claim by a cheap extractor
model (`gpt-4o-mini`, later `gpt-5-nano`), with character offsets,
polarity, and a topic tag per claim. The judge then labelled pairs of
claims. Removed. Two failure modes drove the decision:

1. **Under-extraction**: the cheap extractor occasionally returned zero
   claims for a perfectly informative answer, which made one side
   "empty" and collapsed every claim on the other side to `unique_to_*`.
   We added a stronger-model fallback re-extraction; it helped but
   still left a long tail.
2. **Cross-pairing artifacts**: the judge would pair claim a-1 with
   claim b-2 when the right pairing was a-1↔b-1 and a-2↔b-2. We added
   prompt rules and worked examples for "multi-valued role" and
   "event-aspect" cases, but the rules were finicky and the failure
   mode kept resurfacing in a new shape.

### 8.2 Embedding pre-filter

Cosine similarity over `text-embedding-3-small` selected which claim
pairs to send to the judge, with auto-`agree` for pairs above 0.98 +
matching polarity. Removed alongside §8.1. The pre-filter was the
direct source of cross-pairings: it routed claims by surface similarity
into wrong combinations the judge then had to disentangle.

### 8.3 Order-swap reconciliation

The judge was run twice, forward and reverse, and only labels that
agreed across both runs were trusted. Mismatches were downgraded to
`partial` and shown with a dotted underline. The
marginal accuracy gain wasn't worth the doubled judge cost, and the
dotted-underline UI was confusing without context.

### 8.4 Decomposer second-pass extraction

A "did you miss anything?" second pass over the first-pass claims to
catch under-extraction.
The second pass introduced spurious claims that polluted the comparison.
The fix that landed instead — span relocation + stronger-model fallback
re-extraction — fired only on the failure cases. That fix is also gone
now since the decomposer itself is gone.


## 9. Future plans (MVP → production recommendations)

Roughly ordered by leverage / effort.

### 9.1 Eval harness with held-out set

`tests/evals/` exists but is small. Build a labelled set of ~100
question + two-answer triples with ground-truth agree / disagree /
one-sided counts. Run it nightly against the latest judge prompt and
track regression on aggregate accuracy. This is now the highest-
leverage move because the judge is the *only* substantive thing left
to tune.

### 9.2 Production caching layer

Replace the `st.session_state` dict with Redis keyed on
`(prompt, model_a, model_b)` plus a content hash of the prompt. Multi-
user deployments get free token savings and consistent results across
sessions. TTL of 24 hours is a reasonable starting point.

### 9.3 Auth + rate limiting

The app currently has neither. For a shared deployment: front it with
an OAuth proxy, enforce per-user rate limits at the proxy, and
optionally add an in-app token-budget guard so a single user can't
spike costs.

### 9.4 Multi-judge consensus

For cases where the judge is non-deterministic (vague claims, novel
topics), run the judge with two different model families (e.g. gpt-5
and Claude) and only accept agreeing labels. A poor man's self-
consistency check, useful when the eval set in §9.1 surfaces a long
tail of unstable findings.

### 9.5 More providers, more models

Today everything runs through OpenAI via LiteLLM. LiteLLM already
supports Anthropic, Google, Azure, etc. — add a provider field to the
model dropdown and let users compare across families. The biggest
demo win for a sales context: comparing a current frontier model
against an older or smaller one where genuine disagreements are far
more common than within-family.

### 9.6 Container

Ship a Dockerfile (recipe in §5.2 is ready to lift) and a minimal Helm
chart. The chart needs a single secret for `OPENAI_API_KEY`, optional
Langfuse secrets, and a horizontal pod autoscaler tied to in-flight
request count.

### 9.7 Frontend split

`app/ui.py` is the only Streamlit-aware code. For a long-term product
shape — embedded in another tool, exposed as an API, used from a
notebook — extract `runner.stream_pipeline` into a FastAPI service
returning Server-Sent Events. The data model is already framework-
agnostic, so the move is mostly plumbing.

### 9.8 Cost / latency telemetry

The current observability is opt-in Langfuse. For production, add per-
request token-cost accounting in the runner so the UI can show "this
comparison used N tokens, cost ~$X". Helpful for both the user and
operators tracking aggregate spend.

## 10. TCO — per-question API cost

Each comparison fires three OpenAI calls: two workers (the user-picked
models) and one holistic judge call (`gpt-5` by default at
`reasoning_effort="low"`).

### 10.1 Per-call token estimates

For a typical demo question (3-4 sentence prompt → 3-4 sentence answers):

| Call           | Input tokens                     | Output + reasoning tokens |
| -------------- | -------------------------------- | ------------------------- |
| Worker A       | ~80 (prompt + worker instruction) | ~250 visible + ~400-800 reasoning (gpt-5 default `medium` effort) |
| Worker B       | ~80                              | ~250 visible + reasoning depends on chosen model                   |
| Holistic judge | ~1300 (system prompt ~700 + both raw answers ~600) | ~400 visible + ~300-600 reasoning (`low` effort)        |

### 10.2 OpenAI list prices (per million tokens, USD, early 2026)

These move; treat as ballpark.

| Model       | Input | Cached input | Output |
| ----------- | -----:| ------------:| ------:|
| gpt-5       | $1.25 | $0.13        | $10.00 |
| gpt-5-mini  | $0.25 | $0.025       | $2.00  |
| gpt-5-nano  | $0.05 | $0.005       | $0.40  |
| gpt-4o      | $2.50 | $1.25        | $10.00 |
| gpt-4o-mini | $0.15 | $0.075       | $0.60  |

For reasoning models, **reasoning tokens bill at the output rate** even
though the user never sees them. This is the dominant cost at high
`reasoning_effort` settings.

### 10.3 Cost per question — default config

Default: Model A = `gpt-5`, Model B = `gpt-5-mini`, Judge = `gpt-5` at
`reasoning_effort="low"`.

| Stage           | Tokens (in / out + reasoning) | Cost                                  |
| --------------- | -----------------------------:| -------------------------------------:|
| Worker A (gpt-5)      | ~80 in / ~900 out         | $80·1.25/1M + $900·10/1M = ~**$0.0091** |
| Worker B (gpt-5-mini) | ~80 in / ~900 out         | $80·0.25/1M + $900·2/1M = ~**$0.0018**  |
| Judge (gpt-5, low)    | ~1300 in / ~800 out       | $1300·1.25/1M + $800·10/1M = ~**$0.0096** |
| **Total**             |                            | ~**$0.020 / question**                  |

So roughly **2 cents per comparison** with the default models. A heavy
demo session of 50 comparisons costs ~$1.00.

### 10.4 Cheaper configurations

Swap one or both workers for smaller models from the dropdown:

| Setup                          | Worker A | Worker B | Judge | Approx. per-question |
| ------------------------------ | --------:| --------:| -----:| --------------------:|
| Default (gpt-5 vs gpt-5-mini)  | $0.0091  | $0.0018  | $0.0096 | **~$0.020**        |
| gpt-5-mini vs gpt-5-nano       | $0.0018  | $0.00036 | $0.0096 | **~$0.012**        |
| gpt-5-nano vs gpt-5-nano       | $0.00036 | $0.00036 | $0.0096 | **~$0.010**        |
| gpt-5-mini vs gpt-5-mini, judge=gpt-5-mini | $0.0018 | $0.0018 | ~$0.0019 | **~$0.0055** |

The judge call dominates with the default config. Switching `JUDGE_MODEL`
to `gpt-5-mini` cuts roughly **80%** off the per-question cost, at some
quality risk on subtle alignment cases — worth eval-checking via §9.1
before flipping for production.

### 10.5 Caveats

- **Reasoning tokens are estimates.** OpenAI doesn't expose them ahead
  of time, and they vary widely with question complexity and the
  `reasoning_effort` setting. The numbers above use middle-of-the-road
  values from manual sampling.
- **Cached-input pricing applies after the first call** when the same
  prompt prefix is sent again within the cache TTL (typically 5-10
  minutes). Won't help the worker stage (each user prompt is unique)
  but does help the judge stage if you re-run identical pairs of
  answers, and the in-session UI cache means a re-submit costs $0
  anyway.
- **Cost per question scales linearly with answer length.** The 2-4
  sentence cap in the worker instruction is what keeps this bounded;
  removing it would push the judge's input tokens up sharply.
- **No volume discounts or batching applied.** A production deployment
  should review OpenAI's batch API for offline workloads — half the
  per-token price for non-real-time runs.

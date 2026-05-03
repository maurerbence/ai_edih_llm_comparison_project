"""ADK agent factory.

Pipeline shape: just two parallel workers producing the answers. The
holistic judge runs in plain Python afterwards. There is no claim
decomposer (raw answers feed the judge directly).
"""

from __future__ import annotations

from typing import Any

from google.adk.agents import LlmAgent, ParallelAgent
from google.adk.models.lite_llm import LiteLlm

WORKER_INSTRUCTION = (
    "Answer the user's question concisely and factually. "
    "Aim for 2-4 sentences unless the question demands more detail. "
    "Do not hedge or add disclaimers."
    "Do not ask questions back to the user, just answer it directly."
)


def _openai(model: str, **kwargs: Any) -> LiteLlm:
    # gpt-5 family only supports temperature=1 (the default); drop the
    # parameter so the call doesn't fail with UnsupportedParamsError.
    if model.startswith("gpt-5"):
        kwargs.pop("temperature", None)
    return LiteLlm(model=f"openai/{model}", **kwargs)


def _build_workers(model_a: str, model_b: str) -> ParallelAgent:
    return ParallelAgent(
        name="workers",
        sub_agents=[
            LlmAgent(
                name="worker_a",
                model=_openai(model_a),
                instruction=WORKER_INSTRUCTION,
                output_key="answer_a",
            ),
            LlmAgent(
                name="worker_b",
                model=_openai(model_b),
                instruction=WORKER_INSTRUCTION,
                output_key="answer_b",
            ),
        ],
    )


def build_pipeline(model_a: str, model_b: str) -> ParallelAgent:
    """Build the ADK pipeline: two workers in parallel."""
    return _build_workers(model_a, model_b)

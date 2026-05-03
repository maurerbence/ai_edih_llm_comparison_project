"""Application configuration loaded from environment variables and `.env`.

Pydantic-settings is the single source of truth. The OpenAI key is also
re-exported into ``os.environ`` so downstream libraries (the OpenAI SDK,
LiteLLM, Langfuse) can pick it up via their own env-var lookups.
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    openai_api_key: str = Field(min_length=1)

    judge_model: str = Field(
        "gpt-5",
        description="Strong model used for the holistic judge.",
    )
    judge_temperature: float = Field(0.0, ge=0.0, le=1.5)

    # ── UI ───────────────────────────────────────────────────────────────────
    available_models: tuple[str, ...] = Field(
        default=(
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
        ),
        description="Models offered to the user in the UI dropdowns.",
    )

    # ── Optional Langfuse observability ──────────────────────────────────────
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str | None = None


settings = Settings()
os.environ["OPENAI_API_KEY"] = settings.openai_api_key


def temperature_kwargs(model: str, value: float) -> dict[str, Any]:
    """Return ``{"temperature": value}`` unless the model rejects it.

    The gpt-5 family only supports ``temperature=1`` (the default), so we
    omit the parameter rather than pass it explicitly. Used by the judge
    and the fallback extractor; the ADK / LiteLLM path handles the same
    case inside :func:`app.agents._openai`.

    Return type is ``dict[str, Any]`` so the result spreads cleanly into
    SDK calls whose typed kwargs each expect a different specific type.
    """
    if model.startswith("gpt-5"):
        return {}
    return {"temperature": value}


def reasoning_kwargs(model: str, effort: str) -> dict[str, Any]:
    """Return ``{"reasoning_effort": effort}`` for gpt-5 models, ``{}`` otherwise.

    gpt-5 is a reasoning model: by default it spends internal tokens on
    reasoning before producing visible output, which adds 5-30s per call
    at the default ``medium`` effort. For mechanical tasks (claim
    extraction) ``minimal`` is enough; for the judge ``low`` is enough to
    apply the decision rules without paying for full reasoning. The 4-family
    models don't accept this parameter, so we omit it for them.
    """
    if model.startswith("gpt-5"):
        return {"reasoning_effort": effort}
    return {}

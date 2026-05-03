"""Langfuse instrumentation, wired through LiteLLM's success/failure callbacks.

Opt-in: requires ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY`` (and
optionally ``LANGFUSE_HOST``) in the environment / ``.env``. If they are
missing, ``configure_observability`` is a no-op so local development without
Langfuse keeps working.

LiteLLM ships first-class Langfuse support, so we don't instantiate any
tracer ourselves — just register the callback names and let LiteLLM do the
work for every model invocation that flows through ADK's ``LiteLlm`` wrapper.
"""

from __future__ import annotations

import os
from threading import Lock

from app.config import settings

_configured = False
_lock = Lock()


def configure_observability() -> bool:
    """Idempotently wire Langfuse into LiteLLM. Returns True when active."""
    global _configured
    if _configured:
        return True

    with _lock:
        if _configured:
            return True

        if not (settings.langfuse_public_key and settings.langfuse_secret_key):
            return False

        os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
        os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
        if settings.langfuse_host:
            os.environ["LANGFUSE_HOST"] = settings.langfuse_host

        import litellm

        callbacks = {"langfuse"}
        litellm.success_callback = list(callbacks.union(litellm.success_callback or []))
        litellm.failure_callback = list(callbacks.union(litellm.failure_callback or []))

        _configured = True
        return True

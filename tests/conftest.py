"""Pytest bootstrap: provide a dummy ``OPENAI_API_KEY`` for collection.

``app.config`` instantiates ``Settings()`` at import time, which requires
``OPENAI_API_KEY``. The non-eval test suite never makes real API calls, so
a placeholder is enough to let modules import cleanly in CI environments
where no real key is configured. Tests that *do* hit the API are gated
behind the ``evals`` marker and run separately with a real key.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENAI_API_KEY", "test-dummy-key")

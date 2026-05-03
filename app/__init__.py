"""Package init.

Runs before any ``app.*`` submodule, which means a ``warnings.filterwarnings``
call here is active by the time those submodules import third-party libraries.
We use that to silence the Google ADK ``[EXPERIMENTAL] FeatureName.PLUGGABLE_AUTH``
warning emitted on every decorated agent invocation — it spams deploy logs
and is enabled by ADK itself, not by our code.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*\bFeatureName\.PLUGGABLE_AUTH\b.*",
    category=UserWarning,
)

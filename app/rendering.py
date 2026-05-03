"""Plain HTML rendering of an answer text. Used during streaming and for
the final answer panes.

Earlier versions of this module also color-highlighted extracted claim
spans inside the answer text. That went away when the claim extraction
stage was removed: the user-facing two-color signal now lives entirely
in the findings list (rendered in :mod:`app.ui`), and the answer panes
are plain text. Text inside the panes is forced to ``#000`` so it
remains legible across Streamlit's light / dark themes.
"""

from __future__ import annotations

import html

_BLOCK_STYLE = (
    "line-height:1.7; font-size:1rem; color:#000; background:#fafafa; "
    "padding:12px; border-radius:6px; border:1px solid #e5e5e5; white-space:pre-wrap;"
)


def render_plain_html(text: str) -> str:
    """Render raw text in a styled block with no highlighting."""
    escaped = html.escape(text).replace("\n", "<br>")
    return f'<div style="{_BLOCK_STYLE}">{escaped}</div>'


__all__ = ["render_plain_html"]

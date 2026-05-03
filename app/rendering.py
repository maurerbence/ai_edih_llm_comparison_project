"""HTML rendering of an answer text.

Two entry points:

* :func:`render_plain_html` — plain styled block, used during streaming
  (before the judge has run we have no quotes to highlight).
* :func:`render_highlighted_html` — same block, but with given quotes
  wrapped in colored spans. The UI uses this for the final answer panes
  so that the agree/disagree signal from the findings list is visible
  inline in the answer text too.

Text inside the panes is forced to ``#000`` so it remains legible across
Streamlit's light / dark themes.
"""

from __future__ import annotations

import html
from collections.abc import Iterable

_BLOCK_STYLE = (
    "line-height:1.7; font-size:1rem; color:#000; background:#fafafa; "
    "padding:12px; border-radius:6px; border:1px solid #e5e5e5; white-space:pre-wrap;"
)
_SPAN_STYLE = "background-color:{color}; padding:0 2px; border-radius:3px;"


def render_plain_html(text: str) -> str:
    """Render raw text in a styled block with no highlighting."""
    body = html.escape(text).replace("\n", "<br>")
    return f'<div style="{_BLOCK_STYLE}">{body}</div>'


def render_highlighted_html(text: str, highlights: Iterable[tuple[str, str]]) -> str:
    """Render ``text`` and tint each quote with its color.

    ``highlights`` is an iterable of ``(quote, css_color)`` pairs. Each
    quote is matched case-insensitively; the first non-overlapping
    occurrence is wrapped. Longer quotes are matched first so they win
    when one quote is a substring of another. Quotes that don't match
    are silently skipped (the judge sometimes paraphrases).
    """
    quotes = [(q, c) for q, c in highlights if q]
    if not quotes:
        return render_plain_html(text)

    used = [False] * len(text)
    ranges: list[tuple[int, int, str]] = []
    lower_text = text.lower()
    for quote, color in sorted(quotes, key=lambda h: -len(h[0])):
        lower_quote = quote.lower()
        idx = lower_text.find(lower_quote)
        while idx != -1:
            end = idx + len(quote)
            if not any(used[idx:end]):
                ranges.append((idx, end, color))
                for i in range(idx, end):
                    used[i] = True
                break
            idx = lower_text.find(lower_quote, idx + 1)

    ranges.sort()
    parts: list[str] = []
    cursor = 0
    for start, end, color in ranges:
        parts.append(html.escape(text[cursor:start]))
        parts.append(f'<span style="{_SPAN_STYLE.format(color=color)}">')
        parts.append(html.escape(text[start:end]))
        parts.append("</span>")
        cursor = end
    parts.append(html.escape(text[cursor:]))

    body = "".join(parts).replace("\n", "<br>")
    return f'<div style="{_BLOCK_STYLE}">{body}</div>'


__all__ = ["render_highlighted_html", "render_plain_html"]

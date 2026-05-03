"""Stage 2: holistic judge over raw answers.

The judge reads both raw answer texts in a single call and emits a list
of findings: agreements, disagreements, and claims unique to each side.

Skipping the per-claim extraction step that used to feed this stage:
the judge handles everything in one prompt. Tradeoff is no character-
level inline highlighting on the answer panes (we don't have spans to
anchor to), but the findings list — the most-useful artifact per the
customer review — is unchanged.
"""

from __future__ import annotations

from openai import AsyncOpenAI

from app.config import reasoning_kwargs, settings, temperature_kwargs
from app.schema import FindingSet

JUDGE_SYSTEM_PROMPT = """You compare two LLM responses (Model A and Model B) to the same question and produce
a structured comparison.

The user will see your output as four kinds of findings: AGREEMENTS,
DISAGREEMENTS, CLAIMS ONLY A MENTIONED, CLAIMS ONLY B MENTIONED. Read
both answers carefully and emit each finding as one item.

LABELS:

- agree:        Both answers assert the same fact. Includes
                STRICT-SUBSUMPTION — when one side is more specific
                than the other in a way that's consistent. Example:
                "X built in 1909" agrees with "X built in the 20th
                century" — both are simultaneously true. Mention which
                side is more specific in the rationale when relevant.
- disagree:     Both answers address the same fact but give
                contradictory specifics. Examples: "X is 330m tall" vs
                "X is 280m tall" (incompatible value); "X works" vs "X
                does NOT work" (opposite polarity).
- unique_to_a:  A states a fact that B does not address.
- unique_to_b:  B states a fact that A does not address.

RULES:

1. MULTI-VALUED roles do NOT disagree just because each side names a
   different filler (founders, members, ingredients, list elements).
   "Founded by Gates" and "Founded by Allen" are both true. If both
   names appear on both sides, emit two AGREE findings (one per name).
   If a name appears on only one side, emit unique_to_<that side>.

2. Each finding has:
   - label
   - summary: one short sentence describing the finding (the user
     reads this first)
   - quote_a: a verbatim or near-verbatim excerpt from Model A's
     answer that supports the finding. Empty string for unique_to_b.
   - quote_b: same for Model B. Empty string for unique_to_a.
   - rationale: one sentence explaining your decision; for AGREE,
     mention specificity if one side is more precise than the other;
     for DISAGREE, cite the specific values that conflict.

3. Be exhaustive but don't double-count. Each substantive fact in
   either answer should appear in exactly ONE finding.

4. Ignore filler/hedge phrases ("It is worth noting", "Generally
   speaking"). Focus on factual content.

5. Quotes should be short — a sentence fragment or one sentence —
   enough for the user to recognise the source in the answer pane.
   Use the answer's own wording; don't paraphrase quotes.

OUTPUT: JSON matching the FindingSet schema. Output only the JSON, no
commentary."""


async def judge_answers(answer_a: str, answer_b: str) -> FindingSet:
    """Run the holistic judge against both raw answers."""
    if not answer_a.strip() or not answer_b.strip():
        return FindingSet(findings=[])

    user_prompt = (
        f'Model A\'s answer:\n"""\n{answer_a}\n"""\n\n'
        f'Model B\'s answer:\n"""\n{answer_b}\n"""\n\n'
        f"Produce a FindingSet aligning both answers."
    )

    async with AsyncOpenAI() as client:
        response = await client.beta.chat.completions.parse(
            model=settings.judge_model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format=FindingSet,
            **temperature_kwargs(settings.judge_model, settings.judge_temperature),
            **reasoning_kwargs(settings.judge_model, "low"),
        )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        return FindingSet(findings=[])
    return parsed


__all__ = [
    "JUDGE_SYSTEM_PROMPT",
    "judge_answers",
]

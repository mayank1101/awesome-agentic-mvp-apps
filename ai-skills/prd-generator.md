# PRD Generator Skill

**Description:** An instruction set that turns a product brief into a structured, production-grade Product Requirements Document — outline first, then one section at a time.

**Reference implementation:** [`mvps/ai-prd-generator`](../mvps/ai-prd-generator)

---

## Why two phases

A single call asked for a whole PRD writes the sections it always writes, at whatever length it feels like, and repeats itself across them. Splitting the work fixes all three:

1. **Outline pass** picks the sections *this* product needs and gets a word budget.
2. **Section pass** runs once per section and is handed the *full outline* every time — that is what stops a section from repeating or contradicting a sibling.

Word budget per section = total words ÷ number of sections the outline actually produced, floored at a minimum (~120 words) so a long outline cannot produce stubs.

---

## Shared audience guidance

Prepend to both phases:

> Write for a **{audience}** audience: match their vocabulary and the depth they need. For engineering-leaning readers lead with architecture, data flow, interfaces, and failure modes; for business-leaning readers lead with user impact, outcomes, and trade-offs, keeping technical detail brief and plain.

`{audience}` is user-supplied and lands inside instructions, so defang it first — see [untrusted-input-guardrail](untrusted-input-guardrail.md).

---

## Phase 1: Outline Prompt

You are a senior product management partner who writes crisp, well-structured PRDs.

{audience guidance}

Given the {subject} context you are sent, produce a PRD outline: an overall title and an ordered list of sections tailored to this specific {subject}.

Target length: about {total_words} words total across {section_count} sections. Pick the sections that matter most — do not pad the outline with sections just to hit a count.

{sections_hint}

Reply with one JSON object and nothing else — no prose, no code fence — in exactly this shape:

```json
{"title": "overall PRD title",
 "sections": [{"title": "section title", "summary": "1-2 sentence brief for this section"}]}
```

> Spell the JSON shape out rather than leaving it to the provider: a strict `json_schema` response format is rejected by many free models, so the working default is a plain JSON object plus this shape in the prompt.

---

## Phase 2: Section Prompt

You are a senior product management partner writing one section of a PRD about a {subject}.

{audience guidance}

Write the content for the section **"{section_title}"** ({section_summary}) in clean Markdown. Be specific and concrete, use bullet points and sub-headings where useful, and stay consistent with the rest of the PRD outline. Do not repeat the section title as a heading — start directly with the content.

Keep this section to roughly {word_budget} words, be concise and avoid filler; do not pad to reach the target.

**User message:** the fenced brief, then the full outline as `- {title}: {summary}` lines.

The section title and summary come from the model's own outline, which the user's brief influenced — so defang them like any other untrusted value before interpolating.

---

## Levers

| Lever | Values | What it changes |
| :--- | :--- | :--- |
| `scope` | `product` \| `feature` | The word `{subject}` throughout, and `{sections_hint}`. A feature PRD also takes the parent product as context, and infers conservatively when it is missing. |
| `length` | short / standard / long | `{total_words}` and `{section_count}`, and through them every section budget. |
| `audience` | free text | The audience guidance line. |

---

## Input format

```
Product name: [Name]
One-liner: [Elevator pitch]
Problem statement: [Problem]
Target users: [Users]
Goals:
- [Goal 1]
- [Goal 2]
Additional context / notes: [optional]
```

Omit optional fields rather than sending them empty — a blank heading is an invitation to invent content. Send the whole block fenced, in the user message, with the untrusted-data notice appended to the instructions.

## Output format

The assembled Markdown: the outline's title as `#`, then each section title as `##` followed by that section's content.

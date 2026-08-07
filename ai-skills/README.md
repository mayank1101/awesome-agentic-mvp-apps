# AI Skills

Seven system prompts, written up as Markdown, plus one shared defense pattern. No SDK, no
Python, no framework — a skill here is the instructions themselves, meant to be pasted straight
into whichever AI tool you actually use: ChatGPT custom instructions, a Claude Project or
`CLAUDE.md`, a Cursor rule, a Gem, an OpenRouter system message, your own agent framework's
`instructions=` field.

If you're building in Python and want the working code instead of the prompt, see
[`../ai-agents`](../ai-agents) — the standalone-agent counterpart to this folder. Each skill here
names its reference implementation, and most name a standalone agent file too.

---

## What a "skill" is, here

Each file is a **system prompt**, the **reasoning behind it** (what it's guarding against and
why the wording is what it is, not something else that would read almost the same), the **exact
shape of the user message** it expects, and the **JSON output schema** it's held to. That's
deliberate — a system prompt copied without the reasoning behind it degrades the first time
someone "simplifies" a sentence that was load-bearing.

None of these are "be a helpful expert in X" prompts. Every one of them is built around a
specific failure mode seen in a real, running MVP in this repo, and the prompt exists to close
that failure mode structurally, not to ask the model to please not do it:

- **[Competitor Analyst](competitor-analyst.md)** — writes six sections about one company from
  supplied search evidence only; `"Not found in public sources"` is trained as a *correct*
  answer, not a fallback, because a model that's seen the phrase used approvingly reaches for it
  and one that's only seen it allowed does not.
- **[Review Gap Analyst](review-gap-analyst.md)** — clusters a competitor app's critical reviews
  into cited feature gaps. The model cites review ids; there is no field in its output for a
  quote, so it cannot fabricate an excerpt even if asked to.
- **[PRD Generator](prd-generator.md)** — outline first, then one section at a time, so a long
  document doesn't degrade into vaguer prose the further the model gets from the brief.
- **[PM Interview Coach](pm-interview-coach.md)** — two separate prompts, an interviewer that
  only asks and never grades mid-interview, and an evaluator that only sees the finished
  transcript. Keeping them apart is what stops the interviewer angling questions toward a score
  it's already forming.
- **[Resume ↔ Job Fit](resume-job-fit.md)** — scores a resume against one posting, then rewrites
  it, under an instruction to never state a fact the resume didn't already carry. The number
  itself is never asked of the model (see below).
- **[Resume-Driven Job Search](resume-driven-job-search.md)** — ranks fetched postings against a
  resume with the same no-fabrication discipline, at the scale of a search result list rather
  than one document.
- **[Feature Prioritisation](feature-prioritisation.md)** — the model classifies four RICE/ICE
  factors per feature onto anchored scales; it never produces the score. There's no field for
  one, so arithmetic disagreement between two runs is structurally impossible.
- **[Untrusted Input Guardrail](untrusted-input-guardrail.md)** — not a task prompt. The fencing
  block every skill above ends with, written up on its own so it's reusable anywhere a model is
  about to read text it didn't write.

---

## How to use one

1. **Open the skill file** and read the "two failures" or "one thing worth knowing" section
   first — it's there so you know what happens if you paraphrase a sentence in the next section.
2. **Copy the System Prompt block** verbatim into your tool's system/instructions field.
3. **Match the user message shape** the skill documents — most expect a labelled request line
   followed by a fenced block of evidence or source material, not a single freeform paragraph.
4. **Enforce the output schema** if your tool supports structured output or JSON mode. If it
   doesn't, most skills' prompts already ask for JSON in plain words and describe a repair turn
   for when the reply doesn't parse — resend the documented repair instruction once before giving
   up.
5. **Do the arithmetic and the citation-resolution outside the model.** This is the part that's
   easy to skip when you're just pasting a prompt in — several skills below are only as safe as
   the code around them. Read "What the prompt cannot do" in each skill; it's not filler.

---

## Catalog

| Skill | Description | Reference implementation | Standalone agent |
| :--- | :--- | :--- | :--- |
| [Competitor Analyst](competitor-analyst.md) | Six-section competitor brief from search evidence, with explicit missing-data abstention. | [`ai-competitor-analyzer`](../mvps/ai-competitor-analyzer) | [`ai_competitor_analyser_agent.py`](../ai-agents/ai_competitor_analyser_agent.py) |
| [Review Gap Analyst](review-gap-analyst.md) | Clusters critical App Store / Google Play reviews into cited feature gaps; cites ids, never writes an excerpt. | [`ai-app-store-review-competitor-tracker`](../mvps/ai-app-store-review-competitor-tracker) | [`review_gap_analyser_agent.py`](../ai-agents/review_gap_analyser_agent.py) |
| [PRD Generator](prd-generator.md) | Product brief → structured PRD, outline first, then section-by-section expansion. | [`ai-prd-generator`](../mvps/ai-prd-generator) | [`prd_ai_agent.py`](../ai-agents/prd_ai_agent.py) |
| [PM Interview Coach](pm-interview-coach.md) | Strict interviewer + separate rubric evaluator with quoted evidence, zero safe midpoint. | [`ai-pm-interview-coach`](../mvps/ai-pm-interview-coach) | [`pm_interview_agent.py`](../ai-agents/pm_interview_agent.py) |
| [Resume ↔ Job Fit](resume-job-fit.md) | Scores a resume against one posting and rewrites it with a mechanical no-fabrication guard. | [`ai-job-match`](../mvps/ai-job-match) | [`job_match_agent.py`](../ai-agents/job_match_agent.py) |
| [Resume-Driven Job Search](resume-driven-job-search.md) | Two-tiered ranking of fetched postings against a resume with strict quote provenance. | [`ai-job-search-assistant`](../mvps/ai-job-search-assistant) | [`job_search_agent.py`](../ai-agents/job_search_agent.py) |
| [Feature Prioritisation](feature-prioritisation.md) | RICE/ICE ranking where the model classifies factors and never touches the score. | [`ai-feature-prioritisation-assistant`](../mvps/ai-feature-prioritisation-assistant) | [`feature_prioritisation_agent.py`](../ai-agents/feature_prioritisation_agent.py) |
| [Untrusted Input Guardrail](untrusted-input-guardrail.md) | The shared fencing pattern every skill above uses for retrieved or pasted text. | — | — |

---

## Principles that show up in every skill here

- **Citation, not quotation, wherever a model would otherwise have to reproduce source text
  exactly.** A model asked to cite an id or a source label can't get the *wording* wrong,
  because it never touches the wording. A model asked to quote can, and eventually will.
- **A structurally absent field beats an instruction not to fill one in.** Where a skill's output
  schema has no `score` field, no `url` field, or no `quote` field, that's the actual guarantee —
  not the sentence in the prompt asking the model to leave it out.
- **"Not found" / "insufficient evidence" is trained as a correct, good answer**, with a worked
  example, everywhere a skill can legitimately have nothing to report. A model that has only seen
  the honest answer permitted reaches for a plausible-sounding one instead; a model that's seen it
  modeled as *right* does not.
- **The untrusted-input fence is universal, not skill-specific.** Every skill that reads retrieved
  web content, a resume, a transcript, or review text ends its system prompt with the pattern in
  [untrusted-input-guardrail.md](untrusted-input-guardrail.md), naming the likely source of an
  injection attempt specifically (a competitor's own marketing page, an app's own users, a
  candidate's resume) rather than with a generic warning.
- **What the prompt cannot guarantee, code checks afterward.** URL allowlisting, citation-id
  resolution, and fabricated-fact detection all happen after the model replies, never as a
  request the model could simply decline. Every skill's "Around the prompt" section says exactly
  what that check looks like for that skill.

---

## Adapting a skill for a tool with no system-prompt field

Some surfaces (a plain ChatGPT conversation, a basic chat widget) don't expose a separate system
role. In that case, paste the **System Prompt** and **Untrusted-input notice** sections as the
first message in the conversation, followed by a line making the framing explicit — *"Everything
above this line is your operating instructions for the rest of this conversation."* — then send
the user message in the shape the skill documents as a follow-up turn. The guarantees that depend
on code (citation resolution, URL allowlisting, arithmetic) still need to live in whatever is
reading the model's replies; a skill used this way is only as trustworthy as that surrounding
check.

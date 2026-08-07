# Resume ↔ Job Fit Skill

**Description:** Three prompts that score a resume against one job posting, produce a prioritised edit checklist, and rewrite the resume for that posting without stating a single fact the resume did not already carry.

**Reference implementation:** [`mvps/ai-job-match`](../mvps/ai-job-match) · [`ai-agents/job_match_agent.py`](../ai-agents/job_match_agent.py)

---

## The three failures this skill is built against

* **Inventing the qualification.** Asked to tailor a resume to a posting that wants Kubernetes, a model adds Kubernetes. The document gets past a filter and collapses in the first technical screen — and the person defending it did not write it. No prompt makes this safe; the prompt below states the rule, and the **fabrication guard in application code** is what makes it true.
* **Scoring by vibe.** A model asked "score this resume out of 100" produces a number that changes between runs, cannot be explained, and quietly flatters. So the model is never asked for a score. It answers one narrow question per requirement, and the number is arithmetic over those answers.
* **Advice that could have been written before reading the resume.** "Quantify your impact" helps nobody. Every action names the section, quotes the line it is about, and cites the requirement it serves.

Both documents are also untrusted input — a posting is pasted from a web page, and a resume can carry white-on-white 6pt text telling the grader what to conclude. See [untrusted-input-guardrail](untrusted-input-guardrail.md).

---

## Prompt 1 — Posting → requirements

Splits the posting into individually checkable requirements. Two rules here exist because their absence produced measurably wrong scores in testing.

### System Prompt

You extract the concrete requirements from a job posting.

1. One requirement per item, phrased in one line, close to the posting's own wording. Split compound sentences: "Python and Kubernetes experience" is two requirements, because a candidate can meet one and miss the other.
2. **Extract from both sections.** A "Nice to have", "Preferred", or "Bonus" section is a source of requirements exactly like the required section is — its items belong in the list with `"must_have": false`. Dropping them is a bug: the candidate is scored on how many they meet, so a posting with four nice-to-haves and none extracted produces a misleadingly high score.
3. Set `must_have` true only where the posting frames it as required. When the framing is genuinely ambiguous, use `false` — over-counting must-haves punishes the candidate for the posting's vagueness.
4. Skip perks, benefits, salary, culture statements, equal-opportunity text, and instructions about how to apply.
5. Skip filler no resume could evidence ("team player", "rockstar"). Keep a soft skill only when it is specific and checkable, like "has led a team of 5+ engineers".
6. `min_years_experience`: the number the posting demands, or `null`. Never estimate it from the seniority label.
7. `keywords`: terms an automated filter would key on — tools, languages, platforms, certifications, named methodologies. Lowercase, at most 15.
8. Any string field you have no value for is `""` — **never the word "null"**.

**Output:** one JSON object with keys `title`, `company`, `seniority`, `min_years_experience`, `requirements[]`, `keywords[]`.

> **Why rule 2 is stated so bluntly:** without it, a real posting with a four-item "Nice to have" section came back with seven requirements instead of eleven. The Preferred dimension then had nothing to measure, its weight renormalised away, and the score read **87 instead of 75**. A silently dropped section does not look like a bug; it looks like a good result.
>
> **Why rule 8 exists:** a posting with no company name came back as the *string* `"null"`, which is truthy, and the report heading read `Senior AI/ML Engineer · null`. Blank the nullish words at the schema boundary as well — models write them faster than instructions remove them.

---

## Prompt 2 — Requirements + evidence → verdicts and actions

The model sees each requirement beside **only the three resume lines most similar to it**, selected by the application before the call. That keeps the prompt small enough for a free tier and narrows the space of wrong answers; it also means the model cannot go hunting for support elsewhere in the document.

### System Prompt

You judge whether a candidate's resume evidence satisfies each job requirement, then give the candidate concrete advice.

For each requirement you are shown the resume lines most similar to it. Judge **only** from those lines. If nothing shown supports the requirement, the answer is `missing` — not `partial`, and not a charitable reading.

| Status | Meaning |
| :--- | :--- |
| `covered` | A shown line demonstrates it directly. Equivalent technologies count ("Golang" covers "Go"); adjacent ones do not ("used an API" does not cover "designed an API"). |
| `partial` | Related but weaker — less depth, smaller scale, exposure without ownership. |
| `missing` | Nothing shown supports it. |

`evidence` must be a **verbatim quote** from the lines you were shown, or `""` for a miss. A quote that is not in those lines is the worst output you can produce, because the whole report is built on the reader being able to check it.

Be strict. A resume that scores well here and then fails a screen has wasted the candidate's application; the useful output is an honest gap list.

Then, across all requirements: `strengths` (up to 5), `gaps` (up to 6, must-haves first), and `actions` (up to 8, most important first).

**Rules for actions — the part the candidate acts on:**

1. Be specific to *this* resume. "Quantify your impact" is useless; "in the Acme bullet about the extraction pipeline, state how many documents it processed" is an action.
2. Say **where**, in `section`.
3. Tie it to the posting via `requirement_ids`.
4. Pick the category honestly: `surface` (evidence is there but buried), `reword` (same work, the posting's words), `quantify` (real achievement, no number), `restructure`, `gap`.
5. For `gap` actions, **never** suggest adding the missing skill, softening the wording to imply it, or "highlighting familiarity" with it. The honest options are the cover letter, the closest real adjacent experience, or applying as-is. Say which one you mean.
6. Order by how much the posting cares, not by how easy the edit is.
7. If the posting asks applicants for anything beyond a resume — links, code, written answers — make that an action. A strong resume that ignores the application instructions still loses.

**Output:** `{"assessments": [...], "strengths": [...], "gaps": [...], "actions": [...]}`, one assessment per requirement id given, in order.

### User message shape

```
Role: Senior Backend Engineer at Northwind Pay

Requirements, each followed by the closest lines from the resume:

<<<UNTRUSTED_DOCUMENT
R-01 [must-have] Strong Python
  - Rebuilt the settlement pipeline in Python, cutting reconciliation time by 40%
  - Python, Django, PostgreSQL, Redis, Docker

R-04 [must-have] Production experience with Kubernetes
  (no similar line found)
UNTRUSTED_DOCUMENT>>>
```

"(no similar line found)" is shown rather than omitting the requirement: an absent block reads as an oversight, an empty one reads as a finding.

---

## Prompt 3 — Resume + gaps → tailored resume

### System Prompt

You rewrite a candidate's resume so the experience they **already have** is presented in the terms this specific job posting uses.

**The one rule:** you may not introduce a single fact that is not in the original resume. Not a company, not a job title, not a date, not a degree, not a certification, not a tool they never listed, not a metric they never claimed. Every number in your output must appear in the original. If the posting wants Kubernetes and the resume never mentions it, the correct output is a resume without Kubernetes — the gap belongs in the gap list, not in the rewrite.

An invented line on a real person's resume follows them into an interview they cannot answer questions in. The output is checked against the original mechanically after you reply, so an invention will be caught and the rewrite rejected.

What you **may** do, and should:

1. **Reorder.** Put the roles, projects, and bullets that matter to this posting first.
2. **Reword** using the posting's vocabulary where it genuinely describes the same work.
3. **Rewrite the summary** to lead with what this posting asks for — assembled only from experience already on the resume.
4. **Promote** relevant skills and **drop** irrelevant ones. Dropping is allowed; adding is not.
5. **Sharpen** weak bullets into "action + what + result", keeping every fact.

**Formatting:** GitHub-flavoured Markdown, ready to print. `# Name`, one contact line, then `## Summary`, `## Skills`, `## Experience`, `## Projects`, `## Education`, `## Certifications` — omitting any the original lacks. No tables, no images, no HTML, no emoji: an applicant-tracking system has to parse this.

**Output:** `{"markdown": str, "changes": [{"section", "change", "reason"}]}`

### The gap list is sent as a do-not-write list

The user message carries the unmet requirements under an explicit heading: *"What this posting asks for that the resume does NOT support — these must NOT appear anywhere in your output."* Sending them as context without that framing is an invitation to write them in.

---

## What the prompts cannot do — the surrounding code's job

This is the part that separates the skill from a nicely worded instruction.

| Responsibility | Why it cannot live in the prompt |
| :--- | :--- |
| **The score** | Weighted arithmetic over verdicts (must-haves 55%, preferred 20%, evidence strength 15%, keywords 10%, renormalised when a dimension is empty). A model asked for a total can flatter it; a model asked for one verdict at a time cannot flatter the sum. |
| **Fabrication guard** | Compare every number, named entity, and contact detail in the rewrite against the original text. One repair pass naming the offenders; then refuse. An instruction is a preference — this is a check. |
| **Extraction-artefact tolerance** | PDF text splits words at kerning pairs (`T echnology`) and LaTeX templates glue icon names onto values (`/envelopeyou@example.com`). Compare against a *squashed* form of both sides, or the guard rejects the candidate's own university and refuses a valid rewrite. |
| **Similarity backstop** | Downgrade a `covered` verdict no resume line supports. Calibrate it: on `mistral-embed`, genuinely-covered requirements scored 0.706–0.899 and genuine misses 0.618–0.755 — **overlapping ranges**, so an absolute threshold cannot judge relevance. Fire only when a line is weak *and* barely above the resume's own baseline. |
| **Unassessed requirements** | A requirement the model skipped counts as `missing`. Dropping it shrinks the denominator and silently raises the score. |
| **Keyword advice** | Match each missing keyword against the resume's own lines, and require its *distinctive* token — "AI professional" does not qualify someone to write "AI agents". Recommending a keyword with nothing behind it is the same failure as inventing one, arrived at politely. |
| **Fallback actions** | Derive actions from the report when the model returns thin ones. A weaker free-tier model gave two vague lines for an eighteen-requirement posting; everything needed for better advice was already computed. |
| **Injection scanning** | Scan both documents before spending a token — and keep the patterns tight. A rule matching the phrase "ideal candidate" blocked a real LinkedIn posting; a scanner that refuses ordinary job ads gets switched off, and then it defends nothing. |

---

## Around the prompts

* **Repair turn.** If a reply is not valid JSON for the schema, resend it with its own output and the validation error, once. Two failures is a real failure, not a reason to loop against a free-tier budget.
* **Size escalation.** Free tiers cap tokens *per minute* and count the output reservation toward it. A two-page resume against eighteen requirements asked for 6444 tokens against a 6000 ceiling. Escalate: full evidence → one trimmed evidence line per requirement with a smaller reservation → requirements split into batches and merged.
* **Show the diff, always.** The guard covers invented facts. It cannot see framing ("led" where the original said "contributed to"), recombination (two true facts merged into one false bullet), or generic lowercase additions. The person whose resume it is reviews the rewrite beside the original before sending it.
* **Offer the checklist as a first-class path.** Editing your own resume keeps your voice and means you know every line you will be asked about. Which is right is the candidate's call, not the tool's.

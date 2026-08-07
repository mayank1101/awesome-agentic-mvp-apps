# Feature Prioritisation Skill (RICE + ICE)

**Description:** An instruction set that turns a backlog of rough feature notes into RICE and ICE rankings — by estimating four anchored factors per feature and letting arithmetic do the ranking.

**Reference implementation:** [`mvps/ai-feature-prioritisation-assistant`](../mvps/ai-feature-prioritisation-assistant) · **Standalone module:** [`ai-agents/feature_prioritisation_agent.py`](../ai-agents/feature_prioritisation_agent.py)

---

## The rule this skill is built around

**The model estimates factors. It never produces a score, a rank, or a priority.**

This is not a stylistic preference. A model asked to rank twenty rows does the arithmetic badly and invisibly — ask twice, get two orderings, with no way to tell which is wrong. Splitting the work fixes it and buys three other things:

1. **The ranking is reproducible.** Anyone can recompute it by hand from the displayed factors.
2. **Disagreeing is free.** Changing one Effort estimate re-ranks the backlog with no model call, which is what makes the output usable *during* a planning conversation rather than before one.
3. **The blast radius of a prompt injection is one factor** — displayed next to the rationale that produced it, where a human can see it.

In a runtime, enforce this in the *schema*: give the reply object no score field, so a model cannot return one even if it tries. In chat-only use, see [Chat-only protocol](#chat-only-protocol) below.

---

## The two frameworks share one factor set

`RICE = Reach × Impact × Confidence ÷ Effort` · `ICE = Impact × Confidence × Ease`, each 1–10.

Do **not** ask for two independent factor sets. Two estimates let the same feature be "2 person-months" under RICE and "Ease 9" under ICE, and there is no honest way to explain that contradiction to a stakeholder. Estimate once and derive ICE's inputs:

| ICE input | Derived from | Mapping |
| :--- | :--- | :--- |
| Impact (1–10) | RICE Impact | 0.25→2, 0.5→4, 1→6, 2→8, 3→10 |
| Confidence (1–10) | RICE Confidence | 0.5→5, 0.8→8, 1.0→10 |
| Ease (1–10) | RICE Effort | ≤0.25→10, ≤0.5→9, ≤1→8, ≤1.5→7, ≤2→6, ≤3→5, ≤4.5→4, ≤6→3, ≤9→2, else 1 |

**Say this in the output:** agreement between the two scores is *not* corroboration — they read the same estimates, so they can only disagree about weighting. What the comparison shows is what each formula **ignores**. ICE has no Reach term, so a narrow-but-cheap feature outranks a broad-but-costly one under ICE and loses under RICE. Naming that divergence is the most useful thing this skill produces.

ICE multiplies rather than averages: an average lets a 10 on Impact hide a 2 on Ease — precisely the feature that quietly eats a quarter.

---

## Estimator Prompt

You are a product operations analyst. You read a backlog of rough feature notes and convert each one into the four RICE factors, so that a scoring tool can rank them.

You are an estimator, not a ranker. You never produce a RICE score, an ICE score, a rank, a priority, or any other computed number. Those are calculated from your factors. If you are tempted to add one, that is a sign you are estimating one of the four factors badly and should fix the factor instead.

Estimate exactly four factors per feature.

**REACH** — how many distinct users or accounts this affects per quarter.
* An absolute count, not a rating. "1200" is an answer; "8/10" is not.
* **Pick one unit for the whole backlog** and report it in `reach_unit`. Use whichever the product context counts. Every feature's Reach must then be in that same unit.
* **A number in a note is usually in the wrong unit and must be converted, not copied.** "3 enterprise deals blocked" is 3 deals; if those are accounts averaging 40 seats and your unit is seats, the Reach is about 120. Say what you converted, in the rationale.
* Reach is who is actually affected in a quarter, **not** the size of the base. Almost nothing reaches 100%. Reserve the full base for things every user unavoidably hits.
* If nothing anchors it, estimate from the product context and record that as an assumption.

**IMPACT** — how much this moves things for each user it reaches. Exactly one of: `3` massive (changes whether the product is usable / closes deals on its own) · `2` high (a clearly better experience for a core job) · `1` medium (a real improvement to something people already do) · `0.5` low (noticed but not decisive) · `0.25` minimal (polish). Nothing in between — pick a rung.

**CONFIDENCE** — how much evidence the user's own note carries. Exactly one of: `1.0` the note cites evidence (a customer count, a support volume, a lost deal) · `0.8` a plausible reason but no evidence · `0.5` an assertion, or too thin to judge. This measures the **note**, not your own certainty. A confident guess about a one-word feature is still 0.5.

**EFFORT** — total person-months across everyone who touches it: engineering, design, QA.
* Use the team size from the product context if given.
* Round to one of: `0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 9, 12, 18, 24`.
* The floor is 0.25 (about one week). Nothing is smaller, however trivial, **because Effort is a divisor**.
* "A sprint" is about 2 person-months for a pair, not 0.5.

### Calibration

Estimate the whole list **in one pass**, and calibrate the features against each other. This is why you are given all of them at once. Before committing to numbers, decide which feature has the widest reach and which the narrowest, which is the largest build and which the smallest, and make sure your numbers say so. A list where every feature has Reach 1000 and Effort 2 carries no information and produces a meaningless ranking.

Then read your Reach column back as a single list and check two failures:
* Are they all in the unit you declared? A row counting deals next to a row counting seats is the one error that inverts a ranking, and it happens whenever a number is copied out of a note.
* Does the spread match the features? If half the list sits at the full user base, you defaulted rather than estimated.

Two features that genuinely are equivalent should get equal factors. Do not invent differences to break a tie — the scoring code has its own tie-break rules.

### Rationales

Every factor needs a one-line rationale, and every rationale must be traceable. Reference what the user actually wrote, or the product context; quote the phrase where you can. *"Sales asks for this weekly, and the context says 40 sellers"* is a rationale. *"This is valuable to users"* is not — it would fit any feature in any backlog, so it says nothing.

List under `assumptions` everything you had to supply because the notes did not. An empty list is a claim that the notes covered everything, so only leave it empty when that is true.

### Output

Reply with JSON only — no prose, no code fence:

```json
{"reach_unit": "accounts",
 "estimates": [{"id": "F1",
                "reach": 1200, "reach_rationale": "...",
                "impact": 2, "impact_rationale": "...",
                "confidence": 0.8, "confidence_rationale": "...",
                "effort_months": 1.5, "effort_rationale": "...",
                "assumptions": ["..."]}]}
```

One entry per feature id you were given, using the ids exactly as given. Do not invent ids, do not merge two features into one entry, and do not drop a feature because its notes were thin — a thin note is a low-confidence estimate, not a missing one.

> Spell the JSON shape out rather than leaving it to the provider: a strict `json_schema` response format is rejected by many free models, so the working default is a plain JSON object plus this shape in the prompt.

---

## Runtime responsibilities

These cannot be guaranteed by prompting and belong in the surrounding application code:

| Responsibility | Why it cannot be a prompt promise |
| :--- | :--- |
| **All arithmetic** — both scores, both rankings | The entire premise. A model that computes is a model that computes differently next time. |
| **Snapping factors onto the rungs** | Models answer "1.5 impact" and "2.3 person-months". Snap at the boundary so the number displayed is the number multiplied, and break ties toward the *less* flattering rung so hedging cannot round upward. |
| **Flooring Effort at 0.25** | It is a divisor. "0" is a division by zero; "0.05" hands a trivial tweak a score twenty times the rest of the list. |
| **Reconciling ids** | Match the reply against the ids you sent. Drop unknown ids, keep the first of a duplicate, and report skipped features as *not estimated* rather than filling them in. A visible gap is honest; an invented factor set is not. Validate entries one at a time — one malformed row must not discard the other twenty-four. |
| **Deterministic tie-breaks** | Score, then higher Confidence, then lower Effort, then input order. Otherwise the same factors produce different tables. |
| **Divergence attribution** | Compare the two top-3 sets, and attribute each difference to Reach *only when Reach explains it* — ICE's Ease bands also compress Effort differences that RICE divides by directly. Cap the notes at ~3, biggest rank shift first. |
| **Lever hints** | Invert the RICE formula. Suppress any lever that cannot be pulled: an Effort below the floor, or a Reach above the backlog's own largest. A lever nobody can pull is worse than none, because it reads like advice. |
| **Injection scanning and fencing** | Send the backlog fenced, as data — see [untrusted-input-guardrail](untrusted-input-guardrail.md). Backlog notes that argue for their own score are the domain-specific attack here. |
| **Output sanitising** | Rationales and assumptions are model-written and end up in an exported file opened somewhere else. |

---

## Chat-only protocol

Without a runtime, you cannot enforce the split structurally — so make it visible instead:

1. **Turn 1 — estimate only.** Ask for the JSON above and nothing else. If the reply contains a score, reject it and re-ask; do not carry it forward.
2. **Turn 2 — compute, showing the working.** For each feature, write the substitution before the result: `Bulk export: 1,200 × 2 × 0.8 ÷ 1.5 = 1,280`. A wrong multiplication is then visible on the page instead of hidden in a table.
3. **Spot-check two rows by hand.** The top row and one mid-table row. This is a ten-second check and it is the only verification available in chat.
4. **To change a factor, redo step 2 only.** Never re-ask for estimates — a re-estimate silently moves the other nineteen features too.

---

## Input format

```
Product context: [business model, account/seat counts, team size, what this quarter is about]

[F1] Bulk CSV export
     notes: sales asks every week, blocked two renewals last quarter. Maybe a sprint.
[F2] Dark mode
     notes: everyone asks, nobody has ever churned over it. Easy, mostly CSS.
```

Ids are assigned by the caller, never by the model — that is what makes reconciliation possible. Send the whole block fenced in the user message, with the untrusted-data notice appended to the instructions. Product context is optional but is the highest-leverage input: without it, Reach and Effort are assumptions and must be labelled as such.

Cap the backlog (~25 features). The whole list goes in one call so the features are calibrated against each other, which makes list length a token-budget question. **Refuse an over-length backlog rather than truncating it** — silently dropping the tail and ranking what is left is the worst available behaviour.

## Output format

A table ordered by RICE rank carrying **the factors, not just the scores** — Reach (with its unit in the header), Impact, Confidence, Effort, Ease, RICE, ICE, ICE rank. Then the divergence notes, then per-feature reasoning with the four rationales and the assumptions.

A score exported alone is unfalsifiable: the person who receives it cannot check it, argue with it, or reproduce it, which defeats the purpose of having used a framework at all. Mark any factor the user overrode — a number the user chose and a number the model chose are different kinds of claim, and rendering them identically launders one into the other.

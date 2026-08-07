# AI Feature Prioritisation Assistant

Paste a backlog of rough feature notes. Get it ranked by **RICE** and **ICE**, with every factor
shown, explained, and editable.

The arithmetic was never the hard part of RICE. The blank cells are — a spreadsheet demands four
numbers per feature and offers no help producing them, so they get guessed, and guessed
inconsistently by the time you reach item 18. This app estimates the factors from your own prose,
in one pass over the whole list so the features are calibrated against each other, and then does
the multiplication in Python.

```
Bulk CSV export — sales asks every week, blocked two renewals. Maybe a sprint.
Dark mode — everyone asks in the feedback widget, nobody has churned over it. Easy, mostly CSS.
SSO / SAML — only 3 enterprise deals blocked, but they're our biggest. Big lift.
```

becomes a ranked table with Reach, Impact, Confidence and Effort filled in, a one-line rationale
under each factor quoting your own note, and a plain-language account of where the two frameworks
disagree and which factor is responsible.

---

## The one thing worth knowing

**The model never produces a score.**

It classifies your prose onto four anchored scales. `app/services/scales.py` and
`app/services/scoring.py` do every multiplication and every comparison, and they import no client
and make no call. This is not a stylistic preference:

* A model asked to rank twenty rows will do arithmetic badly and invisibly. Ask it twice, get two
  orderings.
* `FeatureEstimate` — the only model the agent layer produces — **has no score field**. A model
  cannot return a RICE number here even if it tries; there is nowhere for one to go. A test asserts
  this against the generated JSON schema.
* A second test asserts, for every row, that the rendered score equals the formula applied to the
  rendered factors. If that ever fails, the app is broken in the only way that matters.
* Because scoring is pure, **editing a factor re-ranks instantly and calls no model**. The argument
  about one Effort estimate is the whole point of a planning meeting, and it should cost nothing.

The strongest guardrail in the app is this shape, not the pattern matching. A successful prompt
injection can move a *factor* — and every factor is displayed next to the rationale that produced
it, where a human can see it.

## Why both frameworks, and what that actually buys you

`RICE = Reach × Impact × Confidence ÷ Effort` · `ICE = Impact × Confidence × Ease`, each 1–10.

The obvious implementation asks the model for both factor sets. That was rejected. Two independent
estimates let the same feature be "2 person-months" under RICE and "Ease 9" under ICE, and there is
no honest way to explain that contradiction to a stakeholder.

So **one factor set is estimated**, and ICE's inputs are derived from it by published mappings —
Impact and Confidence remapped onto 1–10, Ease a fixed monotone function of Effort.

The trade, stated plainly in the app itself: ICE is no longer an independent second opinion, so
**agreement between the two scores is not corroboration.** What the comparison shows is what each
formula *ignores*. ICE has no Reach term at all, so a narrow-but-cheap feature outranks a
broad-but-costly one under ICE and loses under RICE. Where the two top-3 lists differ, the app names
the feature and the responsible factor:

> **Custom invoice templates** — RICE #9, ICE #3. It reaches 12 accounts/quarter against a backlog
> median of 400, and ICE has no Reach term to notice that. ICE is ranking it on impact and ease alone.

That divergence is the most useful thing on the page. It is where the *choice of framework* is
deciding your roadmap, and it is invisible in a single-framework tool.

## Design decisions with teeth

**Reach is declared in one unit for the whole backlog.** The estimator returns a `reach_unit`
("accounts", "seats") and the table header says it. This exists because the first live run produced
a list where SSO's Reach counted blocked *deals* (3) and Keyboard shortcuts' counted *seats*
(12,000) — both defensible alone, and together they inverted the ranking. One declared unit, and a
prompt rule that a number in a note must be **converted, not copied**, is the fix.

**Factors snap to fixed rungs, ties breaking conservatively.** A model asked for Impact answers
"1.5"; a model asked for Effort answers "2.3 person-months". Both are false precision. Everything
snaps to the published ladder on the way in, so the number displayed is the number multiplied, and
a value exactly between two rungs takes the *less* flattering one.

**Effort has a floor of 0.25 person-months.** It is a divisor. An unconstrained "0" is a division by
zero and an unconstrained "0.05" hands a trivial tweak a score twenty times the rest of the backlog
off the back of a rounding opinion.

**ICE multiplies rather than averages.** An average lets a 10 on Impact hide a 2 on Ease — precisely
the feature that quietly eats a quarter.

**Ordering is fully deterministic**, including ties: score, then higher Confidence, then lower
Effort, then the order you typed them.

**Levers are computed by inverting the formula**, not asked of a model — and capped at the
backlog's own largest Reach. The first live run suggested "overtakes Keyboard shortcuts if Reach
reaches 24,000" for a product with 12,000 seats. A lever nobody can pull is worse than no lever,
because it reads like advice.

**Gaps are shown, never filled.** If the estimator skips a feature or returns an unusable row, that
feature is listed as *not estimated* rather than ranked on invented numbers. One malformed row out
of twenty-five does not discard the other twenty-four.

## Running it

```bash
cd mvps/ai-feature-prioritisation-assistant
cp .env.example .env      # then set GROQ_API_KEY (or switch MODEL_PROVIDER)
pip install -r requirements.txt
streamlit run streamlit_app.py --server.port=8505
```

Docker:

```bash
docker build -t ai-feature-prioritisation-assistant . && docker run --rm -p 8505:8505 --env-file .env ai-feature-prioritisation-assistant
```

Tests and lint:

```bash
pip install -r requirements-dev.txt && ruff check . && ruff format --check . && pytest
```

Any of seven providers works via `MODEL_PROVIDER` — openrouter, groq, openai, anthropic, ollama,
gemini, foundry. Only the credentials for the one you pick matter.

## Known limits, stated plainly

* **RICE structurally undervalues enterprise features.** A feature blocking three very large
  accounts scores on those three accounts; revenue concentration is not a term in the formula. The
  app will rank it low and be arithmetically correct. That is a property of RICE, not a bug here —
  but the ranking is a first pass for a conversation, not the decision.
* **Reach quality depends entirely on the product context you provide.** Without it the estimator
  states its assumed baseline and flags the rows resting on it, which is the honest behaviour, not a
  good one. Two lines of context is the highest-leverage input in the app.
* **Backlogs are capped at 25 features.** The whole list goes out in one call so the features are
  calibrated against each other, and that makes list length a token-budget question. The cap is
  refused rather than truncated — silently dropping the tail of a backlog and ranking what is left
  is the worst available behaviour.
* **Free-tier rate limits are the practical ceiling.** Providers charge `max_tokens` against the
  per-minute limit *as requested*, so the output budget is sized from the backlog rather than fixed.
  Even so, a 25-feature backlog will not fit inside Groq's 8,000 tokens/minute on the smaller
  models; the shipped default (`llama-3.3-70b-versatile`) has more headroom. Groq also enforces a
  *daily* token budget per model, which a handful of full runs will exhaust.
* **Reasoning models need a large fixed output allowance.** They spend a chunk of the budget
  thinking before emitting a character of JSON, and it does not shrink with the backlog. Sized too
  low, the provider returns `400 json_validate_failed` with an empty generation — a symptom that
  names JSON and a cause that is arithmetic.
* **Session-scoped.** No accounts, no saved backlogs, no history. Export is the persistence story,
  and both formats carry the factors and rationales, because a score on its own cannot be checked by
  whoever receives it.
* **No integrations, no roadmap output, no other frameworks.** Ranking a list is the product;
  sequencing it is a different app.

## Layout

```
app/
  core/       config, logging, exceptions
  models/     the schemas — note that FeatureEstimate has no score field
  agents/     the single estimator call, its prompt, the provider registry
  services/   scales + scoring (all the arithmetic), backlog parsing, guardrails, export
ui/           input form, results, factor editor, sidebar — no arithmetic anywhere
```

`ui/` depends on `app/`; `app/` has no awareness of Streamlit. The entire `app` package is testable
without a browser or an API key, which is why 119 tests run in under a second.

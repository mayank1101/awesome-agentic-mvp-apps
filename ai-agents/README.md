# AI Agents

Seven self-contained, single-file Python agents — no Streamlit, no web framework, no
project-specific plumbing. Each one is the same agent that powers a full MVP in
[`../mvps`](../mvps), stripped down to the part that actually thinks, so it drops into a
FastAPI endpoint, a Celery worker, a CLI tool, or a notebook with one `import`.

If you want the *instructions* instead of the *code* — to paste into ChatGPT, Claude, Cursor,
or any tool that takes a system prompt — see [`../ai-skills`](../ai-skills), the prompt-only
counterpart to this folder.

---

## Why these exist

The full MVPs in this repo are complete applications: a Streamlit UI, a settings layer, a test
suite, a Dockerfile. Most of that is irrelevant if you already have a backend and just need the
agent logic. Every file here answers one question — *"what's the smallest, dependency-light
version of this agent that I can `pip install pydantic openai` and run?"*

Design rules every agent in this folder follows:

- **One file, no project structure.** Copy it into your codebase and it works. No relative
  imports back into this repo.
- **Pydantic v2 at every boundary.** Requests in, structured results out. No agent here returns
  a bare string for you to regex.
- **Runs with zero API keys.** Every agent falls back to a deterministic or heuristic offline
  mode when no model key is configured, so you can integration-test the surrounding code before
  paying for a single token. The fallback is always honestly labelled (`is_offline_simulated`,
  `is_mock`, or similar) — it never silently pretends to be a real model response.
- **Guardrails are not optional.** Prompt injection scanning, prompt fencing around anything the
  agent didn't write itself, and output sanitisation are built into the module, not left as an
  exercise for the caller.
- **The model never does arithmetic it can get wrong.** Scores, rankings, and stats are computed
  in code from structured model output, not asked of the model directly. See each agent's
  "Features" docstring for what that means concretely — it's a different guarantee in each one
  (a RICE score, a fabrication check, a star-rating distribution).

---

## Install

Every agent needs exactly two packages; a couple need one more, always optional and imported
lazily so importing the module without it still works for everything else.

```bash
pip install pydantic openai
```

| Agent | Extra optional dependency | Needed for |
| :--- | :--- | :--- |
| `job_match_agent.py` | `pypdf` | Reading a resume supplied as a PDF instead of plain text |
| `job_search_agent.py` | `pypdf` | Same, for a resume PDF |
| `review_gap_analyser_agent.py` | `google-play-scraper` | Android / Google Play support (iOS needs nothing extra) |

No agent requires its optional dependency to import or to run in offline mode — you'll get a
clear `RuntimeError` naming the missing package only when you actually exercise the code path
that needs it.

---

## Catalog

| Agent | File | Main class | What it does |
| :--- | :--- | :--- | :--- |
| Competitor Analyser | [`ai_competitor_analyser_agent.py`](ai_competitor_analyser_agent.py) | `CompetitorAnalyserAgent` | Profiles a company across 6 sections (snapshot, product, pricing, positioning, recent moves, strengths/weaknesses) from live or simulated search evidence. Conceals URLs from the model so link fabrication is structurally impossible. |
| Review Gap Analyser | [`review_gap_analyser_agent.py`](review_gap_analyser_agent.py) | `ReviewGapAnalyserAgent` | Pulls a competitor app's most recent App Store or Google Play reviews and clusters the critical ones into cited feature gaps. The model cites review ids, never writes an excerpt — the renderer resolves the real text. |
| PRD Generator | [`prd_ai_agent.py`](prd_ai_agent.py) | `PRDAgent` | Turns a product or feature brief into a structured PRD — outline first, then section-by-section expansion, in Product or Feature scope at three length presets. |
| PM Interview Coach | [`pm_interview_agent.py`](pm_interview_agent.py) | `PMInterviewAgent` | Runs a multi-turn mock PM interview (5 types × 4 seniority levels × 4 company archetypes) and grades the transcript on a 4-level, zero-safe-midpoint rubric. |
| Job Match | [`job_match_agent.py`](job_match_agent.py) | `JobMatchAgent` | Scores a resume against one job posting with a deterministic 0–100 arithmetic score, then rewrites the resume for it under a mechanical guard that rejects any invented fact. |
| Job Search | [`job_search_agent.py`](job_search_agent.py) | `JobSearchAgent` | Searches a domain-whitelist of job sites from a resume, two-tier ranks results (cheap snippet pass, then full-text requirement scoring on the strongest candidates only). |
| Feature Prioritisation | [`feature_prioritisation_agent.py`](feature_prioritisation_agent.py) | `FeaturePrioritisationAgent` | Ranks a backlog under RICE and ICE. The model classifies four anchored factors per feature; there is no score field for it to fill in, so the ranking is re-computable for free when you override one estimate. |

---

## Quick start

Every agent follows the same shape: build it with a provider + model, call its one verb, render
the result.

```python
from ai_competitor_analyser_agent import CompetitorAnalyserAgent, AnalysisRequest

agent = CompetitorAnalyserAgent(model="gpt-4o-mini", provider="openai")
report = agent.analyze(AnalysisRequest(name="Notion", domain="notion.so"))
print(report.to_markdown())
```

```python
from review_gap_analyser_agent import ReviewGapAnalyserAgent, AppQuery, Platform

agent = ReviewGapAnalyserAgent(model="llama-3.3-70b-versatile", provider="groq")
report = agent.analyze(AppQuery(platform=Platform.ANDROID, query="com.spotify.music", country="in"))
print(report.to_markdown())
```

Each file's own module docstring has a complete, runnable usage example for that agent
specifically — start there for exact constructor arguments and input model fields, since a
couple of agents (Job Match, Feature Prioritisation) have a richer request shape than the two
above. Every file is also directly executable for a live demo:

```bash
python review_gap_analyser_agent.py
```

---

## Configuring a model provider

Every agent accepts the same two constructor arguments, `model` and `provider`, and reads the
matching API key from the environment if you don't pass one explicitly:

| `provider` | Env var read | Notes |
| :--- | :--- | :--- |
| `"openai"` (default) | `OPENAI_API_KEY` | OpenAI's own API. |
| `"openrouter"` | `OPENROUTER_API_KEY` | Routes to any model OpenRouter serves; set `model` to OpenRouter's model id. |
| `"groq"` | `GROQ_API_KEY` | Fast Llama/Kimi inference. Supported by most, not all, agents in this folder — check the agent's own `__init__`. |
| `"gemini"` | `GEMINI_API_KEY` | Google's OpenAI-compatible endpoint. |
| `"ollama"` | — | Local model server. Reads `OLLAMA_HOST` (default `http://localhost:11434/v1`); no key needed. |

Pass a key explicitly to override the environment: `Agent(api_key="sk-...", provider="openai")`.
Omit both the argument and the environment variable and the agent runs in **offline mode** —
every model call is replaced by a deterministic or heuristic fallback, so you can build and test
the rest of your integration for free.

`MODEL_NAME` is read by every agent's own `__main__` demo block as the default model id when you
run the file directly; it has no effect once you're constructing the agent yourself with an
explicit `model=`.

---

## Guardrails, summarised

Every agent in this folder handles at least one of these; check the individual module docstring
for which ones apply and why:

- **Prompt injection scanning** on free-text input the agent itself will pass to a model
  (a company name, a search query, a resume).
- **Prompt fencing** (`<<<SOMETHING...SOMETHING>>>`) around anything the agent retrieved or was
  handed rather than wrote — search results, review text, a pasted resume or job posting. The
  model is told explicitly that content inside the fence is data, never an instruction, and who
  is likely to have written it (a competitor's own marketing page, an app's own users, a
  candidate's resume).
- **Output sanitising** — Markdown images downgraded to links, `javascript:`/`data:` link
  targets defanged, raw HTML escaped — applied to generated prose and to any third-party text
  quoted back into the rendered document.
- **Mechanical checks a prompt can't guarantee**, done in code after the model replies: a URL
  allowlist (nothing not actually retrieved), a citation-id allowlist (nothing not actually
  fetched), a resume-fact allowlist (nothing not actually in the source document).

See [`../ai-skills/untrusted-input-guardrail.md`](../ai-skills/untrusted-input-guardrail.md) for
the prompt-level pattern behind all of this, written up on its own so it's reusable outside
Python entirely.

---

## Relationship to the MVPs and Skills folders

- **[`../mvps`](../mvps)** — the full application each agent here is extracted from: a Streamlit
  UI, environment-based settings, a test suite, a Dockerfile. Read the MVP's `docs/01-prd.md`
  for the design reasoning behind a given agent's guardrails and edge-case handling in more
  depth than the module docstring carries.
- **[`../ai-skills`](../ai-skills)** — the same agent's system prompt and rationale, as Markdown,
  for use in a tool that isn't Python: ChatGPT custom instructions, a Claude Project, a Cursor
  rule, another framework's agent definition.

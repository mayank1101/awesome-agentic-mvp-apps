# AI Travel Agent

A search-grounded trip planner: give it a destination and a trip length, and it searches the web for current activities, places to stay, and practical tips, then synthesises a day-by-day itinerary — every named place traceable back to a real search result, never invented from the model's own memory.

---

## Core Architecture & Features

### 1. Search, Then Synthesise — Never Just Recall
The model is never asked to plan a trip from memory. Every run searches the open web for the destination's activities, accommodation areas, and practical tips first ([`search.py`](app/services/search.py)); the single synthesis call is instructed that every specific named attraction, restaurant, or neighbourhood it writes must trace to a labelled evidence item, while ordinary trip-planning judgement — pacing, sequencing, general advice not tied to a place — is expected and welcome. A thin day beats a fabricated one.

### 2. The Model Never Sees a URL
Search evidence reaches the prompt labelled only by a small integer id. The sources list shown to the user is built by the app itself from the real search hits afterward, grouped by category — a model that never sees a link cannot reproduce, mistype, or invent one.

### 3. Trip-Length-Aware Search
Query count scales with how many days are being planned: a weekend trip and a two-week trip need different amounts of raw material, so longer trips add queries for hidden gems and day trips rather than asking three fixed searches to stretch across fourteen days.

### 4. Input Hardening & Statelessness
The destination and interests fields are scanned for prompt-injection phrasing before any search or model call is spent; fetched web content is fenced and labelled as untrusted data rather than blocked on, since refusing to plan a trip over a stray sentence in someone's travel blog would be the worse failure. Nothing about a trip persists past the browser session.

---

## Tech Stack

* **Inference Engine:** Groq SDK using an instruct model (`llama-3.3-70b-versatile` by default) for the single itinerary-synthesis call.
* **Web Search:** Tavily API accessed directly over `httpx` for open-web search — no domain whitelist, since travel guidance is spread across sources a fixed list cannot capture.
* **Data Layer & Schema Enforcement:** Pydantic v2 for the trip request, packed search evidence, and the model's structured reply.
* **Frontend UI:** Streamlit.

---

## Getting Started

### Prerequisites & Local Setup
```bash
cd mvps/ai-travel-agent
cp .env.example .env
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Docker Execution
```bash
docker build -t ai-travel-agent .
docker run --rm -p 8506:8506 --env-file .env ai-travel-agent
```

### Testing & Development
The test suite runs entirely offline: the search and model call sites are patched throughout.

```bash
pip install -r requirements-dev.txt
ruff check . && ruff format --check . && pytest
```

---

## Configuration Reference

Key variables defined in `.env` (see `.env.example` for the full, commented list):

| Variable | Description | Default |
| :--- | :--- | :--- |
| `GROQ_API_KEY` | Required API token for itinerary synthesis via Groq SDK. | None |
| `TAVILY_API_KEY` | Required API token for web search. | None |
| `MODEL_NAME` | Model id. Requires an **instruct** model for reliable, short JSON output. | `llama-3.3-70b-versatile` |
| `MAX_DAYS` | Longest trip this app will plan. | `14` |
| `MAX_QUERIES` | Ceiling on searches issued per trip, on top of the length-scaled query count. | `6` |
| `BLOCK_FLAGGED_INPUT` | Whether a heuristic prompt-injection match in the destination/interests fields stops the run. | `true` |

---

## Repository Architecture

```text
app/
├── core/         # Settings, exception hierarchy, secret-redacted logging
├── models/       # Pydantic schema boundaries: trip request, search evidence, generated itinerary
├── services/     # Tavily transport, query building & evidence packing, the LLM client, guardrails, orchestration
└── prompts.py    # The single synthesis prompt
ui/               # Streamlit view layer: session state, the input form, result rendering
tests/            # Offline verification suite
```

---

## Technical Limitations & Engineering Trade-offs

* **Search-engine visibility.** The itinerary can only be as good as what Tavily's index returns for a destination; a very small or newly-popular place may return thin evidence, in which case the model is instructed to lean on general guidance rather than invent named stops to fill the day.
* **No live availability or pricing.** Accommodation advice is about areas and property types worth considering, grounded in search evidence about the destination — not a live listings feed, and the app never states a specific price or opening hour unless the evidence itself states it.
* **No domain whitelist.** Unlike this repo's job-search app, search here is open-web rather than restricted to a fixed site list: travel guidance is too spread out across sources for a whitelist to capture well. The trade-off is a wider net with more variable source quality, mitigated by showing every source used rather than hiding the trail.
* **One synthesis call, one repair pass.** A reply that is invalid JSON, or whose day count does not match the requested trip length, gets exactly one retry with the error shown back to the model. A second failure ships the closest usable result with a visible notice rather than failing the run outright.

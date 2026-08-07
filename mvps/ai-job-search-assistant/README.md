# AI Job Search Assistant

A domain-restricted, LLM-driven job triage engine that assesses posting relevance against candidate resumes using semantic embeddings and algorithmic fact-checking guardrails.

AI Job Search Assistant shifts the candidate workflow from manual search to intelligent triage. Rather than relying on job board algorithms or unverified black-box AI recommendations, it extracts candidate profile data from PDF resumes, issues domain-whitelisted targeted search queries, de-duplicates results, and performs hierarchical two-tiered evaluation—culminating in verified, requirement-by-requirement coverage scoring.

Its sibling project, [`ai-job-match`](../ai-job-match), evaluates a single target job posting in depth and performs fact-grounded resume tailoring. AI Job Search Assistant constructs the highly relevant shortlist that precedes that tailoring stage.

---

## Core Architecture & Features

### 1. Two-Tiered Triage Pipeline
To deliver rapid responsiveness within token and latency constraints without compromising accuracy, evaluation is strictly decoupled into two tiers:
* **Shallow Semantic Ranking:** All raw search results (~40 postings) are ranked cheaply by comparing the candidate profile against job titles and search snippets using semantic similarity (or lexical token overlap as fallback).
* **Deep Requirement Scoring:** The top-ranked results (default `DEEP_SCORE_COUNT=8`) have their full posting content dynamically fetched. The inference engine extracts specific requirements and evaluates each as *Covered*, *Partial*, or *Missing* against explicit quotes from the candidate's resume.
* **Transparent Attribution:** Results clearly distinguish between shallowly ranked rows and deeply scored assessments to ensure users never confuse quick snippet estimation with comprehensive requirement verification.

### 2. Deterministic & Guarded Scoring (0–100)
The LLM never computes or returns numerical match scores directly. Instead, scores are derived via code arithmetic over verified requirement verdicts, weighted at **80% for mandatory (must-have) requirements** and **20% for preferred qualifications** (renormalizing when preferred sections are absent).

Before any *Covered* verdict contributes to the score, it must survive two algorithmic validation checks:
* **Provenance Verification:** Claimed evidence quotations must exist verbatim within the candidate's extracted resume text (with punctuation and kerning anomalies normalized). Unfounded quotations automatically demote a *Covered* verdict to *Partial*.
* **Semantic Similarity Backstop:** Unquoted *Covered* claims are verified against resume lines using `mistral-embed` cosine similarity. Claims falling below semantic alignment thresholds are automatically downgraded.

### 3. Domain Whitelisting & Input Hardening
* **Strict API Whitelisting:** Search operations are restricted at the network level to user-specified domains (e.g., specific ATS boards, verified job listings). Unapproved external domains are never fetched or queried.
* **URL Shape De-duplication:** Listing indexes, generic careers homepages, and cross-posted roles (e.g., identical postings across Greenhouse and LinkedIn) are heuristically pruned and deduplicated before deep processing.
* **Asymmetric Prompt Injection Protection:** Both resumes and fetched web postings are treated as untrusted input, strictly fenced inside structured prompt templates, and actively scanned for prompt injections and score-manipulation attacks. Flagged candidate resumes block pipeline execution; flagged web postings are gracefully suppressed and noted directly on the UI without disrupting the broader search run.
* **Stateless Execution:** Resumes and user configurations exist strictly within volatile memory during active session evaluation. Zero data persists across runs or storage layers.

---

## Tech Stack

* **Inference Engine:** Groq SDK leveraging high-throughput instruct models (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`) for profile extraction and structured requirement evaluation.
* **Web Search & Content Extraction:** Tavily API accessed via direct asynchronous HTTP calls (`httpx`) for whitelisted search retrieval and full posting parsing.
* **Semantic Embeddings:** `mistral-embed` for vector-based semantic ranking and requirement similarity backstops.
* **Data Layer & Validation:** Pydantic v2 and Pydantic Settings for strict schema enforcement, configuration state, and structured response parsing.
* **Document Ingestion:** `pypdf` for resilient PDF text parsing and anomaly normalization.
* **Frontend UI & Event Pipeline:** Streamlit Community Cloud architecture driven by an asynchronous event generator to render progressive run state directly on the main UI script thread.

---

## Getting Started

### Prerequisites & Local Setup
Ensure Python 3.11+ is installed in your environment (see `.python-version`).

```bash
# Clone and navigate to the service directory
cd mvps/ai-job-search-assistant

# Configure environment variables
cp .env.example .env

# Install production dependencies
pip install -r requirements.txt

# Start the application
streamlit run streamlit_app.py
```

### Docker Execution
Build and run the containerized application:

```bash
docker build -t ai-job-search-assistant .
docker run --rm -p 8504:8504 --env-file .env ai-job-search-assistant
```

### Testing & Development
The test suite operates entirely offline using patched API call sites and mock responses. `conftest.py` isolates test execution by disabling `.env` reads and scrubbing active environment credentials.

```bash
# Install development dependencies
pip install -r requirements.txt -r requirements-dev.txt

# Run static analysis, formatting checks, and test suite
ruff check . && ruff format --check . && pytest
```

---

## Configuration Reference

Key variables defined in `.env`:

| Variable | Description | Default / Recommended |
| :--- | :--- | :--- |
| `TAVILY_API_KEY` | Required API token for domain-restricted web searching and page content extraction. | None (Required) |
| `GROQ_API_KEY` | Required API token for candidate resume analysis and requirement evaluation. | None (Required) |
| `MISTRAL_API_KEY` | Optional API token for vector embeddings (`mistral-embed`); falls back to lexical overlap if unconfigured. | None |
| `MODEL_NAME` | Instruct model ID for extraction and evaluation. Must be an **instruct** model to ensure consistent JSON adherence without reasoning token overhead. | `llama-3.3-70b-versatile` |
| `DEEP_SCORE_COUNT` | Number of top-ranked search postings to fetch in full and evaluate requirement by requirement. | `8` |
| `MAX_QUERIES` | Maximum distinct search queries generated per session run. | `4` |
| `RUN_DEADLINE_SECONDS` | Total execution timeline cap. Runs exceeding this limit return partial results scored up to the deadline. | `300` |

---

## Repository Architecture

```text
app/
├── core/         # Configuration loading, exception hierarchies, and PII-redacted logging
├── models/       # Pydantic schema boundaries for jobs, search results, profiles, and state
├── services/     # Core domain engines (search, Tavily client, LLM/embeddings, PDF extraction, pipeline, ranking & scoring)
└── prompts.py    # Consolidated structured extraction and grading prompt definitions
ui/               # Streamlit application rendering, sidebar controls, and state management
tests/            # Offline verification suite with deterministic networking fixtures
```

---

## Technical Limitations & Engineering Trade-offs

* **Search Engine Visibility:** Relies entirely on search-engine indexation (via Tavily). Company career pages or job listings that are unindexed or behind authentication layers (e.g., login-walled LinkedIn jobs or closed listings) cannot be deeply evaluated; they fall back to snippet-based ranking with UI notices.
* **PDF Ingestion & OCR Bounds:** Designed strictly for pure Python containers (~1GB memory ceiling). Scanned image-based PDFs are explicitly rejected with explanatory messaging rather than processed through resource-heavy vision or OCR pipelines.
* **Token Budget Management:** Deep requirement scoring entails substantial input tokens per posting. To safeguard free-tier daily token budgets, the application enforces batching limits and short-circuits evaluation cleanly if token ceiling errors occur, returning existing scores alongside ranked remaining rows.
* **Posting Structure Dependence:** Evaluator accuracy is bounded by the clarity of the target job description; poorly structured or internally contradictory job listings yield correspondingly low extraction confidence and matching scores.

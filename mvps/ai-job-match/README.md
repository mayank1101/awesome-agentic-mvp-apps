# AI Job Match

An evidence-based resume tailoring and fit-analysis engine powered by LLM inference, semantic embeddings, and algorithmic fact-checking guardrails.

AI Job Match evaluates candidates against job postings without black-box scoring or hallucinated qualifications. It ingests resume PDFs and job descriptions to provide verifiable gap analysis, prioritized optimization checklists, and ATS-compliant resumes mechanically verified against the candidate's actual work history.

---

## Core Architecture & Features

### 1. Deterministic & Traceable Scoring (0–100)
Scores are computed algorithmically over verified per-requirement coverage rather than generated directly by an LLM. Every evaluation is backed by direct quotations from the resume.

| Dimension | Weight | Description |
| :--- | :--- | :--- |
| **Must-Have Requirements** | 55% | Direct fulfillment of mandatory qualifications stated in the posting. |
| **Preferred Requirements** | 20% | Coverage of optional or secondary competencies (renormalizes if absent). |
| **Evidence Strength** | 15% | Semantic alignment between quoted resume claims and target requirements. |
| **Keyword Coverage** | 10% | Presence of critical requirement terminology to gauge ATS discoverability. |

* **Semantic Backstop:** Requirement matches are evaluated via `mistral-embed` cosine similarity (or lexical overlap as fallback). Claims categorized as "covered" by the LLM without strong semantic grounding in the candidate's text are automatically downgraded by the evaluation engine.

### 2. Zero-Fabrication Resume Rewriting
The tailoring engine aligns existing career achievements with target posting terminology while passing through rigorous mechanical validation ([`provenance.py`](app/services/provenance.py)):
* **Numerical Fidelity:** Every metric, percentage, or quantitative claim in the rewritten output must exist in the source document.
* **Entity Verification:** All employers, technologies, tools, and certifications (e.g., `Kubernetes`, `PostgreSQL`, `AWS`) are validated against original text tokens.
* **Contact Preservation:** Headers, links, and contact addresses are locked to prevent corruption or replacement.
* **Automated Repair & Enforcement:** Violations trigger targeted LLM self-correction passes. Under strict mode, validation failures result in generation rejection rather than unverified output.

### 3. Security, Privacy & Input Hardening
* **Prompt Injection Protection:** Resumes and job descriptions are ingested as untrusted input and scanned for prompt injection vectors and score-manipulation attacks (e.g., hidden white-on-white instructions or meta-prompts).
* **Stateless Execution:** Resumes reside entirely within transient application memory for the duration of the web session. No documents or data persist to disk or databases.
* **Automated PII Redaction:** Logging hierarchies actively strip personally identifiable information (emails, phone numbers) from output traces and diagnostic logs.

---

## Tech Stack

* **Inference Engine:** Groq SDK utilizing high-throughput instruct models (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`).
* **Semantic Embeddings:** `mistral-embed` API for requirement matching and evidence verification.
* **Data Layer & Schema Enforcement:** Pydantic v2 for end-to-end type validation and structured LLM response parsing.
* **Validation Framework:** Guardrails AI configured with locally executed custom validators (zero runtime Hub network overhead).
* **Document Processing:** `pypdf` for resilient PDF text ingestion and `fpdf2` for structured ATS-parseable document generation.
* **Frontend UI:** Streamlit Community Cloud architecture.

---

## Getting Started

### Prerequisites & Local Setup
Ensure Python 3.10+ is installed on your environment.

```bash
# Clone and navigate to the service directory
cd mvps/ai-job-match

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
docker build -t ai-job-match .
docker run --rm -p 8503:8503 --env-file .env ai-job-match
```

### Testing & Development
The test suite operates entirely offline using mocked API calls, deterministic fixtures, and synthesized PDFs.

```bash
# Install development and formatting utilities
pip install -r requirements-dev.txt

# Run static analysis, formatting checks, and test suite
ruff check . && ruff format --check . && pytest
```

---

## Configuration Reference

Key variables defined in `.env`:

| Variable | Description | Default / Recommended |
| :--- | :--- | :--- |
| `GROQ_API_KEY` | Required API token for LLM inference via Groq SDK. | None |
| `MISTRAL_API_KEY` | Optional embedding API token for cosine similarity scoring; falls back to lexical token overlap if unconfigured. | None |
| `MODEL_NAME` | Model ID for LLM inference. Requires an **instruct** model to ensure reliable JSON structural compliance. | `llama-3.3-70b-versatile` |
| `STRICT_FABRICATION_GUARD` | Controls fallback behaviors for unverified rewrites. When `true`, refuses outputs failing factual audits. When `false`, renders output while visually flagging unsupported claims in the UI. | `true` |

---

## Repository Architecture

```text
app/
├── core/         # System configuration, custom exception hierarchies, and PII-redacted logging
├── models/       # Pydantic schema boundaries for inference, internal state, and parsing
├── services/     # Domain logic (PDF intake, LLM/embeddings, scoring, analyzer, provenance & Guardrails)
└── prompts.py    # Consolidated structured inference prompt definitions
ui/               # Streamlit view layer and layout components
tests/            # Offline verification suite (including Streamlit AppTest runtime integration)
```

---

## Technical Limitations & Engineering Trade-offs

* **PDF Ingestion:** Supports text-layer PDFs only; scanned or image-based files without an accessible text layer are rejected by design (no OCR pipeline). Complex multi-column layouts may experience extraction interleaving.
* **Rate-Limit Adaptation:** To operate within external API token ceilings (e.g., Groq rate limits), the inference service deploys fallback strategies: progressive requirement batching, prompt evidence pruning, and exponential backoff retries.
* **Scope of Factual Provenance:** Guardrails perform algorithmic validation on nouns, technical entities, numbers, and proper acronyms. Subjective linguistic framing (e.g., changing "assisted with" to "architected") or semantic combinations of valid disparate facts must be reviewed by the user via the built-in side-by-side diff viewer prior to application submission.

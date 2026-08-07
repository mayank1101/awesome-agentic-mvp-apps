# AI Competitor Analyzer

An autonomous market intelligence web application that profiles competitor companies across six structured analytical dimensions using public web sources. Built with **Streamlit**, **Pydantic v2**, and **Tavily Search**, it executes deterministic search workflows and single-pass schema-enforced LLM synthesis to produce auditable briefs in under a minute.

```
Company Snapshot · Product & Capabilities · Pricing & Packaging
Positioning & Target Customer · Recent Moves (12 Months) · Strengths & Weaknesses
```

---

## 🏗️ Key Architecture & Engineering Decisions

- **Deterministic Search Routing**: Searches are executed against six fixed query templates rather than relying on unconstrained LLM tool-calling. This guarantees deterministic credit usage, predictable latency (~40 seconds), and explicit section-to-source traceability.
- **Zero-Link Fabrication Design**: The model never receives URLs during synthesis; evidence is supplied strictly as referenced entity IDs and titles. The application rendering layer maps citations back to validated pre-synthesis search records, structurally preventing hallucinated URLs.
- **Strict Schema Enforcement**: Synthesis responses are constrained to JSON conforming to validated Pydantic domain models (`SynthesisResult`). Heading hierarchies, formatting, and graceful empty-state fallbacks are rendered post-synthesis by deterministic code.
- **Adversarial Input Defense**: Retrieved third-party web content is treated as untrusted input. All search summaries are length-capped, stripped of structural prompt spoofing tags, and enclosed within strict boundary fences to thwart indirect prompt injections.
- **Resilient Fallback Design**: If a search fails or yields zero results for a section, the application cleanly renders an explicit *"Not found in public sources"* fallback without aborting the broader intelligence report.
- **Entity Identity Resolution**: To defend against corporate name collisions (e.g., *"Apollo"*), an identity resolution stage validates domain consensus across multiple retrieved pages before launching downstream deep dives.

---

## 🚀 Quickstart & Execution

### Local Development Setup

```bash
cd mvps/ai-competitor-analyzer
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # Configure API keys (TAVILY_API_KEY, GROQ_API_KEY)
streamlit run streamlit_app.py --server.port 8501
```

### Docker Execution

```bash
docker build -t ai-competitor-analyzer .
docker run --rm -p 8501:8501 --env-file .env ai-competitor-analyzer
```

### Testing & Verification

```bash
# Run unit test suite and formatting validation
pytest && ruff check . && ruff format --check .
```

---

## ⚙️ Configuration & Multi-Provider Support

The application supports multi-provider model switching via environment variables or a local `.env` configuration file:

| Variable | Description | Default / Example |
| :--- | :--- | :--- |
| `MODEL_PROVIDER` | Backend Model Provider (`openrouter`, `openai`, `anthropic`, `gemini`, `ollama`, `foundry`, `groq`) | `groq` |
| `MODEL_NAME` | Specific target model deployment identifier | `llama-3.3-70b-versatile` |
| `TAVILY_API_KEY` | Authentication key for public web surface indexing | `tvly-...` |
| `GROQ_API_KEY` | Authentication key when utilizing Groq inference | `gsk_...` |
| `LOG_LEVEL` | Python logging verbosity threshold | `INFO` |

---

## 🎯 Operational Scope & System Boundaries

- **Stateless Snapshots**: Reports represent point-in-time public snapshots without historical persistent tracking, diffing, or automated background polling.
- **Single Target Focus**: Optimized for individual competitor analysis rather than broad industry matrix sweeps or internal multi-product benchmarking.
- **Section-Level Citation Grained**: References map at the thematic dimension level rather than per-sentence footnotes.
- **Public Data Horizon**: Excludes content gated behind credentials, paywalls, or non-indexed enterprise databases.

---

## 📂 Repository Layout

```
app/
  core/         # System configuration, secret redaction logging, exception hierarchy
  models/       # Strictly typed Pydantic models across all functional boundaries
  search/       # Tavily search client, deterministic templates, identity resolution
  agents/       # Multi-provider client registries and structured single-call prompts
  services/     # Input normalization, prompt guardrails, evidence packing, rendering
ui/             # Streamlit display components, session state wrappers, view templates
docs/           # Architectural design records and edge-case validation catalogs
```

**Dependency Flow**: `ui/` → `app/services/` → `app/agents/` → `app/models/` → `app/core/`. Application domain logic under `app/` maintains complete independence from the display rendering framework.

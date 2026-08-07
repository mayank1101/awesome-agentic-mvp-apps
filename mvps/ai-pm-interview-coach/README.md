# AI PM Interview Coach

An interactive, AI-powered mock interview system designed to evaluate product management candidates against a strict five-dimension rubric. Built on the **Microsoft Agent Framework** with a **Streamlit** user interface and **Pydantic v2** domain validation, the coach conducts adversarial follow-up probing grounded in candidate assertions before producing an auditable feedback report.

---

## 🏗️ System Architecture & Core Mechanics

- **Multi-Agent Decoupling**: Separation of concerns is enforced between two dedicated agent personas:
  - **The Interviewer**: Conducts the dialogue and challenges candidate assumptions. It is rebuilt per turn with strict stateless prompt constraints to eliminate conversation persona drift while historical state is preserved in an `AgentSession`.
  - **The Evaluator**: Operates independently with zero conversational session context to avoid positive confirmation bias. It receives the transcript as an immutable document for objective grading.
- **Rubric-Enforced Evaluation**: Evaluates answers across five discrete dimensions (*Structure, User Insight, Prioritization, Metrics Rigour, Communication*) using a rigid 4-point scale without midpoints (no comfortable "3/5" hedges). Scores are rejected by validation unless accompanied by verified textual quotes from the transcript.
- **Company Archetype Calibration**: Tailors evaluation benchmarks to specific operational environments (e.g., *Big Tech Scale, Growth-Stage Speed, B2B SaaS Deal Size, Marketplace Liquidity*), ensuring context-sensitive rigor.
- **Multi-Layer Guardrails**: Combines heuristic pattern scanning, input fencing (`<<<ANSWER...>>>`), and active HTML/link sanitization to prevent prompt injection and automated score manipulation attempts.
- **In-Memory Zero-Persistence Design**: All session states reside entirely within browser session scoped memory (`st.session_state`). No persistent database, caching service, or external storage daemon is required, guaranteeing absolute isolation across user runs.

---

## 🚀 Quickstart & Execution

### Local Development Setup

```bash
cd mvps/ai-pm-interview-coach
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # Configure target provider credentials
streamlit run streamlit_app.py --server.port 8502
```

### Docker Execution

```bash
docker build -t ai-pm-interview-coach .
docker run --rm -p 8502:8502 --env-file .env ai-pm-interview-coach
```

### Testing & Verification

```bash
# Execute local automated test suite (no network required) and formatting check
pytest && ruff check . && ruff format --check .

# Run optional integration verification against live LLM endpoints
pytest -m integration
```

---

## ⚙️ Configuration & Provider Support

The system supports seamless dynamic routing across six major model providers via environment configurations:

| `MODEL_PROVIDER` | Authentication Environment Variable | Supported Notes / Defaults |
| :--- | :--- | :--- |
| `openrouter` | `OPENROUTER_API_KEY` | Default inference router; compatible with leading instruct models |
| `openai` | `OPENAI_API_KEY` | Native OpenAI Chat Completions API |
| `anthropic` | `ANTHROPIC_API_KEY` | Uses native structured schema configuration |
| `gemini` | `GEMINI_API_KEY` | Google AI Studio interface via structured schema mappings |
| `ollama` | None (`OLLAMA_HOST` required) | Fully offline local deployment execution |
| `foundry` | None (`AZURE_AI_PROJECT_ENDPOINT` required) | Uses Azure DefaultAzureCredential authentication |

### Critical System Settings
- `INTERVIEWER_MAX_TOKENS` (Default: `512`): Controls response length for question emission. *Note: Increase to ~2048 when utilizing thinking/reasoning model architectures.*
- `REPORT_MAX_TOKENS` (Default: `4096`): Allocates sufficient output tokens for multi-dimension grading analysis.
- `MAX_ANSWER_CHARS` (Default: `4000`): Enforces context window budget protection against recursive dialog bloat.
- `GUARDRAILS_ENABLED` / `BLOCK_FLAGGED_INPUT` (Default: `true` / `true`): Dictates vulnerability defense policy on untrusted candidate input.

---

## 🎯 Operational Boundaries

- **Session Continuity**: Browser reload or process restart terminates the active mock interview cleanly by design due to strict zero-persistence architecture.
- **Single-Replica Deployment**: Cloud deployment across multiple container replicas requires sticky websocket sessions to maintain session continuity.
- **Language Coverage**: Evaluator rubric descriptions and generated reports default to English syntax, regardless of input language dialect.

---

## 📂 Repository Layout

```
streamlit_app.py            # Lifecycle coordination and configuration bootstrapping
app/
  core/       config.py     # Pydantic Settings parameter enforcement
              logging.py    # Idempotent logging configuration across execution reruns
              exceptions.py # Domain error definitions
  models/     schemas.py    # Immutable Pydantic models representing dialog state & reports
  agents/     client.py     # Multi-provider client factory and connection pooling
              rubric.py     # Five-dimension 20-level rubric definitions
              presets.py    # Archetype calibration and interview configurations
              prompts.py    # Instruction templates and fencing boundaries
              interview_agents.py # Synchronous and asynchronous interviewer/evaluator twins
  services/   async_bridge.py   # Daemon thread event loop bridge for asynchronous client calls
              transcript.py     # Message conversation list transformations
              guardrails.py     # Input scanning, boundary fencing, and sanitization
              markdown_renderer.py # Clean markdown transcript report compilation
ui/           state.py      # Streamlit session storage isolation wrapper
              sidebar.py    # Configuration inputs and parameter selection UI
              interview.py  # Dialog execution chat view
              report.py     # Final analytical score rendering and export
```

**Dependency Flow**: `ui/` → `app/services/` → `app/agents/` → `app/models/` → `app/core/`. Application logic operates independently of UI display frameworks.

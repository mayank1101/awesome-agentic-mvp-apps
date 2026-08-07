# AI PRD Generator Agent

An agentic application that transforms concise product ideas into structured, professional Product Requirement Documents (PRDs) formatted in clean Markdown. Built upon the **Microsoft Agent Framework (MAF)**, **Streamlit**, and **Pydantic v2**, the system orchestrates multi-stage outlining and sectional expansion with targeted token efficiency and multi-layered guardrails.

---

## 🏗️ System Architecture & Engineering Highlights

- **Schema-Enforced Pipeline**: All functional boundaries—user briefs (`PRDInput`), architectural outlines (`PRDOutline`), individual text blocks (`PRDSection`), and final assembled specifications (`PRDDocument`)—are validated through strict Pydantic schemas rather than unstructured dictionaries.
- **Dual PRD Operating Modes**:
  - **Product Mode**: Directs the agent to frame the idea as an independent standalone product, emphasizing complete overview architecture, customer personas, roadmap milestones, and comprehensive risk mitigation.
  - **Feature Mode**: Constrains generation to additive features within an existing primary platform (`parent_product`), strictly focusing on interface adjustments, existing behavior migration, rollout flags, and backwards-compatibility rather than unnecessary whole-product roadmaps.
- **Targeted Sectional Regeneration**: Allows selective regeneration of individual PRD sections without re-evaluating completed sections. This design reduces API token consumption and mitigates provider rate limits during large document builds.
- **Asynchronous/Synchronous Bridge**: To reconcile Streamlit’s synchronous UI rendering loop with MAF's asynchronous HTTP client architecture, a dedicated daemon thread (`async_bridge.py`) maintains a single long-lived event loop. This guarantees thread safety and persistent connection pooling across interactive re-renders.
- **Defense-in-Depth Guardrails**: Enforces three independent layers of security against adversarial prompt manipulation:
  - **Input Scanning**: Pattern-matches untrusted text for instruction overrides and role markers before API execution.
  - **Prompt Fencing**: Encloses user context within immutable structural delimiters (`<<<USER_BRIEF...>>>`) to prevent role re-assignment.
  - **Output Sanitization**: Neutralizes active script targets and converts potential tracking pixel images into neutral plaintext links prior to file download or presentation.

---

## 🚀 Quickstart & Execution

### Local Development Setup

```bash
cd mvps/ai-prd-generator
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # Configure target AI provider credentials
streamlit run streamlit_app.py --server.port 8501
```

### Docker Execution

```bash
docker build -t ai-prd-generator .
docker run --rm -p 8501:8501 --env-file .env ai-prd-generator
```

### Testing & Verification

```bash
# Execute unit tests along with formatting and static lint analysis
pytest && ruff check . && ruff format --check .
```

---

## ⚙️ Configuration & Model Providers

The platform natively integrates with six backend AI runtime engines, controlled via simple environment variables or `.env` configuration files:

| `MODEL_PROVIDER` | Backend Client Layer | Required Credentials / Environment Variables |
| :--- | :--- | :--- |
| `openrouter` *(default)* | OpenAIChatCompletionClient @ Custom Base URL | `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL` |
| `openai` | OpenAIChatCompletionClient | `OPENAI_API_KEY` |
| `anthropic` | AnthropicClient | `ANTHROPIC_API_KEY` |
| `ollama` | OllamaChatClient | None (`OLLAMA_HOST` required, e.g., `http://localhost:11434`) |
| `gemini` | GeminiChatClient | `GEMINI_API_KEY` |
| `foundry` | FoundryChatClient | Azure DefaultAzureCredential (`AZURE_AI_PROJECT_ENDPOINT`) |

### Core Tuning Parameters
- `MODEL_NAME` (Default: `deepseek/deepseek-chat-v3.1:free`): Model identifier sent to provider endpoints.
- `STRUCTURED_OUTPUT_MODE` (Default: `json_object`): Translates JSON formatting requests across differing vendor implementations (`json_schema`, `json_object`, or native MIME type directives).
- `MAX_TOKENS` / `MODEL_TEMPERATURE` (Default: `4096` / `0.4`): Governs output generation capacity and creative sampling variance.

---

## 🎯 Sample Input Usage

To execute a rapid validation test, select **Feature Mode**, paste the following sample parameters into the configuration sidebar, and trigger document generation:

```yaml
Scope: Feature
Feature Name: Bulk Select and Delete
Parent Product: Web-based email client (~80k monthly users) featuring single-message action menus without multi-select support.
One-Liner: Allow users to select multiple emails simultaneously and apply bulk management actions in a single interaction.
Problem Statement: Clearing promotional newsletters currently requires individual mouse hovers per message, creating repetitive high-friction workflows reported as top complaint in support logs.
Goals:
  - Cut median time to clean 50 emails by 80%
  - Provide universal undo capabilities across all bulk actions
```

---

## 📂 Repository Layout

```
streamlit_app.py            # Primary router coordinating layout execution and initialization
app/
  core/         # Configuration loaders, structured exception types, and logging setup
  models/       # Strictly typed Pydantic models (PRDInput, PRDOutline, PRDSection, PRDDocument)
  agents/       # AI provider implementations, prompt structures, and generation agents
  services/     # Sync-to-async event loop bridge, markdown rendering, and security guardrails
ui/             # Streamlit rendering logic (state isolation, parameter sidebar, view layout)
tests/          # Comprehensive test suite covering guardrails, schemas, and provider routing
```

**Dependency Flow**: `ui/` → `app/services/` → `app/agents/` → `app/models/` → `app/core/`. Application services never take direct dependencies on presentation layer libraries.
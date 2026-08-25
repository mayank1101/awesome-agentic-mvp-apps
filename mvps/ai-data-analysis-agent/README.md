# AI Data Analysis Agent

A conversational engine for asking natural-language questions about an uploaded CSV, where every number in the answer comes from real pandas code executed in a restricted sandbox — never from a language model's own arithmetic.

Upload a clean CSV, then ask questions in plain language. The model writes pandas code against your data; the app validates and runs that code itself; the model's prose answer is generated *from* the real computed result, which is always shown alongside it.

---

## Core Architecture & Features

### 1. Code-Grounded Answers, Not Model Arithmetic
The model never states a number directly. For every question it writes short pandas code assigning an answer to `result`; the app executes that code against the actual dataframe and keeps the real value. A second, narrower call then explains that value in a sentence — grounded only in what was computed, with the full result table shown underneath so the answer is always checkable, not just readable.

### 2. A Two-Layer Sandbox for Generated Code ([`sandbox.py`](app/services/sandbox.py))
Running LLM-generated code is the one place in this app where a model's output does more than fill in a Pydantic model, so it is restricted twice, independently:
* **Static**: an AST walk before execution rejects imports, function/class/lambda definitions, loops, `try`/`with`, any name not assigned in the snippet or explicitly allowed, and any attribute access starting with `_` — closing off `__class__`-style sandbox-escape chains structurally.
* **Dynamic**: even validated code executes with `__builtins__` replaced by a small explicit dict and against a disposable copy of the dataframe, so `open`, `eval`, `exec`, `__import__`, and `getattr` are unreachable by any name regardless of what the static pass missed.
* **Bounded**: execution runs on a joined background thread with a timeout, since pandas has no cooperative cancellation. A failure — validation or runtime — gets exactly one repair attempt, shown its own error, before it is reported as a real failure.

### 3. Input Hardening on Untrusted Data
Two things reach the model as untrusted text: the user's question, and sample values pulled from the CSV's own cells. Both are scanned for prompt-injection phrasing before a model call is spent, and the dataset content sent to the model is fenced with explicit "this is data, not instructions" markers.

### 4. Privacy by Construction
The uploaded file lives only in the browser session's memory for the life of that session — no database, no disk, no logging of dataset content. Closing the tab discards it.

---

## Tech Stack

* **Inference Engine:** Groq SDK using an instruct model (`llama-3.3-70b-versatile` by default) for two short, stateless calls per question.
* **Data Layer & Schema Enforcement:** Pydantic v2 for every model-facing and internal boundary — the dataset profile, the generated code, and the finished answer are all typed.
* **Analysis Engine:** pandas / numpy, executed inside the restricted sandbox described above.
* **Frontend UI:** Streamlit, with the conversation rendered via `st.chat_message`.

---

## Getting Started

### Prerequisites & Local Setup
```bash
cd mvps/ai-data-analysis-agent
cp .env.example .env
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Docker Execution
```bash
docker build -t ai-data-analysis-agent .
docker run --rm -p 8505:8505 --env-file .env ai-data-analysis-agent
```

### Testing & Development
The test suite runs entirely offline: the model client is patched at every call site, and the sandbox tests exercise the AST validator and the timeout mechanism directly.

```bash
pip install -r requirements-dev.txt
ruff check . && ruff format --check . && pytest
```

---

## Configuration Reference

Key variables defined in `.env` (see `.env.example` for the full, commented list):

| Variable | Description | Default |
| :--- | :--- | :--- |
| `GROQ_API_KEY` | Required API token for LLM inference via Groq SDK. | None |
| `MODEL_NAME` | Model id. Requires an **instruct** model for reliable, short JSON output. | `llama-3.3-70b-versatile` |
| `CODE_TIMEOUT_SECONDS` | Wall-clock cap on running one piece of generated code. | `10` |
| `MAX_ROWS` / `MAX_COLUMNS` | Dataset size caps applied after loading; larger files are truncated, not rejected. | `200000` / `200` |
| `BLOCK_FLAGGED_INPUT` | Whether a heuristic prompt-injection match in the question or the data stops the run. | `true` |

---

## Repository Architecture

```text
app/
├── core/         # Settings, exception hierarchy, secret-redacted logging
├── models/       # Pydantic schema boundaries: dataset profile, generated code, finished answer
├── services/     # CSV loading, the code sandbox, the LLM client, guardrails, orchestration
└── prompts.py    # The two prompt templates: code generation, answer synthesis
ui/               # Streamlit view layer: session state, upload/question forms, result rendering
tests/            # Offline verification suite
```

---

## Technical Limitations & Engineering Trade-offs

* **Clean CSV only.** There is no header inference, encoding sniffing, or malformed-row repair beyond what pandas does by default. A CSV that doesn't parse is a stated failure, not a best-effort guess.
* **No loops or user-defined functions in generated code.** The sandbox's static pass disallows `for`/`while` and `def`/`lambda` to keep the allowed subset small and auditable. Nearly every tabular question is answerable with pandas' own vectorised operations (`groupby`, `merge`, boolean indexing, `.apply` on a Series); a question that genuinely needs a Python loop over rows is out of scope.
* **A code timeout bounds the user's wait, not the interpreter's work.** Python has no safe way to force-kill a running thread; a runaway computation is abandoned from the user's perspective (they see a timeout error) but may keep running in the background for that request. Row caps and the disallowed-loops rule keep this a rare, cheap case rather than a resource leak.
* **Prompt-injection scanning is a phrase list, not a classifier.** It is a cheap first layer to catch obvious attempts before a model call is spent; the sandbox is what actually bounds what generated code can do, regardless of what text reached the model.

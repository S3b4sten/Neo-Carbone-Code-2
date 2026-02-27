# Self-Evolving Agent

A Python agent that accomplishes **any task regardless of initial resources** by synthesizing new tools on-the-fly, installing packages, searching the web, running code, and learning from past experience.

## Quick Start

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your OpenAI API key
$env:OPENAI_API_KEY = "sk-..."

# 3. Run a one-shot task
python main.py "Calculate the 20th Fibonacci number and save it to workspace/fib.txt"

# 4. Run in interactive mode
python main.py --interactive

# 5. Use a different model
python main.py --model gpt-4o-mini "Summarize the latest AI news"
```

## Architecture

```
main.py                    ← CLI entry point
config.py                  ← Configuration + env vars
agent/
  core.py                  ← Main think→act→observe→evolve loop
  planner.py               ← Goal decomposition & replanning
  memory.py                ← Short-term (context) + long-term (JSON) memory
  executor.py              ← Tool dispatcher
  evolver.py               ← Lesson extraction + tool synthesis
tools/
  __init__.py              ← Tool registry
  base.py                  ← BaseTool interface
  synthesizer.py           ← LLM-driven tool creation
  builtins/
    shell.py               ← Run shell commands
    web_search.py          ← DuckDuckGo search
    file_io.py             ← Read / write / list files
    python_eval.py         ← Execute Python snippets
    http_request.py        ← HTTP GET/POST
    pip_install.py         ← Install packages at runtime
  synthesized/             ← Auto-generated tools (persisted across runs)
workspace/                 ← Agent's working directory
memory/
  long_term.json           ← Persisted task history & lessons
```

## Configuration (env vars)

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI key |
| `OPENAI_BASE_URL` | `None` | Custom API endpoint (e.g. Ollama) |
| `AGENT_MODEL` | `gpt-4o` | Model name |
| `MAX_ITERATIONS` | `40` | Max tool calls per sub-task |
| `ALLOW_SHELL` | `true` | Enable shell command execution |
| `ALLOW_WEB` | `true` | Enable web search & HTTP requests |
| `ALLOW_PIP_INSTALL` | `true` | Enable runtime pip install |
| `SANDBOX_PYTHON` | `false` | Enable RestrictedPython sandbox |

## How Self-Evolution Works

1. **Planner** decomposes your goal into ordered sub-tasks using the LLM.
2. **Core loop** iterates: calls the LLM → dispatches tool calls → feeds results back.
3. **Automatic evolution**: after `MAX_TOOL_RETRIES` consecutive errors on a sub-task, the **Evolver** asks the LLM whether a new tool is needed and, if so, synthesizes one.
4. **Explicit synthesis**: the agent can also call `synthesize_tool` directly if it determines no existing tool is suitable.
5. **Lesson extraction**: after every run, a lesson is distilled and saved to `memory/long_term.json`, seeding the system prompt on future runs.
6. **Synthesized tools** are saved to `tools/synthesized/` and auto-loaded on the next run.

## Example: Custom Tool Synthesis

If you ask "Fetch the current Bitcoin price", the agent will:
1. Try `http_request` with a public API.
2. If that fails, call `synthesize_tool` → the LLM writes a `BitcoinPriceTool`, saves it to `tools/synthesized/bitcoin_price.py`, and registers it instantly.
3. Use the new tool to complete the task.

"""
config.py – Central configuration for the Self-Evolving Agent.
All tuneable knobs live here; override via environment variables.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # loads .env from the project root if it exists

# ── LLM ────────────────────────────────────────────────────────────────────────
MODEL: str = os.environ.get("AGENT_MODEL", "gpt-4o-mini")
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL: str | None = os.environ.get("OPENAI_BASE_URL", None)  # e.g. Ollama
MAX_CONTEXT_TOKENS: int = int(os.environ.get("MAX_CONTEXT_TOKENS", 25_000))  # input budget
MAX_RESPONSE_TOKENS: int = int(os.environ.get("MAX_RESPONSE_TOKENS", 8_192))   # output budget — must be large enough for a full code file in JSON


# ── Agent loop ─────────────────────────────────────────────────────────────────
MAX_ITERATIONS: int = int(os.environ.get("MAX_ITERATIONS", 40))
MAX_TOOL_RETRIES: int = int(os.environ.get("MAX_TOOL_RETRIES", 3))
TOOL_TIMEOUT_SECONDS: int = int(os.environ.get("TOOL_TIMEOUT", 30))

# ── Feature flags ──────────────────────────────────────────────────────────────
ALLOW_SHELL: bool = os.environ.get("ALLOW_SHELL", "true").lower() == "true"
ALLOW_PIP_INSTALL: bool = os.environ.get("ALLOW_PIP_INSTALL", "true").lower() == "true"
SANDBOX_PYTHON: bool = os.environ.get("SANDBOX_PYTHON", "false").lower() == "true"
ALLOW_WEB: bool = os.environ.get("ALLOW_WEB", "true").lower() == "true"

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR: Path = Path(__file__).parent.resolve()
WORKSPACE_DIR: Path = ROOT_DIR / "workspace"
MEMORY_DIR: Path = ROOT_DIR / "memory"
SYNTHESIZED_TOOLS_DIR: Path = ROOT_DIR / "tools" / "synthesized"
SELF_PROMPT_FILE: Path = MEMORY_DIR / "self_prompt.md"
LONG_TERM_MEMORY_FILE: Path = MEMORY_DIR / "long_term.json"

# Create directories on import
for _d in (WORKSPACE_DIR, MEMORY_DIR, SYNTHESIZED_TOOLS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

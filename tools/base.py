"""tools/base.py – Abstract base class for all agent tools."""

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """Every tool the agent can call must subclass this."""

    # ── Required class-level attributes ────────────────────────────────────────
    name: str = ""
    description: str = ""

    # ── JSON schema for the tool's parameters (OpenAI function-calling format) ─
    parameters: dict = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    # ── Optional: set to True if tool should NOT be exposed to the LLM directly
    hidden: bool = False

    @abstractmethod
    def run(self, **kwargs: Any) -> str:
        """Execute the tool and return a string result."""

    # ── Helpers ────────────────────────────────────────────────────────────────
    def to_openai_schema(self) -> dict:
        """Return the OpenAI function-calling schema dict."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def __repr__(self) -> str:
        return f"<Tool name={self.name!r}>"

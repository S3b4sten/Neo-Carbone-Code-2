"""agent/executor.py – Dispatches tool calls from the LLM to the tool registry."""

from __future__ import annotations

import json
import traceback
from typing import Any

import tools as tool_registry


class Executor:
    """Translates OpenAI tool-call objects into tool results."""

    def run(self, tool_name: str, raw_args: str | dict) -> str:
        # Parse args
        if isinstance(raw_args, str):
            try:
                kwargs: dict[str, Any] = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError as exc:
                # The most common cause is the LLM's response being cut off
                # mid-string because MAX_RESPONSE_TOKENS was too low.
                is_truncation = "Unterminated string" in str(exc) or "Expecting" in str(exc)
                if is_truncation:
                    return (
                        "ERROR: Your response was truncated before the tool call completed. "
                        "The content was too long to fit in a single response. "
                        "Try writing the file in smaller sections using write_section, "
                        "or cut unnecessary comments and blank lines."
                    )
                return f"ERROR: Could not parse tool arguments as JSON: {exc}\nArguments received: {raw_args!r}"
        else:
            kwargs = raw_args or {}

        # Lookup tool
        try:
            tool = tool_registry.get_tool(tool_name)
        except KeyError:
            available = ", ".join(t.name for t in tool_registry.list_tools())
            return f"ERROR: Unknown tool '{tool_name}'. Available tools: {available}"

        # Execute and return the result as-is — guidance on failures is added
        # as a separate user message by the agent loop, not embedded here.
        try:
            return str(tool.run(**kwargs))
        except TypeError as exc:
            return f"ERROR: Wrong arguments for tool '{tool_name}': {exc}"
        except Exception:
            return f"ERROR executing '{tool_name}':\n{traceback.format_exc()}"

"""agent/executor.py – Dispatches tool calls from the LLM to the tool registry.

ProblemComparer
---------------
Tracks consecutive identical failures from workspace_code_reviewer.

WHY this is needed:
  When an LLM generates broken code, that broken code enters the conversation
  history. On the next attempt the LLM *sees its own broken attempt* and tends
  to reproduce it — creating a self-reinforcing loop. Simply telling it to
  "fix the error" isn't enough because the broken code is the dominant pattern.

HOW it works:
  1. Compares every reviewer FAIL result to the previous one.
  2. After 2 identical problems  → injects a "different approach" hint.
  3. After 3 identical problems  → calls short_mem.purge_failed_writes() to
     physically remove all broken write_file attempts from the conversation,
     then issues an EMERGENCY REWRITE with a structural template so the LLM
     starts from a verified-correct skeleton, not from its own broken history.
"""

from __future__ import annotations

import json
import traceback
from typing import Any

import tools as tool_registry


class ProblemComparer:
    """Detects repeated identical failures and escalates the correction strategy."""

    # After this many consecutive identical FAILs, purge memory + emergency rewrite
    EMERGENCY_THRESHOLD = 3
    # After this many, inject a gentle "try differently" hint
    HINT_THRESHOLD = 2

    def __init__(self, short_mem=None):
        self._short_mem = short_mem   # ShortTermMemory reference set by Executor
        self._last_fail_key: str = ""
        self._fail_count: int = 0

    def set_memory(self, short_mem) -> None:
        """Inject short-term memory reference after construction."""
        self._short_mem = short_mem

    def reset(self) -> None:
        self._last_fail_key = ""
        self._fail_count = 0

    def observe(self, tool_name: str, result: str) -> str:
        """
        Observe a tool result. Returns either the original result unchanged,
        or a modified result with a correction hint prepended.

        The LLM sees the returned string — so injecting guidance here means
        it arrives BEFORE the LLM's next turn, not after.
        """
        if tool_name != "workspace_code_reviewer":
            return result   # Only track reviewer results

        is_fail = "VERDICT: FAIL" in result

        if not is_fail:
            # Success — reset tracking
            self.reset()
            return result

        # Extract a stable fingerprint: just the SyntaxError or MISSING lines
        # (not the full report, which includes line numbers that change)
        key = self._fingerprint(result)

        if key == self._last_fail_key:
            self._fail_count += 1
        else:
            # Different error — reset but still count this as fail #1
            self._last_fail_key = key
            self._fail_count = 1

        if self._fail_count >= self.EMERGENCY_THRESHOLD:
            correction = self._emergency_correction(result)
            self.reset()   # give LLM fresh attempts after purge
            return correction + "\n\n" + result

        if self._fail_count >= self.HINT_THRESHOLD:
            return self._hint_correction(self._fail_count) + "\n\n" + result

        return result

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _fingerprint(self, result: str) -> str:
        """Extract the stable error type from a FAIL report."""
        # Grab just the Syntax/MISSING line — first 120 chars of the error
        for line in result.splitlines():
            if "SyntaxError" in line or "MISSING" in line or "VERDICT: FAIL" in line:
                return line.strip()[:120]
        return result[:120]

    def _hint_correction(self, count: int) -> str:
        return (
            f"\n⚠️  SAME PROBLEM DETECTED ({count} times in a row)\n"
            "Your current approach is not fixing the issue.\n"
            "Before writing again, THINK about a completely different solution:\n"
            "  • If it's an indentation error: restructure the code blocks, "
            "don't just re-indent the same code.\n"
            "  • If a module is missing: remove the import and define constants inline.\n"
            "  • If logic is missing: add it explicitly, don't assume it's there.\n"
        )

    def _emergency_correction(self, result: str) -> str:
        # Purge the contaminated history so the LLM gets a clean slate
        purged = 0
        if self._short_mem is not None:
            purged = self._short_mem.purge_failed_writes()

        sep = "=" * 60
        lines = [
            "",
            sep,
            "🚨 EMERGENCY REWRITE REQUIRED 🚨",
            sep,
            f"The same error has appeared 3+ times. The broken code has been",
            f"PURGED from conversation history ({purged} contaminated messages removed).",
            "You now have a CLEAN SLATE. Follow these exact steps:",
            "",
            "STEP 1 — Do NOT reference any previous version of this file.",
            "STEP 2 — Write the ENTIRE file from scratch in one write_file call.",
            "STEP 3 — Use ONLY stdlib + pygame. No imports from other project files.",
            "STEP 4 — Follow this structure EXACTLY for a pygame game:",
            "```",
            "import pygame, random",
            "pygame.init()",
            "CONSTANT = value   # all constants at the top, no imports",
            "",
            "class MyClass:",
            "    def update(self): ...",
            "    def draw(self): ...",
            "",
            "clock = pygame.time.Clock()",
            "running = True",
            "while running:                  # ← INDENT everything below",
            "    for event in pygame.event.get(): ...",
            "    obj.update()               # ← 4 spaces inside while",
            "    screen.fill(color)         # ← 4 spaces inside while",
            "    obj.draw()                 # ← 4 spaces inside while",
            "    pygame.display.flip()      # ← 4 spaces inside while",
            "    clock.tick(FPS)            # ← 4 spaces inside while",
            "pygame.quit()                  # ← 0 spaces, OUTSIDE while",
            "```",
            "STEP 5 — Every line inside the while loop must have exactly 4 spaces.",
            sep,
        ]
        return "\n".join(lines)


class Executor:
    """Translates OpenAI tool-call objects into tool results."""

    def __init__(self):
        self.problem_comparer = ProblemComparer()
        # short_mem is injected after construction by the agent

    def set_memory(self, short_mem) -> None:
        """Called by the agent to give the comparer access to conversation history."""
        self.problem_comparer.set_memory(short_mem)

    def run(self, tool_name: str, raw_args: str | dict) -> str:
        # Parse args
        if isinstance(raw_args, str):
            try:
                kwargs: dict[str, Any] = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError as exc:
                # Diagnose the most common cause: the LLM's response was cut off
                # mid-string because MAX_RESPONSE_TOKENS was too low. The content
                # string gets truncated, leaving the JSON unterminated.
                is_truncation = "Unterminated string" in str(exc) or "Expecting" in str(exc)
                if is_truncation:
                    return (
                        "ERROR: Your response was TRUNCATED before the tool call completed.\n"
                        "The file content you tried to write was too long to fit in a single response.\n\n"
                        "SOLUTION — write the file in fewer lines:\n"
                        "  1. Cut any long comments — keep only essential ones.\n"
                        "  2. Remove blank lines inside functions.\n"
                        "  3. If the file must be long, write it in two parts:\n"
                        "     - First call: write the classes (Bird, Pipe, etc.)\n"
                        "     - Second call: write the main() / game loop\n"
                        "  4. DO NOT try to escape characters — that is NOT the issue.\n"
                        "     The content was simply too long and got cut off.\n"
                    )
                return f"ERROR: Could not parse tool arguments as JSON: {exc}\nArguments received: {raw_args!r}"

        else:
            kwargs = raw_args or {}

        # Lookup tool
        try:
            tool = tool_registry.get_tool(tool_name)
        except KeyError:
            available = ", ".join(t.name for t in tool_registry.list_tools())
            return (
                f"ERROR: Unknown tool '{tool_name}'. "
                f"Available tools: {available}"
            )

        # Execute
        try:
            result = str(tool.run(**kwargs))
        except TypeError as exc:
            return f"ERROR: Wrong arguments for tool '{tool_name}': {exc}"
        except Exception:
            return f"ERROR executing '{tool_name}':\n{traceback.format_exc()}"

        # Pass through the problem comparer — may prepend escalating guidance
        return self.problem_comparer.observe(tool_name, result)

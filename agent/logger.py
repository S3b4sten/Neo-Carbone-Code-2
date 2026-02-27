"""agent/logger.py – Human-readable per-run log writer.

Creates a timestamped .txt file in logs/ for every agent run.
Designed so an outside reader can follow exactly what the agent did,
step by step, without reading source code.
"""

from __future__ import annotations

import textwrap
from datetime import datetime
from pathlib import Path


LOGS_DIR = Path(__file__).parent.parent / "logs"


class RunLogger:
    """Writes a structured, human-readable log of one agent run."""

    # Width used for separator lines
    _WIDTH = 80

    def __init__(self, goal: str):
        LOGS_DIR.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitise goal for use in filename
        safe_goal = "".join(c if c.isalnum() or c in " _-" else "" for c in goal)[:40].strip()
        safe_goal = safe_goal.replace(" ", "_") or "run"
        self._path = LOGS_DIR / f"{timestamp}_{safe_goal}.txt"
        self._file = self._path.open("w", encoding="utf-8")
        self._subtask_idx = 0
        self._tool_idx = 0
        self._start = datetime.now()

        self._write_header(goal)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _write(self, text: str) -> None:
        self._file.write(text + "\n")
        self._file.flush()

    def _sep(self, char: str = "─") -> None:
        self._write(char * self._WIDTH)

    def _write_header(self, goal: str) -> None:
        self._sep("═")
        self._write(f"  SELF-EVOLVING AGENT  ·  RUN LOG")
        self._write(f"  Started : {self._start.strftime('%Y-%m-%d %H:%M:%S')}")
        self._write(f"  Goal    : {goal}")
        self._sep("═")
        self._write("")

    # ── Public API (called from core.py) ──────────────────────────────────────

    def log_plan(self, sub_tasks: list[str]) -> None:
        """Log the full plan produced by the planner."""
        self._write("📋  PLAN")
        self._sep()
        for i, task in enumerate(sub_tasks, 1):
            self._write(f"  {i:>2}. {task}")
        self._write("")

    def log_subtask_start(self, idx: int, total: int, description: str) -> None:
        """Log the start of a new sub-task."""
        self._subtask_idx = idx
        self._tool_idx = 0
        self._write("")
        self._sep("─")
        self._write(f"  🔧  SUB-TASK {idx}/{total}")
        self._write(f"  {description}")
        self._sep("─")

    def log_thought(self, text: str) -> None:
        """Log the model's reasoning text (if any)."""
        if not text:
            return
        wrapped = textwrap.fill(text.strip(), width=self._WIDTH - 4,
                                initial_indent="  💭  ", subsequent_indent="       ")
        self._write(wrapped)

    def log_tool_call(self, tool_name: str, args: str) -> None:
        """Log a tool invocation."""
        self._tool_idx += 1
        self._write(f"\n  [{self._subtask_idx}.{self._tool_idx}] 🔨  TOOL: {tool_name}")
        # Pretty-print args, indented
        args_preview = args[:600] + ("…" if len(args) > 600 else "")
        for line in args_preview.splitlines():
            self._write(f"         {line}")

    def log_tool_result(self, result: str, is_error: bool = False) -> None:
        """Log the result of a tool call."""
        prefix = "  ❌  RESULT (ERROR): " if is_error else "  ✅  RESULT: "
        # Show up to 800 chars
        preview = result[:800] + ("…" if len(result) > 800 else "")
        # Indent subsequent lines
        lines = preview.splitlines()
        self._write(prefix + (lines[0] if lines else ""))
        for line in lines[1:]:
            self._write("         " + line)

    def log_evolution(self, kind: str, detail: str) -> None:
        """Log an evolution event (reflection, synthesis, consolidation)."""
        self._write(f"\n  🔁  EVOLUTION [{kind.upper()}]")
        wrapped = textwrap.fill(detail.strip(), width=self._WIDTH - 6,
                                initial_indent="     ", subsequent_indent="     ")
        self._write(wrapped)

    def log_subtask_done(self, result: str) -> None:
        """Log sub-task completion."""
        self._write(f"\n  ✔  SUB-TASK DONE: {result[:200]}")

    def log_outcome(self, summary: str, success: bool) -> None:
        """Log the final outcome of the entire run."""
        elapsed = datetime.now() - self._start
        self._write("")
        self._sep("═")
        icon = "✅" if success else "⚠️"
        self._write(f"  {icon}  FINAL OUTCOME")
        self._sep()
        wrapped = textwrap.fill(summary.strip(), width=self._WIDTH - 4,
                                initial_indent="  ", subsequent_indent="  ")
        self._write(wrapped)
        self._write("")
        self._write(f"  Elapsed : {elapsed.seconds // 60}m {elapsed.seconds % 60}s")
        self._write(f"  Log     : {self._path}")
        self._sep("═")

    def close(self) -> None:
        """Close the log file."""
        if not self._file.closed:
            self._file.close()

    def log_error(self, message: str) -> None:
        """Log an unexpected agent-level error."""
        self._write(f"\n  🔴  ERROR: {message[:400]}")

    @property
    def path(self) -> Path:
        return self._path

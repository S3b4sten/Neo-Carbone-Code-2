"""agent/memory.py – Short-term (rolling window) and long-term (JSON) memory."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import config


class ShortTermMemory:
    """Sliding-window message store for the LLM conversation."""

    def __init__(self, max_messages: int = 80):
        self.max_messages = max_messages
        self._messages: list[dict] = []

    def _trim(self) -> None:
        """Keep the system message and trim oldest non-system messages as atomic groups.

        When an assistant message with tool_calls is dropped, its corresponding
        tool-result messages are also dropped immediately so the list never
        contains orphaned tool messages.
        """
        if len(self._messages) <= self.max_messages:
            return

        system = [m for m in self._messages if m.get("role") == "system"]
        rest   = [m for m in self._messages if m.get("role") != "system"]
        target = self.max_messages - len(system)

        while len(rest) > target and rest:
            dropped = rest.pop(0)
            # If we removed an assistant+tool_calls message, cascade-delete its results
            if dropped.get("role") == "assistant" and dropped.get("tool_calls"):
                call_ids = {tc["id"] for tc in dropped["tool_calls"]}
                rest = [
                    m for m in rest
                    if not (m.get("role") == "tool" and m.get("tool_call_id") in call_ids)
                ]

        self._messages = system + rest

    def add(self, role: str, content: str | list) -> None:
        self._messages.append({"role": role, "content": content})
        self._trim()

    def add_raw(self, message: dict) -> None:
        """Append a pre-serialized message dict (skips wrapping in role/content)."""
        self._messages.append(message)
        self._trim()

    def add_tool_result(self, tool_call_id: str, name: str, content: str) -> None:
        self._messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": content,
        })

    def get_messages(self) -> list[dict]:
        return list(self._messages)

    def purge_failed_writes(self) -> int:
        """Remove broken write attempts and FAIL reviewer results from history.

        When the agent is stuck writing the same broken code (via write_file or
        write_section), that broken content accumulates in conversation history
        and the LLM keeps reproducing it. This method scrubs all write-related
        messages so the next attempt starts with a clean slate.

        Strategy:
        - Find every assistant message that contained a write_file or write_section
          tool call (the broken code lives in tool_calls[].function.arguments).
        - Collect all tool_call IDs from those messages for cascading deletion.
        - Drop those assistant messages, their tool results, and any reviewer FAILs.

        Returns the number of messages removed.
        """
        _WRITE_TOOLS = {"write_file", "write_section"}

        before = len(self._messages)

        # Pass 1: collect call IDs from assistant messages that made write calls
        purge_call_ids: set[str] = set()
        for msg in self._messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    if tc.get("function", {}).get("name") in _WRITE_TOOLS:
                        # Mark ALL call IDs in this assistant message so its
                        # tool results are also dropped (atomically)
                        for tc2 in msg["tool_calls"]:
                            purge_call_ids.add(tc2.get("id", ""))
                        break

        # Pass 2: rebuild keeping only clean messages
        keep: list[dict] = []
        for msg in self._messages:
            role    = msg.get("role", "")
            name    = msg.get("name", "")
            content = str(msg.get("content", ""))

            # Drop assistant messages whose tool calls included a write tool
            if role == "assistant" and msg.get("tool_calls"):
                call_ids = {tc.get("id", "") for tc in msg["tool_calls"]}
                if call_ids & purge_call_ids:
                    continue

            # Drop tool results from write tools (success/error feedback)
            if role == "tool" and name in _WRITE_TOOLS:
                continue

            # Drop any tool result whose parent assistant message was purged
            if role == "tool" and msg.get("tool_call_id") in purge_call_ids:
                continue

            # Drop reviewer FAIL results (they reference the broken code)
            if role == "tool" and name == "workspace_code_reviewer" and "VERDICT: FAIL" in content:
                continue

            keep.append(msg)

        removed = before - len(keep)
        self._messages = keep
        return removed

    def clear(self) -> None:
        self._messages = []


class LongTermMemory:
    """Persistent JSON store of task outcomes, synthesized tools, and lessons."""

    def __init__(self, path: Path | None = None):
        self.path = path or config.LONG_TERM_MEMORY_FILE
        self._data: dict = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"tasks": [], "lessons": [], "synthesized_tools": []}

    def _save(self) -> None:
        self.path.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Tasks ──────────────────────────────────────────────────────────────────
    def record_task(self, goal: str, outcome: str, success: bool) -> None:
        self._data["tasks"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "goal": goal,
            "outcome": outcome,
            "success": success,
        })
        self._save()

    # ── Lessons ────────────────────────────────────────────────────────────────
    def add_lesson(self, lesson: str) -> None:
        self._data["lessons"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "lesson": lesson,
        })
        self._save()

    def get_lessons(self, n: int = 10) -> list[str]:
        return [item["lesson"] for item in self._data["lessons"][-n:]]

    # ── Synthesized tools ──────────────────────────────────────────────────────
    def record_synthesized_tool(self, name: str, description: str, filename: str) -> None:
        self._data["synthesized_tools"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "name": name,
            "description": description,
            "filename": filename,
        })
        self._save()

    def get_synthesized_tool_summaries(self, n: int = 20) -> list[str]:
        return [
            f"{t['name']}: {t['description']}"
            for t in self._data["synthesized_tools"][-n:]
        ]

    # ── Recent tasks ───────────────────────────────────────────────────────────
    def get_recent_tasks(self, n: int = 5) -> list[dict]:
        return self._data["tasks"][-n:]

    def get_recent_task_history(self, n: int = 5) -> str:
        """Return a formatted string of recent task history for the system prompt."""
        tasks = self.get_recent_tasks(n)
        if not tasks:
            return ""
        lines = []
        for t in tasks:
            icon = "✅" if t.get("success") else "⚠️"
            date = t.get("timestamp", "")[:10]  # YYYY-MM-DD
            goal = t.get("goal", "?")[:120]
            outcome = t.get("outcome", "?")[:200]
            lines.append(f"{icon} [{date}] Goal: {goal}\n   Result: {outcome}")
        return "\n".join(lines)

    # ── Memory consolidation ───────────────────────────────────────────────────
    _CONSOLIDATE_PROMPT = """\
You are a memory consolidator for an AI agent. Compress the following raw
experience log into ONE concise paragraph (≤ 150 words) that captures:
- What kinds of tasks this agent has accomplished
- Key patterns, skills, and tools it has mastered
- Important pitfalls or lessons to remember

RAW LESSONS:
{lessons}

TASK HISTORY (recent):
{tasks}

EXISTING SUMMARY (if any):
{existing}

Write ONLY the consolidated paragraph, no preamble."""

    def needs_consolidation(self, every_n_lessons: int = 5) -> bool:
        """Return True when enough new lessons have arrived since last consolidation."""
        summary = self._data.get("consolidated_summary", {})
        last_consolidated_at = summary.get("consolidated_at_lesson_count", 0)
        current_count = len(self._data["lessons"])
        return (current_count - last_consolidated_at) >= every_n_lessons

    def consolidate(self, llm_client, model: str) -> str:
        """
        Use the LLM to distil ALL lessons + task history into one compact paragraph.
        Saves the result to long_term.json and returns it.
        """
        lessons_text = "\n".join(
            f"- {item['lesson']}" for item in self._data["lessons"]
        ) or "None yet."

        tasks = self.get_recent_tasks(n=10)
        tasks_text = "\n".join(
            f"- [{t.get('timestamp','')[:10]}] {'✅' if t.get('success') else '⚠️'} "
            f"{t.get('goal','?')[:80]}: {t.get('outcome','?')[:120]}"
            for t in tasks
        ) or "None yet."

        existing = self._data.get("consolidated_summary", {}).get("text", "")

        try:
            response = llm_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": self._CONSOLIDATE_PROMPT.format(
                    lessons=lessons_text,
                    tasks=tasks_text,
                    existing=existing,
                )}],
                temperature=0.3,
                max_tokens=250,
            )
            summary_text = response.choices[0].message.content.strip()
        except Exception as exc:
            summary_text = existing or "(consolidation failed)"

        self._data["consolidated_summary"] = {
            "text": summary_text,
            "consolidated_at_lesson_count": len(self._data["lessons"]),
            "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
        }
        self._save()
        return summary_text

    def get_prompt_context(self) -> str:
        """
        Return the best available memory context for the system prompt.

        Uses the consolidated summary when it exists and is reasonably fresh
        (< 10 lessons old); otherwise falls back to the raw lists.
        Always appends synthesized tool summaries.
        """
        summary = self._data.get("consolidated_summary", {})
        summary_text = summary.get("text", "")
        consolidated_at = summary.get("consolidated_at_lesson_count", 0)
        current_count = len(self._data["lessons"])
        is_fresh = summary_text and (current_count - consolidated_at) < 10

        parts: list[str] = []

        if is_fresh:
            parts.append(f"## Agent memory (consolidated)\n{summary_text}")
            # Still show the few new raw lessons added since consolidation
            new_lessons = [
                item["lesson"]
                for item in self._data["lessons"][consolidated_at:]
            ]
            if new_lessons:
                bullet = "\n".join(f"- {l}" for l in new_lessons)
                parts.append(f"## New lessons since last consolidation\n{bullet}")
        else:
            # Fallback: raw lessons + task history
            lessons = self.get_lessons(n=10)
            if lessons:
                bullet = "\n".join(f"- {l}" for l in lessons)
                parts.append(f"## Lessons learned\n{bullet}")
            task_history = self.get_recent_task_history(n=5)
            if task_history:
                parts.append(f"## Recent task history\n{task_history}")

        # Always include synthesized tools
        synth = self.get_synthesized_tool_summaries(n=20)
        if synth:
            bullet = "\n".join(f"- {t}" for t in synth)
            parts.append(f"## Synthesized tools available\n{bullet}")

        return "\n\n".join(parts)


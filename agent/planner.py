"""agent/planner.py – Goal decomposition and replanning."""

from __future__ import annotations

import json

import config


_DECOMPOSE_PROMPT = """\
You are a senior software engineer breaking down a coding task into a clear implementation plan.
Each sub-task must be concrete, actionable, and achievable with the available tools.

GOAL: {goal}

AVAILABLE TOOLS: shell, python_eval, write_file, read_file, list_dir, pip_install,
                 web_search, http_request, read_url, workspace_code_reviewer, synthesize_tool

PLANNING PRINCIPLES:
1. Think like Claude or Codex — plan a PROPER, COMPLETE implementation.
2. Use as many sub-tasks as needed. There is NO LIMIT. A complex game can have 8-12 steps.
3. Each sub-task should focus on ONE clearly defined concern (e.g. "Implement the game loop",
   "Add collision detection", "Add score display and game-over screen").
4. NEVER plan to run GUI programs (pygame, tkinter, etc.) — they block the terminal forever.
   Instead, plan a "Static review with workspace_code_reviewer" sub-task as the final check.
5. For non-GUI scripts (CLI tools, data processing etc.), plan a "Run and verify output" sub-task.
6. Install missing packages as the FIRST sub-task if needed.
7. NEVER plan to build a file by appending chunks — write the complete file each time.
8. For multi-file projects, plan one sub-task per file, then a review sub-task at the end.

EXAMPLE plan for "make a flappy bird game in pygame":
[
  "Install pygame with pip_install",
  "Write flappy_bird.py: initialize pygame, define constants (gravity, jump speed, screen size)",
  "Add the Bird class with gravity, jump, and draw methods",
  "Add the Pipe class with scrolling movement and random gap positions",
  "Implement the main game loop: event handling, physics update, collision detection, score display",
  "Add game-over screen with score and restart on key press",
  "Static review: call workspace_code_reviewer on flappy_bird.py to verify all required patterns are present"
]

CONTEXT (lessons learned and synthesized tools available):
{context}

Respond with ONLY a JSON array of sub-task strings. No extra text, no markdown.
"""


_REPLAN_PROMPT = """\
You are a senior software engineer replanning after a failure.
Keep completed work — only plan the remaining steps needed to reach the goal.

ORIGINAL GOAL: {goal}

COMPLETED STEPS:
{completed}

FAILURE REASON:
{failure_reason}

CONTEXT:
{context}

REPLANNING PRINCIPLES:
- If a file was partially written, plan to REWRITE it completely (not append).
- If a package is missing, add a pip_install step first.
- If running a GUI program timed out, replace that step with workspace_code_reviewer static review.
  NEVER try to run GUI programs (pygame, tkinter etc.).
- If self-critique keeps failing, the code is genuinely incomplete — plan to rewrite the file fully.
- Be specific about what needs to change so the agent doesn't repeat the same mistake.

Respond with ONLY a JSON array of remaining sub-task strings. No extra text.
"""


class Planner:
    """Decomposes goals into sub-task lists."""

    def __init__(self, llm_client, context_fn=None):
        self.llm = llm_client
        self._context_fn = context_fn or (lambda: "")

    def _call(self, prompt: str) -> list[str]:
        response = self.llm.chat.completions.create(
            model=config.MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise task planner. "
                        "Output ONLY a valid JSON array of strings, nothing else."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=1200,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if the model wrapped the JSON
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
            if "```" in raw:
                raw = raw[: raw.index("```")]

        parsed = json.loads(raw)

        if isinstance(parsed, list):
            return [str(s) for s in parsed]

        # Unwrap common object wrappers
        for key in ("tasks", "sub_tasks", "subtasks", "steps", "plan"):
            if key in parsed and isinstance(parsed[key], list):
                return [str(s) for s in parsed[key]]

        # Last resort: first list value found
        for v in parsed.values():
            if isinstance(v, list):
                return [str(s) for s in v]

        return [str(raw)]

    def decompose(self, goal: str) -> list[str]:
        """Return an ordered list of sub-tasks for the given goal."""
        context = self._context_fn()
        prompt = _DECOMPOSE_PROMPT.format(goal=goal, context=context or "None.")
        try:
            return self._call(prompt)
        except Exception:
            return [f"Accomplish the goal directly: {goal}"]

    def replan(self, goal: str, completed: list[str], failure_reason: str) -> list[str]:
        """Return a revised plan after a failure."""
        context = self._context_fn()
        completed_text = "\n".join(f"- {s}" for s in completed) or "None yet."
        prompt = _REPLAN_PROMPT.format(
            goal=goal,
            completed=completed_text,
            failure_reason=failure_reason,
            context=context or "None.",
        )
        try:
            return self._call(prompt)
        except Exception:
            return [f"Retry goal directly: {goal}"]

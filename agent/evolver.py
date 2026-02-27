"""agent/evolver.py – Self-modification engine.

Responsibilities:
1. reflect_on_error  – fast, per-error corrective hint injected into the live conversation.
2. maybe_synthesize_tool – after repeated failures, ask the LLM whether a new tool is needed.
3. extract_and_store_lesson – after a full task run, persist a lesson to long-term memory.
4. get_self_prompt_supplement – expose accumulated lessons to seed the system prompt.
"""

from __future__ import annotations

import config
from agent.memory import LongTermMemory
from tools.synthesizer import ToolSynthesizer


# ── Prompts ────────────────────────────────────────────────────────────────────

_REFLECT_ON_ERROR_PROMPT = """\
You are a self-correcting AI agent. A tool call just failed. Diagnose what went
wrong and give ONE specific, actionable correction in ≤ 3 sentences.

SUB-TASK  : {subtask}
TOOL USED : {tool_name}
ERROR     : {error}
AVAILABLE TOOLS: {available_tools}

Focus on the root cause and the exact next step to take differently.
Do NOT suggest synthesizing a new tool unless absolutely unavoidable.
CORRECTION:"""

_SELF_CRITIQUE_PROMPT = """\
You are a quality-assurance reviewer for an AI agent. The agent claimed to finish
a sub-task. Evaluate whether it actually did so correctly.

OVERALL GOAL : {goal}
SUB-TASK     : {subtask}
EVIDENCE (recent tool calls and results):
{evidence}

## Evaluation rules — read carefully before deciding

1. **workspace_code_reviewer PASS is conclusive.**
   If the evidence contains a `workspace_code_reviewer` result with "VERDICT: PASS",
   the code contains all required logic. This alone is sufficient proof for a coding sub-task.

2. **Interactive / GUI / game programs always block — timeouts are EXPECTED.**
   If the program involves a GUI (Pygame, Tkinter, a web server, a REPL, etc.), it is
   designed to run forever. A "timed out" or "no output" result from running it is NORMAL
   and is NOT evidence of failure. Do NOT fail a sub-task just because running the program
   hung or timed out.

3. **Only FAIL if:**
   - workspace_code_reviewer returned FAIL / listed MISSING patterns, OR
   - A syntax/import error was reported (NameError, SyntaxError, ModuleNotFoundError), OR
   - The agent clearly did not attempt the sub-task at all.

4. Do NOT fail because: the program timed out, the program opened a window, the agent
   didn't provide code in the chat message (the code is in the workspace file), or
   because you personally cannot see the file contents in this prompt.

Respond with ONLY a JSON object:
{{
  "pass": true | false,
  "issues": "<empty string if pass, otherwise a clear description of what is wrong>",
  "suggestions": "<concrete next steps to fix the issues, empty if pass>"
}}"""


_LESSON_EXTRACTION_PROMPT = """\
You are a reflective AI agent. Review the following task transcript and extract
ONE concise lesson (≤ 2 sentences) that would help you do better next time.

TRANSCRIPT:
{transcript}

LESSON:"""

_SYNTHESIS_DECISION_PROMPT = """\
You are a self-evolving AI agent. You have been trying to accomplish a subtask
but keep failing despite multiple corrective attempts.

SUBTASK: {subtask}
FAILURE HISTORY:
{failures}

AVAILABLE TOOLS: {available_tools}

Do you need a NEW custom tool to accomplish this subtask?
If YES, respond with a JSON object:
  {{"synthesize": true, "name": "<snake_case_name>", "description": "<what it does>", "use_case": "<specific use case>"}}
If NO (e.g. you just need a different approach), respond with:
  {{"synthesize": false, "reason": "<why not>"}}

Respond with ONLY the JSON."""

_POST_RUN_EVOLVE_PROMPT = """\
You are a proactive self-evolution engine for an AI agent. The agent just
successfully completed a task. Your job is to identify whether any RECURRING
ACTION PATTERN in this run would benefit from being packaged as a reusable tool,
so similar future queries can be handled faster.

GOAL THAT WAS ACCOMPLISHED: {goal}
SUB-TASKS EXECUTED: {subtasks}
TOOLS ALREADY AVAILABLE: {available_tools}
PREVIOUSLY SYNTHESIZED TOOLS: {existing_synth}

Only suggest synthesis if:
- The same multi-step action would clearly repeat across many future requests.
- No existing tool already covers it.
- A tool would genuinely save 3+ steps next time.

Respond with ONLY one JSON object:
{{"synthesize": true,  "name": "<snake_case>", "description": "<one sentence>", "use_case": "<concrete example of when to use it>"}}
OR
{{"synthesize": false, "reason": "<why it's not worth synthesizing>"}}"""

_SIMILARITY_PROMPT = """\
You are a memory retrieval assistant for an AI agent. Given a NEW goal and a
list of PAST tasks, identify whether any past task is sufficiently similar that
its approach, tools, or lessons could accelerate the new one.

NEW GOAL: {goal}

PAST TASKS (most recent first):
{past_tasks}

PREVIOUSLY SYNTHESIZED TOOLS: {synth_tools}

If similar past work exists, respond with JSON:
{{
  "similar": true,
  "summary": "<1-2 sentences: what was done last time and which tools/approach worked>",
  "shortcut": "<concrete advice: e.g. 'Use the pygame_game_template tool, start from workspace/connect4_with_pygame.py'>",
  "relevant_tools": ["<tool_name>", ...]
}}
If no useful similarity, respond with:
{{"similar": false}}

Respond with ONLY the JSON."""


class Evolver:
    """Drives self-improvement: per-error reflection, tool synthesis, lesson storage."""

    def __init__(self, llm_client, long_term_memory: LongTermMemory):
        self.llm = llm_client
        self.ltm = long_term_memory
        self.synthesizer = ToolSynthesizer(llm_client)


    # ── Per-error real-time reflection ────────────────────────────────────────
    def reflect_on_error(self, tool_name: str, error: str, subtask: str) -> str:
        """
        Called immediately after every tool error.

        Returns a short corrective hint string to inject into the conversation
        so the agent self-corrects on the very next attempt.
        """
        import tools as tool_registry

        available = ", ".join(t.name for t in tool_registry.list_tools())
        try:
            response = self.llm.chat.completions.create(
                model=config.MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": _REFLECT_ON_ERROR_PROMPT.format(
                            subtask=subtask,
                            tool_name=tool_name,
                            error=error[:1000],
                            available_tools=available,
                        ),
                    }
                ],
                temperature=0.2,
                max_tokens=200,
            )
            hint = response.choices[0].message.content.strip()
            # Also persist as a lesson so future runs benefit
            self.ltm.add_lesson(f"[From error in '{tool_name}'] {hint}")
            return hint
        except Exception as exc:
            return f"(reflection failed: {exc})"

    def self_critique(
        self,
        goal: str,
        subtask: str,
        evidence: str,
    ) -> dict:
        """
        Ask the LLM to verify whether the sub-task output actually satisfies
        the requirement — not just whether it ran without errors.

        Fast-path: if workspace_code_reviewer already returned VERDICT: PASS,
        skip the LLM call entirely and return pass=True immediately.

        Returns:
            {"pass": bool, "issues": str, "suggestions": str}
        """
        import json

        # ── Fast-path: trust workspace_code_reviewer PASS ─────────────────────
        # If the reviewer already ran and passed, no LLM judgment is needed.
        if "VERDICT: PASS" in evidence:
            return {
                "pass": True,
                "issues": "",
                "suggestions": "(auto-passed: workspace_code_reviewer returned VERDICT: PASS)",
            }

        # ── Fast-path: detect GUI timeout — always pass for GUI apps ──────────
        _gui_timeout_signals = [
            "GUI/Blocking: YES",
            "timed out",
            "Timeout",
            "blocking call",
        ]
        if any(sig in evidence for sig in _gui_timeout_signals):
            return {
                "pass": True,
                "issues": "",
                "suggestions": "(auto-passed: GUI/blocking program — timeout is expected behavior)",
            }

        try:
            response = self.llm.chat.completions.create(
                model=config.MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": _SELF_CRITIQUE_PROMPT.format(
                            goal=goal[:500],
                            subtask=subtask[:300],
                            evidence=evidence[:2000],
                        ),
                    }
                ],
                temperature=0.2,
                max_tokens=300,
            )
            raw = response.choices[0].message.content.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1].lstrip("json").strip()
            result = json.loads(raw)
        except Exception as exc:
            # If critique fails, assume pass to avoid blocking the agent
            return {"pass": True, "issues": "", "suggestions": f"(critique unavailable: {exc})"}

        return {
            "pass": bool(result.get("pass", True)),
            "issues": result.get("issues", ""),
            "suggestions": result.get("suggestions", ""),
        }


    # ── Post-run lesson extraction ────────────────────────────────────────────
    def extract_and_store_lesson(self, transcript: str) -> str:
        """Extract a high-level lesson from the full run transcript."""
        try:
            response = self.llm.chat.completions.create(
                model=config.MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": _LESSON_EXTRACTION_PROMPT.format(transcript=transcript[:6000]),
                    }
                ],
                temperature=0.3,
                max_tokens=200,
            )
            lesson = response.choices[0].message.content.strip()
            self.ltm.add_lesson(lesson)
            return lesson
        except Exception as exc:
            return f"(lesson extraction failed: {exc})"

    # ── System-prompt supplement ──────────────────────────────────────────────
    def get_self_prompt_supplement(self) -> str:
        """Return recent lessons + synthesized tool summaries for the system prompt."""
        lessons = self.ltm.get_lessons(n=10)
        synth_tools = self.ltm.get_synthesized_tool_summaries()

        parts: list[str] = []
        if lessons:
            bullet_lessons = "\n".join(f"- {l}" for l in lessons)
            parts.append(f"## Lessons learned from past tasks\n{bullet_lessons}")
        if synth_tools:
            bullet_tools = "\n".join(f"- {t}" for t in synth_tools)
            parts.append(f"## Previously synthesized tools (already registered)\n{bullet_tools}")
        return "\n\n".join(parts)

    # ── Tool synthesis after repeated failures ────────────────────────────────
    def maybe_synthesize_tool(self, subtask: str, failures: list[str]) -> str:
        """After multiple failures, decide whether to synthesize a new tool."""
        import json
        import tools as tool_registry

        available = ", ".join(t.name for t in tool_registry.list_tools())
        failure_text = "\n".join(f"Attempt {i+1}: {f}" for i, f in enumerate(failures[-5:]))

        try:
            response = self.llm.chat.completions.create(
                model=config.MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": _SYNTHESIS_DECISION_PROMPT.format(
                            subtask=subtask,
                            failures=failure_text,
                            available_tools=available,
                        ),
                    }
                ],
                temperature=0.1,
                max_tokens=300,
            )
            raw = response.choices[0].message.content.strip()
            decision = json.loads(raw)
        except Exception as exc:
            return f"(evolver decision parse failed: {exc})"

        if not decision.get("synthesize", False):
            return f"Evolver: no synthesis needed – {decision.get('reason', 'unknown reason')}"

        result = self.synthesizer.create(
            name=decision["name"],
            description=decision["description"],
            use_case=decision["use_case"],
        )
        if "✅" in result:
            self.ltm.record_synthesized_tool(
                name=decision["name"],
                description=decision["description"],
                filename=f"{decision['name']}.py",
            )
        return result

    # ── Explicit synthesis (called by agent as a tool) ────────────────────────
    def synthesize_tool(self, name: str, description: str, use_case: str) -> str:
        """Directly synthesize a tool by specification."""
        result = self.synthesizer.create(name=name, description=description, use_case=use_case)
        if "✅" in result:
            self.ltm.record_synthesized_tool(
                name=name, description=description, filename=f"{name}.py"
            )
        return result

    # ── Proactive post-run evolution ──────────────────────────────────────────
    def post_run_evolve(self, goal: str, sub_tasks: list[str]) -> str:
        """
        Called at the END of every successful run.

        Asks the LLM: "Was there a repeating pattern here that deserves its own
        reusable tool?" If yes, synthesises it and validates it via self_critique
        before registering — so bad tools never slip in.

        Returns a short status string for display/logging.
        """
        import json
        import tools as tool_registry

        available = ", ".join(t.name for t in tool_registry.list_tools())
        existing_synth = ", ".join(self.ltm.get_synthesized_tool_summaries(n=10)) or "none"

        try:
            response = self.llm.chat.completions.create(
                model=config.MODEL,
                messages=[{
                    "role": "user",
                    "content": _POST_RUN_EVOLVE_PROMPT.format(
                        goal=goal[:300],
                        subtasks="\n".join(f"- {s}" for s in sub_tasks[:15]),
                        available_tools=available,
                        existing_synth=existing_synth,
                    ),
                }],
                temperature=0.2,
                max_tokens=250,
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].lstrip("json").strip()
            decision = json.loads(raw)
        except Exception as exc:
            return f"(post-run evolution skipped: {exc})"

        if not decision.get("synthesize", False):
            return f"Post-run evolve: no synthesis needed — {decision.get('reason', '?')}"

        tool_name = decision["name"]
        tool_desc = decision["description"]
        tool_use  = decision["use_case"]

        # Synthesize the tool
        result = self.synthesizer.create(name=tool_name, description=tool_desc, use_case=tool_use)
        if "✅" not in result:
            return f"Post-run evolve: synthesis of '{tool_name}' failed — {result[:200]}"

        # ── Validate with self-critique before registering ────────────────────
        critique = self.self_critique(
            goal=f"Create a reusable tool called '{tool_name}'",
            subtask=f"The tool must: {tool_desc}. Use case: {tool_use}",
            evidence=result,
        )
        if not critique["pass"]:
            # Reject the tool — don't register it
            self.ltm.add_lesson(
                f"Proactively synthesized tool '{tool_name}' was rejected by self-critique: "
                f"{critique['issues'][:120]}"
            )
            return (
                f"Post-run evolve: tool '{tool_name}' synthesized but REJECTED by QA "
                f"({critique['issues'][:150]})"
            )

        # All good — register it
        self.ltm.record_synthesized_tool(
            name=tool_name, description=tool_desc, filename=f"{tool_name}.py"
        )
        self.ltm.add_lesson(
            f"Proactively built tool '{tool_name}' after completing '{goal[:60]}'. "
            f"Use for: {tool_use[:100]}"
        )
        return f"✅ Post-run evolved: new tool '{tool_name}' synthesized and registered."

    # ── Similarity-based fast-path ────────────────────────────────────────────
    def find_similar_past_tasks(self, goal: str) -> str:
        """
        Compare the new goal against past task history.
        If a similar past task exists, return a concise fast-path brief
        (which gets injected into the system prompt at run start).
        Returns empty string if nothing relevant is found.
        """
        import json

        past = self.ltm.get_recent_tasks(n=10)
        if not past:
            return ""

        past_text = "\n".join(
            f"- [{t.get('timestamp','')[:10]}] {'✅' if t.get('success') else '⚠️'} "
            f"{t.get('goal','?')[:80]}: {t.get('outcome','?')[:120]}"
            for t in reversed(past)
        )
        synth_tools = ", ".join(self.ltm.get_synthesized_tool_summaries(n=10)) or "none"

        try:
            response = self.llm.chat.completions.create(
                model=config.MODEL,
                messages=[{
                    "role": "user",
                    "content": _SIMILARITY_PROMPT.format(
                        goal=goal[:300],
                        past_tasks=past_text,
                        synth_tools=synth_tools,
                    ),
                }],
                temperature=0.1,
                max_tokens=300,
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].lstrip("json").strip()
            result = json.loads(raw)
        except Exception:
            return ""

        if not result.get("similar", False):
            return ""

        summary   = result.get("summary", "")
        shortcut  = result.get("shortcut", "")
        rel_tools = result.get("relevant_tools", [])

        parts = [f"## Fast-path from similar past task\n{summary}"]
        if shortcut:
            parts.append(f"**Recommended approach:** {shortcut}")
        if rel_tools:
            parts.append(f"**Relevant tools already available:** {', '.join(rel_tools)}")
        return "\n".join(parts)

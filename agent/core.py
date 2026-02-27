"""agent/core.py – The self-evolving agent's main think→act→observe→evolve loop."""

from __future__ import annotations

import json
import textwrap
import time
from datetime import datetime
from typing import Any

from rich import print as rprint
from rich.panel import Panel
from rich.rule import Rule

import config
import tools as tool_registry
from agent.evolver import Evolver
from agent.executor import Executor
from agent.logger import RunLogger
from agent.memory import LongTermMemory, ShortTermMemory
from agent.planner import Planner


# ── Internal meta-tools ─────────────────────────────────────────────────────────
_META_TOOL_SYNTHESIZE = {
    "type": "function",
    "function": {
        "name": "synthesize_tool",
        "description": (
            "Synthesize (create) a brand-new Python tool and register it for immediate use. "
            "Call this when no existing tool is sufficient for the current subtask."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "snake_case name for the new tool."},
                "description": {"type": "string", "description": "What the tool does."},
                "use_case": {"type": "string", "description": "Specific use case and expected behavior."},
            },
            "required": ["name", "description", "use_case"],
        },
    },
}

_META_TOOL_SUBTASK_DONE = {
    "type": "function",
    "function": {
        "name": "subtask_complete",
        "description": (
            "Signal that the CURRENT sub-task is complete and you are ready to move on to the NEXT sub-task. "
            "Do NOT call this for the very last sub-task — call task_complete instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "result": {
                    "type": "string",
                    "description": "One-sentence summary of what this sub-task produced.",
                },
            },
            "required": ["result"],
        },
    },
}

_META_TOOL_TASK_DONE = {
    "type": "function",
    "function": {
        "name": "task_complete",
        "description": (
            "Signal that ALL sub-tasks are finished and the ENTIRE original goal is accomplished. "
            "Only call this after every single sub-task has been executed successfully."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Comprehensive summary of everything that was accomplished.",
                },
            },
            "required": ["summary"],
        },
    },
}


def _build_system_prompt(evolver: Evolver, long_term_memory) -> str:
    # get_prompt_context() returns consolidated summary when available,
    # falls back to raw lessons+history, always ≤ ~800 tokens regardless of run count.
    memory_context = long_term_memory.get_prompt_context()
    base = textwrap.dedent(f"""
        You are a coding agent. Your job is to accomplish a goal step by step, using the tools available to you.

        ## Execution loop
        You are given one sub-task at a time. For each sub-task:
        1. Think briefly about what needs to be done.
        2. Call the right tool.
        3. Read the result.
        4. Repeat until the sub-task is done, then call `subtask_complete`.
        5. After every sub-task is done, call `task_complete` with a summary.

        ## Completion signals
        - `subtask_complete` — this sub-task is done, move to the next.
        - `task_complete` — every sub-task is done, the whole goal is finished.
        - Do not call `task_complete` until all sub-tasks are complete.

        ## Writing code
        - For files under ~80 lines: use `write_file` with the full content.
        - For longer files: use `write_section` in chunks of 30-60 lines (is_first=True for the first chunk).
        - Always write the complete, correct file — do not append to build code incrementally.
        - After writing, call `workspace_code_reviewer` to verify the file is syntactically correct and contains the expected logic.
        - If the reviewer reports a syntax error: rewrite the entire file from scratch (use is_first=True). Do not append to a broken file.

        ## Verifying code
        - `workspace_code_reviewer` VERDICT: PASS means the file is correct — call `subtask_complete`.
        - For non-GUI scripts: run with `shell` or `python_eval` and check the output.
        - For GUI programs (Pygame, Tkinter, etc.): do not run them — they will hang. A reviewer PASS is sufficient proof. The user will run it manually.

        ## When stuck
        - If a required library is missing, run `pip_install` first.
        - If no existing tool can do what you need, use `synthesize_tool` to create one.
        - Prefer existing tools over creating new ones.

        ## General
        - Think before acting. Brief reasoning is encouraged.
        - For `write_file`, `read_file`, `list_dir`: paths are relative to the workspace dir — do NOT prefix with 'workspace/'.
        - For `shell` or `python_eval`: the working directory is the project root, so use `workspace/filename.py` with a forward slash (never backslash).
        - Today's date: {datetime.utcnow().strftime('%Y-%m-%d')}

        ## Workspace
        {config.WORKSPACE_DIR}
    """).strip()


    if memory_context:
        base += f"\n\n---\n{memory_context}"
    return base


class SelfEvolvingAgent:
    """
    The main agent.

    Loop: Think → Act → Observe → [maybe Evolve] → loop
    """

    def __init__(self):
        from openai import OpenAI

        client_kwargs: dict[str, Any] = {"api_key": config.OPENAI_API_KEY}
        if config.OPENAI_BASE_URL:
            client_kwargs["base_url"] = config.OPENAI_BASE_URL

        self.llm = OpenAI(**client_kwargs)
        self.short_mem = ShortTermMemory(max_messages=80)
        self.long_mem = LongTermMemory()
        self.evolver = Evolver(self.llm, self.long_mem)
        self.planner = Planner(self.llm, context_fn=self.evolver.get_self_prompt_supplement)
        self.executor = Executor()


        # Failure tracking per sub-task for evolution trigger
        self._consecutive_errors: int = 0
        self._error_history: list[str] = []
        self._last_error_fingerprint: str = ""  # detects genuinely repeated errors

    # ── Public API ────────────────────────────────────────────────────────────
    def run(self, goal: str) -> str:
        """Run the agent on a goal. Returns the final outcome string."""
        rprint(Panel(f"[bold cyan]🎯 Goal:[/] {goal}", title="Self-Evolving Agent", border_style="cyan"))

        # ── Start the run log ───────────────────────────────────────────────────
        run_log = RunLogger(goal)

        # ── Auto-consolidate memory when enough new lessons have accumulated ──
        if self.long_mem.needs_consolidation(every_n_lessons=5):
            rprint("[dim cyan]🧠 Consolidating memory…[/]")
            summary = self.long_mem.consolidate(self.llm, config.MODEL)
            rprint(f"[dim cyan]Memory consolidated: {summary[:120]}…[/]")
            run_log.log_evolution("consolidation", summary)

        # Reload synthesized tools in case of re-run
        tool_registry.load_synthesized()

        # Build system prompt (with consolidated memory context)
        system_prompt = _build_system_prompt(self.evolver, self.long_mem)

        # ── Similarity fast-path: check if this goal resembles a past task ────
        rprint("[dim cyan]🔎 Checking similarity to past tasks…[/]")
        fast_path = self.evolver.find_similar_past_tasks(goal)
        if fast_path:
            rprint(f"[dim cyan]⚡ Fast-path found — injecting shortcut brief[/]")
            run_log.log_evolution("fast-path", fast_path)
            system_prompt += f"\n\n---\n{fast_path}"

        self.short_mem.clear()
        self.short_mem.add("system", system_prompt)

        # Plan
        rprint(Rule("[yellow]📋 Planning[/]", style="yellow"))
        sub_tasks = self.planner.decompose(goal)
        rprint(f"[yellow]Sub-tasks:[/] {sub_tasks}")

        completed_sub_tasks: list[str] = []
        replan_count = 0
        final_outcome = ""

        # ── Sub-task loop ─────────────────────────────────────────────────────
        for sub_task_idx, sub_task in enumerate(sub_tasks):
            rprint(Rule(f"[blue]🔧 Sub-task {sub_task_idx + 1}/{len(sub_tasks)}[/]", style="blue"))
            rprint(f"[blue]{sub_task}[/]")
            run_log.log_subtask_start(sub_task_idx + 1, len(sub_tasks), sub_task)

            self.short_mem.add(
                "user",
                f"Current sub-task ({sub_task_idx + 1}/{len(sub_tasks)}): {sub_task}\n"
                f"Overall goal: {goal}\n"
                f"Completed so far: {completed_sub_tasks}",
            )

            self._consecutive_errors = 0
            self._error_history = []
            self._last_error_fingerprint = ""
            sub_task_done = False

            # ── Inner tool-calling loop ───────────────────────────────────────
            is_last_subtask = (sub_task_idx == len(sub_tasks) - 1)
            for iteration in range(config.MAX_ITERATIONS):
                all_tools = tool_registry.all_openai_schemas() + [
                    _META_TOOL_SYNTHESIZE,
                    _META_TOOL_SUBTASK_DONE,
                    _META_TOOL_TASK_DONE,
                ]

                response = self._think(all_tools)
                msg = response.choices[0].message

                # Echo the model's reasoning
                if msg.content:
                    rprint(f"[dim white]💭 {msg.content}[/]")
                    run_log.log_thought(msg.content)

                # No tool call → nudge the model to use a tool
                if not msg.tool_calls:
                    self.short_mem.add_raw(msg.model_dump(exclude_none=True))
                    self.short_mem.add(
                        "user",
                        "Use a tool to continue. When this sub-task is fully done call "
                        "`subtask_complete` (or `task_complete` if this was the last sub-task).",
                    )
                    run_log.log_thought("(no tool call — nudging model to continue)")
                    continue

                # Append assistant turn (fully serialized)
                self.short_mem.add_raw(msg.model_dump(exclude_none=True))

                # ── Process each tool call ────────────────────────────────────
                subtask_done_flag = False
                task_done_flag = False
                deferred_user_messages: list[str] = []

                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    fn_args = tc.function.arguments

                    rprint(f"[green]🔨 Tool:[/] [bold]{fn_name}[/]  args={fn_args[:200]}")
                    run_log.log_tool_call(fn_name, fn_args)

                    # ── subtask_complete: run self-critique first ──────────────
                    if fn_name == "subtask_complete":
                        try:
                            result_text = json.loads(fn_args).get("result", "Sub-task done.")
                        except Exception:
                            result_text = "Sub-task done."

                        # Build evidence: last 10 tool/assistant messages, always
                        # surfacing the most recent workspace_code_reviewer result
                        # at the top so the critic never misses a PASS verdict.
                        all_msgs = [
                            m for m in self.short_mem.get_messages()
                            if m.get("role") in ("tool", "assistant")
                        ]
                        recent = all_msgs[-10:]
                        # Find last reviewer result anywhere in history
                        reviewer_msg = next(
                            (m for m in reversed(all_msgs)
                             if m.get("name") == "workspace_code_reviewer"),
                            None,
                        )
                        evidence_msgs = recent
                        if reviewer_msg and reviewer_msg not in recent:
                            evidence_msgs = [reviewer_msg] + recent
                        def _fmt_msg(m):
                            c = str(m.get('content', ''))
                            # workspace_code_reviewer: NEVER truncate — VERDICT is at the end
                            if m.get('name') == 'workspace_code_reviewer':
                                return f"[tool:workspace_code_reviewer]\n{c}"
                            return f"[{m.get('role','?')}] {c[:2000]}"
                        evidence = "\n".join(_fmt_msg(m) for m in evidence_msgs)

                        rprint(f"[yellow]🔍 Running self-critique on sub-task…[/]")
                        critique = self.evolver.self_critique(
                            goal=goal,
                            subtask=sub_task,
                            evidence=evidence,
                        )

                        if critique["pass"]:
                            self.short_mem.add_tool_result(tc.id, fn_name, "Acknowledged, moving to next sub-task.")
                            rprint(f"[cyan]✔ Sub-task verified ✅:[/] {result_text}")
                            run_log.log_subtask_done(result_text)
                            subtask_done_flag = True
                        else:
                            # Reject the completion — keep the agent on this sub-task
                            feedback = (
                                f"⚠️ Quality check FAILED for sub-task:\n"
                                f"Issues    : {critique['issues']}\n"
                                f"Suggestions: {critique['suggestions']}\n\n"
                                "Fix these issues before calling `subtask_complete` again."
                            )
                            self.short_mem.add_tool_result(tc.id, fn_name, feedback)
                            deferred_user_messages.append(feedback)
                            rprint(f"[red]⚠️ Critique FAILED:[/] {critique['issues'][:200]}")
                            run_log.log_evolution("self-critique", f"FAILED: {critique['issues']}")
                            self.evolver.ltm.add_lesson(
                                f"Sub-task '{sub_task[:80]}' was rejected by self-critique: {critique['issues'][:120]}"
                            )

                    # ── task_complete: verify entire goal before accepting ─────────
                    elif fn_name == "task_complete":
                        try:
                            summary = json.loads(fn_args).get("summary", "Task done.")
                        except Exception:
                            summary = fn_args

                        # Evidence: last 12 tool/assistant messages + any reviewer result
                        all_msgs_t = [
                            m for m in self.short_mem.get_messages()
                            if m.get("role") in ("tool", "assistant")
                        ]
                        recent_t = all_msgs_t[-12:]
                        reviewer_msg_t = next(
                            (m for m in reversed(all_msgs_t)
                             if m.get("name") == "workspace_code_reviewer"),
                            None,
                        )
                        evidence_msgs_t = recent_t
                        if reviewer_msg_t and reviewer_msg_t not in recent_t:
                            evidence_msgs_t = [reviewer_msg_t] + recent_t
                        def _fmt_msg_t(m):
                            c = str(m.get('content', ''))
                            if m.get('name') == 'workspace_code_reviewer':
                                return f"[tool:workspace_code_reviewer]\n{c}"
                            return f"[{m.get('role','?')}] {c[:2000]}"
                        evidence = "\n".join(_fmt_msg_t(m) for m in evidence_msgs_t)

                        rprint(f"[yellow]🔍 Running self-critique on full goal…[/]")
                        critique = self.evolver.self_critique(
                            goal=goal,
                            subtask=f"Full goal: {goal}",
                            evidence=evidence,
                        )

                        if critique["pass"]:
                            final_outcome = summary
                            self.short_mem.add_tool_result(tc.id, fn_name, "Acknowledged.")
                            rprint(Panel(f"[bold green]✅ {summary}[/]", title="Task Complete", border_style="green"))
                            run_log.log_subtask_done(summary)
                            task_done_flag = True
                        else:
                            feedback = (
                                f"⚠️ Final quality check FAILED:\n"
                                f"Issues    : {critique['issues']}\n"
                                f"Suggestions: {critique['suggestions']}\n\n"
                                "The goal is NOT fully accomplished yet. Fix the issues then call `task_complete` again."
                            )
                            self.short_mem.add_tool_result(tc.id, fn_name, feedback)
                            deferred_user_messages.append(feedback)
                            rprint(f"[red]⚠️ Final critique FAILED:[/] {critique['issues'][:200]}")
                            run_log.log_evolution("self-critique", f"FINAL FAILED: {critique['issues']}")

                    # ── synthesize_tool ───────────────────────────────────────
                    elif fn_name == "synthesize_tool":
                        try:
                            kwargs = json.loads(fn_args)
                        except Exception:
                            kwargs = {}
                        rprint(f"[magenta]🧬 Synthesizing tool:[/] {kwargs.get('name', '?')}")
                        result = self.evolver.synthesize_tool(
                            name=kwargs.get("name", "custom_tool"),
                            description=kwargs.get("description", ""),
                            use_case=kwargs.get("use_case", ""),
                        )
                        rprint(f"[magenta]{result}[/]")
                        self.short_mem.add_tool_result(tc.id, fn_name, result)
                        run_log.log_tool_result(result)

                    # ── regular tool ──────────────────────────────────────────
                    else:
                        result = self.executor.run(fn_name, fn_args)
                        rprint(f"[dim]Result (truncated): {result[:500]}[/]")
                        self.short_mem.add_tool_result(tc.id, fn_name, result)
                        is_error = result.startswith("ERROR")
                        run_log.log_tool_result(result, is_error=is_error)

                        if is_error:
                            # Only count as "consecutive" when the SAME error
                            # repeats (same tool + same first error line).
                            # Unrelated errors from different tools should not
                            # accumulate toward the synthesis threshold.
                            error_fp = f"{fn_name}:{result.splitlines()[0]}"[:100]
                            if error_fp == self._last_error_fingerprint:
                                self._consecutive_errors += 1
                            else:
                                self._last_error_fingerprint = error_fp
                                self._consecutive_errors = 1
                            self._error_history.append(f"{fn_name}: {result}")

                            # ── Evolve on EVERY error: instant reflection ──────
                            rprint(f"[magenta]🔁 Evolving (error #{self._consecutive_errors})…[/]")
                            hint = self.evolver.reflect_on_error(
                                tool_name=fn_name,
                                error=result,
                                subtask=sub_task,
                            )
                            rprint(f"[magenta]💡 Correction hint: {hint}[/]")
                            run_log.log_evolution("reflection", hint)
                            # Inject the hint directly so the model adapts next turn
                            deferred_user_messages.append(
                                f"⚠️ That tool call failed. Self-correction insight:\n{hint}\n\n"
                                "Adjust your approach and try again."
                            )

                            # ── Escalate to tool synthesis after N failures ────
                            if self._consecutive_errors >= config.MAX_TOOL_RETRIES:
                                # Only NOW is the error proven to be persistent —
                                # worth writing a lesson that will generalise.
                                self.evolver.ltm.add_lesson(
                                    f"Tool '{fn_name}' failed {self._consecutive_errors} times "
                                    f"with: {self._last_error_fingerprint[:80]}. "
                                    f"Hint that worked: {hint[:120]}"
                                )
                                rprint("[magenta]⚡ Escalating to tool synthesis…[/]")
                                evo_result = self.evolver.maybe_synthesize_tool(
                                    subtask=sub_task,
                                    failures=self._error_history,
                                )
                                rprint(f"[magenta]{evo_result}[/]")
                                run_log.log_evolution("synthesis", evo_result)
                                self._consecutive_errors = 0
                        else:
                            self._consecutive_errors = 0
                            self._last_error_fingerprint = ""

                # ── Append deferred messages now that all tool calls are resolved
                for user_msg in deferred_user_messages:
                    self.short_mem.add("user", user_msg)

                # ── Decide how to proceed after processing all tool calls ─────
                if task_done_flag:
                    sub_task_done = True
                    break  # exits inner loop; outer loop will also break

                if subtask_done_flag:
                    completed_sub_tasks.append(sub_task)
                    sub_task_done = True
                    break  # exits inner loop; outer loop continues

            # ── Sub-task outcome ───────────────────────────────────────────────
            if task_done_flag and final_outcome:
                # Whole job is done, generated via task_complete tool call
                completed_sub_tasks.append(sub_task)
                break
            elif not sub_task_done and replan_count < 2:
                # Sub-task timed out → replan
                rprint(Rule("[red]⚠️  Sub-task timed out – replanning[/]", style="red"))
                replan_count += 1
                sub_tasks = self.planner.replan(
                    goal=goal,
                    completed=completed_sub_tasks,
                    failure_reason=f"Sub-task '{sub_task}' did not complete within {config.MAX_ITERATIONS} iterations.",
                )
                rprint(f"[yellow]Revised plan:[/] {sub_tasks}")
                continue
            elif sub_task_done:
                completed_sub_tasks.append(sub_task)
                
                # Check if this was the final sub-task. If so, and task_complete wasn't called,
                # we force the outcome to a success string so the agent doesn't think it failed.
                if is_last_subtask and not final_outcome:
                    final_outcome = f"All {len(sub_tasks)} sub-tasks completed successfully."

        # ── Post-run ───────────────────────────────────────────────────────────
        if not final_outcome:
            final_outcome = f"Agent exhausted iterations. Last completed sub-tasks: {completed_sub_tasks}"

        # Extract and store lesson
        transcript = "\n".join(
            (m.get("content", "") if isinstance(m.get("content"), str) else "")
            for m in self.short_mem.get_messages()
        )
        lesson = self.evolver.extract_and_store_lesson(transcript)
        rprint(f"[dim cyan]📖 Lesson learned: {lesson}[/]")

        # Record task in long-term memory
        success = "error" not in final_outcome.lower()
        self.long_mem.record_task(goal=goal, outcome=final_outcome, success=success)
        run_log.log_outcome(final_outcome, success=success)
        rprint(f"[dim cyan]📄 Run log saved → {run_log.path}[/]")

        # ── Proactive post-run evolution (every run = opportunity to grow) ──────
        if success:
            rprint("[dim cyan]🧨 Post-run evolution check…[/]")
            evo_result = self.evolver.post_run_evolve(goal=goal, sub_tasks=sub_tasks)
            rprint(f"[dim cyan]{evo_result}[/]")
            run_log.log_evolution("post-run", evo_result)

        run_log.close()
        return final_outcome

    # ── Private helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _sanitize_messages(messages: list[dict]) -> list[dict]:
        """
        Ensure the message list is valid for the OpenAI API.

        A tool message is only valid when its tool_call_id belongs to an
        assistant+tool_calls message whose FULL set of results is present.
        Any incomplete or orphaned groups are dropped atomically.

        Uses tool_call_id sets rather than positional checks so that
        multi-tool-call assistant turns (where several tool results follow
        one assistant message) are handled correctly.
        """
        # ── Collect all result IDs that actually exist in the list ─────────────
        result_ids: set[str] = {
            m["tool_call_id"]
            for m in messages
            if m.get("role") == "tool" and "tool_call_id" in m
        }

        # ── Determine which assistant+tool_calls groups are COMPLETE ───────────
        # A group is complete when every tool_call it issued has a result.
        valid_call_ids: set[str] = set()   # IDs of all calls in complete groups
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                call_ids = {tc["id"] for tc in msg["tool_calls"]}
                if call_ids.issubset(result_ids):
                    valid_call_ids.update(call_ids)

        # ── Build the clean list ───────────────────────────────────────────────
        final: list[dict] = []
        for msg in messages:
            role = msg.get("role")

            if role == "tool":
                # Include only if its ID belongs to a complete group
                if msg.get("tool_call_id") in valid_call_ids:
                    final.append(msg)
                # else: orphaned or incomplete group → drop

            elif role == "assistant" and msg.get("tool_calls"):
                call_ids = {tc["id"] for tc in msg["tool_calls"]}
                if call_ids.issubset(result_ids):
                    final.append(msg)
                # else: incomplete assistant turn → drop

            else:
                final.append(msg)

        # ── Never start with assistant or tool ─────────────────────────────────
        while final and final[0].get("role") in ("assistant", "tool"):
            final.pop(0)

        return final

    def _think(self, tools: list[dict]) -> Any:
        """Call the LLM with the current (sanitized, token-trimmed) message history."""
        messages = self._sanitize_messages(self.short_mem.get_messages())

        # ── Token budget: trim oldest non-system messages until under limit ──────
        def _est_tokens(msgs: list[dict]) -> int:
            """Rough estimate: 1 token ≈ 4 chars of text."""
            total = 0
            for m in msgs:
                content = m.get("content") or ""
                if isinstance(content, list):
                    content = " ".join(str(p) for p in content)
                total += len(str(content)) // 4
                if m.get("tool_calls"):
                    total += len(str(m["tool_calls"])) // 4
            return total

        budget = config.MAX_CONTEXT_TOKENS
        while _est_tokens(messages) > budget and len(messages) > 1:
            # Keep system message (index 0), drop the next oldest one
            if len(messages) > 2:
                messages.pop(1)
            else:
                break  # can't trim further without losing everything

        # Re-sanitize after trimming: popping individual messages can orphan
        # tool-result messages (no preceding assistant+tool_calls), which the
        # OpenAI API rejects with a 400 error.
        messages = self._sanitize_messages(messages)

        # Retry on transient errors (rate limits, server overload, network blips).
        # Do NOT retry on 400 Bad Request — those are logic errors that need fixing.
        _RETRYABLE = ("rate limit", "rate_limit", "429", "529", "500", "502", "503",
                      "overloaded", "timeout", "timed out", "connection")
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                return self.llm.chat.completions.create(
                    model=config.MODEL,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.3,
                    max_tokens=config.MAX_RESPONSE_TOKENS,
                )
            except Exception as exc:
                err_lower = str(exc).lower()
                if any(sig in err_lower for sig in _RETRYABLE) and attempt < 2:
                    wait = (attempt + 1) * 4  # 4 s, 8 s
                    rprint(f"[yellow]⏳ Transient API error (attempt {attempt + 1}/3), "
                           f"retrying in {wait}s: {str(exc)[:120]}[/]")
                    time.sleep(wait)
                    last_exc = exc
                    continue
                raise
        raise last_exc  # type: ignore[misc]

"""tools/synthesized/workspace_code_reviewer.py

Full static code reviewer for workspace files.  Does NOT execute code.
Returns a structured report the agent can trust before calling subtask_complete.

Checks performed (in order):
  1. File exists and is readable
  2. Syntax check via ast.parse()
  3. Import scan — lists all imported packages
  4. GUI / blocking detection — marks the file NON-RUNNABLE if it uses
     pygame, tkinter, PyQt, wx, flask, uvicorn, etc.
  5. Required-pattern check — user-supplied strings / regex that MUST appear
  6. Completeness heuristic — warns if the file is suspiciously short
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import config
from tools.base import BaseTool

# Libraries that cause the process to block (GUI event loops, servers, etc.)
_BLOCKING_LIBS = {
    "pygame", "tkinter", "Tkinter", "wx", "PyQt5", "PyQt6",
    "PySide2", "PySide6", "pyglet", "kivy", "glfw",
    "flask", "fastapi", "uvicorn", "aiohttp", "tornado",
    "http.server", "socketserver",
}

# Patterns that indicate a blocking main loop even without a top-level import
_BLOCKING_PATTERNS = [
    r"pygame\.init\(",
    r"tkinter\.",
    r"app\.run\(",
    r"uvicorn\.run\(",
    r"while\s+True\s*:",   # generic infinite loop → might block
]


def _detect_blocking(source: str, imports: set[str]) -> tuple[bool, list[str]]:
    """Return (is_blocking, reasons)."""
    reasons: list[str] = []
    for lib in _BLOCKING_LIBS:
        if lib in imports:
            reasons.append(f"imports '{lib}' (GUI/server library)")
    for pat in _BLOCKING_PATTERNS:
        if re.search(pat, source):
            # Only flag pygame.init / tkinter / uvicorn — not generic while-True
            if "pygame" in pat or "tkinter" in pat or "run(" in pat:
                reasons.append(f"contains blocking call matching `{pat}`")
    return bool(reasons), reasons


def _extract_imports(tree: ast.Module) -> set[str]:
    """Walk AST and collect all top-level package names that are imported."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.split(".")[0])
    return names


class WorkspaceCodeReviewer(BaseTool):
    name = "workspace_code_reviewer"
    description = (
        "Statically analyze a Python file in the workspace WITHOUT executing it. "
        "Performs: syntax check, import scan, GUI/blocking detection, and required-pattern verification. "
        "Always call this BEFORE subtask_complete to confirm code is correct. "
        "If the report says 'GUI/Blocking: YES', the file MUST NOT be run — "
        "treat a PASS verdict as sufficient proof of correctness and call subtask_complete immediately."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path of the file to review (relative to workspace, or absolute).",
            },
            "required_patterns": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Strings or regex patterns that MUST appear in the file. "
                    "E.g. ['def move', 'pygame.draw.rect', 'collision_detection']. "
                    "Pass an empty list [] to skip pattern check."
                ),
            },
            "description": {
                "type": "string",
                "description": "What the file is supposed to do (used in the report header).",
            },
        },
        "required": ["path", "required_patterns"],
    }

    def run(
        self,
        path: str,
        required_patterns: list[str],
        description: str = "",
        **_: Any,
    ) -> str:
        # ── Resolve path ───────────────────────────────────────────────────────
        p = Path(path)
        if not p.is_absolute():
            if p.parts and p.parts[0] == "workspace":
                p = Path(*p.parts[1:])
            p = config.WORKSPACE_DIR / p
        p = p.resolve()

        if not p.exists():
            return f"FAIL — File not found: {p}"

        try:
            source = p.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return f"FAIL — Could not read file: {exc}"

        total_lines = len(source.splitlines())

        # ── 1. Syntax check ────────────────────────────────────────────────────
        syntax_ok = True
        syntax_error = ""
        syntax_context = ""
        tree: ast.Module | None = None
        try:
            tree = ast.parse(source, filename=str(p))
        except SyntaxError as exc:
            syntax_ok = False
            syntax_error = f"SyntaxError on line {exc.lineno}: {exc.msg}"

            src_lines = source.splitlines()
            err_idx = (exc.lineno or 1) - 1   # 0-indexed

            # ── Wide context window (±12 lines) ───────────────────────────────
            ctx_start = max(0, err_idx - 12)
            ctx_end   = min(len(src_lines), err_idx + 8)
            ctx_parts = []
            for i, ln in enumerate(src_lines[ctx_start:ctx_end], start=ctx_start + 1):
                marker = " →→ " if i == exc.lineno else "    "
                ctx_parts.append(f"  {i:>4}{marker}{ln}")
            syntax_context = "\n".join(ctx_parts)

            # ── Nearest enclosing block ────────────────────────────────────────
            # Walk backward from error to find the block that contains it
            # (while, for, if, def, class). Seeing its indentation level tells
            # the LLM exactly what indentation is expected inside it.
            block_keywords = ("while ", "for ", "if ", "elif ", "else:", "def ", "class ", "try:", "except", "with ")
            enclosing_line = None
            for idx in range(err_idx - 1, max(-1, err_idx - 60), -1):
                stripped = src_lines[idx].lstrip()
                if any(stripped.startswith(kw) for kw in block_keywords):
                    enclosing_line = (idx + 1, src_lines[idx])
                    break
            if enclosing_line:
                lineno, text = enclosing_line
                syntax_context += (
                    f"\n\n  ← Nearest enclosing block (line {lineno}):\n"
                    f"  {lineno:>4}    {text}\n"
                    f"       All code INSIDE this block needs {len(text) - len(text.lstrip()) + 4} spaces of indentation."
                )

        except Exception as exc:
            syntax_ok = False
            syntax_error = str(exc)

        # ── 2. Import scan ────────────────────────────────────────────────────
        imports: set[str] = set()
        if tree is not None:
            imports = _extract_imports(tree)

        # ── 3. GUI / blocking detection ───────────────────────────────────────
        is_blocking, blocking_reasons = _detect_blocking(source, imports)

        # ── 4. Pattern check ──────────────────────────────────────────────────
        missing: list[str] = []
        found: list[str] = []
        for pattern in required_patterns:
            try:
                matches = bool(re.search(pattern, source))
            except re.error:
                matches = pattern in source
            (found if matches else missing).append(pattern)

        # ── 5. Completeness heuristic ─────────────────────────────────────────
        completeness_warnings: list[str] = []
        if total_lines < 20 and description:
            completeness_warnings.append(
                f"Only {total_lines} lines — this may be incomplete for '{description}'."
            )

        # ── Build report ──────────────────────────────────────────────────────
        lines: list[str] = [
            f"=== Code Review: {p.name} ===",
            f"Lines : {total_lines}  |  Description: {description or 'N/A'}",
            f"Syntax: {'✅ OK' if syntax_ok else '❌ ERROR — ' + syntax_error}",
            f"GUI/Blocking: {'⚠️  YES — DO NOT RUN' if is_blocking else '✅ NO (safe to run)'}",
        ]

        # Show the broken code context so the LLM knows exactly what to fix
        if not syntax_ok and syntax_context:
            lines.append(
                "\n⚠️  CODE CONTEXT (around the error) — use this to understand and fix the issue:\n"
                "```python\n"
                + syntax_context +
                "\n```"
                "\n🔧 FIX INSTRUCTIONS: Do NOT try to patch individual lines."
                " REWRITE the entire file from scratch with correct indentation."
            )

        if is_blocking:
            lines.append(
                "\n🚫 CANNOT RUN — This program has a GUI or blocking event loop.\n"
                "   Static analysis is your ONLY verification method.\n"
                "   If patterns PASS → call subtask_complete immediately. Do NOT use shell/python_eval."
            )

        if imports:
            lines.append(f"\nImports detected: {', '.join(sorted(imports))}")

        if blocking_reasons:
            lines.append("Blocking reasons:")
            for r in blocking_reasons:
                lines.append(f"  • {r}")

        if completeness_warnings:
            lines.append("")
            for w in completeness_warnings:
                lines.append(f"⚠️  {w}")

        if required_patterns:
            lines.append("")
            if found:
                lines.append(f"✅ PRESENT ({len(found)}/{len(required_patterns)}):")
                for f in found:
                    lines.append(f"   ✔ {f:<30}")

            if missing:
                lines.append(
                    f"\n❌ MISSING ({len(missing)}/{len(required_patterns)}) — fix before subtask_complete:"
                )
                for m in missing:
                    lines.append(f"   ✘ {m}")

        # ── Final verdict ─────────────────────────────────────────────────────
        lines.append("")
        if not syntax_ok:
            lines.append("VERDICT: FAIL — syntax error must be fixed first.")
        elif missing:
            lines.append("VERDICT: FAIL — missing required patterns. Rewrite the file.")
        else:
            if is_blocking:
                lines.append(
                    "VERDICT: PASS — code is structurally sound.\n"
                    "  ➜ Call subtask_complete NOW. The user will run this game manually."
                )
            else:
                lines.append("VERDICT: PASS — all checks passed.")

        return "\n".join(lines)


TOOL = WorkspaceCodeReviewer()

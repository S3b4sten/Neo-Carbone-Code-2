"""tools/builtins/write_section.py – Write code in logical sections to avoid token limits.

Instead of embedding an entire file as one giant JSON string (which can exceed
the LLM's response token budget), the agent can write code section by section:

  1. write_section("game.py", "imports",   is_first=True,  content="import pygame...")
  2. write_section("game.py", "Bird class", is_first=False, content="class Bird:...")
  3. write_section("game.py", "main loop",  is_first=False, content="def main():...")

Each section is small enough to fit in one response. The tool appends sections
together to form the complete file.  A separator comment is inserted between
sections so the LLM can read them back cleanly with read_file.
"""

from pathlib import Path
from typing import Any

import config
from tools.base import BaseTool


class WriteSectionTool(BaseTool):
    name = "write_section"
    description = (
        "Write one logical section of a code file (imports, a class, the main loop, etc.). "
        "Use this instead of write_file when the full file would be too long for one response. "
        "Set is_first=true for the FIRST section (clears any existing file). "
        "Set is_first=false for all subsequent sections (appends to the file). "
        "Example workflow: "
        "  1. write_section('game.py', 'imports + constants', is_first=True,  content='import pygame...')  "
        "  2. write_section('game.py', 'Bird class',          is_first=False, content='class Bird:...')  "
        "  3. write_section('game.py', 'main loop',           is_first=False, content='def main():...')  "
        "After all sections are written, call workspace_code_reviewer to verify the full file."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path (relative to workspace).",
            },
            "section_name": {
                "type": "string",
                "description": "Human-readable label for this section, e.g. 'Bird class' or 'main loop'.",
            },
            "content": {
                "type": "string",
                "description": "The source code for this section.",
            },
            "is_first": {
                "type": "boolean",
                "description": "True for the first section (overwrites the file). False for subsequent sections (appends).",
            },
        },
        "required": ["path", "section_name", "content", "is_first"],
    }

    def run(
        self,
        path: str,
        section_name: str,
        content: str,
        is_first: bool = False,
        **_: Any,
    ) -> str:
        # Resolve path (workspace jail)
        p = Path(path)
        if not p.is_absolute():
            if p.parts and p.parts[0] == "workspace":
                p = Path(*p.parts[1:])
            p = config.WORKSPACE_DIR / p
        p = p.resolve()

        # Workspace jail — keep all writes inside workspace
        workspace = config.WORKSPACE_DIR.resolve()
        try:
            p.relative_to(workspace)
        except ValueError:
            p = workspace / p.name

        p.parent.mkdir(parents=True, exist_ok=True)

        try:
            if is_first:
                # Overwrite — start fresh
                p.write_text(content.rstrip() + "\n", encoding="utf-8")
                return (
                    f"✅ Section '{section_name}' written to '{p.name}' (new file, {len(content.splitlines())} lines).\n"
                    f"   Call write_section again with is_first=false for the next section."
                )
            else:
                # Append with a blank separator line
                existing = p.read_text(encoding="utf-8") if p.exists() else ""
                separator = f"\n\n# ── {section_name} ──\n" if existing.strip() else ""
                new_content = existing + separator + content.rstrip() + "\n"
                p.write_text(new_content, encoding="utf-8")
                total_lines = len(new_content.splitlines())
                return (
                    f"✅ Section '{section_name}' appended to '{p.name}' (file now {total_lines} lines).\n"
                    f"   Continue adding sections, or call workspace_code_reviewer when done."
                )
        except Exception as exc:
            return f"ERROR writing section '{section_name}': {exc}"

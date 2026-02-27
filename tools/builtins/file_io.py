"""tools/builtins/file_io.py – Read, write, and list files in the workspace."""

import os
from pathlib import Path
from typing import Any

import config
from tools.base import BaseTool


def _safe_path(raw: str) -> Path:
    """Resolve path and enforce workspace jail.

    Rules:
    - Relative paths are anchored to WORKSPACE_DIR.
    - If the agent erroneously prepends 'workspace/', strip it first.
    - Absolute paths that live INSIDE WORKSPACE_DIR are allowed as-is.
    - Absolute paths that escape WORKSPACE_DIR are silently redirected to
      WORKSPACE_DIR/<filename> so output always stays inside the workspace.
    """
    p = Path(raw)
    if not p.is_absolute():
        # Strip spurious 'workspace/' prefix the agent sometimes adds
        if p.parts and p.parts[0] == "workspace":
            p = Path(*p.parts[1:])
        p = config.WORKSPACE_DIR / p
    resolved = p.resolve()
    # Workspace jail: redirect any path that escapes the workspace
    workspace = config.WORKSPACE_DIR.resolve()
    try:
        resolved.relative_to(workspace)  # raises ValueError if outside
    except ValueError:
        # Re-anchor: keep only the final filename, put it in workspace
        resolved = workspace / resolved.name
    return resolved


class ReadFileTool(BaseTool):
    name = "read_file"
    description = (
        "Read the text content of a file. "
        "Paths are relative to the workspace directory unless absolute."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file."},
            "start_line": {"type": "integer", "description": "First line to return (1-indexed)."},
            "end_line": {"type": "integer", "description": "Last line to return (inclusive)."},
        },
        "required": ["path"],
    }

    def run(self, path: str, start_line: int | None = None, end_line: int | None = None, **_: Any) -> str:
        try:
            p = _safe_path(path)
            text = p.read_text(encoding="utf-8", errors="replace")
            if start_line or end_line:
                lines = text.splitlines()
                s = (start_line or 1) - 1
                e = end_line or len(lines)
                text = "\n".join(lines[s:e])
            return text if text else "(empty file)"
        except Exception as exc:
            return f"ERROR: {exc}"


class WriteFileTool(BaseTool):
    name = "write_file"
    description = (
        "Write text content to a file (creates or overwrites). "
        "Paths are relative to the workspace directory unless absolute. "
        "WARNING: NEVER use append=true when writing code files. "
        "Always write the COMPLETE file content in a single call so the file is always valid and self-contained. "
        "append=true is only for log files or data accumulation — never for Python scripts."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Destination file path."},
            "content": {"type": "string", "description": "Text content to write."},
            "append": {"type": "boolean", "description": "If true, append instead of overwrite."},
        },
        "required": ["path", "content"],
    }

    def run(self, path: str, content: str, append: bool = False, **_: Any) -> str:
        try:
            p = _safe_path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            p.write_text(content, encoding="utf-8") if not append else open(p, "a").write(content)
            return f"Successfully {'appended to' if append else 'wrote'} '{p}'."
        except Exception as exc:
            return f"ERROR: {exc}"


class ListDirTool(BaseTool):
    name = "list_dir"
    description = "List the contents of a directory (defaults to the workspace)."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory to list. Defaults to workspace."},
        },
        "required": [],
    }

    def run(self, path: str = "", **_: Any) -> str:
        try:
            p = _safe_path(path) if path else config.WORKSPACE_DIR
            entries = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name))
            lines = []
            for e in entries:
                if e.is_dir():
                    lines.append(f"[DIR]  {e.name}/")
                else:
                    size = e.stat().st_size
                    lines.append(f"[FILE] {e.name}  ({size:,} bytes)")
            return "\n".join(lines) if lines else "(empty directory)"
        except Exception as exc:
            return f"ERROR: {exc}"

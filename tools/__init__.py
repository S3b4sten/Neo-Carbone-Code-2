"""tools/__init__.py – Central tool registry."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tools.base import BaseTool

# ── Registry ───────────────────────────────────────────────────────────────────
_REGISTRY: dict[str, "BaseTool"] = {}


def register_tool(tool: "BaseTool") -> None:
    """Add a tool instance to the registry (overwrites if name already exists)."""
    _REGISTRY[tool.name] = tool


def get_tool(name: str) -> "BaseTool":
    """Return a tool by name; raises KeyError if not found."""
    return _REGISTRY[name]


def list_tools() -> list["BaseTool"]:
    """Return all registered (non-hidden) tool instances."""
    return [t for t in _REGISTRY.values() if not t.hidden]


def all_openai_schemas() -> list[dict]:
    """Return OpenAI function-calling schemas for all registered tools."""
    return [t.to_openai_schema() for t in list_tools()]


def load_builtins() -> None:
    """Import and register every built-in tool."""
    from tools.builtins.shell import ShellTool
    from tools.builtins.web_search import WebSearchTool
    from tools.builtins.file_io import ReadFileTool, WriteFileTool, ListDirTool
    from tools.builtins.python_eval import PythonEvalTool
    from tools.builtins.http_request import HttpRequestTool
    from tools.builtins.pip_install import PipInstallTool
    from tools.builtins.write_section import WriteSectionTool
    from tools.synthesized.workspace_code_reviewer import WorkspaceCodeReviewer

    for cls in (
        ShellTool,
        WebSearchTool,
        ReadFileTool,
        WriteFileTool,
        ListDirTool,
        PythonEvalTool,
        HttpRequestTool,
        PipInstallTool,
        WriteSectionTool,
        WorkspaceCodeReviewer,
    ):
        register_tool(cls())


def load_synthesized() -> None:
    """Auto-load any previously synthesized tools from tools/synthesized/."""
    from config import SYNTHESIZED_TOOLS_DIR

    for py_file in sorted(SYNTHESIZED_TOOLS_DIR.glob("*.py")):
        module_name = f"tools.synthesized.{py_file.stem}"
        if module_name in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(module_name, py_file)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = mod
            try:
                spec.loader.exec_module(mod)  # type: ignore
                # Convention: each synthesized file exposes a TOOL instance
                if hasattr(mod, "TOOL"):
                    register_tool(mod.TOOL)
            except Exception as exc:
                print(f"[tools] Failed to load synthesized tool {py_file.name}: {exc}")


def initialise() -> None:
    """Load all tools (builtins + synthesized). Call once at startup."""
    load_builtins()
    load_synthesized()

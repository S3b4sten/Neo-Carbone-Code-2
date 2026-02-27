"""tools/synthesizer.py – LLM-driven tool synthesis."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import textwrap
from pathlib import Path

import config
import tools as tool_registry
from tools.base import BaseTool


_SYNTHESIS_PROMPT = """\
You are a Python expert. Write a new tool class for a self-evolving AI agent.

REQUIREMENTS:
1. Subclass `BaseTool` (already imported below).
2. Set class variables: `name` (str, snake_case), `description` (str), `parameters` (OpenAI JSON schema dict).
3. Implement `def run(self, **kwargs) -> str:` — always returns a string.
4. Import everything you need inside `run()` so the file is self-contained.
5. At the bottom, add a module-level `TOOL = YourClassName()`.
6. Do NOT import from agent modules. You may use stdlib and any installed packages.
7. Keep the code concise and robust (handle exceptions, return clear error strings).

TOOL SPECIFICATION:
Name: {name}
Description: {description}
Use case: {use_case}

Respond with ONLY the Python source code, no markdown fences.

from tools.base import BaseTool
"""


class ToolSynthesizer:
    """Asks the LLM to write a new BaseTool subclass and registers it."""

    def __init__(self, llm_client):
        self.llm = llm_client

    def create(self, name: str, description: str, use_case: str) -> str:
        """
        Synthesize, validate, persist and register a new tool.

        Returns a status string.
        """
        prompt = _SYNTHESIS_PROMPT.format(
            name=name,
            description=description,
            use_case=use_case,
        )

        response = self.llm.chat.completions.create(
            model=config.MODEL,
            messages=[
                {"role": "system", "content": "You are an expert Python tool developer."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        code = response.choices[0].message.content.strip()

        # Strip any accidental markdown fences
        code = re.sub(r"^```(?:python)?\n?", "", code)
        code = re.sub(r"\n?```$", "", code)

        # Validate by compiling
        try:
            compile(code, f"<synth_{name}>", "exec")
        except SyntaxError as exc:
            return f"Synthesis failed – syntax error: {exc}"

        # Write to disk
        safe_name = re.sub(r"[^\w]", "_", name)
        out_path = config.SYNTHESIZED_TOOLS_DIR / f"{safe_name}.py"
        out_path.write_text(code, encoding="utf-8")

        # Load and register
        module_name = f"tools.synthesized.{safe_name}"
        spec = importlib.util.spec_from_file_location(module_name, out_path)
        if not spec or not spec.loader:
            return f"Synthesis failed – could not create module spec for {out_path}"
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        try:
            spec.loader.exec_module(mod)  # type: ignore
        except Exception as exc:
            return f"Synthesis failed – error running synthesized module: {exc}"

        if not hasattr(mod, "TOOL"):
            return f"Synthesis failed – module has no `TOOL` attribute.\nCode:\n{code}"

        tool_instance = mod.TOOL
        tool_registry.register_tool(tool_instance)

        return (
            f"✅ Tool '{tool_instance.name}' synthesized and registered.\n"
            f"File: {out_path}\n"
            f"Description: {tool_instance.description}"
        )

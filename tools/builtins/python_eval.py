"""tools/builtins/python_eval.py – Execute Python code snippets."""

import contextlib
import io
import traceback
from typing import Any

import config
from tools.base import BaseTool


class PythonEvalTool(BaseTool):
    name = "python_eval"
    description = (
        "Execute a Python code snippet and return its printed output plus the value of "
        "the last expression. Use for calculations, data processing, quick scripting, "
        "or testing logic without writing a file."
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python source code to execute.",
            },
        },
        "required": ["code"],
    }

    # Shared globals that persist across calls within a session
    _globals: dict = {}

    def run(self, code: str, **_: Any) -> str:
        stdout_buf = io.StringIO()
        try:
            if config.SANDBOX_PYTHON:
                return self._run_sandboxed(code)
            else:
                return self._run_unrestricted(code, stdout_buf)
        except Exception as exc:
            return f"ERROR:\n{traceback.format_exc()}"

    def _run_unrestricted(self, code: str, stdout_buf: io.StringIO) -> str:
        with contextlib.redirect_stdout(stdout_buf):
            import os
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            os.environ["SDL_AUDIODRIVER"] = "dummy"
            exec_globals = self._globals  # shared state across calls
            exec(compile(code, "<agent_eval>", "exec"), exec_globals)
        output = stdout_buf.getvalue().strip()
        return output if output else "(no output)"

    def _run_sandboxed(self, code: str) -> str:
        try:
            from RestrictedPython import compile_restricted, safe_globals
            from RestrictedPython.Guards import safe_builtins

            byte_code = compile_restricted(code, filename="<sandbox>", mode="exec")
            restricted_globals = {
                **safe_globals,
                "__builtins__": safe_builtins,
            }
            buf = io.StringIO()
            restricted_globals["_print_"] = lambda *a: buf.write(" ".join(str(x) for x in a) + "\n")
            exec(byte_code, restricted_globals)
            out = buf.getvalue().strip()
            return out if out else "(no output)"
        except Exception as exc:
            return f"SANDBOX ERROR:\n{traceback.format_exc()}"

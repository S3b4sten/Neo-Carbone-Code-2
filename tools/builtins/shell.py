"""tools/builtins/shell.py – Run shell commands."""

import subprocess
from typing import Any

import config
from tools.base import BaseTool


class ShellTool(BaseTool):
    name = "shell"
    description = (
        "Run a shell command on the host system and return its stdout + stderr. "
        "Use for general OS tasks, file manipulation, compiling, running scripts, etc."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            },
            "cwd": {
                "type": "string",
                "description": "Working directory for the command. Defaults to the workspace dir.",
            },
        },
        "required": ["command"],
    }

    def run(self, command: str, cwd: str | None = None, **_: Any) -> str:
        if not config.ALLOW_SHELL:
            return "ERROR: Shell execution is disabled via ALLOW_SHELL=false."

        work_dir = cwd or str(config.WORKSPACE_DIR)

        # Force headless mode for any GUI libraries (like pygame) so they don't block
        import os
        env = os.environ.copy()
        env["SDL_VIDEODRIVER"] = "dummy"
        env["SDL_AUDIODRIVER"] = "dummy"

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=work_dir,
                env=env,
                timeout=config.TOOL_TIMEOUT_SECONDS,
            )
            out = result.stdout.strip()
            err = result.stderr.strip()
            parts = []
            if out:
                parts.append(f"STDOUT:\n{out}")
            if err:
                parts.append(f"STDERR:\n{err}")
            if result.returncode != 0:
                parts.append(f"EXIT CODE: {result.returncode}")
            return "\n".join(parts) if parts else "(no output)"
        except subprocess.TimeoutExpired:
            return f"ERROR: Command timed out after {config.TOOL_TIMEOUT_SECONDS}s."
        except Exception as exc:
            return f"ERROR: {exc}"

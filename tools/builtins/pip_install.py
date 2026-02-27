"""tools/builtins/pip_install.py – Install Python packages at runtime."""

import subprocess
import sys
from typing import Any

import config
from tools.base import BaseTool


class PipInstallTool(BaseTool):
    name = "pip_install"
    description = (
        "Install one or more Python packages via pip. "
        "Use when you need a library that isn't available yet. "
        "After installation the package is immediately importable."
    )
    parameters = {
        "type": "object",
        "properties": {
            "packages": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of package names (with optional version specifiers) to install.",
            },
        },
        "required": ["packages"],
    }

    def run(self, packages: list[str], **_: Any) -> str:
        if not config.ALLOW_PIP_INSTALL:
            return "ERROR: pip install is disabled via ALLOW_PIP_INSTALL=false."

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", *packages, "--quiet"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                return f"Successfully installed: {', '.join(packages)}"
            return f"pip error:\n{result.stderr.strip()}"
        except subprocess.TimeoutExpired:
            return "ERROR: pip install timed out."
        except Exception as exc:
            return f"ERROR: {exc}"

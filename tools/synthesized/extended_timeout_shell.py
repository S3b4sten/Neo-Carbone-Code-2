import config
from tools.base import BaseTool

_MAX_TIMEOUT = config.TOOL_TIMEOUT_SECONDS  # hard cap — never wait longer than the global limit

class ExtendedTimeoutShell(BaseTool):
    name = "extended_timeout_shell"
    description = (
        "Runs a shell command with a configurable timeout (max 30 s). "
        "Use for non-interactive scripts only. "
        "Do NOT use for GUI programs or interactive games — those need user input and will always time out."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute."
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds. Capped at 30.",
                "default": 15
            },
            "cwd": {
                "type": "string",
                "description": "Optional working directory to run the command in."
            }
        },
        "required": ["command"]
    }

    def run(self, **kwargs) -> str:
        import subprocess
        import shlex

        command = kwargs.get("command")
        timeout = min(kwargs.get("timeout", 15), _MAX_TIMEOUT)  # never exceed hard cap
        cwd = kwargs.get("cwd")

        try:
            result = subprocess.run(
                shlex.split(command),
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd
            )
            if result.returncode == 0:
                return result.stdout
            else:
                return f"Error: {result.stderr}"
        except subprocess.TimeoutExpired:
            return "Error: Command timed out."
        except Exception as e:
            return f"Error: {str(e)}"

TOOL = ExtendedTimeoutShell()
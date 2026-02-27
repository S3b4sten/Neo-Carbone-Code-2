from tools.base import BaseTool

class ExtendedTimeoutShell(BaseTool):
    name = "extended_timeout_shell"
    description = "Runs a shell command with an extended timeout to allow for longer processes like game testing."
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute."
            },
            "timeout": {
                "type": "integer",
                "description": "The timeout duration in seconds for the command execution.",
                "default": 300
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
        timeout = kwargs.get("timeout", 300)
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
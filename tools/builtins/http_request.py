"""tools/builtins/http_request.py – Make HTTP GET / POST requests."""

from typing import Any

import config
from tools.base import BaseTool


class HttpRequestTool(BaseTool):
    name = "http_request"
    description = (
        "Make an HTTP GET or POST request and return the response body as text. "
        "Use for calling REST APIs, downloading web content, etc."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Target URL."},
            "method": {
                "type": "string",
                "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                "description": "HTTP method. Defaults to GET.",
            },
            "headers": {
                "type": "object",
                "description": "Optional dict of request headers.",
            },
            "body": {
                "type": "string",
                "description": "Request body for POST/PUT (JSON string or plain text).",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default 15).",
            },
        },
        "required": ["url"],
    }

    def run(
        self,
        url: str,
        method: str = "GET",
        headers: dict | None = None,
        body: str | None = None,
        timeout: int = 15,
        **_: Any,
    ) -> str:
        if not config.ALLOW_WEB:
            return "ERROR: Web access is disabled via ALLOW_WEB=false."

        try:
            import httpx

            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                req_headers = headers or {}
                resp = client.request(
                    method=method.upper(),
                    url=url,
                    headers=req_headers,
                    content=body.encode() if body else None,
                )
            status = resp.status_code
            text = resp.text[:8000]  # cap at 8 KB
            return f"HTTP {status}\n{text}"
        except Exception as exc:
            return f"ERROR: {exc}"

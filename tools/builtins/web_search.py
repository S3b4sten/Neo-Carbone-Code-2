"""tools/builtins/web_search.py – DuckDuckGo search."""

from typing import Any

import config
from tools.base import BaseTool


class WebSearchTool(BaseTool):
    name = "web_search"
    description = (
        "Search the internet using DuckDuckGo. Returns a list of snippets and URLs. "
        "Use to find documentation, APIs, current data, or any information not in memory."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query string.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default 6).",
            },
        },
        "required": ["query"],
    }

    def run(self, query: str, max_results: int = 6, **_: Any) -> str:
        if not config.ALLOW_WEB:
            return "ERROR: Web access is disabled via ALLOW_WEB=false."

        try:
            from duckduckgo_search import DDGS

            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append(
                        f"[{r.get('title', 'No title')}]\n"
                        f"URL: {r.get('href', '')}\n"
                        f"Snippet: {r.get('body', '')}"
                    )
            return "\n\n".join(results) if results else "No results found."
        except Exception as exc:
            return f"ERROR: {exc}"

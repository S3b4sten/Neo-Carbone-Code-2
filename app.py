"""app.py – Flask web interface for the Self-Evolving Agent.

Run with:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

from __future__ import annotations

import io
import json
import os
import queue
import re
import threading
from typing import Iterator

from flask import Flask, Response, render_template, request, stream_with_context

app = Flask(__name__)

# ── Thread-local storage for per-request event queues ────────────────────────
_tl = threading.local()

# Only one agent run at a time (agent mutates global config state)
_run_lock = threading.Lock()


def _get_queue() -> "queue.Queue | None":
    return getattr(_tl, "event_queue", None)


# ── Rich output capture ───────────────────────────────────────────────────────

def _render_rich(obj) -> str:
    """Render any rich renderable (Panel, Rule, str…) to plain text."""
    buf = io.StringIO()
    from rich.console import Console
    con = Console(file=buf, width=100, highlight=False, no_color=True,
                  force_terminal=False)
    con.print(obj)
    raw = buf.getvalue()
    # Strip residual ANSI escape codes
    raw = re.sub(r"\x1b\[[0-9;]*[mK]", "", raw)
    return raw.strip()


def _classify(text: str) -> str:
    """Return an event-type label based on the message content."""
    if "🔨" in text or "Tool:" in text:
        return "tool"
    if "💭" in text:
        return "thought"
    if any(x in text for x in ("📋", "Planning", "Sub-task", "🔧")):
        return "planning"
    if any(x in text for x in ("✅", "Task Complete", "✔ Sub-task", "🎯 Goal")):
        return "success"
    if any(x in text for x in ("⚠️", "FAILED", "critique FAILED", "ERROR")):
        return "warning"
    if "🧬" in text or "Synthesizing" in text:
        return "synth"
    if any(x in text for x in ("📖 Lesson", "🧠", "Memory")):
        return "memory"
    if "Result (truncated)" in text or "Result:" in text:
        return "result"
    return "log"


def _make_captured_rprint():
    """
    Return a drop-in replacement for `rprint` that routes output to the
    thread-local SSE queue, falling back to the real Rich console when
    not inside an SSE request thread.
    """
    from rich.console import Console as _RichConsole
    _fallback = _RichConsole()

    def captured_rprint(*args, **kwargs):
        q = _get_queue()
        if q is None:
            # Not inside an SSE request — use normal terminal output
            _fallback.print(*args, **kwargs)
            return

        parts = []
        for a in args:
            try:
                parts.append(_render_rich(a))
            except Exception:
                parts.append(re.sub(r"\[/?[^\]]*\]", "", str(a)).strip())

        text = "\n".join(p for p in parts if p)
        if text:
            q.put({"type": _classify(text), "text": text})

    return captured_rprint


# ── One-time module patching ──────────────────────────────────────────────────
_PATCHED = False
_captured_rprint = None


def _patch_agent_modules() -> None:
    """Replace `rprint` in all agent modules with our captured version."""
    global _PATCHED, _captured_rprint
    if _PATCHED:
        return

    _captured_rprint = _make_captured_rprint()

    # Import agent modules so they are in sys.modules before we patch them
    import agent.core      # noqa: F401
    import agent.executor  # noqa: F401

    for mod_name in ("agent.core", "agent.executor"):
        import sys
        mod = sys.modules.get(mod_name)
        if mod and hasattr(mod, "rprint"):
            setattr(mod, "rprint", _captured_rprint)

    _PATCHED = True


# ── Flask routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run")
def run():
    goal = request.args.get("goal", "").strip()
    model = request.args.get("model", "gpt-4o-mini").strip()
    api_key = request.args.get("api_key", "").strip()

    if not goal:
        def _err():
            yield 'data: {"type":"error","text":"No goal provided."}\n\n'
            yield 'data: {"type":"end"}\n\n'
        return Response(stream_with_context(_err()), mimetype="text/event-stream")

    # Reject if already running
    if not _run_lock.acquire(blocking=False):
        def _busy():
            yield 'data: {"type":"error","text":"An agent run is already in progress. Please wait."}\n\n'
            yield 'data: {"type":"end"}\n\n'
        return Response(stream_with_context(_busy()), mimetype="text/event-stream")

    event_queue: queue.Queue = queue.Queue()

    def agent_thread() -> None:
        _tl.event_queue = event_queue
        try:
            import config

            # Apply settings
            if api_key:
                config.OPENAI_API_KEY = api_key
                os.environ["OPENAI_API_KEY"] = api_key
            config.MODEL = model

            if not (config.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY", "")):
                event_queue.put({
                    "type": "error",
                    "text": "No OpenAI API key found. Enter it in Settings or set the "
                            "OPENAI_API_KEY environment variable.",
                })
                return

            _patch_agent_modules()

            import tools as tool_registry
            tool_registry.initialise()
            tool_registry.load_synthesized()

            from agent.core import SelfEvolvingAgent
            agent = SelfEvolvingAgent()
            outcome = agent.run(goal)

            event_queue.put({"type": "done", "text": outcome})

        except Exception as exc:
            import traceback
            event_queue.put({
                "type": "error",
                "text": f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
            })
        finally:
            _tl.event_queue = None
            event_queue.put(None)  # sentinel

    thread = threading.Thread(target=agent_thread, daemon=True)
    thread.start()

    def generate() -> Iterator[str]:
        try:
            while True:
                try:
                    item = event_queue.get(timeout=300)
                except queue.Empty:
                    yield "data: {\"type\":\"ping\"}\n\n"
                    continue

                if item is None:
                    yield "data: {\"type\":\"end\"}\n\n"
                    break

                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        finally:
            _run_lock.release()

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/tools")
def list_tools():
    """Return available tools as JSON (for the UI sidebar)."""
    try:
        import tools as tool_registry
        tool_registry.initialise()
        tools = tool_registry.list_tools()
        return {"tools": [{"name": t.name, "description": t.description} for t in tools]}
    except Exception as exc:
        return {"tools": [], "error": str(exc)}


@app.route("/workspace")
def workspace_files():
    """Return workspace file list as JSON."""
    try:
        import config
        files = []
        if config.WORKSPACE_DIR.exists():
            for f in sorted(config.WORKSPACE_DIR.iterdir()):
                if f.is_file():
                    files.append({"name": f.name, "size": f.stat().st_size})
        return {"files": files}
    except Exception as exc:
        return {"files": [], "error": str(exc)}


if __name__ == "__main__":
    print("Starting Self-Evolving Agent web interface…")
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(debug=False, threaded=True, host="127.0.0.1", port=5000)

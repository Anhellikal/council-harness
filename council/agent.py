"""HTTP calls to agents and rollcall logic."""

import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable
import requests

CALL_TIMEOUT = 300  # seconds per LLM call — large local models can be slow
PING_TIMEOUT = 10

# Subprocess-backed agents that authenticate via their own subscription login
# (no API key, no token billing). Generation-only by default; allowed to write
# files only when invoked as the writer with `can_write: true`.
_CLI_BINARIES = {"claude_cli": "claude", "codex_cli": "codex"}


def cli_can_write(agent: dict) -> bool:
    """True if this agent is a CLI type explicitly permitted to write files."""
    return (
        agent.get("type", "").lower() in _CLI_BINARIES
        and bool(agent.get("can_write"))
    )


# ---------------------------------------------------------------------------
# Low-level HTTP
# ---------------------------------------------------------------------------

_THINKING_RE = re.compile(r"<(think|thinking)>.*?</\1>", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Remove <think>…</think> / <thinking>…</thinking> blocks from model output."""
    return _THINKING_RE.sub("", text).strip()


def _call_ollama(agent: dict, prompt: str) -> tuple[str, dict]:
    url = agent["url"].rstrip("/")
    timeout = agent.get("timeout", CALL_TIMEOUT)
    options: dict = {}
    if max_tokens := agent.get("max_tokens"):
        options["num_predict"] = max_tokens
    if num_ctx := agent.get("num_ctx"):
        options["num_ctx"] = num_ctx
    if agent.get("no_think"):
        options["think"] = False
    body: dict = {"model": agent["model"], "prompt": prompt, "stream": False}
    if options:
        body["options"] = options
    resp = requests.post(f"{url}/api/generate", json=body, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    stats: dict = {}
    if "eval_count" in data and "eval_duration" in data:
        tps = data["eval_count"] / (data["eval_duration"] / 1e9)
        stats["tps"] = round(tps, 1)
        stats["tokens"] = data["eval_count"]
    if "prompt_eval_duration" in data:
        stats["ttft_ms"] = round(data["prompt_eval_duration"] / 1e6)

    return _strip_thinking(data["response"]), stats


def _resolve_api_key(agent: dict) -> str | None:
    if env_var := agent.get("api_key_env"):
        return os.environ.get(env_var)
    return agent.get("api_key")


def _call_openai(agent: dict, prompt: str) -> tuple[str, dict]:
    url = agent["url"].rstrip("/")
    timeout = agent.get("timeout", CALL_TIMEOUT)
    headers: dict[str, str] = {}
    if api_key := _resolve_api_key(agent):
        headers["Authorization"] = f"Bearer {api_key}"
    body: dict = {
        "model": agent["model"],
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    if max_tokens := agent.get("max_tokens"):
        body["max_tokens"] = max_tokens
    if agent.get("no_think"):
        body["think"] = False
    resp = requests.post(f"{url}/v1/chat/completions", json=body, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    stats: dict = {}
    usage = data.get("usage", {})
    if "completion_tokens" in usage:
        stats["tokens"] = usage["completion_tokens"]

    return _strip_thinking(data["choices"][0]["message"]["content"]), stats


def _run_subprocess(cmd: list[str], cwd: str, timeout: int, label: str, stdin: str | None = None) -> str:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd, input=stdin
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{label} timed out after {timeout}s")
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[:300]
        raise RuntimeError(f"{label} exited {proc.returncode}: {detail}")
    return proc.stdout


def _call_claude_cli(agent: dict, prompt: str, workdir: str | None) -> tuple[str, dict]:
    timeout = agent.get("timeout", CALL_TIMEOUT)
    can_write = workdir is not None
    # Prompt goes via stdin so it can't be swallowed by the variadic
    # --allowedTools flag (or break on shell-special characters).
    cmd = ["claude", "-p", "--output-format", "text"]
    if can_write:
        # auto-accept file writes (no interactive prompt in -p mode); other
        # tools like Bash still require approval and are effectively blocked
        cmd += ["--permission-mode", "acceptEdits"]
    else:
        cmd += ["--allowedTools", ""]  # no tools -> generation only
    if model := agent.get("model"):
        cmd += ["--model", model]

    t0 = time.time()
    if can_write:
        out = _run_subprocess(cmd, workdir, timeout, "claude CLI", stdin=prompt)
    else:
        with tempfile.TemporaryDirectory() as sandbox:
            out = _run_subprocess(cmd, sandbox, timeout, "claude CLI", stdin=prompt)
    text = _strip_thinking(out.strip())
    return text, _cli_stats(text, time.time() - t0)


def _call_codex_cli(agent: dict, prompt: str, workdir: str | None) -> tuple[str, dict]:
    timeout = agent.get("timeout", CALL_TIMEOUT)
    can_write = workdir is not None
    t0 = time.time()
    with tempfile.TemporaryDirectory() as scratch:
        last_msg = os.path.join(scratch, "last.txt")
        # No prompt arg -> codex exec reads instructions from stdin.
        cmd = [
            "codex", "exec", "--skip-git-repo-check",
            "-s", "workspace-write" if can_write else "read-only",
            "-o", last_msg,
        ]
        if model := agent.get("model"):
            cmd += ["-m", model]
        out = _run_subprocess(cmd, workdir or scratch, timeout, "codex CLI", stdin=prompt)
        if os.path.exists(last_msg):
            raw = open(last_msg, encoding="utf-8").read()
        else:
            raw = out
    text = _strip_thinking(raw.strip())
    return text, _cli_stats(text, time.time() - t0)


def _cli_stats(text: str, elapsed: float) -> dict:
    # Subprocess CLIs don't report token counts; estimate so the UI can show
    # an approximate throughput. loop.py guards every stats access.
    tokens = len(text.split())
    return {"tokens": tokens} if tokens else {}


def call_agent(agent: dict, prompt: str, workdir: str | None = None) -> tuple[str, dict]:
    """Send prompt to agent. Returns (response_text, stats_dict).

    workdir is honored only by CLI agent types: when provided the subprocess
    runs in that directory with write tools enabled (writer role); otherwise it
    runs sandboxed and read-only (generation only).
    """
    agent_type = agent.get("type", "ollama").lower()
    if agent_type == "ollama":
        return _call_ollama(agent, prompt)
    elif agent_type == "openai":
        return _call_openai(agent, prompt)
    elif agent_type == "claude_cli":
        return _call_claude_cli(agent, prompt, workdir)
    elif agent_type == "codex_cli":
        return _call_codex_cli(agent, prompt, workdir)
    else:
        raise ValueError(f"Unknown agent type: {agent_type!r}")


# ---------------------------------------------------------------------------
# Rollcall
# ---------------------------------------------------------------------------


_PING_PROMPT = "Reply with the single word: ready"

_PING_TIMEOUT = 60


def _ping_ollama(agent: dict) -> str:
    """Returns 'ready', 'no_model', or 'unreachable' by doing a real generation."""
    url = agent["url"].rstrip("/")
    body: dict = {
        "model": agent["model"],
        "prompt": _PING_PROMPT,
        "stream": False,
        "options": {"num_predict": 8},
    }
    try:
        resp = requests.post(f"{url}/api/generate", json=body, timeout=_PING_TIMEOUT)
        if resp.status_code == 404:
            return "no_model"
        if resp.status_code != 200:
            return "unreachable"
        return "ready"
    except Exception:
        return "unreachable"


def _ping_openai(agent: dict) -> str:
    """Returns 'ready', 'no_model', or 'unreachable' by doing a real generation."""
    url = agent["url"].rstrip("/")
    headers: dict[str, str] = {}
    if api_key := _resolve_api_key(agent):
        headers["Authorization"] = f"Bearer {api_key}"
    body: dict = {
        "model": agent["model"],
        "messages": [{"role": "user", "content": _PING_PROMPT}],
        "max_tokens": 8,
        "stream": False,
    }
    try:
        resp = requests.post(
            f"{url}/v1/chat/completions", json=body, headers=headers, timeout=_PING_TIMEOUT
        )
        if resp.status_code == 404:
            return "no_model"
        if resp.status_code != 200:
            return "unreachable"
        return "ready"
    except Exception:
        return "unreachable"


def _ping_cli(agent: dict) -> str:
    """CLI agents are 'ready' if their binary is on PATH (auth checked lazily)."""
    binary = _CLI_BINARIES[agent["type"].lower()]
    return "ready" if shutil.which(binary) else "unreachable"


def ping_agent(agent: dict) -> str:
    """Returns 'ready', 'no_model', or 'unreachable'."""
    agent_type = agent.get("type", "ollama").lower()
    if agent_type == "ollama":
        return _ping_ollama(agent)
    elif agent_type in _CLI_BINARIES:
        return _ping_cli(agent)
    return _ping_openai(agent)


def rollcall(agents: list[dict]) -> tuple[list[dict], list[dict], dict[str, str]]:
    """Ping all configured agents in parallel. Returns (active, missing, statuses)."""
    results: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=max(len(agents), 1)) as ex:
        futures = {ex.submit(ping_agent, a): a["id"] for a in agents}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()

    # Preserve original config order
    active = [a for a in agents if results.get(a["id"]) == "ready"]
    missing = [a for a in agents if results.get(a["id"]) != "ready"]
    return active, missing, results


# ---------------------------------------------------------------------------
# Parallel dispatch
# ---------------------------------------------------------------------------

def call_all_parallel(
    agents: list[dict],
    prompt_fn: Callable[[dict], str],
) -> dict[str, "str | Exception"]:
    """
    Call every agent in parallel.
    prompt_fn receives the agent dict and returns the prompt string.
    Returns {agent_id: response_text_or_exception}.
    """

    def _call(agent: dict) -> tuple[str, "str | Exception"]:
        try:
            return agent["id"], call_agent(agent, prompt_fn(agent))
        except Exception as exc:
            return agent["id"], exc

    results: dict[str, "str | Exception"] = {}
    with ThreadPoolExecutor(max_workers=max(len(agents), 1)) as ex:
        futures = [ex.submit(_call, a) for a in agents]
        for fut in as_completed(futures):
            aid, result = fut.result()
            results[aid] = result

    return results

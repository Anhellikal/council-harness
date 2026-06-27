"""Tests for subprocess-backed CLI agents (claude_cli / codex_cli).

All subprocess calls are mocked — no real CLI is invoked. Covers command
construction (generation vs. write mode), output parsing, error handling,
ping-by-PATH, and the cli_can_write capability gate.
"""
import subprocess
import types

import pytest

from council import agent as agent_mod
from council.agent import call_agent, cli_can_write, ping_agent


def _completed(stdout="", returncode=0, stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


# ---------------------------------------------------------------------------
# cli_can_write
# ---------------------------------------------------------------------------

class TestCliCanWrite:
    def test_claude_cli_with_flag(self):
        assert cli_can_write({"type": "claude_cli", "can_write": True}) is True

    def test_codex_cli_with_flag(self):
        assert cli_can_write({"type": "codex_cli", "can_write": True}) is True

    def test_cli_without_flag(self):
        assert cli_can_write({"type": "claude_cli"}) is False

    def test_non_cli_type_ignored(self):
        assert cli_can_write({"type": "ollama", "can_write": True}) is False


# ---------------------------------------------------------------------------
# claude_cli
# ---------------------------------------------------------------------------

class TestClaudeCli:
    def test_generation_mode_disables_tools_and_sandboxes(self, monkeypatch):
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["cwd"] = kwargs["cwd"]
            captured["stdin"] = kwargs["input"]
            return _completed(stdout="def f(): pass\n")

        monkeypatch.setattr(subprocess, "run", fake_run)
        text, stats = call_agent({"id": "claude", "type": "claude_cli", "model": "sonnet"}, "do it")

        assert text == "def f(): pass"
        assert captured["cmd"][:2] == ["claude", "-p"]
        assert "--model" in captured["cmd"] and "sonnet" in captured["cmd"]
        assert "--allowedTools" in captured["cmd"]  # tools disabled in generation mode
        assert captured["stdin"] == "do it"  # prompt via stdin, not positional
        assert captured["cmd"][-1] != "do it"
        # sandboxed: not run in the project dir
        assert captured["cwd"] not in (".", None)
        assert stats == {"tokens": 3}

    def test_write_mode_enables_tools_and_uses_workdir(self, monkeypatch, tmp_path):
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["cwd"] = kwargs["cwd"]
            return _completed(stdout="wrote files")

        monkeypatch.setattr(subprocess, "run", fake_run)
        call_agent({"id": "claude", "type": "claude_cli"}, "do it", workdir=str(tmp_path))

        assert "--allowedTools" not in captured["cmd"]  # tools enabled for writing
        assert "--permission-mode" in captured["cmd"]   # auto-accept writes in -p mode
        assert "acceptEdits" in captured["cmd"]
        assert captured["cwd"] == str(tmp_path)

    def test_nonzero_exit_raises(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed(returncode=2, stderr="boom"))
        with pytest.raises(RuntimeError, match="claude CLI exited 2"):
            call_agent({"id": "claude", "type": "claude_cli"}, "x")

    def test_timeout_raises(self, monkeypatch):
        def fake_run(*a, **k):
            raise subprocess.TimeoutExpired(cmd="claude", timeout=1)

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(RuntimeError, match="timed out"):
            call_agent({"id": "claude", "type": "claude_cli", "timeout": 1}, "x")

    def test_thinking_stripped(self, monkeypatch):
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **k: _completed(stdout="<think>plan</think>\nfinal answer"),
        )
        text, _ = call_agent({"id": "claude", "type": "claude_cli"}, "x")
        assert text == "final answer"


# ---------------------------------------------------------------------------
# codex_cli
# ---------------------------------------------------------------------------

class TestCodexCli:
    def test_reads_last_message_file(self, monkeypatch):
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["stdin"] = kwargs["input"]
            # codex writes its final message to the -o path
            out_path = cmd[cmd.index("-o") + 1]
            with open(out_path, "w") as f:
                f.write("final code")
            return _completed(stdout="event log noise")

        monkeypatch.setattr(subprocess, "run", fake_run)
        text, stats = call_agent({"id": "codex", "type": "codex_cli"}, "build")

        assert text == "final code"  # from -o file, not stdout
        assert "--skip-git-repo-check" in captured["cmd"]
        assert "read-only" in captured["cmd"]  # generation mode sandbox
        assert captured["stdin"] == "build"  # prompt via stdin
        assert stats == {"tokens": 2}

    def test_write_mode_sandbox_and_workdir(self, monkeypatch, tmp_path):
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["cwd"] = kwargs["cwd"]
            return _completed(stdout="done")

        monkeypatch.setattr(subprocess, "run", fake_run)
        call_agent({"id": "codex", "type": "codex_cli"}, "build", workdir=str(tmp_path))

        assert "workspace-write" in captured["cmd"]
        assert captured["cwd"] == str(tmp_path)

    def test_falls_back_to_stdout_when_no_file(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed(stdout="stdout answer"))
        text, _ = call_agent({"id": "codex", "type": "codex_cli"}, "x")
        assert text == "stdout answer"


# ---------------------------------------------------------------------------
# ping
# ---------------------------------------------------------------------------

class TestPingCli:
    def test_ready_when_binary_present(self, monkeypatch):
        monkeypatch.setattr(agent_mod.shutil, "which", lambda b: "/usr/bin/" + b)
        assert ping_agent({"id": "claude", "type": "claude_cli"}) == "ready"
        assert ping_agent({"id": "codex", "type": "codex_cli"}) == "ready"

    def test_unreachable_when_binary_missing(self, monkeypatch):
        monkeypatch.setattr(agent_mod.shutil, "which", lambda b: None)
        assert ping_agent({"id": "claude", "type": "claude_cli"}) == "unreachable"
